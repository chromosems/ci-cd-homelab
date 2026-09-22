"""
Loan Decision Support API
FastAPI backend with JWT auth, guardrails, RAG pipeline, and advanced features.
"""

import io
import json
import os
import re
import sqlite3
import time
import uuid
from datetime import datetime, timedelta, timezone
from functools import lru_cache

import pandas as pd
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from fpdf import FPDF
from jose import JWTError, jwt
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from passlib.context import CryptContext
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SECRET_KEY = "demo-secret-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 120

CSV_PATH = os.environ.get("LOANS_CSV", "data/loans.csv")
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "qwen3:8b")  # Advanced 8B model from host
FALLBACK_MODEL = os.environ.get("FALLBACK_MODEL", "loan-qwen")  # Fallback
INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "data/faiss_index")
DB_PATH = os.environ.get("LOANASSIST_DB", "data/loanassist.db")
CACHE_SIZE = int(os.environ.get("CACHE_SIZE", "128"))

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

VALID_USERS = {
    "loanadmin": {
        "username": "loanadmin",
        "full_name": "Loan Administrator",
        "hashed_password": pwd_context.hash("securepassword123"),
        "disabled": False,
    }
}


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_user(db, username: str):
    return db.get(username)


def authenticate_user(db, username: str, password: str):
    user = get_user(db, username)
    if not user or not verify_password(password, user["hashed_password"]):
        return False
    return user


def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = get_user(VALID_USERS, username)
    if user is None:
        raise credentials_exception
    return user


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str
    content: str


class QueryRequest(BaseModel):
    question: str
    history: list[ChatMessage] | None = []


class QueryResponse(BaseModel):
    question: str
    answer: str  # Keep raw LLM output for debugging
    decision: str
    confidence: str
    reason: str
    cited_case_ids: list[int]
    risk_factors: list[str]
    comparison_summary: str
    context: list[str]
    mode: str


class Token(BaseModel):
    access_token: str
    token_type: str


class LoanDecision(BaseModel):
    decision: str
    confidence: str
    reason: str
    cited_case_ids: list[int]
    risk_factors: list[str]
    comparison_summary: str

    @staticmethod
    def normalize_decision(value: str) -> str:
        v = value.strip().upper()
        if v in ("APPROVE", "REJECT", "INSUFFICIENT_DATA"):
            return v.lower()
        return "neutral"

    @staticmethod
    def normalize_confidence(value: str) -> str:
        v = value.strip().lower()
        if v in ("high", "medium", "low"):
            return v
        return "medium"


class StatsResponse(BaseModel):
    total_queries: int
    rag_queries: int
    direct_queries: int
    approve_count: int
    reject_count: int
    avg_response_ms: int


class ScheduleRequest(BaseModel):
    loan_amount_ugx: int
    interest_rate_annual: float
    term_months: int


class ScheduleResponse(BaseModel):
    monthly_payment: int
    total_repayment: int
    total_interest: int
    schedule: list[dict]


class BatchResult(BaseModel):
    customer_id: int
    question: str
    answer: str
    decision: str
    mode: str


class FeedbackRequest(BaseModel):
    query: str
    decision: str
    confidence: str | None = ""
    reason: str | None = ""
    cited_case_ids: list[int] | None = []
    feedback: str  # 'up' or 'down'


class FeedbackResponse(BaseModel):
    status: str
    message: str


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
LOAN_KEYWORDS = [
    "loan",
    "credit",
    "approve",
    "reject",
    "income",
    "score",
    "mortgage",
    "personal",
    "business",
    "car",
    "default",
    "lending",
    "borrower",
    "application",
    "debt",
    "interest",
    "collateral",
    "risk",
    "underwriting",
    "principal",
    "repayment",
    "emi",
    "apr",
    "delinquent",
    "foreclosure",
    "guarantor",
    "co-sign",
    "dti",
    "lvr",
    "fico",
    "experian",
    "transunion",
    # Ugandan context
    "ugx",
    "uganda",
    "shilling",
    "mobile money",
    "sacco",
    "vsla",
    "agricultural",
    "education",
    "boda boda",
    "market",
    "trader",
    "civil servant",
    "teacher",
    "farmer",
    "self-employed",
    # Repayment schedule
    "monthly payment",
    "total repayment",
    "interest rate",
    "term",
    "months",
    "payment history",
    "missed payment",
    "defaulted after",
    "repayment schedule",
    "amortization",
    "installment",
    "schedule",
    "payment plan",
    "arrears",
]


def is_loan_query(question: str) -> bool:
    q = question.lower()
    if any(kw in q for kw in LOAN_KEYWORDS):
        return True
    return bool(re.search(r"\d", q) or any(sym in q for sym in ["$", "£", "€", "ugx", "shilling"]))


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
SYSTEM_IDENTITY = """You are LoanAssist, a Ugandan loan decision support AI. Your ONLY source of knowledge is the historical loan cases provided below.

MANDATORY INSTRUCTIONS:
1. You MUST base your decision ONLY on the patterns in the historical cases shown below
2. You MUST respond ONLY in valid JSON. Do not use markdown code blocks. Do not write any text before or after the JSON.
3. Decision must be exactly "APPROVE", "REJECT", or "INSUFFICIENT_DATA"
4. Confidence must be exactly "high", "medium", or "low"
5. Give exactly ONE reason citing specific similar cases from the historical data
6. Compare the applicant to similar cases: credit score, income, loan amount, Debt to Income Ratio, employment, collateral, payment history
7. If similar cases were approved, recommend APPROVE. If rejected/defaulted, recommend REJECT.
8. NEVER use the abbreviation "DTI". Always write "Debt to Income Ratio" in full.
9. NEVER give generic advice. NEVER mention debt management, budgeting, or financial planning.

RESPONSE FORMAT — output ONLY this exact JSON structure. Write the JSON across multiple lines:
{
  "decision": "APPROVE",
  "confidence": "high",
  "reason": "Customer 391 had similar income and was approved.",
  "cited_case_ids": [391],
  "risk_factors": ["Large loan amount relative to income"],
  "comparison_summary": "Applicant profile matches approved historical cases."
}"""


def build_rag_prompt(context: str, question: str) -> str:
    return (
        f"{SYSTEM_IDENTITY}\n\n"
        f"HISTORICAL LOAN CASES FROM YOUR DATABASE (this is your ONLY knowledge):\n"
        f"{'=' * 60}\n{context}\n{'=' * 60}\n\n"
        f"NEW LOAN APPLICATION TO EVALUATE:\n{question}\n\n"
        f"Based ONLY on the historical cases above, respond with valid JSON following the RESPONSE FORMAT above.\n"
        f"REMEMBER: Do not use markdown. Do not write 'DTI' — always write 'Debt to Income Ratio'."
    )


def build_direct_prompt(question: str) -> str:
    return (
        f"{SYSTEM_IDENTITY}\n\n"
        f"LOAN APPLICATION:\n{question}\n\n"
        f"Respond with valid JSON following the RESPONSE FORMAT above.\n"
        f"REMEMBER: Do not use markdown. Do not write 'DTI' — always write 'Debt to Income Ratio'."
    )


# ---------------------------------------------------------------------------
# RAG pipeline setup
# ---------------------------------------------------------------------------
print("[Startup] Loading loan data...")
df = pd.read_csv(CSV_PATH)

records = []
for _, row in df.iterrows():
    season_part = f", season: {row['season']}" if pd.notna(row["season"]) else ""
    record = (
        f"Customer {row['customer_id']}, age {row['age']}, income UGX {row['income_ugx']:,}, "
        f"credit score {row['credit_score']}, loan UGX {row['loan_amount_ugx']:,} "
        f"({row['loan_type']}, {row['loan_purpose']}, {row['loan_term_months']} months at {row['interest_rate_annual']}% APR), "
        f"monthly payment UGX {row['monthly_payment_ugx']:,}, total repayment UGX {row['total_repayment_ugx']:,}, "
        f"Debt to Income Ratio {row['debt_to_income_ratio']}%, existing debt UGX {row['existing_debt_ugx']:,}, "
        f"employment: {row['employment_status']}, region: {row['district']} ({row['region']}), "
        f"collateral: {row['collateral']}, payment method: {row['payment_method']}, "
        f"payments made: {row['payments_made']}/{row['loan_term_months']}, missed: {row['payments_missed']}, "
        f"history: {row['payment_history']}, schedule: {row['repayment_schedule']}, "
        f"status: {row['status']}."
    )
    records.append(record)

docs = []
for record, row in zip(records, df.itertuples()):
    docs.append(
        Document(
            page_content=record,
            metadata={
                "customer_id": row.customer_id,
                "status": row.status,
                "loan_type": row.loan_type,
                "region": row.region,
                "district": row.district,
            },
        )
    )

# ---------------------------------------------------------------------------
# FAISS index: load from disk if available, otherwise build and save
# ---------------------------------------------------------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})

if os.path.exists(INDEX_PATH):
    print(f"[Startup] Loading FAISS index from {INDEX_PATH}...")
    db = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    print("[Startup] FAISS index loaded.")
else:
    print("[Startup] Building FAISS index...")
    db = FAISS.from_documents(docs, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    os.makedirs(os.path.dirname(INDEX_PATH) or ".", exist_ok=True)
    db.save_local(INDEX_PATH)
    print(f"[Startup] FAISS index built and saved to {INDEX_PATH}.")

# ---------------------------------------------------------------------------
# Hybrid retrieval: BM25 + FAISS
# ---------------------------------------------------------------------------
try:
    import numpy as np
    from rank_bm25 import BM25Okapi

    tokenized_docs = [doc.page_content.lower().split() for doc in docs]
    bm25 = BM25Okapi(tokenized_docs)

    def bm25_retrieve(query: str, k: int = 5):
        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[-k:][::-1]
        return [docs[i] for i in top_indices]

    def hybrid_retrieve(query: str, k: int = 3):
        faiss_results = retriever.invoke(query)[:k]
        bm25_results = bm25_retrieve(query, k)
        seen = set()
        merged = []
        for doc in faiss_results + bm25_results:
            cid = doc.metadata.get("customer_id")
            if cid not in seen:
                seen.add(cid)
                merged.append(doc)
        return merged[:k]

    print("[Startup] Hybrid retrieval (FAISS + BM25) enabled.")
except ImportError:
    print("[Startup] rank-bm25 not installed. Using FAISS only.")

    def hybrid_retrieve(query: str, k: int = 3):
        return retriever.invoke(query)[:k]


OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://192.168.222.1:11434")

# Try primary model, fallback if unavailable
llm = None
active_model = MODEL_NAME

try:
    print(f"[Startup] Loading Ollama model ({MODEL_NAME}) from {OLLAMA_BASE_URL}...")
    llm = OllamaLLM(
        model=MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
        temperature=0.3,
        num_ctx=8192,
        num_predict=2000,
    )
    # Test invocation
    llm.invoke("Hi")
    print(f"[Startup] Model {MODEL_NAME} loaded successfully.")
except Exception as e:
    print(f"[Startup] Warning: Could not load {MODEL_NAME}: {e}")
    print(f"[Startup] Falling back to {FALLBACK_MODEL}...")
    try:
        llm = OllamaLLM(
            model=FALLBACK_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.3,
            num_ctx=512,
        )
        active_model = FALLBACK_MODEL
        print(f"[Startup] Fallback model {FALLBACK_MODEL} loaded.")
    except Exception as e2:
        print(f"[Startup] Error: Could not load any model: {e2}")
        raise

assert llm is not None


# ---------------------------------------------------------------------------
# Caching wrapper for LLM calls
# ---------------------------------------------------------------------------
@lru_cache(maxsize=CACHE_SIZE)
def _cached_llm_invoke(prompt_hash: str) -> str:
    """Cached LLM call. Accepts hash string to keep args hashable."""
    return llm.invoke(prompt_hash)  # type: ignore[union-attr]


def invoke_llm(prompt: str) -> str:
    """Invoke LLM with in-memory caching."""
    result = _cached_llm_invoke(prompt)
    if not result or not result.strip():
        # Empty cached result — evict and retry once
        _cached_llm_invoke.cache_clear()
        result = _cached_llm_invoke(prompt)
    return result


# ---------------------------------------------------------------------------
# SQLite feedback database
# ---------------------------------------------------------------------------
def _init_db():
    """Initialize SQLite database with feedback table."""
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            decision TEXT NOT NULL,
            confidence TEXT,
            reason TEXT,
            cited_case_ids TEXT,
            officer_id TEXT,
            feedback TEXT CHECK(feedback IN ('up', 'down')),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id TEXT UNIQUE NOT NULL,
            query TEXT NOT NULL,
            decision TEXT NOT NULL,
            confidence TEXT,
            reason TEXT,
            cited_case_ids TEXT,
            risk_factors TEXT,
            context TEXT,
            mode TEXT,
            raw_response TEXT,
            response_time_ms INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()
    print(f"[Startup] Database initialized at {DB_PATH}")


_init_db()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def calculate_amortization(loan_amount, annual_rate, term_months):
    """Generate full amortization schedule."""
    monthly_rate = annual_rate / 100 / 12
    if monthly_rate == 0:
        monthly_payment = loan_amount / term_months
    else:
        monthly_payment = loan_amount * (monthly_rate * (1 + monthly_rate) ** term_months) / ((1 + monthly_rate) ** term_months - 1)

    monthly_payment = round(monthly_payment)
    total_repayment = monthly_payment * term_months
    total_interest = total_repayment - loan_amount

    schedule = []
    balance = loan_amount
    for month in range(1, term_months + 1):
        interest_payment = round(balance * monthly_rate)
        principal_payment = monthly_payment - interest_payment
        balance -= principal_payment
        balance = max(balance, 0)

        schedule.append({"month": month, "payment": monthly_payment, "principal": principal_payment, "interest": interest_payment, "balance": balance})

    return {"monthly_payment": monthly_payment, "total_repayment": total_repayment, "total_interest": total_interest, "schedule": schedule}


def generate_pdf_report(query, answer, context, mode, decision):
    """Generate a PDF loan decision report."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Title
    pdf.set_font("Arial", "B", 20)
    pdf.set_text_color(102, 126, 234)
    pdf.cell(0, 15, "Loan Decision Report", ln=True, align="C")
    pdf.ln(5)

    # Metadata
    pdf.set_font("Arial", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(0, 8, f"Report ID: {str(uuid.uuid4())[:8]}", ln=True)
    pdf.cell(0, 8, f"Mode: {mode.upper()}", ln=True)
    pdf.ln(5)

    # Decision
    pdf.set_font("Arial", "B", 14)
    if decision == "approve":
        pdf.set_text_color(21, 87, 36)
        pdf.cell(0, 12, "DECISION: APPROVE", ln=True)
    elif decision == "reject":
        pdf.set_text_color(114, 28, 36)
        pdf.cell(0, 12, "DECISION: REJECT", ln=True)
    else:
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 12, "DECISION: NEUTRAL", ln=True)
    pdf.ln(5)

    # Query
    pdf.set_font("Arial", "B", 12)
    pdf.set_text_color(50, 50, 50)
    pdf.cell(0, 10, "Loan Application Question:", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 8, query)
    pdf.ln(3)

    # Answer
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "AI Analysis & Recommendation:", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 8, answer)
    pdf.ln(5)

    # Context (if RAG)
    if context and mode == "rag":
        pdf.set_font("Arial", "B", 12)
        pdf.set_text_color(50, 50, 50)
        pdf.cell(0, 10, "RAG Retrieved Cases:", ln=True)
        pdf.set_font("Arial", "", 9)
        pdf.set_text_color(80, 80, 80)

        for i, ctx in enumerate(context[:3], 1):
            pdf.set_font("Arial", "B", 10)
            pdf.set_text_color(102, 126, 234)
            pdf.cell(0, 8, f"Case #{i}:", ln=True)
            pdf.set_font("Arial", "", 9)
            pdf.set_text_color(80, 80, 80)
            # Truncate if too long
            ctx_text = ctx[:500] + "..." if len(ctx) > 500 else ctx
            pdf.multi_cell(0, 6, ctx_text)
            pdf.ln(2)

    # Footer
    pdf.set_y(-30)
    pdf.set_font("Arial", "I", 8)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 10, "Generated by LoanAssist RAG System | This is a decision support tool, not a final lending decision.", ln=True, align="C")

    pdf_bytes = pdf.output(dest="S")
    if isinstance(pdf_bytes, bytearray):
        return bytes(pdf_bytes)
    return pdf_bytes.encode("latin-1")


def _strip_markdown(text: str) -> str:
    """Remove markdown code fences."""
    text = text.strip()
    for prefix in ["```json", "```"]:
        text = text.removeprefix(prefix)
    text = text.removesuffix("```")
    return text.strip()


def _post_process_reason(text: str) -> str:
    """Enforce 'Debt to Income Ratio' everywhere."""
    import re as _re

    # Word-boundary replace: DTI not preceded/followed by letters
    return _re.sub(r"\bDTI\b", "Debt to Income Ratio", text)


def parse_response(answer: str) -> dict:
    """Parse LLM response. Try JSON first, fall back to regex."""
    # Debug: log first 200 chars of raw response
    print(f"[DEBUG] Raw answer (first 200 chars): {answer[:200]!r}")

    clean = _strip_markdown(answer)

    # Attempt 1: direct json.loads + pydantic validation (on UNMODIFIED clean text)
    try:
        data = json.loads(clean)
        decision = LoanDecision.model_validate(data) if hasattr(LoanDecision, "model_validate") else LoanDecision.parse_obj(data)

        raw_decision = decision.decision.lower()
        # Normalize NEUTRAL -> reject (model should not return NEUTRAL per prompt)
        if raw_decision == "neutral":
            raw_decision = "reject"

        print(f"[DEBUG] JSON parsed successfully. Decision: {raw_decision}")
        return {
            "decision": raw_decision,
            "confidence": decision.confidence.lower(),
            "reason": _post_process_reason(decision.reason) or "No explanation provided.",
            "cited_case_ids": decision.cited_case_ids,
            "risk_factors": [_post_process_reason(r) for r in decision.risk_factors],
            "comparison_summary": _post_process_reason(decision.comparison_summary),
        }
    except Exception as e:
        print(f"[DEBUG] JSON parse failed: {e}")

    # Attempt 2: regex extract JSON object from anywhere in the text
    json_match = re.search(r"\{.*\}", clean, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            decision = LoanDecision.model_validate(data) if hasattr(LoanDecision, "model_validate") else LoanDecision.parse_obj(data)

            raw_decision = decision.decision.lower()
            if raw_decision == "neutral":
                raw_decision = "reject"

            print(f"[DEBUG] Regex JSON extract success. Decision: {raw_decision}")
            return {
                "decision": raw_decision,
                "confidence": decision.confidence.lower(),
                "reason": _post_process_reason(decision.reason) or "No explanation provided.",
                "cited_case_ids": decision.cited_case_ids,
                "risk_factors": [_post_process_reason(r) for r in decision.risk_factors],
                "comparison_summary": _post_process_reason(decision.comparison_summary),
            }
        except Exception as e2:
            print(f"[DEBUG] Regex JSON extract failed: {e2}")

    # Final fallback: regex parse for backward compatibility
    text = (answer or "").upper()
    if "REJECT" in text:
        fallback_decision = "reject"
    elif "APPROVE" in text:
        fallback_decision = "approve"
    elif "INSUFFICIENT" in text:
        fallback_decision = "reject"
    else:
        fallback_decision = "reject"  # Default to reject instead of neutral

    print(f"[DEBUG] Fallback parser used. Decision: {fallback_decision}")
    raw = (answer or "").strip()
    if not raw:
        reason_text = "The model returned an empty response. This may indicate the prompt was too long or the model timed out."
    else:
        reason_text = _post_process_reason(raw)
    return {
        "decision": fallback_decision,
        "confidence": "medium",
        "reason": reason_text,
        "cited_case_ids": [],
        "risk_factors": [],
        "comparison_summary": "",
    }


# ---------------------------------------------------------------------------
# Session stats
# ---------------------------------------------------------------------------
session_stats = {
    "total_queries": 0,
    "rag_queries": 0,
    "direct_queries": 0,
    "approve_count": 0,
    "reject_count": 0,
    "total_response_ms": 0,
}


print("[Startup] Ready.")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Loan Decision Support",
    description="Local LLM + RAG for loan decisions with advanced features",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(VALID_USERS, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": user["username"]}, expires_delta=access_token_expires)
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/query", response_model=QueryResponse)
async def query_llm(req: QueryRequest, current_user: dict = Depends(get_current_user)):
    start_ms = time.time() * 1000
    question = req.question.strip()

    if is_loan_query(question):
        retrieved_docs = hybrid_retrieve(question)
        context = [d.page_content for d in retrieved_docs]
        context_str = "\n".join(context)
        prompt = build_rag_prompt(context_str, question)
        answer = invoke_llm(prompt)
        mode = "rag"
        session_stats["rag_queries"] += 1
    else:
        context = []
        prompt = build_direct_prompt(question)
        answer = invoke_llm(prompt)
        mode = "direct"
        session_stats["direct_queries"] += 1

    session_stats["total_queries"] += 1
    elapsed_ms = int(time.time() * 1000 - start_ms)
    session_stats["total_response_ms"] += elapsed_ms

    parsed = parse_response(answer)
    if parsed["decision"] == "approve":
        session_stats["approve_count"] += 1
    elif parsed["decision"] == "reject":
        session_stats["reject_count"] += 1

    # Audit log
    request_id = str(uuid.uuid4())
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            """
            INSERT INTO audit_log (request_id, query, decision, confidence, reason, cited_case_ids, risk_factors, context, mode, raw_response, response_time_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                request_id,
                question,
                parsed["decision"],
                parsed["confidence"],
                parsed["reason"],
                json.dumps(parsed["cited_case_ids"]),
                json.dumps(parsed["risk_factors"]),
                json.dumps(context[:3]),
                mode,
                answer,
                elapsed_ms,
            ),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[Audit Warning] Could not write audit log: {e}")

    return QueryResponse(
        question=question,
        answer=answer,
        decision=parsed["decision"],
        confidence=parsed["confidence"],
        reason=parsed["reason"],
        cited_case_ids=parsed["cited_case_ids"],
        risk_factors=parsed["risk_factors"],
        comparison_summary=parsed["comparison_summary"],
        context=context,
        mode=mode,
    )


@app.post("/calculate-schedule", response_model=ScheduleResponse)
async def calculate_schedule(req: ScheduleRequest, current_user: dict = Depends(get_current_user)):
    """Calculate loan amortization schedule."""
    result = calculate_amortization(req.loan_amount_ugx, req.interest_rate_annual, req.term_months)
    return ScheduleResponse(**result)


@app.post("/export-decision")
async def export_decision(req: QueryRequest, current_user: dict = Depends(get_current_user)):
    """Generate PDF report for a loan decision."""
    # Process the query
    question = req.question.strip()

    if is_loan_query(question):
        retrieved_docs = hybrid_retrieve(question)
        context = [d.page_content for d in retrieved_docs]
        context_str = "\n".join(context)
        prompt = build_rag_prompt(context_str, question)
        answer = invoke_llm(prompt)
        mode = "rag"
    else:
        context = []
        prompt = build_direct_prompt(question)
        answer = invoke_llm(prompt)
        mode = "direct"

    parsed = parse_response(answer)
    decision = parsed["decision"]

    # Generate PDF
    pdf_bytes = generate_pdf_report(question, answer, context, mode, decision)

    filename = f"loan_decision_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.pdf"
    return StreamingResponse(io.BytesIO(pdf_bytes), media_type="application/pdf", headers={"Content-Disposition": f"attachment; filename={filename}"})


@app.post("/batch-process")
async def batch_process(file: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
    """Process a CSV file of loan applications."""
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

    results = []
    for _, row in df.iterrows():
        # Build question from CSV row
        question = (
            f"Should we approve a UGX {row.get('loan_amount_ugx', 0):,} "
            f"{row.get('loan_type', 'personal')} loan for a {row.get('age', 0)}-year-old "
            f"{row.get('employment_status', 'applicant')} with income UGX {row.get('income_ugx', 0):,}, "
            f"credit score {row.get('credit_score', 0)}, Debt to Income Ratio {row.get('debt_to_income_ratio', 0)}%, "
            f"and {row.get('collateral', 'no')} collateral?"
        )

        if is_loan_query(question):
            retrieved_docs = hybrid_retrieve(question)
            context = [d.page_content for d in retrieved_docs]
            context_str = "\n".join(context)
            prompt = build_rag_prompt(context_str, question)
            answer = invoke_llm(prompt)
            mode = "rag"
        else:
            context = []
            prompt = build_direct_prompt(question)
            answer = invoke_llm(prompt)
            mode = "direct"

        parsed = parse_response(answer)

        results.append(
            {
                "customer_id": int(row.get("customer_id", 0)),
                "question": question,
                "answer": answer,
                "decision": parsed["decision"].upper(),
                "confidence": parsed["confidence"],
                "reason": parsed["reason"],
                "cited_case_ids": parsed["cited_case_ids"],
                "risk_factors": parsed["risk_factors"],
                "mode": mode,
            }
        )

    return {"total_processed": len(results), "results": results}


@app.get("/health")
async def health():
    return {"status": "ok", "model": active_model, "dataset_size": len(df)}


@app.get("/stats", response_model=StatsResponse)
async def get_stats(current_user: dict = Depends(get_current_user)):
    avg_ms = session_stats["total_response_ms"] // session_stats["total_queries"] if session_stats["total_queries"] > 0 else 0
    return StatsResponse(
        total_queries=session_stats["total_queries"],
        rag_queries=session_stats["rag_queries"],
        direct_queries=session_stats["direct_queries"],
        approve_count=session_stats["approve_count"],
        reject_count=session_stats["reject_count"],
        avg_response_ms=avg_ms,
    )


@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(req: FeedbackRequest, current_user: dict = Depends(get_current_user)):
    """Record officer feedback (thumbs up/down) on a decision."""
    officer_id = current_user.get("username", "anonymous")
    if req.feedback not in ("up", "down"):
        raise HTTPException(status_code=400, detail="feedback must be 'up' or 'down'")

    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        INSERT INTO feedback (query, decision, confidence, reason, cited_case_ids, officer_id, feedback)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            req.query,
            req.decision,
            req.confidence,
            req.reason,
            json.dumps(req.cited_case_ids),
            officer_id,
            req.feedback,
        ),
    )
    conn.commit()
    conn.close()
    return {"status": "ok", "message": "Feedback recorded."}


@app.get("/feedback-summary")
async def feedback_summary(current_user: dict = Depends(get_current_user)):
    """Return aggregated feedback statistics."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute("SELECT feedback, COUNT(*) FROM feedback GROUP BY feedback")
    rows = cursor.fetchall()
    conn.close()
    summary = {"up": 0, "down": 0}
    for feedback_type, count in rows:
        summary[feedback_type] = count
    return summary


# Serve static dashboard
app.mount("/", StaticFiles(directory="static", html=True), name="static")
