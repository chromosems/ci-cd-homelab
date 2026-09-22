from fastapi import FastAPI

app = FastAPI()


@app.get("/add/{a}/{b}")
def add(a: int, b: int) -> int:
    return a + b


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
