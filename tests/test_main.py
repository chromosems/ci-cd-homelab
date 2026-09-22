from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_add_endpoint() -> None:
    response = client.get("/add/2/3")
    assert response.status_code == 200
    assert response.json() == 5


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
