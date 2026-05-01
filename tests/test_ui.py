from fastapi.testclient import TestClient

from app.main import create_app


def test_ui_index_is_served() -> None:
    client = TestClient(create_app())

    response = client.get("/")

    assert response.status_code == 200
    assert "AI Knowledge Base Assistant" in response.text
    assert "Waiting for the answer" in response.text
    assert "/static/app.js" in response.text
