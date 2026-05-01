import pytest

from app.services.ollama_client import OllamaClient, OllamaClientError


def test_ollama_client_returns_empty_list_for_empty_embedding_input() -> None:
    client = OllamaClient(base_url="http://localhost:11434", timeout_seconds=1)

    embeddings = client.embed(model="test-embed", texts=[])

    assert embeddings == []


def test_ollama_client_parses_embedding_response() -> None:
    client = OllamaClient(base_url="http://localhost:11434", timeout_seconds=1)

    embeddings = client._parse_embeddings(
        {
            "embeddings": [
                [1, "2.5", 3.0],
            ]
        }
    )

    assert embeddings == [[1.0, 2.5, 3.0]]


def test_ollama_client_rejects_embedding_response_without_embeddings() -> None:
    client = OllamaClient(base_url="http://localhost:11434", timeout_seconds=1)

    with pytest.raises(OllamaClientError):
        client._parse_embeddings({"model": "test-embed"})


def test_ollama_client_rejects_invalid_embedding_item() -> None:
    client = OllamaClient(base_url="http://localhost:11434", timeout_seconds=1)

    with pytest.raises(OllamaClientError):
        client._parse_embeddings({"embeddings": ["not-a-vector"]})


def test_ollama_client_parses_generation_response() -> None:
    client = OllamaClient(base_url="http://localhost:11434", timeout_seconds=1)

    answer = client._parse_generation({"response": "  Generated answer.  "})

    assert answer == "Generated answer."


def test_ollama_client_rejects_generation_response_without_text() -> None:
    client = OllamaClient(base_url="http://localhost:11434", timeout_seconds=1)

    with pytest.raises(OllamaClientError):
        client._parse_generation({"done": True})
