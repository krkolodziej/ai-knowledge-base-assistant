import pytest

from app.services.embedding_service import EmbeddingService, EmbeddingServiceError


class FakeOllamaClient:
    def __init__(self, embeddings: list[list[float]]) -> None:
        self.embeddings = embeddings

    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        return self.embeddings


def test_embedding_service_returns_embeddings_with_expected_dimension() -> None:
    service = EmbeddingService(
        client=FakeOllamaClient([[0.1, 0.2], [0.3, 0.4]]),
        model="test-embed",
        expected_dimension=2,
    )

    embeddings = service.embed_texts(["first", "second"])

    assert embeddings == [[0.1, 0.2], [0.3, 0.4]]


def test_embedding_service_rejects_wrong_embedding_dimension() -> None:
    service = EmbeddingService(
        client=FakeOllamaClient([[0.1, 0.2, 0.3]]),
        model="test-embed",
        expected_dimension=2,
    )

    with pytest.raises(EmbeddingServiceError):
        service.embed_texts(["first"])
