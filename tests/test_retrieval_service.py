import uuid
from types import SimpleNamespace

import pytest
from sqlalchemy.exc import SQLAlchemyError

from app.repositories.chunk_repository import SimilarChunk
from app.services.embedding_service import EmbeddingServiceError
from app.services.retrieval_service import (
    RetrievalEmbeddingError,
    RetrievalError,
    RetrievalService,
)


class FakeEmbeddingService:
    model = "test-embed"

    def __init__(
        self,
        embedding: list[float] | None = None,
        *,
        raise_error: bool = False,
    ) -> None:
        self.embedding = embedding or [0.1, 0.2]
        self.raise_error = raise_error
        self.texts: list[str] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if self.raise_error:
            raise EmbeddingServiceError("Embedding failed.")
        self.texts = texts
        return [self.embedding]


class FakeChunkRepository:
    def __init__(
        self,
        *,
        results: list[SimilarChunk] | None = None,
        raise_error: bool = False,
    ) -> None:
        self.results = results
        self.raise_error = raise_error
        self.query_embedding: list[float] | None = None
        self.limit: int | None = None

    def search_similar(
        self,
        query_embedding: list[float],
        limit: int,
    ) -> list[SimilarChunk]:
        if self.raise_error:
            raise SQLAlchemyError("Database failed.")

        self.query_embedding = query_embedding
        self.limit = limit
        if self.results is not None:
            return self.results

        return [
            SimilarChunk(
                chunk=SimpleNamespace(
                    id=uuid.UUID("22222222-2222-2222-2222-222222222222"),
                    document_id=uuid.UUID("11111111-1111-1111-1111-111111111111"),
                    chunk_index=3,
                    content="RAG uses retrieval before generation.",
                ),
                distance=0.25,
            )
        ]


def test_retrieval_service_embeds_question_and_returns_ranked_chunks() -> None:
    embedding_service = FakeEmbeddingService([0.3, 0.4])
    chunk_repository = FakeChunkRepository()
    service = RetrievalService(
        db=SimpleNamespace(),
        embedding_service=embedding_service,
        chunk_repository=chunk_repository,
    )

    result = service.retrieve("What is RAG?", top_k=4)

    assert embedding_service.texts == ["What is RAG?"]
    assert chunk_repository.query_embedding == [0.3, 0.4]
    assert chunk_repository.limit == 4
    assert result.embedding_model == "test-embed"
    assert result.chunks[0].content == "RAG uses retrieval before generation."
    assert result.chunks[0].distance == 0.25
    assert result.chunks[0].similarity == 0.75


def test_retrieval_service_wraps_embedding_errors() -> None:
    service = RetrievalService(
        db=SimpleNamespace(),
        embedding_service=FakeEmbeddingService(raise_error=True),
        chunk_repository=FakeChunkRepository(),
    )

    with pytest.raises(RetrievalEmbeddingError):
        service.retrieve("What is RAG?", top_k=4)


def test_retrieval_service_returns_empty_result_when_no_chunks_match() -> None:
    service = RetrievalService(
        db=SimpleNamespace(),
        embedding_service=FakeEmbeddingService([0.3, 0.4]),
        chunk_repository=FakeChunkRepository(results=[]),
    )

    result = service.retrieve("What is RAG?", top_k=4)

    assert result.chunks == []
    assert result.embedding_model == "test-embed"


def test_retrieval_service_wraps_database_errors() -> None:
    service = RetrievalService(
        db=SimpleNamespace(),
        embedding_service=FakeEmbeddingService([0.3, 0.4]),
        chunk_repository=FakeChunkRepository(raise_error=True),
    )

    with pytest.raises(RetrievalError):
        service.retrieve("What is RAG?", top_k=4)
