import uuid
from dataclasses import dataclass
from typing import Protocol

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.repositories.chunk_repository import ChunkRepository, SimilarChunk
from app.services.embedding_service import EmbeddingService, EmbeddingServiceError


class RetrievalError(Exception):
    pass


class RetrievalEmbeddingError(RetrievalError):
    pass


class ChunkSearchRepository(Protocol):
    def search_similar(
        self,
        query_embedding: list[float],
        limit: int,
    ) -> list[SimilarChunk]:
        pass


@dataclass(frozen=True)
class RetrievedChunk:
    document_id: uuid.UUID
    chunk_id: uuid.UUID
    chunk_index: int
    content: str
    distance: float
    similarity: float


@dataclass(frozen=True)
class RetrievalResult:
    question: str
    chunks: list[RetrievedChunk]
    embedding_model: str


class RetrievalService:
    def __init__(
        self,
        db: Session,
        embedding_service: EmbeddingService,
        chunk_repository: ChunkSearchRepository | None = None,
    ) -> None:
        self.db = db
        self.embedding_service = embedding_service
        self.chunks = chunk_repository or ChunkRepository(db)

    def retrieve(self, question: str, top_k: int) -> RetrievalResult:
        try:
            question_embedding = self.embedding_service.embed_texts([question])[0]
        except EmbeddingServiceError as exc:
            raise RetrievalEmbeddingError(
                "Could not generate question embedding with Ollama."
            ) from exc

        try:
            similar_chunks = self.chunks.search_similar(
                query_embedding=question_embedding,
                limit=top_k,
            )
        except SQLAlchemyError as exc:
            raise RetrievalError("Could not retrieve similar chunks.") from exc

        return RetrievalResult(
            question=question,
            chunks=[
                RetrievedChunk(
                    document_id=item.chunk.document_id,
                    chunk_id=item.chunk.id,
                    chunk_index=item.chunk.chunk_index,
                    content=item.chunk.content,
                    distance=item.distance,
                    similarity=1.0 - item.distance,
                )
                for item in similar_chunks
            ],
            embedding_model=self.embedding_service.model,
        )
