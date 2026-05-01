import uuid
from dataclasses import dataclass

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.models.document_chunk import DocumentChunk


@dataclass(frozen=True)
class SimilarChunk:
    chunk: DocumentChunk
    distance: float


class ChunkRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def add_many(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        self.db.add_all(chunks)
        self.db.flush()
        return chunks

    def delete_for_document(self, document_id: uuid.UUID) -> None:
        statement = delete(DocumentChunk).where(DocumentChunk.document_id == document_id)
        self.db.execute(statement)

    def list_for_document(self, document_id: uuid.UUID) -> list[DocumentChunk]:
        statement = (
            select(DocumentChunk)
            .where(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.chunk_index.asc())
        )
        return list(self.db.scalars(statement).all())

    def search_similar(
        self,
        query_embedding: list[float],
        limit: int,
    ) -> list[SimilarChunk]:
        distance = DocumentChunk.embedding.cosine_distance(query_embedding)
        statement = (
            select(DocumentChunk, distance.label("distance"))
            .where(DocumentChunk.embedding.is_not(None))
            .order_by(distance.asc())
            .limit(limit)
        )

        rows = self.db.execute(statement).all()
        return [
            SimilarChunk(chunk=chunk, distance=float(chunk_distance))
            for chunk, chunk_distance in rows
        ]
