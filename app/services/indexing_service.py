import uuid
from dataclasses import dataclass

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.repositories.chunk_repository import ChunkRepository
from app.repositories.document_repository import DocumentRepository
from app.services.chunking_service import ChunkingService
from app.services.embedding_service import EmbeddingService, EmbeddingServiceError


class DocumentIndexingError(Exception):
    pass


class DocumentNotFoundError(DocumentIndexingError):
    pass


class DocumentEmbeddingError(DocumentIndexingError):
    pass


@dataclass(frozen=True)
class IndexingResult:
    document: Document
    chunks_indexed: int
    embedding_model: str


class IndexingService:
    def __init__(
        self,
        db: Session,
        chunking_service: ChunkingService,
        embedding_service: EmbeddingService,
    ) -> None:
        self.db = db
        self.documents = DocumentRepository(db)
        self.chunks = ChunkRepository(db)
        self.chunking_service = chunking_service
        self.embedding_service = embedding_service

    def index_document(self, document_id: uuid.UUID) -> IndexingResult:
        document = self.documents.get_by_id(document_id)
        if document is None:
            raise DocumentNotFoundError("Document not found.")

        text_chunks = self.chunking_service.split_text(document.content)
        if not text_chunks:
            raise DocumentIndexingError("Document does not contain indexable text.")

        try:
            embeddings = self.embedding_service.embed_texts(
                [text_chunk.content for text_chunk in text_chunks]
            )
        except EmbeddingServiceError as exc:
            raise DocumentEmbeddingError(
                f"Could not generate embeddings with Ollama. {exc}"
            ) from exc

        chunk_models = [
            DocumentChunk(
                document_id=document.id,
                chunk_index=text_chunk.index,
                content=text_chunk.content,
                embedding=embedding,
                chunk_metadata=text_chunk.metadata,
            )
            for text_chunk, embedding in zip(text_chunks, embeddings, strict=True)
        ]

        try:
            self.chunks.delete_for_document(document.id)
            self.chunks.add_many(chunk_models)
            document.status = "indexed"
            self.db.commit()
            self.db.refresh(document)
        except SQLAlchemyError as exc:
            self.db.rollback()
            raise DocumentIndexingError("Could not save document chunks.") from exc

        return IndexingResult(
            document=document,
            chunks_indexed=len(chunk_models),
            embedding_model=self.embedding_service.model,
        )
