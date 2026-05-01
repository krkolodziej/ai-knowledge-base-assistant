from typing import Annotated

from fastapi import Depends
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.db.session import get_db_session
from app.services.chunking_service import ChunkingService
from app.services.document_service import DocumentService
from app.services.embedding_service import EmbeddingService
from app.services.indexing_service import IndexingService
from app.services.ollama_client import OllamaClient
from app.services.rag_service import RagService
from app.services.retrieval_service import RetrievalService

DatabaseSession = Annotated[Session, Depends(get_db_session)]
AppSettings = Annotated[Settings, Depends(get_settings)]


def get_ollama_client(settings: AppSettings) -> OllamaClient:
    return OllamaClient(
        base_url=settings.ollama_base_url,
        timeout_seconds=settings.ollama_request_timeout_seconds,
    )


def get_embedding_service(
    settings: AppSettings,
    ollama_client: Annotated[OllamaClient, Depends(get_ollama_client)],
) -> EmbeddingService:
    return EmbeddingService(
        client=ollama_client,
        model=settings.ollama_embedding_model,
        expected_dimension=settings.embedding_dimension,
    )


def get_chunking_service(settings: AppSettings) -> ChunkingService:
    return ChunkingService(
        chunk_size_chars=settings.chunk_size_chars,
        chunk_overlap_chars=settings.chunk_overlap_chars,
    )


def get_document_service(db: DatabaseSession) -> DocumentService:
    return DocumentService(db)


def get_indexing_service(
    db: DatabaseSession,
    chunking_service: Annotated[ChunkingService, Depends(get_chunking_service)],
    embedding_service: Annotated[EmbeddingService, Depends(get_embedding_service)],
) -> IndexingService:
    return IndexingService(
        db=db,
        chunking_service=chunking_service,
        embedding_service=embedding_service,
    )


def get_retrieval_service(
    db: DatabaseSession,
    embedding_service: Annotated[EmbeddingService, Depends(get_embedding_service)],
) -> RetrievalService:
    return RetrievalService(db=db, embedding_service=embedding_service)


def get_rag_service(
    settings: AppSettings,
    retrieval_service: Annotated[RetrievalService, Depends(get_retrieval_service)],
    ollama_client: Annotated[OllamaClient, Depends(get_ollama_client)],
) -> RagService:
    return RagService(
        retrieval_service=retrieval_service,
        generation_client=ollama_client,
        chat_model=settings.ollama_chat_model,
    )
