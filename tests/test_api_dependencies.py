from types import SimpleNamespace

from app.api.dependencies import (
    get_chunking_service,
    get_embedding_service,
    get_indexing_service,
    get_ollama_client,
    get_rag_service,
    get_retrieval_service,
)
from app.core.config import Settings
from app.services.chunking_service import ChunkingService
from app.services.embedding_service import EmbeddingService
from app.services.indexing_service import IndexingService
from app.services.ollama_client import OllamaClient
from app.services.rag_service import RagService
from app.services.retrieval_service import RetrievalService


def make_settings() -> Settings:
    return Settings(
        ollama_base_url="http://localhost:11434/",
        ollama_request_timeout_seconds=5,
        ollama_embedding_model="test-embed",
        ollama_chat_model="test-chat",
        embedding_dimension=3,
        chunk_size_chars=100,
        chunk_overlap_chars=10,
    )


def test_dependencies_build_ollama_client_from_settings() -> None:
    client = get_ollama_client(make_settings())

    assert isinstance(client, OllamaClient)
    assert client.base_url == "http://localhost:11434"
    assert client.timeout_seconds == 5


def test_dependencies_build_embedding_service_from_settings() -> None:
    settings = make_settings()
    client = get_ollama_client(settings)

    service = get_embedding_service(settings=settings, ollama_client=client)

    assert isinstance(service, EmbeddingService)
    assert service.client is client
    assert service.model == "test-embed"
    assert service.expected_dimension == 3


def test_dependencies_build_chunking_service_from_settings() -> None:
    service = get_chunking_service(make_settings())

    assert isinstance(service, ChunkingService)
    assert service.chunk_size_chars == 100
    assert service.chunk_overlap_chars == 10


def test_dependencies_build_application_services() -> None:
    settings = make_settings()
    db = SimpleNamespace()
    ollama_client = get_ollama_client(settings)
    embedding_service = get_embedding_service(
        settings=settings,
        ollama_client=ollama_client,
    )
    chunking_service = get_chunking_service(settings)

    indexing_service = get_indexing_service(
        db=db,
        chunking_service=chunking_service,
        embedding_service=embedding_service,
    )
    retrieval_service = get_retrieval_service(
        db=db,
        embedding_service=embedding_service,
    )
    rag_service = get_rag_service(
        settings=settings,
        retrieval_service=retrieval_service,
        ollama_client=ollama_client,
    )

    assert isinstance(indexing_service, IndexingService)
    assert isinstance(retrieval_service, RetrievalService)
    assert isinstance(rag_service, RagService)
    assert rag_service.chat_model == "test-chat"
