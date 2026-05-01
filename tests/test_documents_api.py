import uuid
from datetime import UTC, datetime
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.api.dependencies import get_document_service, get_indexing_service
from app.main import create_app
from app.services.document_service import DocumentNotFoundError as DocumentServiceNotFoundError
from app.services.document_service import DocumentServiceError
from app.services.indexing_service import (
    DocumentEmbeddingError,
    DocumentIndexingError,
    DocumentNotFoundError,
    IndexingResult,
)


def make_document(**overrides: object) -> SimpleNamespace:
    data = {
        "id": uuid.UUID("11111111-1111-1111-1111-111111111111"),
        "title": "RAG notes",
        "content": "RAG connects retrieval with generation.",
        "content_type": "text/markdown",
        "status": "pending",
        "document_metadata": {"source": "notes.md"},
        "created_at": datetime(2026, 4, 28, 12, 0, tzinfo=UTC),
        "updated_at": datetime(2026, 4, 28, 12, 0, tzinfo=UTC),
    }
    data.update(overrides)
    return SimpleNamespace(**data)


class FakeDocumentService:
    def __init__(
        self,
        *,
        raise_delete_not_found: bool = False,
        raise_error: bool = False,
    ) -> None:
        self.raise_delete_not_found = raise_delete_not_found
        self.raise_error = raise_error
        self.documents = [make_document()]

    def create_document(self, payload: object) -> SimpleNamespace:
        if self.raise_error:
            raise DocumentServiceError("Could not save document.")
        return make_document(
            title=payload.title,
            content=payload.content,
            content_type=payload.content_type,
            document_metadata=payload.metadata,
        )

    def list_documents(self) -> list[SimpleNamespace]:
        if self.raise_error:
            raise DocumentServiceError("Could not list documents.")
        return self.documents

    def delete_document(self, document_id: uuid.UUID) -> None:
        if self.raise_delete_not_found:
            raise DocumentServiceNotFoundError("Document not found.")
        if self.raise_error:
            raise DocumentServiceError("Could not delete document.")


class FakeIndexingService:
    def __init__(
        self,
        *,
        raise_embedding_error: bool = False,
        raise_indexing_error: bool = False,
        raise_not_found: bool = False,
    ) -> None:
        self.raise_embedding_error = raise_embedding_error
        self.raise_indexing_error = raise_indexing_error
        self.raise_not_found = raise_not_found

    def index_document(self, document_id: uuid.UUID) -> IndexingResult:
        if self.raise_not_found:
            raise DocumentNotFoundError("Document not found.")
        if self.raise_embedding_error:
            raise DocumentEmbeddingError("Could not generate embeddings with Ollama.")
        if self.raise_indexing_error:
            raise DocumentIndexingError("Could not save document chunks.")

        document = make_document(id=document_id, status="indexed")
        return IndexingResult(
            document=document,
            chunks_indexed=2,
            embedding_model="nomic-embed-text",
        )


def create_test_client(
    service: FakeDocumentService | None = None,
    indexing_service: FakeIndexingService | None = None,
) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_document_service] = lambda: service or FakeDocumentService()
    app.dependency_overrides[get_indexing_service] = (
        lambda: indexing_service or FakeIndexingService()
    )
    return TestClient(app)


def test_create_document_returns_created_document() -> None:
    client = create_test_client()

    response = client.post(
        "/api/v1/documents",
        json={
            "title": "  RAG notes  ",
            "content": "  RAG connects retrieval with generation.  ",
            "content_type": "text/markdown",
            "metadata": {"source": "notes.md"},
        },
    )

    assert response.status_code == 201
    assert response.json() == {
        "id": "11111111-1111-1111-1111-111111111111",
        "title": "RAG notes",
        "content": "RAG connects retrieval with generation.",
        "content_type": "text/markdown",
        "status": "pending",
        "metadata": {"source": "notes.md"},
        "created_at": "2026-04-28T12:00:00Z",
        "updated_at": "2026-04-28T12:00:00Z",
    }


def test_list_documents_returns_summaries_without_full_content() -> None:
    client = create_test_client()

    response = client.get("/api/v1/documents")

    assert response.status_code == 200
    assert response.json() == {
        "items": [
            {
                "id": "11111111-1111-1111-1111-111111111111",
                "title": "RAG notes",
                "content_type": "text/markdown",
                "status": "pending",
                "created_at": "2026-04-28T12:00:00Z",
                "updated_at": "2026-04-28T12:00:00Z",
            }
        ],
        "total": 1,
    }


def test_create_document_rejects_blank_content() -> None:
    client = create_test_client()

    response = client.post(
        "/api/v1/documents",
        json={
            "title": "RAG notes",
            "content": "   ",
            "content_type": "text/plain",
        },
    )

    assert response.status_code == 422


def test_create_document_rejects_unsupported_content_type() -> None:
    client = create_test_client()

    response = client.post(
        "/api/v1/documents",
        json={
            "title": "RAG notes",
            "content": "Some content",
            "content_type": "application/pdf",
        },
    )

    assert response.status_code == 422


def test_index_document_returns_indexing_summary() -> None:
    client = create_test_client()

    response = client.post("/api/v1/documents/11111111-1111-1111-1111-111111111111/index")

    assert response.status_code == 200
    assert response.json() == {
        "document_id": "11111111-1111-1111-1111-111111111111",
        "status": "indexed",
        "chunks_indexed": 2,
        "embedding_model": "nomic-embed-text",
    }


def test_index_document_returns_404_when_document_does_not_exist() -> None:
    client = create_test_client(indexing_service=FakeIndexingService(raise_not_found=True))

    response = client.post("/api/v1/documents/11111111-1111-1111-1111-111111111111/index")

    assert response.status_code == 404


def test_create_document_returns_500_when_service_fails() -> None:
    client = create_test_client(service=FakeDocumentService(raise_error=True))

    response = client.post(
        "/api/v1/documents",
        json={
            "title": "RAG notes",
            "content": "Some content",
            "content_type": "text/plain",
        },
    )

    assert response.status_code == 500


def test_list_documents_returns_500_when_service_fails() -> None:
    client = create_test_client(service=FakeDocumentService(raise_error=True))

    response = client.get("/api/v1/documents")

    assert response.status_code == 500


def test_index_document_returns_502_when_embedding_fails() -> None:
    client = create_test_client(
        indexing_service=FakeIndexingService(raise_embedding_error=True)
    )

    response = client.post("/api/v1/documents/11111111-1111-1111-1111-111111111111/index")

    assert response.status_code == 502


def test_index_document_returns_500_when_indexing_fails() -> None:
    client = create_test_client(
        indexing_service=FakeIndexingService(raise_indexing_error=True)
    )

    response = client.post("/api/v1/documents/11111111-1111-1111-1111-111111111111/index")

    assert response.status_code == 500


def test_delete_document_returns_204() -> None:
    client = create_test_client()

    response = client.delete("/api/v1/documents/11111111-1111-1111-1111-111111111111")

    assert response.status_code == 204
    assert response.content == b""


def test_delete_document_returns_404_when_document_does_not_exist() -> None:
    client = create_test_client(
        service=FakeDocumentService(raise_delete_not_found=True)
    )

    response = client.delete("/api/v1/documents/11111111-1111-1111-1111-111111111111")

    assert response.status_code == 404


def test_delete_document_returns_500_when_service_fails() -> None:
    client = create_test_client(service=FakeDocumentService(raise_error=True))

    response = client.delete("/api/v1/documents/11111111-1111-1111-1111-111111111111")

    assert response.status_code == 500
