import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_document_service, get_indexing_service
from app.schemas.document import (
    DocumentCreate,
    DocumentIndexResponse,
    DocumentListResponse,
    DocumentResponse,
    DocumentSummary,
)
from app.services.document_service import (
    DocumentNotFoundError as DocumentServiceNotFoundError,
)
from app.services.document_service import DocumentService, DocumentServiceError
from app.services.indexing_service import (
    DocumentEmbeddingError,
    DocumentIndexingError,
    IndexingService,
)
from app.services.indexing_service import (
    DocumentNotFoundError as IndexingDocumentNotFoundError,
)

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
def create_document(
    payload: DocumentCreate,
    service: Annotated[DocumentService, Depends(get_document_service)],
) -> DocumentResponse:
    try:
        document = service.create_document(payload)
    except DocumentServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    return DocumentResponse.from_document(document)


@router.get("", response_model=DocumentListResponse)
def list_documents(
    service: Annotated[DocumentService, Depends(get_document_service)],
) -> DocumentListResponse:
    try:
        documents = service.list_documents()
    except DocumentServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    items = [DocumentSummary.from_document(document) for document in documents]
    return DocumentListResponse(items=items, total=len(items))


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    document_id: uuid.UUID,
    service: Annotated[DocumentService, Depends(get_document_service)],
) -> None:
    try:
        service.delete_document(document_id)
    except DocumentServiceNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except DocumentServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc


@router.post("/{document_id}/index", response_model=DocumentIndexResponse)
def index_document(
    document_id: uuid.UUID,
    service: Annotated[IndexingService, Depends(get_indexing_service)],
) -> DocumentIndexResponse:
    try:
        result = service.index_document(document_id)
    except IndexingDocumentNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except DocumentEmbeddingError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
    except DocumentIndexingError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    return DocumentIndexResponse(
        document_id=result.document.id,
        status=result.document.status,
        chunks_indexed=result.chunks_indexed,
        embedding_model=result.embedding_model,
    )
