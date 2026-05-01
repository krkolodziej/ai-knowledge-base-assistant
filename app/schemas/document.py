import uuid
from datetime import datetime
from typing import Any, Literal, Self

from pydantic import BaseModel, Field, field_validator

DocumentContentType = Literal["text/plain", "text/markdown"]


class DocumentCreate(BaseModel):
    title: str = Field(min_length=1, max_length=255)
    content: str = Field(min_length=1)
    content_type: DocumentContentType = "text/plain"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("title", "content", mode="before")
    @classmethod
    def strip_text_fields(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("title", "content")
    @classmethod
    def reject_blank_text_fields(cls, value: str) -> str:
        if not value:
            raise ValueError("Field cannot be blank.")
        return value


class DocumentResponse(BaseModel):
    id: uuid.UUID
    title: str
    content: str
    content_type: str
    status: str
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_document(cls, document: Any) -> Self:
        return cls(
            id=document.id,
            title=document.title,
            content=document.content,
            content_type=document.content_type,
            status=document.status,
            metadata=document.document_metadata,
            created_at=document.created_at,
            updated_at=document.updated_at,
        )


class DocumentSummary(BaseModel):
    id: uuid.UUID
    title: str
    content_type: str
    status: str
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_document(cls, document: Any) -> Self:
        return cls(
            id=document.id,
            title=document.title,
            content_type=document.content_type,
            status=document.status,
            created_at=document.created_at,
            updated_at=document.updated_at,
        )


class DocumentListResponse(BaseModel):
    items: list[DocumentSummary]
    total: int


class DocumentIndexResponse(BaseModel):
    document_id: uuid.UUID
    status: str
    chunks_indexed: int
    embedding_model: str
