import uuid
from typing import Any, Self

from pydantic import BaseModel, Field, field_validator


class QuestionCreate(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)

    @field_validator("question", mode="before")
    @classmethod
    def strip_question(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("question")
    @classmethod
    def reject_blank_question(cls, value: str) -> str:
        if not value:
            raise ValueError("Question cannot be blank.")
        return value


class RetrievedChunkResponse(BaseModel):
    document_id: uuid.UUID
    chunk_id: uuid.UUID
    chunk_index: int
    content: str
    distance: float
    similarity: float

    @classmethod
    def from_retrieved_chunk(cls, chunk: Any) -> Self:
        return cls(
            document_id=chunk.document_id,
            chunk_id=chunk.chunk_id,
            chunk_index=chunk.chunk_index,
            content=chunk.content,
            distance=chunk.distance,
            similarity=chunk.similarity,
        )


class QuestionAnswerResponse(BaseModel):
    question: str
    answer: str
    chat_model: str
    embedding_model: str
    sources: list[RetrievedChunkResponse]
    source_count: int
