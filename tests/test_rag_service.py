import uuid

import pytest

from app.services.ollama_client import OllamaClientError
from app.services.rag_service import (
    RagEmbeddingError,
    RagGenerationError,
    RagRetrievalError,
    RagService,
)
from app.services.retrieval_service import (
    RetrievalEmbeddingError,
    RetrievalError,
    RetrievalResult,
    RetrievedChunk,
)


class FakeRetrievalService:
    def __init__(
        self,
        chunks: list[RetrievedChunk],
        *,
        raise_embedding_error: bool = False,
        raise_retrieval_error: bool = False,
    ) -> None:
        self.chunks = chunks
        self.raise_embedding_error = raise_embedding_error
        self.raise_retrieval_error = raise_retrieval_error
        self.question: str | None = None
        self.top_k: int | None = None

    def retrieve(self, question: str, top_k: int) -> RetrievalResult:
        if self.raise_embedding_error:
            raise RetrievalEmbeddingError("Embedding failed.")
        if self.raise_retrieval_error:
            raise RetrievalError("Retrieval failed.")
        self.question = question
        self.top_k = top_k
        return RetrievalResult(
            question=question,
            chunks=self.chunks,
            embedding_model="test-embed",
        )


class FakeGenerationClient:
    def __init__(self, answer: str = "Generated answer.", *, raise_error: bool = False) -> None:
        self.answer = answer
        self.raise_error = raise_error
        self.model: str | None = None
        self.prompt: str | None = None

    def generate(self, model: str, prompt: str) -> str:
        if self.raise_error:
            raise OllamaClientError("Generation failed.")
        self.model = model
        self.prompt = prompt
        return self.answer


def make_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        document_id=uuid.UUID("11111111-1111-1111-1111-111111111111"),
        chunk_id=uuid.UUID("22222222-2222-2222-2222-222222222222"),
        chunk_index=1,
        content="RAG combines retrieval with generation.",
        distance=0.15,
        similarity=0.85,
    )


def test_rag_service_builds_prompt_and_returns_answer_with_sources() -> None:
    retrieval_service = FakeRetrievalService([make_chunk()])
    generation_client = FakeGenerationClient("RAG combines retrieval with generation.")
    service = RagService(
        retrieval_service=retrieval_service,
        generation_client=generation_client,
        chat_model="test-chat",
    )

    result = service.answer_question("What is RAG?", top_k=3)

    assert retrieval_service.question == "What is RAG?"
    assert retrieval_service.top_k == 3
    assert generation_client.model == "test-chat"
    assert generation_client.prompt is not None
    assert "RAG combines retrieval with generation." in generation_client.prompt
    assert "Do not invent facts or sources." in generation_client.prompt
    assert result.answer == "RAG combines retrieval with generation."
    assert result.sources == [make_chunk()]
    assert result.chat_model == "test-chat"
    assert result.embedding_model == "test-embed"


def test_rag_service_returns_safe_answer_when_no_sources_are_found() -> None:
    generation_client = FakeGenerationClient()
    service = RagService(
        retrieval_service=FakeRetrievalService([]),
        generation_client=generation_client,
        chat_model="test-chat",
    )

    result = service.answer_question("What is RAG?", top_k=3)

    assert result.answer == "I do not know based on the indexed documents."
    assert result.sources == []
    assert generation_client.prompt is None


def test_rag_service_wraps_generation_errors() -> None:
    service = RagService(
        retrieval_service=FakeRetrievalService([make_chunk()]),
        generation_client=FakeGenerationClient(raise_error=True),
        chat_model="test-chat",
    )

    with pytest.raises(RagGenerationError):
        service.answer_question("What is RAG?", top_k=3)


def test_rag_service_wraps_embedding_errors() -> None:
    service = RagService(
        retrieval_service=FakeRetrievalService([make_chunk()], raise_embedding_error=True),
        generation_client=FakeGenerationClient(),
        chat_model="test-chat",
    )

    with pytest.raises(RagEmbeddingError):
        service.answer_question("What is RAG?", top_k=3)


def test_rag_service_wraps_retrieval_errors() -> None:
    service = RagService(
        retrieval_service=FakeRetrievalService([make_chunk()], raise_retrieval_error=True),
        generation_client=FakeGenerationClient(),
        chat_model="test-chat",
    )

    with pytest.raises(RagRetrievalError):
        service.answer_question("What is RAG?", top_k=3)
