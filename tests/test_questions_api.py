import uuid

from fastapi.testclient import TestClient

from app.api.dependencies import get_rag_service
from app.main import create_app
from app.services.rag_service import (
    RagEmbeddingError,
    RagGenerationError,
    RagResult,
    RagRetrievalError,
    RagTrace,
)
from app.services.retrieval_service import RetrievedChunk


class FakeRagService:
    def __init__(
        self,
        *,
        raise_embedding_error: bool = False,
        raise_generation_error: bool = False,
        raise_retrieval_error: bool = False,
    ) -> None:
        self.raise_embedding_error = raise_embedding_error
        self.raise_generation_error = raise_generation_error
        self.raise_retrieval_error = raise_retrieval_error

    def answer_question(self, question: str, top_k: int) -> RagResult:
        if self.raise_embedding_error:
            raise RagEmbeddingError("Could not generate question embedding with Ollama.")
        if self.raise_generation_error:
            raise RagGenerationError("Could not generate answer with Ollama.")
        if self.raise_retrieval_error:
            raise RagRetrievalError("Could not retrieve context for the question.")

        return RagResult(
            question=question,
            answer="RAG retrieves relevant context before generating an answer.",
            chat_model="llama3.1:8b",
            embedding_model="nomic-embed-text",
            sources=[
                RetrievedChunk(
                    document_id=uuid.UUID("11111111-1111-1111-1111-111111111111"),
                    chunk_id=uuid.UUID("22222222-2222-2222-2222-222222222222"),
                    chunk_index=0,
                    content="RAG retrieves relevant context before generation.",
                    distance=0.2,
                    similarity=0.8,
                )
            ][:top_k],
            trace=RagTrace(
                top_k=top_k,
                retrieved_chunks=1,
                context_characters=49,
                steps=[
                    "question_embedding_generated",
                    "similar_chunks_retrieved",
                    "prompt_built",
                    "answer_generated",
                ],
            ),
        )


def create_test_client(service: FakeRagService | None = None) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_rag_service] = lambda: service or FakeRagService()
    return TestClient(app)


def test_answer_question_returns_answer_with_sources() -> None:
    client = create_test_client()

    response = client.post(
        "/api/v1/questions",
        json={"question": "  What is RAG?  ", "top_k": 3},
    )

    assert response.status_code == 200
    assert response.json() == {
        "question": "What is RAG?",
        "answer": "RAG retrieves relevant context before generating an answer.",
        "chat_model": "llama3.1:8b",
        "embedding_model": "nomic-embed-text",
        "sources": [
            {
                "document_id": "11111111-1111-1111-1111-111111111111",
                "chunk_id": "22222222-2222-2222-2222-222222222222",
                "chunk_index": 0,
                "content": "RAG retrieves relevant context before generation.",
                "distance": 0.2,
                "similarity": 0.8,
            }
        ],
        "source_count": 1,
        "trace": {
            "top_k": 3,
            "retrieved_chunks": 1,
            "context_characters": 49,
            "steps": [
                "question_embedding_generated",
                "similar_chunks_retrieved",
                "prompt_built",
                "answer_generated",
            ],
        },
    }


def test_answer_question_rejects_blank_question() -> None:
    client = create_test_client()

    response = client.post(
        "/api/v1/questions",
        json={"question": "   "},
    )

    assert response.status_code == 422


def test_answer_question_rejects_invalid_top_k() -> None:
    client = create_test_client()

    response = client.post(
        "/api/v1/questions",
        json={"question": "What is RAG?", "top_k": 0},
    )

    assert response.status_code == 422


def test_answer_question_returns_502_when_generation_fails() -> None:
    client = create_test_client(FakeRagService(raise_generation_error=True))

    response = client.post(
        "/api/v1/questions",
        json={"question": "What is RAG?"},
    )

    assert response.status_code == 502


def test_answer_question_returns_502_when_embedding_fails() -> None:
    client = create_test_client(FakeRagService(raise_embedding_error=True))

    response = client.post(
        "/api/v1/questions",
        json={"question": "What is RAG?"},
    )

    assert response.status_code == 502


def test_answer_question_returns_500_when_retrieval_fails() -> None:
    client = create_test_client(FakeRagService(raise_retrieval_error=True))

    response = client.post(
        "/api/v1/questions",
        json={"question": "What is RAG?"},
    )

    assert response.status_code == 500
