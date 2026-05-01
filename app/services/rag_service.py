from dataclasses import dataclass
from typing import Protocol

from app.services.ollama_client import OllamaClient, OllamaClientError
from app.services.retrieval_service import (
    RetrievalEmbeddingError,
    RetrievalError,
    RetrievalResult,
    RetrievedChunk,
)


class RagError(Exception):
    pass


class RagRetrievalError(RagError):
    pass


class RagEmbeddingError(RagError):
    pass


class RagGenerationError(RagError):
    pass


class ContextRetrievalService(Protocol):
    def retrieve(self, question: str, top_k: int) -> RetrievalResult:
        pass


class TextGenerationClient(Protocol):
    def generate(self, model: str, prompt: str) -> str:
        pass


@dataclass(frozen=True)
class RagResult:
    question: str
    answer: str
    sources: list[RetrievedChunk]
    chat_model: str
    embedding_model: str


class RagService:
    def __init__(
        self,
        retrieval_service: ContextRetrievalService,
        generation_client: TextGenerationClient | OllamaClient,
        chat_model: str,
    ) -> None:
        self.retrieval_service = retrieval_service
        self.generation_client = generation_client
        self.chat_model = chat_model

    def answer_question(self, question: str, top_k: int) -> RagResult:
        try:
            retrieval_result = self.retrieval_service.retrieve(question=question, top_k=top_k)
        except RetrievalEmbeddingError as exc:
            raise RagEmbeddingError("Could not generate question embedding with Ollama.") from exc
        except RetrievalError as exc:
            raise RagRetrievalError("Could not retrieve context for the question.") from exc

        if not retrieval_result.chunks:
            return RagResult(
                question=question,
                answer="I do not know based on the indexed documents.",
                sources=[],
                chat_model=self.chat_model,
                embedding_model=retrieval_result.embedding_model,
            )

        prompt = self._build_prompt(retrieval_result)

        try:
            answer = self.generation_client.generate(model=self.chat_model, prompt=prompt)
        except OllamaClientError as exc:
            raise RagGenerationError("Could not generate answer with Ollama.") from exc

        return RagResult(
            question=question,
            answer=answer,
            sources=retrieval_result.chunks,
            chat_model=self.chat_model,
            embedding_model=retrieval_result.embedding_model,
        )

    def _build_prompt(self, retrieval_result: RetrievalResult) -> str:
        context = "\n\n".join(
            self._format_source(index=index, chunk=chunk)
            for index, chunk in enumerate(retrieval_result.chunks, start=1)
        )
        return (
            "You are an AI knowledge base assistant.\n"
            "Answer the user's question using only the context below.\n"
            "If the context does not contain the answer, say that you do not know "
            "based on the provided documents.\n"
            "Do not invent facts or sources.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION:\n{retrieval_result.question}\n\n"
            "ANSWER:"
        )

    def _format_source(self, index: int, chunk: RetrievedChunk) -> str:
        return (
            f"[Source {index}]\n"
            f"document_id: {chunk.document_id}\n"
            f"chunk_id: {chunk.chunk_id}\n"
            f"chunk_index: {chunk.chunk_index}\n"
            f"content:\n{chunk.content}"
        )
