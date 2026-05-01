from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_rag_service
from app.schemas.question import (
    QuestionAnswerResponse,
    QuestionCreate,
    RagTraceResponse,
    RetrievedChunkResponse,
)
from app.services.rag_service import (
    RagEmbeddingError,
    RagGenerationError,
    RagRetrievalError,
    RagService,
)
from app.services.retrieval_service import RetrievalError

router = APIRouter(prefix="/questions", tags=["questions"])


@router.post("", response_model=QuestionAnswerResponse)
def answer_question(
    payload: QuestionCreate,
    service: Annotated[RagService, Depends(get_rag_service)],
) -> QuestionAnswerResponse:
    try:
        result = service.answer_question(question=payload.question, top_k=payload.top_k)
    except (RagEmbeddingError, RagGenerationError) as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
    except (RetrievalError, RagRetrievalError) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    sources = [
        RetrievedChunkResponse.from_retrieved_chunk(chunk)
        for chunk in result.sources
    ]
    return QuestionAnswerResponse(
        question=result.question,
        answer=result.answer,
        chat_model=result.chat_model,
        embedding_model=result.embedding_model,
        sources=sources,
        source_count=len(sources),
        trace=RagTraceResponse.from_trace(result.trace),
    )
