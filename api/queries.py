"""Question-answering endpoint backed by the indexed RAG corpus."""

import logging

from fastapi import APIRouter, HTTPException

from api.schemas import QuestionRequest, QuestionResponse
from core.config import Settings
from services.rag import RAGService

LOGGER = logging.getLogger(__name__)
router = APIRouter(prefix="/queries", tags=["queries"])


@router.post("", response_model=QuestionResponse)
def answer_question(request: QuestionRequest) -> QuestionResponse:
    """Retrieve indexed context and generate a grounded answer."""
    try:
        result = RAGService.instance(Settings()).answer(request.question, request.top_k_parents)
    except Exception:
        LOGGER.exception("RAG query failed", extra={"question_characters": len(request.question)})
        raise HTTPException(status_code=503, detail="Answer generation is temporarily unavailable")
    return QuestionResponse(answer=result["generated_answer"], sources=result["matched_candidates"], scores=result["rag_triad_scores"])
