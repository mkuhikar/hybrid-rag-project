"""Request and response contracts exposed by the HTTP API."""

from typing import Any

from pydantic import BaseModel, Field


class UploadRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    content_type: str = Field(default="text/plain", max_length=100)


class UploadResponse(BaseModel):
    document_id: str
    status: str
    upload: dict[str, Any]


class DocumentStatusResponse(BaseModel):
    document_id: str
    status: str
    error_message: str | None = None


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=10_000)
    top_k_parents: int = Field(default=2, ge=1, le=10)


class QuestionResponse(BaseModel):
    answer: str
    sources: list[str]
    scores: dict[str, float]
