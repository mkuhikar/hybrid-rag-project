"""Endpoints for direct-to-S3 document uploads and ingestion status."""

import logging
from uuid import uuid4

from fastapi import APIRouter, HTTPException, status

from api.schemas import DocumentStatusResponse, UploadRequest, UploadResponse
from core.config import Settings
from repositories.documents import DocumentRepository
from services.storage import S3Storage

LOGGER = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload-url", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
def create_upload_url(request: UploadRequest) -> UploadResponse:
    """Create a document record and return a short-lived direct S3 upload form."""
    settings = Settings()
    if not settings.document_bucket:
        raise HTTPException(status_code=503, detail="Document storage is not configured")
    document_id = str(uuid4())
    key = f"uploads/{document_id}/{request.filename}"
    repository = DocumentRepository(settings.application_database_path)
    repository.create(document_id, key)
    try:
        upload = S3Storage(settings).create_presigned_post(key, request.content_type)
    except Exception:
        repository.set_status(document_id, "FAILED", "Unable to create upload URL")
        LOGGER.exception("Failed to create S3 upload URL", extra={"document_id": document_id})
        raise HTTPException(status_code=502, detail="Unable to prepare document upload")
    repository.set_status(document_id, "UPLOADING")
    return UploadResponse(document_id=document_id, status="UPLOADING", upload=upload)


@router.get("/{document_id}", response_model=DocumentStatusResponse)
def get_document_status(document_id: str) -> DocumentStatusResponse:
    document = DocumentRepository(Settings().application_database_path).get(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentStatusResponse(**document)
