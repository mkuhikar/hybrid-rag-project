"""API Gateway Lambda handler for document upload and RAG query routes."""

import json
import logging
from uuid import uuid4

from pydantic import ValidationError

from api.schemas import QuestionRequest, UploadRequest
from core.config import Settings
from core.logging import configure_logging
from repositories.documents import DocumentRepository
from services.rag import RAGService
from services.storage import S3Storage

LOGGER = logging.getLogger(__name__)


def _response(status_code: int, body: dict) -> dict:
    return {"statusCode": status_code, "headers": {"content-type": "application/json"}, "body": json.dumps(body)}


def _request_body(event: dict) -> dict:
    body = event.get("body") or "{}"
    return json.loads(body) if isinstance(body, str) else body


def api_handler(event: dict, _: object) -> dict:
    """Route HTTP API v2 requests without an always-on web framework."""
    settings = Settings()
    configure_logging(settings.log_level)
    method = event.get("requestContext", {}).get("http", {}).get("method", "")
    path = event.get("rawPath", "")
    try:
        if method == "GET" and path == "/health":
            return _response(200, {"status": "ok"})
        if method == "POST" and path == "/documents/upload-url":
            request = UploadRequest.model_validate(_request_body(event))
            if not settings.document_bucket:
                return _response(503, {"detail": "Document storage is not configured"})
            document_id = str(uuid4())
            key = f"uploads/{document_id}/{request.filename}"
            repository = DocumentRepository(settings)
            repository.create(document_id, key)
            try:
                upload = S3Storage(settings).create_presigned_post(key, request.content_type)
            except Exception:
                repository.set_status(document_id, "FAILED", "Unable to create upload URL")
                LOGGER.exception("Failed to create S3 upload URL", extra={"document_id": document_id})
                return _response(502, {"detail": "Unable to prepare document upload"})
            repository.set_status(document_id, "UPLOADING")
            return _response(201, {"document_id": document_id, "status": "UPLOADING", "upload": upload})
        if method == "GET" and path.startswith("/documents/"):
            document_id = event.get("pathParameters", {}).get("document_id") or path.rsplit("/", 1)[-1]
            document = DocumentRepository(settings).get(document_id)
            return _response(200, document) if document else _response(404, {"detail": "Document not found"})
        if method == "POST" and path == "/queries":
            request = QuestionRequest.model_validate(_request_body(event))
            result = RAGService.instance(settings).answer(request.question, request.top_k_parents)
            return _response(200, {"answer": result["generated_answer"], "sources": result["matched_candidates"], "scores": result["rag_triad_scores"]})
        return _response(404, {"detail": "Not found"})
    except (ValidationError, json.JSONDecodeError) as error:
        return _response(422, {"detail": str(error)})
    except Exception:
        LOGGER.exception("Lambda API request failed", extra={"method": method, "path": path})
        return _response(503, {"detail": "Service temporarily unavailable"})
