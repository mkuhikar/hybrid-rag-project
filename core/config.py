"""Environment-backed application settings."""

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    aws_region: str = os.getenv("AWS_REGION", "us-east-1")
    aws_endpoint_url: str | None = os.getenv("AWS_ENDPOINT_URL") or None
    document_bucket: str = os.getenv("DOCUMENT_BUCKET", "")
    sqs_queue_url: str = os.getenv("SQS_QUEUE_URL", "")
    documents_table: str = os.getenv("DOCUMENTS_TABLE", "")
    rag_table: str = os.getenv("RAG_TABLE", "")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
