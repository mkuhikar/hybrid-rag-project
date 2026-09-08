"""Environment-backed application settings."""

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    aws_region: str = os.getenv("AWS_REGION", "us-east-1")
    document_bucket: str = os.getenv("DOCUMENT_BUCKET", "")
    sqs_queue_url: str = os.getenv("SQS_QUEUE_URL", "")
    rag_database_path: str = os.getenv("RAG_DATABASE_PATH", "data/rag.db")
    application_database_path: str = os.getenv("APPLICATION_DATABASE_PATH", "data/application.db")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
