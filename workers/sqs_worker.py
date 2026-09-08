"""Consume S3 upload events from SQS and populate the RAG index."""

import json
import logging
from urllib.parse import unquote_plus

import boto3

from core.config import Settings
from core.logging import configure_logging
from repositories.documents import DocumentRepository
from services.rag import RAGService
from services.storage import S3Storage

LOGGER = logging.getLogger(__name__)


class SQSIngestionWorker:
    def __init__(self, settings: Settings):
        if not settings.sqs_queue_url or not settings.document_bucket:
            raise ValueError("SQS_QUEUE_URL and DOCUMENT_BUCKET must be configured")
        self.settings = settings
        self.queue = boto3.client("sqs", region_name=settings.aws_region)
        self.repository = DocumentRepository(settings.application_database_path)
        self.storage = S3Storage(settings)
        self.rag = RAGService.instance(settings)

    def run_forever(self) -> None:
        while True:
            response = self.queue.receive_message(QueueUrl=self.settings.sqs_queue_url, MaxNumberOfMessages=10, WaitTimeSeconds=20, VisibilityTimeout=300)
            for message in response.get("Messages", []):
                self._process(message)

    def _process(self, message: dict) -> None:
        document_id: str | None = None
        try:
            event = json.loads(message["Body"])
            for record in event.get("Records", []):
                key = unquote_plus(record["s3"]["object"]["key"])
                document_id = key.split("/", 2)[1]
                self.repository.set_status(document_id, "PROCESSING")
                self.rag.ingest(document_id, self.storage.read_text(key), key)
                self.repository.set_status(document_id, "READY")
                LOGGER.info("Document indexed", extra={"document_id": document_id})
        except Exception:
            if document_id is not None:
                self.repository.set_status(document_id, "FAILED", "Ingestion failed; queued for retry")
            LOGGER.exception("Document ingestion failed")
            raise
        self.queue.delete_message(QueueUrl=self.settings.sqs_queue_url, ReceiptHandle=message["ReceiptHandle"])


def main() -> None:
    settings = Settings()
    configure_logging(settings.log_level)
    SQSIngestionWorker(settings).run_forever()


if __name__ == "__main__":
    main()
