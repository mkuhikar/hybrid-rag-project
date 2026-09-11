"""DynamoDB-backed document lifecycle persistence for Lambda handlers."""

from datetime import datetime, timezone

import boto3

from core.config import Settings


class DocumentRepository:
    """Persist ingestion state independently of the RAG vector index."""

    def __init__(self, settings: Settings):
        if not settings.documents_table:
            raise ValueError("DOCUMENTS_TABLE must be configured")
        self.table = boto3.resource(
            "dynamodb",
            region_name=settings.aws_region,
            endpoint_url=settings.aws_endpoint_url,
        ).Table(settings.documents_table)

    def create(self, document_id: str, object_key: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.table.put_item(
            Item={
                "document_id": document_id,
                "object_key": object_key,
                "status": "PENDING_UPLOAD",
                "created_at": now,
                "updated_at": now,
            }
        )

    def set_status(self, document_id: str, status: str, error_message: str | None = None) -> None:
        self.table.update_item(
            Key={"document_id": document_id},
            UpdateExpression="SET #status = :status, error_message = :error, updated_at = :updated",
            ExpressionAttributeNames={"#status": "status"},
            ExpressionAttributeValues={
                ":status": status,
                ":error": error_message,
                ":updated": datetime.now(timezone.utc).isoformat(),
            },
        )

    def get(self, document_id: str) -> dict[str, str | None] | None:
        item = self.table.get_item(Key={"document_id": document_id}).get("Item")
        if item is None:
            return None
        return {
            "document_id": item["document_id"],
            "status": item["status"],
            "error_message": item.get("error_message"),
        }
