"""S3 integration used by the API and the ingestion worker."""

from typing import Any

import boto3

from core.config import Settings


class S3Storage:
    def __init__(self, settings: Settings):
        self.bucket = settings.document_bucket
        self.client = boto3.client("s3", region_name=settings.aws_region)

    def create_presigned_post(self, object_key: str, content_type: str) -> dict[str, Any]:
        return self.client.generate_presigned_post(Bucket=self.bucket, Key=object_key, Fields={"Content-Type": content_type}, Conditions=[{"Content-Type": content_type}], ExpiresIn=900)

    def read_text(self, object_key: str) -> str:
        response = self.client.get_object(Bucket=self.bucket, Key=object_key)
        return response["Body"].read().decode("utf-8")
