"""Long-lived RAG component used by the API and SQS worker."""

from core.config import Settings
from db import DynamoVectorStore
from rag import HybridRAGPipeline
from services.storage import S3Storage


class RAGService:
    _instance: "RAGService | None" = None

    def __init__(self, settings: Settings):
        self.store = DynamoVectorStore(settings)
        self.pipeline = HybridRAGPipeline(self.store, S3Storage(settings))

    @classmethod
    def instance(cls, settings: Settings) -> "RAGService":
        if cls._instance is None:
            cls._instance = cls(settings)
        return cls._instance

    def ingest(self, document_id: str, content: str, object_key: str) -> None:
        self.store.ingest_document(document_id, content, {"object_key": object_key})

    def answer(self, question: str, top_k_parents: int) -> dict:
        return self.pipeline.query(question, top_k_parents)
