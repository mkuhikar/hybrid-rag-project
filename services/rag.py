"""Long-lived RAG component used by the API and SQS worker."""

from core.config import Settings
from db import SQLVectorStore
from rag import HybridRAGPipeline


class RAGService:
    _instance: "RAGService | None" = None

    def __init__(self, settings: Settings):
        self.store = SQLVectorStore(db_path=settings.rag_database_path)
        self.pipeline = HybridRAGPipeline(self.store)

    @classmethod
    def instance(cls, settings: Settings) -> "RAGService":
        if cls._instance is None:
            cls._instance = cls(settings)
        return cls._instance

    def ingest(self, document_id: str, content: str, object_key: str) -> None:
        self.store.ingest_document(document_id, content, {"object_key": object_key})

    def answer(self, question: str, top_k_parents: int) -> dict:
        return self.pipeline.query(question, top_k_parents)
