"""Unit tests for serverless persistence, indexing idempotency, and validation."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from api.schemas import QuestionRequest, UploadRequest
from core.config import Settings
from db import DynamoVectorStore
from repositories.documents import DocumentRepository


class FakeBatchWriter:
    def __init__(self, table):
        self.table = table

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def put_item(self, Item):
        self.table.put_item(Item=Item)

    def delete_item(self, Key):
        self.table.items.pop((Key["parent_id"], Key["record_id"]), None)


class FakeTable:
    def __init__(self):
        self.items = {}

    def put_item(self, Item):
        key = (Item.get("parent_id") or Item["document_id"], Item.get("record_id", "DOCUMENT"))
        self.items[key] = Item

    def get_item(self, Key):
        key = (Key.get("parent_id") or Key["document_id"], Key.get("record_id", "DOCUMENT"))
        return {"Item": self.items[key]} if key in self.items else {}

    def update_item(self, Key, ExpressionAttributeValues, **_):
        item = self.get_item(Key)["Item"]
        item.update(
            status=ExpressionAttributeValues[":status"],
            error_message=ExpressionAttributeValues[":error"],
            updated_at=ExpressionAttributeValues[":updated"],
        )

    def query(self, **_):
        return {"Items": list(self.items.values())}

    def scan(self, **_):
        return {"Items": list(self.items.values())}

    def batch_writer(self):
        return FakeBatchWriter(self)


class FakeDynamoResource:
    def __init__(self):
        self.tables = {}

    def Table(self, name):
        return self.tables.setdefault(name, FakeTable())


class FakeEmbedder:
    def embed(self, _: str) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float32)


class DynamoVectorStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.resource = FakeDynamoResource()
        self.resource_patch = patch("db.boto3.resource", return_value=self.resource)
        self.embedder_patch = patch("db.ProductionEmbedder", return_value=FakeEmbedder())
        self.groq_patch = patch("db.Groq", return_value=MagicMock())
        self.resource_patch.start()
        self.embedder_patch.start()
        self.groq_patch.start()
        self.addCleanup(self.resource_patch.stop)
        self.addCleanup(self.embedder_patch.stop)
        self.addCleanup(self.groq_patch.stop)
        self.store = DynamoVectorStore(Settings(rag_table="rag"))

    def test_empty_store_has_no_candidates(self) -> None:
        self.assertEqual(self.store.search_all_sources("anything"), [])

    def test_reingestion_replaces_derived_records(self) -> None:
        self.store._generate_hyde_questions = lambda _: ["What does this say?"]
        self.store.ingest_document("document-1", " ".join(f"first-{i}" for i in range(16)), {"version": 1})
        self.store.ingest_document("document-1", "replacement document", {"version": 2})

        records = self.resource.Table("rag").items
        self.assertEqual(len(records), 3)  # parent, one chunk, one HyDE question
        self.assertEqual(self.store.get_parent_metadata("document-1"), {"version": 2})


class DocumentRepositoryTests(unittest.TestCase):
    def test_document_lifecycle_state_is_persisted(self) -> None:
        resource = FakeDynamoResource()
        with patch("repositories.documents.boto3.resource", return_value=resource):
            repository = DocumentRepository(Settings(documents_table="documents"))
            repository.create("document-1", "uploads/document-1/notes.txt")
            repository.set_status("document-1", "FAILED", "could not read object")
            self.assertEqual(
                repository.get("document-1"),
                {"document_id": "document-1", "status": "FAILED", "error_message": "could not read object"},
            )


class ApiValidationTests(unittest.TestCase):
    def test_request_models_reject_invalid_upload_and_query_values(self) -> None:
        with self.assertRaises(ValueError):
            UploadRequest(filename="")
        with self.assertRaises(ValueError):
            QuestionRequest(question="", top_k_parents=2)
        with self.assertRaises(ValueError):
            QuestionRequest(question="valid", top_k_parents=11)


if __name__ == "__main__":
    unittest.main()
