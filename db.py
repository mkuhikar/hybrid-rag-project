"""DynamoDB-backed RAG index shared by Lambda invocations."""

import json
import logging
import os
from typing import Any

import boto3
from boto3.dynamodb.conditions import Key
from boto3.dynamodb.types import Binary
from groq import Groq
import numpy as np

from core.config import Settings

LOGGER = logging.getLogger(__name__)


class ProductionEmbedder:
    """Dense vector embedder using Sentence Transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> np.ndarray:
        return np.array(self.model.encode(text, normalize_embeddings=True), dtype=np.float32)


class DynamoVectorStore:
    """DynamoDB vector records for a small, serverless RAG corpus.

    Searches scan the prototype corpus and score vectors in-process. Use a
    dedicated vector service before supporting a large corpus.
    """

    def __init__(self, settings: Settings, embedding_model_name: str = "all-MiniLM-L6-v2"):
        if not settings.rag_table:
            raise ValueError("RAG_TABLE must be configured")
        self.table = boto3.resource(
            "dynamodb", region_name=settings.aws_region, endpoint_url=settings.aws_endpoint_url
        ).Table(settings.rag_table)
        self.embedder = ProductionEmbedder(embedding_model_name)
        self.llm_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    def _generate_hyde_questions(self, text: str, num_questions: int = 3) -> list[str]:
        prompt = (
            f"Generate exactly {num_questions} realistic user search questions that can be directly "
            f"answered by this text chunk. Output ONLY questions separated by newlines.\n\nText:\n{text}"
        )
        try:
            result = self.llm_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="openai/gpt-oss-20b",
                temperature=0.3,
            )
            return [line.strip() for line in result.choices[0].message.content.splitlines() if line.strip()][
                :num_questions
            ]
        except Exception:
            LOGGER.warning("Reverse HyDE generation failed", exc_info=True)
            return []

    def _delete_document_records(self, parent_id: str) -> None:
        response = self.table.query(KeyConditionExpression=Key("parent_id").eq(parent_id))
        records = response.get("Items", [])
        while "LastEvaluatedKey" in response:
            response = self.table.query(
                KeyConditionExpression=Key("parent_id").eq(parent_id),
                ExclusiveStartKey=response["LastEvaluatedKey"],
            )
            records.extend(response.get("Items", []))
        with self.table.batch_writer() as batch:
            for record in records:
                batch.delete_item(Key={"parent_id": record["parent_id"], "record_id": record["record_id"]})

    def ingest_document(self, parent_id: str, content: str, metadata: dict[str, Any]) -> None:
        self._delete_document_records(parent_id)
        self.table.put_item(
            Item={
                "parent_id": parent_id,
                "record_id": "PARENT",
                "record_type": "PARENT",
                "metadata": json.dumps(metadata),
            }
        )
        words = content.split()
        with self.table.batch_writer() as batch:
            for index in range(0, len(words), 15):
                chunk = " ".join(words[index : index + 15])
                chunk_id = index // 15
                batch.put_item(
                    Item={
                        "parent_id": parent_id,
                        "record_id": f"CHUNK#{chunk_id}",
                        "record_type": "CHUNK",
                        "text": chunk,
                        "embedding": Binary(self.embedder.embed(chunk).tobytes()),
                    }
                )
                for question_index, question in enumerate(self._generate_hyde_questions(chunk)):
                    batch.put_item(
                        Item={
                            "parent_id": parent_id,
                            "record_id": f"HYDE#{chunk_id}#{question_index}",
                            "record_type": "HYDE",
                            "text": question,
                            "embedding": Binary(self.embedder.embed(question).tobytes()),
                        }
                    )

    def search_all_sources(self, query: str) -> list[dict[str, Any]]:
        query_vector = self.embedder.embed(query)
        response = self.table.scan()
        records = response.get("Items", [])
        while "LastEvaluatedKey" in response:
            response = self.table.scan(ExclusiveStartKey=response["LastEvaluatedKey"])
            records.extend(response.get("Items", []))
        candidates = []
        for record in records:
            if record.get("record_type") not in {"CHUNK", "HYDE"}:
                continue
            vector = np.frombuffer(bytes(record["embedding"]), dtype=np.float32)
            text = record["text"]
            candidates.append(
                {
                    "parent_id": record["parent_id"],
                    "text": text if record["record_type"] == "CHUNK" else f"[Q2Q Match: '{text}']",
                    "score": float(np.dot(query_vector, vector)),
                    "type": record["record_type"].lower(),
                }
            )
        return candidates

    def get_parent_metadata(self, parent_id: str) -> dict[str, Any]:
        item = self.table.get_item(Key={"parent_id": parent_id, "record_id": "PARENT"}).get("Item")
        return json.loads(item["metadata"]) if item else {}
