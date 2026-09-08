import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

from groq import Groq
import numpy as np
from sentence_transformers import SentenceTransformer

LOGGER = logging.getLogger(__name__)


class ProductionEmbedder:
    """Dense vector embedder using Sentence Transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        LOGGER.info("Loading embedding model", extra={"model_name": model_name})
        self.model = SentenceTransformer(model_name)
        LOGGER.info("Embedding model loaded", extra={"model_name": model_name})

    def embed(self, text: str) -> np.ndarray:
        """Return a normalized float32 embedding for a text value."""
        vector = self.model.encode(text, normalize_embeddings=True)
        return np.array(vector, dtype=np.float32)


class SQLVectorStore:
    """Relational SQL Store handling parent docs, child chunks, and Reverse HyDE indices."""

    def __init__(
        self,
        db_path: str = ":memory:",
        embedding_model_name: str = "all-MiniLM-L6-v2",
    ):
        LOGGER.info("Initializing SQL vector store", extra={"db_path": db_path})
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.embedder = ProductionEmbedder(embedding_model_name)
        self.llm_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self._init_db()

    def _init_db(self):
        LOGGER.info("Initializing database schema")
        with self.conn:
            # SQS standard queues can redeliver messages. Replacing a document
            # must also replace its derived index records to remain idempotent.
            self.conn.execute(
                "DELETE FROM hypothetical_questions WHERE parent_id = ?", (parent_id,)
            )
            self.conn.execute("DELETE FROM child_chunks WHERE parent_id = ?", (parent_id,))
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS parent_documents (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    metadata JSON
                )
            """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS child_chunks (
                    id TEXT PRIMARY KEY,
                    parent_id TEXT NOT NULL,
                    chunk_text TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    FOREIGN KEY (parent_id) REFERENCES parent_documents(id)
                )
            """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS hypothetical_questions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    parent_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    FOREIGN KEY (parent_id) REFERENCES parent_documents(id)
                )
            """
            )
        LOGGER.info("Database schema initialized")

    def _generate_hyde_questions(
        self, text: str, num_questions: int = 3
    ) -> List[str]:
        """Generates hypothetical questions AT INGESTION TIME."""
        prompt = (
            f"Generate exactly {num_questions} realistic user search questions that can be directly "
            f"answered by this text chunk.\nOutput ONLY the questions separated by newlines.\n\nText:\n{text}"
        )
        try:
            res = self.llm_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="openai/gpt-oss-20b",
                temperature=0.3,
            )
            questions = [
                q.strip()
                for q in res.choices[0].message.content.split("\n")
                if q.strip()
            ][:num_questions]
            LOGGER.info("Generated hypothetical questions", extra={"question_count": len(questions)})
            return questions
        except Exception:
            LOGGER.warning("Failed to generate hypothetical questions; continuing without HyDE questions", exc_info=True)
            return []

    def ingest_document(
        self, parent_id: str, content: str, metadata: Dict[str, Any]
    ):
        """Ingests parent text, creates child chunks, and builds Q2Q Reverse HyDE index."""
        LOGGER.info("Starting document ingestion", extra={"parent_id": parent_id, "content_characters": len(content)})
        chunk_count = 0
        question_count = 0
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO parent_documents VALUES (?, ?, ?)",
                (parent_id, content, json.dumps(metadata)),
            )

            # Ingest Child Chunks & Generate HyDE Questions
            words = content.split()
            chunk_size = 15
            for i in range(0, len(words), chunk_size):
                chunk_count += 1
                chunk_str = " ".join(words[i : i + chunk_size])
                child_id = f"{parent_id}_chunk_{i // chunk_size}"

                # Dense Chunk Embedding
                chunk_vec = self.embedder.embed(chunk_str).tobytes()
                self.conn.execute(
                    "INSERT OR REPLACE INTO child_chunks VALUES (?, ?, ?, ?)",
                    (child_id, parent_id, chunk_str, chunk_vec),
                )

                # Ingestion-Time HyDE Indexing (Q2Q)
                questions = self._generate_hyde_questions(chunk_str)
                for q in questions:
                    question_count += 1
                    q_vec = self.embedder.embed(q).tobytes()
                    self.conn.execute(
                        "INSERT INTO hypothetical_questions (parent_id, question, embedding) VALUES (?, ?, ?)",
                        (parent_id, q, q_vec),
                    )
        LOGGER.info("Document ingestion completed", extra={"parent_id": parent_id, "chunk_count": chunk_count, "hyde_question_count": question_count})

    def search_all_sources(self, query: str) -> List[Dict[str, Any]]:
        """Queries child chunks AND ingestion-time hypothetical questions."""
        LOGGER.info("Searching indexed sources", extra={"query_characters": len(query)})
        q_vec = self.embedder.embed(query)
        cursor = self.conn.cursor()

        candidates = []

        # 1. Direct Child Chunk Vector Search
        cursor.execute("SELECT parent_id, chunk_text, embedding FROM child_chunks")
        for pid, text, blob_vec in cursor.fetchall():
            vec = np.frombuffer(blob_vec, dtype=np.float32)
            candidates.append(
                {
                    "parent_id": pid,
                    "text": text,
                    "score": float(np.dot(q_vec, vec)),
                    "type": "direct_chunk",
                }
            )

        # 2. Reverse HyDE Q2Q Vector Search
        cursor.execute(
            "SELECT parent_id, question, embedding FROM hypothetical_questions"
        )
        for pid, question_text, blob_vec in cursor.fetchall():
            vec = np.frombuffer(blob_vec, dtype=np.float32)
            candidates.append(
                {
                    "parent_id": pid,
                    "text": f"[Q2Q Match: '{question_text}']",
                    "score": float(np.dot(q_vec, vec)),
                    "type": "hyde_q2q",
                }
            )

        LOGGER.info("Source search completed", extra={"candidate_count": len(candidates)})
        return candidates

    def get_parent_doc(self, parent_id: str) -> str:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT content FROM parent_documents WHERE id = ?", (parent_id,)
        )
        row = cursor.fetchone()
        if row is None:
            LOGGER.warning("Parent document was not found", extra={"parent_id": parent_id})
        return row[0] if row else ""
