"""Document lifecycle persistence for the local development deployment."""

import sqlite3
from pathlib import Path


class DocumentRepository:
    """Persist ingestion state independently of the RAG vector index."""

    def __init__(self, database_path: str):
        Path(database_path).parent.mkdir(parents=True, exist_ok=True)
        self.database_path = database_path
        with self._connect() as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS documents (document_id TEXT PRIMARY KEY, object_key TEXT NOT NULL UNIQUE, status TEXT NOT NULL, error_message TEXT, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)""")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.database_path)

    def create(self, document_id: str, object_key: str) -> None:
        with self._connect() as connection:
            connection.execute("INSERT INTO documents (document_id, object_key, status) VALUES (?, ?, ?)", (document_id, object_key, "PENDING_UPLOAD"))

    def set_status(self, document_id: str, status: str, error_message: str | None = None) -> None:
        with self._connect() as connection:
            connection.execute("UPDATE documents SET status = ?, error_message = ?, updated_at = CURRENT_TIMESTAMP WHERE document_id = ?", (status, error_message, document_id))

    def get(self, document_id: str) -> dict[str, str | None] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT document_id, status, error_message FROM documents WHERE document_id = ?", (document_id,)).fetchone()
        return None if row is None else {"document_id": row[0], "status": row[1], "error_message": row[2]}
