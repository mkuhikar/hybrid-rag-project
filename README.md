# Hybrid RAG Project

Project documentation and the prioritized delivery plan are in
[`docs/PROJECT.md`](docs/PROJECT.md) and [`docs/TODO.md`](docs/TODO.md).

A Python prototype for a hybrid RAG workflow combining dense chunk retrieval,
ingestion-time Reverse HyDE questions, cross-encoder reranking, and simple RAG
quality metrics.

## Project layout

- `app.py` — FastAPI application entry point.
- `api/` — upload, document status, and question-answering endpoints.
- `workers/` — independent SQS ingestion worker.
- `services/` — S3 and RAG application services.
- `repositories/` — document lifecycle persistence.
- `rag.py` — retrieval, reranking, generation, and evaluation pipeline.
- `db.py` — SQLite vector-store implementation for local development.
- `data/` — local SQLite databases (ignored by Git).
- `logs/` — runtime log files (ignored by Git).
- `scripts/` — operational and maintenance scripts.
- `tests/` — automated tests.

## Setup and run

Create a virtual environment, install dependencies, and configure environment
variables before starting either process:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export GROQ_API_KEY="your-key"
export DOCUMENT_BUCKET="your-upload-bucket"
export SQS_QUEUE_URL="your-sqs-queue-url"
uvicorn app:app --reload
```

Run the ingestion worker separately:

```bash
python -m workers.sqs_worker
```

The browser first requests `POST /documents/upload-url`, uploads directly to
the returned S3 form, and polls `GET /documents/{document_id}` until the status
is `READY`. An S3 ObjectCreated notification must be configured to publish to
the configured SQS queue. Submit questions to `POST /queries`.

## Logging

The application writes operational events to standard error and to
`logs/hybrid_rag.log`. Set `LOG_LEVEL` (for example, `DEBUG` or `WARNING`) or
`LOG_DIR` to change its behavior. Logs deliberately contain only identifiers,
counts, and text lengths; they do not record document contents, user queries,
or credentials.
