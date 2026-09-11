# Hybrid RAG Project

## Purpose

This repository is a Python prototype for question answering over uploaded
text documents. It uses a hybrid retrieval design:

- chunks from each source document are embedded and searched directly;
- likely user questions are generated for each chunk at ingestion time and
  embedded as a reverse-HyDE (question-to-question) index;
- candidates are reranked with a cross-encoder before an answer is generated;
- lightweight context relevance, faithfulness, and answer relevance scores are
  returned with the answer.

The intended upload path is browser -> S3 -> SQS -> ingestion worker -> RAG
index. The API does not proxy document bytes.

## Repository map

| Path | Responsibility |
| --- | --- |
| `app.py` | FastAPI application creation, lifespan logging setup, and router registration. |
| `api/documents.py` | Creates direct-to-S3 presigned upload forms and exposes document status. |
| `api/queries.py` | Exposes question-answering over the indexed corpus. |
| `api/schemas.py` | Pydantic request and response contracts. |
| `workers/sqs_worker.py` | Long-polling SQS worker that consumes S3 event records and indexes documents. |
| `services/storage.py` | S3 client wrapper for presigned POST creation and UTF-8 text reads. |
| `services/rag.py` | Process-local singleton that connects the API/worker to the RAG store and pipeline. |
| `repositories/documents.py` | SQLite persistence for document lifecycle status. |
| `db.py` | SQLite schema, embeddings, ingestion, direct search, and Reverse HyDE indexing. |
| `rag.py` | Cross-encoder reranking, Groq answer generation, and quality metrics. |
| `core/config.py` | Environment-backed settings. |
| `core/logging.py` | Application logging configuration. |

## HTTP API

### `GET /health`

Returns `{"status": "ok"}`. It is only a liveness check; it does not verify
AWS credentials, Groq connectivity, the database, or model availability.

### `POST /documents/upload-url`

Accepts:

```json
{"filename": "notes.txt", "content_type": "text/plain"}
```

The API creates a document ID and a row in the application database, then
returns an S3 presigned POST payload. The object key is:

```text
uploads/<document-id>/<filename>
```

The client must submit the returned POST fields and file directly to S3. The
presigned form expires after 15 minutes and requires the requested content
type.

### `GET /documents/{document_id}`

Returns the stored document state and any worker error message. The client is
expected to poll this endpoint after uploading.

### `POST /queries`

Accepts a non-empty question and optional `top_k_parents` (1 through 10,
default 2). It returns the generated answer, displayed matching candidates,
and three heuristic quality scores.

## Document lifecycle

The current lifecycle states are:

```text
PENDING_UPLOAD -> UPLOADING -> PROCESSING -> READY
                                \-> FAILED
UPLOADING ------------------------> FAILED  (if presigned form creation fails)
```

`PENDING_UPLOAD` is written first. `UPLOADING` means the API successfully
issued the form, not that S3 has received the file. `PROCESSING`, `READY`, and
worker-originated `FAILED` are set by the SQS worker. There is currently no
expiry or reconciliation job for abandoned `UPLOADING` records.

## Ingestion and retrieval

1. S3 emits an `ObjectCreated` notification to SQS.
2. The worker reads each S3 record, extracts the document ID from the key, and
   marks it `PROCESSING`.
3. It downloads the object and decodes it as UTF-8 text.
4. The store splits content into fixed 15-word chunks.
5. Each chunk receives a normalized embedding from
   `sentence-transformers/all-MiniLM-L6-v2`.
6. Groq is asked to produce up to three hypothetical questions per chunk;
   those questions are embedded and retained as Reverse HyDE entries.
7. A query searches both chunk and question embeddings with an in-process
   NumPy dot product. The `BAAI/bge-reranker-base` cross-encoder reranks all
   candidates.
8. The full text of the highest-ranked distinct parent documents is sent to
   Groq (`openai/gpt-oss-20b`) with instructions to answer from context only.

The answer quality values are heuristic application metrics, not a formal RAG
evaluation suite. In particular, `faithfulness` compares overlapping words,
and `answer_relevance` is embedding cosine similarity.

## Configuration

Copy `.env.example` into your deployment’s secret/configuration mechanism.

| Variable | Default | Used for |
| --- | --- | --- |
| `GROQ_API_KEY` | none | Reverse-HyDE question and final-answer generation. |
| `AWS_REGION` | `us-east-1` | S3 and SQS boto3 clients. |
| `DOCUMENT_BUCKET` | none | Direct uploads and worker reads. |
| `SQS_QUEUE_URL` | none | Worker receive/delete operations. |
| `RAG_DATABASE_PATH` | `data/rag.db` | SQLite RAG/index database. |
| `APPLICATION_DATABASE_PATH` | `data/application.db` | SQLite document-status database. |
| `LOG_LEVEL` | `INFO` | Console and file logging threshold. |
| `LOG_DIR` | `logs` | File-log directory. |

The model downloads performed by Sentence Transformers and CrossEncoder are
also runtime dependencies. A first start may need internet access and take
noticeably longer.

## AWS resources required for the intended flow

For a development or integration environment, provision:

- an S3 document bucket with an event notification for `s3:ObjectCreated:*`
  filtered to the `uploads/` prefix;
- an SQS ingestion queue and an SQS dead-letter queue with an appropriate
  redrive policy;
- an SQS resource policy allowing only that S3 bucket to send events;
- S3 CORS rules permitting the browser origin to submit the presigned POST;
- an API runtime identity permitted to create presigned S3 uploads;
- a worker runtime identity permitted to read the bucket and receive/delete
  messages from the ingestion queue;
- hosting for the API process and the separate worker process; and
- a secrets/configuration store for the Groq key and resource identifiers.

Recommended hardening includes S3 default encryption, TLS-only bucket policy,
blocked public access, object lifecycle rules, CloudWatch logs/alarms, and
least-privilege IAM policies scoped to the bucket prefix and queue ARNs.

## Local development

Install the packages, set `GROQ_API_KEY`, configure AWS credentials and the
bucket/queue variables, then run these in separate terminals:

```bash
uvicorn app:app --reload
python -m workers.sqs_worker
```

The API service must be able to call S3 to generate the presigned form. The
worker must be able to call SQS, S3, Groq, and download the embedding/reranker
models if they are not already cached.

## Current limitations and known blocker

This is a prototype, not yet a production-ready deployment.

- `db.py` currently executes cleanup statements using `parent_id` inside
  `SQLVectorStore._init_db`, where that variable is undefined. Constructing the
  RAG store therefore raises `NameError`; this must be fixed before worker
  ingestion or query execution can work.
- The cleanup logic belongs in `ingest_document`, before replacing a document,
  so duplicate SQS deliveries replace its chunks and Reverse HyDE records
  safely.
- SQLite is local to the process/filesystem. Separately deployed API and worker
  processes will not share the RAG or document-status data unless they are
  deliberately placed on shared persistent storage. Production needs a shared
  transactional metadata store and a shared vector-search solution, or a
  consciously single-host architecture.
- Only UTF-8 text objects are supported. PDF, DOCX, OCR, size limits, malware
  scanning, and content validation are not implemented.
- SQS messages are deleted only after successful processing, which is correct,
  but visibility is fixed at 300 seconds and is not extended during long model
  work. Long jobs can be delivered again.
- There are no automated tests or infrastructure definitions in the repository.
