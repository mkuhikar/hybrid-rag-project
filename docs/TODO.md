# TODO

This list is ordered by dependency and operational risk.

## P0 — make the prototype runnable

- [x] Fix the undefined `parent_id` reference in `SQLVectorStore._init_db`.
- [x] Move document-derived-record cleanup into `ingest_document`, before the
  parent document is replaced, and verify duplicate SQS delivery is idempotent.
- [x] Add unit tests for schema initialization, document ingestion, duplicate
  ingestion, empty search, lifecycle-state repository operations, and API
  validation.
- [ ] Add a Serverless Framework infrastructure definition for the shared AWS
  stack: S3 upload bucket, SQS ingestion queue, DLQ, S3-to-SQS notification,
  IAM policies, configuration, and the chosen shared metadata/vector stores.
- [ ] Add separate `local` (Floci) and AWS deployment stages that use the same
  resource definition and never allow a local command to target a real AWS
  account.
- [ ] Prove the Serverless Framework/Floci compatibility path with a minimal
  stack deployment before modeling the complete system; verify the deployed
  S3 notification creates an SQS message after an object upload.
- [ ] Decide and document the compute deployment: the existing FastAPI API and
  long-running SQS worker need container/long-running compute (for example
  ECS/Fargate), or must be redesigned as Lambda functions. Serverless
  Framework can provision supporting resources, but this choice determines
  how the application itself runs.
- [ ] Add a local integration test that mocks S3, SQS, Groq, and model calls.
- [ ] Add a startup/readiness check that reports required configuration and
  model/database initialization failures without exposing secrets.

## P1 — validate the AWS ingestion path

- [ ] Install Floci and start its local AWS emulator for integration testing.
- [ ] Add documented test configuration that directs boto3 S3 and SQS clients
  to Floci without affecting real AWS deployments.
- [ ] Add a repeatable Floci bootstrap script that creates a test bucket,
  ingestion queue, DLQ, and the S3 `ObjectCreated` -> SQS notification.
- [ ] Add a local end-to-end test using Floci: request a presigned upload,
  upload a UTF-8 text object, consume its SQS event, and verify the document
  reaches `READY` and is queryable. Mock Groq and embedding models in this
  test so it has no external AI dependency.
- [ ] Add a developer command and documentation for starting, resetting, and
  troubleshooting the Floci test environment.
- [ ] Configure S3 CORS for the actual frontend origins and test a browser
  presigned POST end to end.
- [ ] Set a dead-letter redrive policy and alerts for DLQ messages and queue
  backlog.
- [ ] Verify the worker handles S3 event payloads, URL-encoded filenames,
  retries, and poisoned messages correctly.
- [ ] Decide and document limits for file size, permitted MIME types, and
  request rates.

## P1 — choose a deployable data architecture

- [ ] Decide whether this remains a single-host prototype or becomes a
  horizontally deployable service.
- [ ] For a multi-process/multi-host deployment, replace local SQLite with a
  shared metadata database and a shared vector-search store (or define a
  supported shared-volume design with its availability and locking tradeoffs).
- [ ] Define backup, retention, migration, restore, and tenant/data-deletion
  requirements before creating production data stores.
- [ ] Ensure API readers and worker writers use the same persistent corpus.

## P2 — build the user interface

- [ ] Choose the UI delivery model (a standalone web app or server-rendered
  FastAPI pages) and document its local-development and deployment workflow.
- [ ] Build an upload screen that requests a presigned form, submits the file
  directly to S3, and handles upload failures without sending file bytes to
  the API service.
- [ ] Poll and display document lifecycle states (`UPLOADING`, `PROCESSING`,
  `READY`, and `FAILED`) with clear retry/error messaging.
- [ ] Build a question-answering screen with question input, loading/error
  states, generated answer, retrieval sources, and RAG quality scores.
- [ ] Add client-side file-type and size guidance that matches server-side S3
  upload policy once those limits are defined.
- [ ] Add accessible form labels, keyboard navigation, responsive layout, and
  user-facing empty states.
- [ ] Add UI tests for upload submission, lifecycle polling, query results,
  and failure handling.

## P2 — strengthen ingestion and retrieval

- [ ] Add parsers for approved document formats; reject unsupported/binary
  content clearly.
- [ ] Validate object keys and MIME types server-side, add a content-length
  policy to the presigned POST, and consider malware scanning.
- [ ] Improve chunking with character/token-aware overlap and document
  metadata instead of fixed 15-word chunks.
- [ ] Add retrieval thresholds and an explicit “no supporting context” path.
- [ ] Preserve citations: return document IDs, object keys/names, and relevant
  excerpts rather than only raw candidate text.
- [ ] Make model names and retrieval parameters configurable.
- [ ] Measure ingestion duration; extend SQS visibility or split work when a
  job can exceed the current five-minute visibility timeout.

## P2 — deploy and operate safely

- [ ] Add infrastructure as code for S3, SQS/DLQ, IAM, configuration, and
  compute.
- [ ] Containerize the API and worker, with separate deployment definitions.
- [ ] Store `GROQ_API_KEY` in a secrets manager; never commit it or log it.
- [ ] Enable bucket encryption, public-access blocking, TLS-only access, and
  lifecycle rules.
- [ ] Add structured metrics/alerts for API errors, worker failures, ingestion
  latency, model errors, queue depth, DLQ count, and retrieval quality.
- [ ] Add CI for formatting, linting, type checks, tests, dependency scanning,
  and deployment validation.
- [ ] Define authentication, authorization, tenancy boundaries, audit logging,
  and rate limiting before exposing uploads or queries publicly.

## Suggested first milestone

1. Complete the two P0 database fixes and tests.
2. Create only sandbox S3/SQS/DLQ resources and validate one text upload.
3. Decide the shared database/vector-store architecture.
4. Add infrastructure as code and deploy API plus worker to a non-production
   environment.
