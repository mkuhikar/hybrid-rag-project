"""FastAPI entry point for the Hybrid RAG service."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.documents import router as documents_router
from api.queries import router as queries_router
from core.config import Settings
from core.logging import configure_logging


@asynccontextmanager
async def lifespan(_: FastAPI):
    configure_logging(Settings().log_level)
    yield


app = FastAPI(title="Hybrid RAG API", version="0.1.0", lifespan=lifespan)
app.include_router(documents_router)
app.include_router(queries_router)


@app.get("/health", tags=["health"])
def health_check() -> dict[str, str]:
    """Return service liveness without touching model or cloud dependencies."""
    return {"status": "ok"}
