"""Central logging configuration shared by API and worker processes."""

import logging
import os
from pathlib import Path


def configure_logging(log_level: str = "INFO") -> None:
    """Configure concise operational logs for a process entry point."""
    log_directory = Path(os.getenv("LOG_DIR", "logs"))
    log_directory.mkdir(parents=True, exist_ok=True)
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_directory / "hybrid_rag.log", encoding="utf-8")],
        force=True,
    )
