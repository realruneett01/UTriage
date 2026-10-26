"""Logging configuration with JSON-lines formatting and rotating file handlers."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

_STANDARD_RECORD_ATTRS: frozenset[str] = frozenset(
    {
        "args",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "message",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
    }
)


def _serialize_extra(val: Any) -> Any:
    try:
        json.dumps(val)
        return val
    except (TypeError, OverflowError):  # fallback to string for non-serializable objects
        return str(val)


class JsonLinesFormatter(logging.Formatter):
    """JSON-lines formatter emitting timestamp, level, name, message, and extra fields."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        for key, val in record.__dict__.items():
            if key not in _STANDARD_RECORD_ATTRS and not key.startswith("_"):
                log_entry[key] = _serialize_extra(val)
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


def _setup_file_handler(log_dir: Path, formatter: logging.Formatter) -> RotatingFileHandler:
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        log_dir / "utriage.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    handler.setFormatter(formatter)
    return handler


def configure_logging(log_dir: Path | None = None, level: str = "INFO") -> logging.Logger:
    """Configure the utriage logger with stderr and optional rotating file handler."""
    logger = logging.getLogger("utriage")
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    formatter = JsonLinesFormatter()
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(formatter)
    logger.addHandler(stderr_handler)

    if log_dir is not None:
        logger.addHandler(_setup_file_handler(Path(log_dir), formatter))

    logger.propagate = False
    return logger
