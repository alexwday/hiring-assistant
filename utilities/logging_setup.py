"""Logging setup with simple secret redaction."""

from __future__ import annotations

import logging
import re
import sys
from datetime import datetime
from pathlib import Path

SECRET_PATTERNS = (
    re.compile(
        r"(?i)(api[_-]?key|access[_-]?token|client[_-]?secret|password)"
        r"([\"']?\s*[:=]\s*[\"']?)([^\"'\s,;]+)"
    ),
)
_LOGGING_STATE = {"configured": False}


class ConsoleFormatter(logging.Formatter):
    """Compact console formatter."""

    def formatTime(
        self,
        record: logging.LogRecord,
        datefmt: str | None = None,
    ) -> str:
        """Format record timestamps in local time."""
        created = datetime.fromtimestamp(record.created)
        return created.strftime(datefmt or "%Y-%m-%d %H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        """Return one log line."""
        timestamp = self.formatTime(record)
        return f"{timestamp} | {record.levelname:<8} | {record.getMessage()}"


def setup_logging(
    level: str = "INFO",
    output_logs: bool = False,
    log_dir: str | Path = "logs",
    force: bool = False,
) -> Path | None:
    """Configure root logging and return the file log path when enabled."""
    if _LOGGING_STATE["configured"] and not force:
        return None

    logger = logging.getLogger()
    logger.setLevel(_coerce_level(level))
    logger.handlers.clear()

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(_coerce_level(level))
    console.setFormatter(ConsoleFormatter())
    console.addFilter(redact_record)
    logger.addHandler(console)

    log_file = None
    if output_logs:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        log_file = log_path / f"llm_framework_{datetime.now():%Y%m%d_%H%M%S}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
            )
        )
        file_handler.addFilter(redact_record)
        logger.addHandler(file_handler)

    for noisy_logger in ("httpcore", "httpx", "openai", "urllib3"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    _LOGGING_STATE["configured"] = True
    return log_file


def redact_record(record: logging.LogRecord) -> bool:
    """Redact common secret-bearing key/value pairs from a log record."""
    message = record.getMessage()
    redacted = redact(message)
    if redacted != message:
        record.msg = redacted
        record.args = ()
    return True


def redact(message: str) -> str:
    """Redact known secret patterns from ``message``."""
    redacted = message
    for pattern in SECRET_PATTERNS:
        redacted = pattern.sub(r"\1\2***", redacted)
    return redacted


def _coerce_level(level: int | str) -> int:
    """Convert a logging level name or integer into a logging level value."""
    if isinstance(level, int):
        return level
    value = logging.getLevelName(level.upper())
    if isinstance(value, int):
        return value
    raise ValueError(f"Unknown logging level: {level!r}")
