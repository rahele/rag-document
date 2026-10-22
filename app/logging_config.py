"""
Structured logging configuration.

Outputs JSON-formatted log lines in production so that log aggregators
(Datadog, CloudWatch, GCP Logging) can parse and index fields natively.
Falls back to a human-readable format when LOG_FORMAT=text is set.
"""

import logging
import os
import sys


class _JsonFormatter(logging.Formatter):
    """Emit each log record as a single JSON object."""

    import json as _json

    def format(self, record: logging.LogRecord) -> str:
        import json
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging(level: str = "INFO") -> None:
    """
    Configure the root logger for the application.

    Args:
        level: One of DEBUG, INFO, WARNING, ERROR, CRITICAL.
               Reads LOG_FORMAT env var: 'json' (default) or 'text'.
    """
    fmt = os.getenv("LOG_FORMAT", "json").lower()

    handler = logging.StreamHandler(sys.stdout)
    if fmt == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
        )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Silence overly verbose third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
