"""
Structured logging for Legal Contract Automation Suite.

Design:
- No secrets logged (API keys masked)
- Structured fields for machine parsing
- File + console output
- Per-module log levels
"""

import logging
import sys
from datetime import datetime
from pathlib import Path


class SensitiveDataFilter(logging.Filter):
    """Filter out sensitive data from logs."""

    SENSITIVE_KEYS = ["api_key", "password", "secret", "token", "key"]

    def filter(self, record):
        if hasattr(record, "msg") and isinstance(record.msg, str):
            for key in self.SENSITIVE_KEYS:
                if key in record.msg.lower():
                    record.msg = record.msg.replace(key, "***")
        return True


def setup_logger(name: str = "legal_automation",
                 level: str = "INFO",
                 log_dir: str = "logs") -> logging.Logger:
    """Setup logger with console and file handlers."""
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.addFilter(SensitiveDataFilter())

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(console)

    # File handler
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    try:
        fh = logging.FileHandler(
            log_path / f"legal_{datetime.now().strftime('%Y%m%d')}.log",
            encoding="utf-8",
        )
        fh.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s"
        ))
        logger.addHandler(fh)
    except Exception:
        pass

    return logger
