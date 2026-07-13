"""Professional logging setup with console + rotating file handlers."""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


class _ContextFilter(logging.Filter):
    """Inject run_id into every log record for traceability."""

    def __init__(self, run_id: str) -> None:
        super().__init__()
        self.run_id = run_id

    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = self.run_id
        return True


def setup_logger(
    name: str,
    *,
    level: str = "INFO",
    log_dir: str | Path = "./logs",
    filename: str = "app.log",
    max_bytes: int = 10_485_760,
    backup_count: int = 5,
    console: bool = True,
    run_id: str = "-",
) -> logging.Logger:
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    logger.propagate = False

    fmt = (
        "%(asctime)s | %(levelname)-8s | run=%(run_id)s | "
        "%(name)s | %(funcName)s:%(lineno)d | %(message)s"
    )
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=datefmt)
    ctx_filter = _ContextFilter(run_id)

    file_handler = RotatingFileHandler(
        log_path / filename,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.addFilter(ctx_filter)
    logger.addHandler(file_handler)

    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.addFilter(ctx_filter)
        logger.addHandler(console_handler)

    return logger
