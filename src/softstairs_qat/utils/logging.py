from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from loguru import logger

_LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)

_FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
)


@dataclass(frozen=True)
class LoggingPaths:
    log_dir: Path
    log_file: Path


def configure_logging(
    *,
    name: str = "softstairs_qat",
    level: str = "INFO",
    log_dir: str | Path = "logs",
    rotation: str = "10 MB",
    retention: str = "7 days",
) -> LoggingPaths:
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir_path / f"{name}_{timestamp}.log"

    logger.remove()
    logger.add(sys.stderr, level=level, format=_LOG_FORMAT)
    logger.add(
        log_file,
        level=level,
        format=_FILE_FORMAT,
        rotation=rotation,
        retention=retention,
        encoding="utf-8",
    )

    return LoggingPaths(log_dir=log_dir_path, log_file=log_file)
