"""
Logging Utility Module
Structured logging with file rotation and JSON event logging.
"""

import json
import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
) -> logging.Logger:
    """
    Set up a logger with console + rotating file handlers.
    
    Args:
        name:         Logger name
        log_file:     Path to log file (optional)
        level:        Log level
        max_bytes:    Max file size before rotation
        backup_count: Number of backup files to keep
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger  # Avoid duplicate handlers

    # ── Formatter ──
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Console Handler ──
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    # ── File Handler ──
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger


def log_detection_event(
    logger: logging.Logger,
    request_id: str,
    filename: Optional[str],
    num_detections: int,
    class_counts: dict,
    inference_ms: float,
    severity: str,
) -> None:
    """Log a structured detection event to both text log and JSON log."""
    event = {
        "event": "detection",
        "request_id": request_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "filename": filename,
        "num_detections": num_detections,
        "class_counts": class_counts,
        "inference_ms": round(inference_ms, 2),
        "severity": severity,
    }

    logger.info(
        f"[{request_id}] detections={num_detections} "
        f"classes={class_counts} "
        f"severity={severity} "
        f"time={inference_ms:.1f}ms"
    )

    # Append to JSON log file
    json_log_path = Path("logs/detections.jsonl")
    json_log_path.parent.mkdir(exist_ok=True)
    try:
        with open(json_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
    except Exception as e:
        logger.warning(f"Failed to write JSON log: {e}")


def load_detection_logs(limit: int = 100) -> list[dict]:
    """Load recent detection events from JSON log."""
    json_log_path = Path("logs/detections.jsonl")
    if not json_log_path.exists():
        return []

    events = []
    try:
        with open(json_log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines[-limit:]:
            try:
                events.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                pass
    except Exception:
        pass

    return list(reversed(events))  # Most recent first
