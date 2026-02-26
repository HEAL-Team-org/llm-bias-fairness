#!/usr/bin/env python3
"""Shared logging utilities for local helper services."""

from __future__ import annotations

import logging
import logging.handlers
import sys
import time
from functools import lru_cache
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

_LOG_DIR = Path(__file__).parent / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

_APP_LOG = _LOG_DIR / "app.log"
_ERROR_LOG = _LOG_DIR / "error.log"

# ── ANSI colours ─────────────────────────────────────────────────────────────
_RESET   = "\033[0m"
_BOLD    = "\033[1m"
_DIM     = "\033[2m"
_RED     = "\033[91m"
_YELLOW  = "\033[93m"
_GREEN   = "\033[92m"
_CYAN    = "\033[96m"
_BLUE    = "\033[94m"
_GREY    = "\033[37m"
_MAGENTA = "\033[95m"
_WHITE   = "\033[97m"

_LEVEL_COLOURS: dict[str, str] = {
    "DEBUG":    _DIM + _CYAN,
    "INFO":     _BOLD + _GREEN,
    "WARNING":  _BOLD + _YELLOW,
    "ERROR":    _BOLD + _RED,
    "CRITICAL": _BOLD + _MAGENTA,
}

_LEVEL_ICONS: dict[str, str] = {
    "DEBUG":    "○",
    "INFO":     "●",
    "WARNING":  "▲",
    "ERROR":    "✖",
    "CRITICAL": "☠",
}


class _ColourFormatter(logging.Formatter):
    """Coloured, icon-prefixed formatter for console output."""

    _DATE_FMT = "%H:%M:%S"

    def format(self, record: logging.LogRecord) -> str:
        colour = _LEVEL_COLOURS.get(record.levelname, _RESET)
        icon   = _LEVEL_ICONS.get(record.levelname, " ")

        # timestamp  dim grey
        time_str = self.formatTime(record, self._DATE_FMT)
        time_part = f"{_DIM}{_GREY}{time_str}{_RESET}"

        # level badge  bold + colour
        level_part = f"{colour}{icon} {record.levelname:<8}{_RESET}"

        # logger name  dim cyan
        name_part = f"{_DIM}{_CYAN}{record.name:<28}{_RESET}"

        # message  white (errors get bold red)
        msg = record.getMessage()
        if record.levelno >= logging.ERROR:
            msg_part = f"{_BOLD}{_RED}{msg}{_RESET}"
        elif record.levelno == logging.WARNING:
            msg_part = f"{_YELLOW}{msg}{_RESET}"
        else:
            msg_part = f"{_WHITE}{msg}{_RESET}"

        line = f"{time_part} │ {level_part} │ {name_part} │ {msg_part}"

        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)

        return line


def _enable_windows_ansi() -> None:
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception:
            pass


@lru_cache(maxsize=1)
def _configure_root_logger() -> None:
    _enable_windows_ansi()

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    if root.handlers:
        root.handlers.clear()

    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(_ColourFormatter())

    file_handler = logging.handlers.TimedRotatingFileHandler(
        _APP_LOG,
        when="midnight",
        backupCount=7,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)

    error_handler = logging.handlers.RotatingFileHandler(
        _ERROR_LOG,
        maxBytes=2 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_formatter)

    root.addHandler(stream_handler)
    root.addHandler(file_handler)
    root.addHandler(error_handler)


def get_logger(name: str) -> logging.Logger:
    _configure_root_logger()
    return logging.getLogger(name)


@contextmanager
def log_timer(logger: logging.Logger, label: str) -> Generator[None, None, None]:
    start = time.perf_counter()
    logger.info("START: %s", label)
    try:
        yield
    except Exception:
        elapsed = time.perf_counter() - start
        logger.exception("FAILED: %s (%.2fs)", label, elapsed)
        raise
    elapsed = time.perf_counter() - start
    logger.info("DONE: %s (%.2fs)", label, elapsed)
