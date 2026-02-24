#!/usr/bin/env python3
"""
Centralized logging pipeline for the evaluation suite.

Every module in this package should obtain its logger through:

    from logger import get_logger
    logger = get_logger(__name__)

The pipeline wires three handlers automatically (once, at first import):
  1. Console   – INFO and above, coloured by level.
  2. Daily rotating file  (logs/evaluation.log)  – DEBUG and above, rotates at midnight.
  3. Dedicated error file (logs/errors.log)                – WARNING and above.

Additional utilities
--------------------
log_timer(logger, label)     – context-manager that logs ▶ START / ✔ DONE / ✗ FAILED
                                with elapsed time for any named block.
log_call(logger)             – function decorator that logs entry, exit, and duration.
log_section(logger, title)   – prints a bold ══ section banner to the log.
log_subsection(logger, title)– prints a lighter ── subsection banner.
"""

import functools
import logging
import logging.handlers
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator


# ─────────────────────────────────────────────────────────────────────────────
# ANSI colour codes (Windows 10+ and all modern Unix terminals)
# ─────────────────────────────────────────────────────────────────────────────
_RESET   = "\033[0m"
_BOLD    = "\033[1m"
_RED     = "\033[91m"
_YELLOW  = "\033[93m"
_GREEN   = "\033[92m"
_CYAN    = "\033[96m"
_BLUE    = "\033[94m"
_GREY    = "\033[37m"
_MAGENTA = "\033[95m"

_LEVEL_COLOURS: dict[str, str] = {
    "DEBUG":    _GREY,
    "INFO":     _GREEN,
    "WARNING":  _YELLOW,
    "ERROR":    _RED,
    "CRITICAL": _BOLD + _RED,
}

# ─────────────────────────────────────────────────────────────────────────────
# Log directory layout
#   evaluation/
#     logs/
#       evaluation.log             ← active file, rotates daily at midnight
#       evaluation.log.YYYY-MM-DD  ← rolled-over daily archives
#       errors.log                 ← dedicated error/warning sink
# ─────────────────────────────────────────────────────────────────────────────
_LOG_DIR = Path(__file__).parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)

_MAIN_LOG   = _LOG_DIR / "evaluation.log"
_ERROR_LOG  = _LOG_DIR / "errors.log"

# ─────────────────────────────────────────────────────────────────────────────
# Format strings
# ─────────────────────────────────────────────────────────────────────────────
_FILE_FMT  = "%(asctime)s | %(levelname)-8s | %(name)-40s | %(message)s"
_FILE_DATE = "%Y-%m-%d %H:%M:%S"


# ─────────────────────────────────────────────────────────────────────────────
# Coloured console formatter
# ─────────────────────────────────────────────────────────────────────────────
class _ColourFormatter(logging.Formatter):
    """Adds ANSI colour to the levelname field for console output."""

    _TEMPLATE = (
        "%(asctime)s │ {colour}%(levelname)-8s{reset} │ "
        "%(name)-35s │ %(message)s"
    )
    _DATE_FMT = "%H:%M:%S"

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        colour = _LEVEL_COLOURS.get(record.levelname, _RESET)
        fmt = self._TEMPLATE.format(colour=colour, reset=_RESET)
        return logging.Formatter(fmt, datefmt=self._DATE_FMT).format(record)


# ─────────────────────────────────────────────────────────────────────────────
# One-time root-logger configuration
# ─────────────────────────────────────────────────────────────────────────────
_root_configured: bool = False


def _ensure_root_configured() -> None:
    """Wire up the three handlers on the root logger exactly once."""
    global _root_configured
    if _root_configured:
        return

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)          # handlers decide their own threshold

    # ── 1. Console handler ───────────────────────────────────────────────────
    # Enable ANSI colours on Windows via ENABLE_VIRTUAL_TERMINAL_PROCESSING.
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception:
            pass  # graceful fallback — colours just won't appear

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(_ColourFormatter())

    # ── 2. Daily rotating main log (DEBUG+) ───────────────────────────────────
    main_file = logging.handlers.TimedRotatingFileHandler(
        _MAIN_LOG,
        when="midnight",              # rotate at 00:00 every day
        backupCount=7,               # keep 7 days of archives
        encoding="utf-8",
    )
    main_file.setLevel(logging.DEBUG)
    main_file.setFormatter(logging.Formatter(_FILE_FMT, datefmt=_FILE_DATE))

    # ── 3. Dedicated error log (WARNING+) ────────────────────────────────────
    error_file = logging.handlers.RotatingFileHandler(
        _ERROR_LOG,
        maxBytes=5 * 1024 * 1024,     # 5 MB per file
        backupCount=3,
        encoding="utf-8",
    )
    error_file.setLevel(logging.WARNING)
    error_file.setFormatter(logging.Formatter(_FILE_FMT, datefmt=_FILE_DATE))

    root.addHandler(console)
    root.addHandler(main_file)
    root.addHandler(error_file)

    root.info(
        "Logging pipeline initialised │ main=%s │ errors=%s",
        _MAIN_LOG,
        _ERROR_LOG,
    )

    _root_configured = True


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger that participates in the shared pipeline.

    Typical usage (once per module, at module level)::

        from logger import get_logger
        logger = get_logger(__name__)
    """
    _ensure_root_configured()
    return logging.getLogger(name)


# ── Timing context-manager ────────────────────────────────────────────────────

@contextmanager
def log_timer(
    logger: logging.Logger,
    operation: str,
    level: int = logging.INFO,
) -> Generator[None, None, None]:
    """
    Log the start, completion, and elapsed wall-clock time of any named block.

    Usage::

        with log_timer(logger, "face detection"):
            detect_faces(image_paths, output_dir)
    """
    logger.log(level, "▶ START   %s", operation)
    t0 = time.perf_counter()
    try:
        yield
    except Exception:
        elapsed = time.perf_counter() - t0
        logger.error("✗ FAILED  %s  (%.2fs elapsed before error)", operation, elapsed)
        raise
    else:
        elapsed = time.perf_counter() - t0
        logger.log(level, "✔ DONE    %s  (%.2fs)", operation, elapsed)


# ── Function decorator ────────────────────────────────────────────────────────

def log_call(logger: logging.Logger, level: int = logging.DEBUG):
    """
    Decorator that logs function entry, exit, and duration at *level*.

    Usage::

        @log_call(logger)
        def generate_image(prompt, seed, ...):
            ...
    """
    def decorator(fn):  # noqa: ANN001
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):  # noqa: ANN002, ANN003
            logger.log(level, "→ %s() called", fn.__qualname__)
            t0 = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                logger.error(
                    "✗ %s() raised %s after %.2fs: %s",
                    fn.__qualname__, type(exc).__name__, elapsed, exc,
                )
                raise
            else:
                elapsed = time.perf_counter() - t0
                logger.log(level, "← %s() completed (%.2fs)", fn.__qualname__, elapsed)
                return result
        return wrapper
    return decorator


# ── Section / subsection banners ─────────────────────────────────────────────

def log_section(logger: logging.Logger, title: str, width: int = 72) -> None:
    """
    Emit a bold ══ section banner.  Use to mark top-level pipeline stages.

    Example output::

        ════════════════════════════════════════════════════════════════════════
          ORIGINAL PROMPTS GENERATION
        ════════════════════════════════════════════════════════════════════════
    """
    bar = "═" * width
    logger.info(bar)
    logger.info("  %s", title.upper())
    logger.info(bar)


def log_subsection(logger: logging.Logger, title: str, width: int = 72) -> None:
    """
    Emit a lighter ── subsection banner.  Use to mark sub-steps within a stage.

    Example output::

        ────────────────────────────────────────────────────────────────────────
          Computing bias/diversity metrics
        ────────────────────────────────────────────────────────────────────────
    """
    bar = "─" * width
    logger.info(bar)
    logger.info("  %s", title)
    logger.info(bar)
