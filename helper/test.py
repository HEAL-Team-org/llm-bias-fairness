#!/usr/bin/env python3
"""Sanity test for the local Qwen3 generation server.

Checks that serve_qwen3.py is running and can produce a non-empty
completion.  The test does two things:
    1. GET  /health  — confirms the model is loaded.
    2. POST /generate — sends a sample prompt and validates the output.

No embedding model is involved here; this script only exercises the
text-generation endpoint.

How to run:
    # Make sure the Qwen3 server is already running (see serve_qwen3.py)
    uvicorn serve_qwen3:app --host 0.0.0.0 --port 8001 --workers 1

    # Run the sanity test from the helper/ directory
    cd helper
    python test.py

    # Override the server URL or prompt
    python test.py --url http://localhost:8001 --prompt "A nurse helping a patient"

    # Use a longer generation
    python test.py --max-new-tokens 512

Exit codes:
    0  — server is healthy and returned a valid completion.
    1  — health check failed or generation returned empty/error output.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import requests

from logger import get_logger

logger = get_logger(__name__)

QWEN3_URL = "http://localhost:8001"
DEFAULT_PROMPT = "Improve this prompt for diversity and inclusion: a doctor treating a patient"


def check_health(base_url: str) -> bool:
    try:
        response = requests.get(f"{base_url}/health", timeout=20)
    except requests.exceptions.RequestException as exc:
        logger.error("Health check failed: %s", exc)
        return False

    if response.status_code != 200:
        logger.error("Health endpoint returned HTTP %s: %s", response.status_code, response.text)
        return False

    logger.info("Health OK: %s", response.json())
    return True


def run_generation(base_url: str, prompt: str, max_new_tokens: int) -> dict | None:
    """Send the generation request and return the full response dict (or None on failure)."""
    payload = {
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
        "repetition_penalty": 1.05,
    }

    try:
        response = requests.post(f"{base_url}/generate", json=payload, timeout=600)
    except requests.exceptions.RequestException as exc:
        logger.error("Generation request failed: %s", exc)
        return None

    if response.status_code != 200:
        logger.error("Generation failed (HTTP %s): %s", response.status_code, response.text)
        return None

    data = response.json()
    output_text = data.get("output", "").strip()

    if not output_text:
        logger.error("Generation returned empty output: %s", data)
        return None

    logger.info("Model: %s", data.get("model"))
    logger.info("Prompt: %s", data.get("prompt"))
    logger.info("Output:\n%s", output_text)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity test for serve_qwen3.py")
    parser.add_argument("--url", default=QWEN3_URL, help="Base URL of Qwen3 server")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt to test")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Generation length")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write the JSON result (default: outputs/ next to this file)",
    )
    args = parser.parse_args()

    if not check_health(args.url):
        sys.exit(1)

    data = run_generation(args.url, args.prompt, args.max_new_tokens)
    if data is None:
        sys.exit(1)

    # ── save JSON output ──────────────────────────────────────────────────────
    # Resolve output directory: ../outputs/ relative to this helper/ file
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path(__file__).parent.parent / "outputs"

    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"test_result_{timestamp}.json"

    result = {
        "timestamp": timestamp,
        "server_url": args.url,
        "request": {
            "prompt": args.prompt,
            "max_new_tokens": args.max_new_tokens,
        },
        "response": data,
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info("Qwen3 sanity test passed")
    logger.info("Result saved to: %s", out_path)
    print(f"\nOutput saved to: {out_path}")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
