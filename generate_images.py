#!/usr/bin/env python3
"""
Image Generation Pipeline
=========================

Generates base and enhanced images for every row in the prompts CSV using the
Qwen-Image-2512 FastAPI server (serve_qwen.py).

ALL images are written to disk before any metric calculation begins.

Directory layout produced
--------------------------
outputs/
  directional_similarity/
    row_<ID>/
      base_seed<SEED:05d>.png
      enhanced_seed<SEED:05d>.png
      ...

Usage
-----
    # generate with Qwen (default)
    python generate_images.py --model qwen

    # generate with FLUX.1-dev
    python generate_images.py --model flux-dev

    # 10 images per variant
    python generate_images.py --model qwen --num-images 10

    # force re-generation even if images already exist
    python generate_images.py --model qwen --force-regenerate

    # choose which enhanced prompt variant to use (0-indexed)
    python generate_images.py --model qwen --enhanced-idx 0

CLI flags
---------
  --model           which model to use: qwen | flux-dev  (default: qwen)
  --csv             prompts CSV  (default: inputs/prompts.csv)
  --image-dir       root output folder (default: outputs/directional_similarity)
                    images are written to <image-dir>/<model>/row_<ID>/
  --server          override generation server URL (auto-set from --model)
  --num-images      number of images to generate per prompt variant (default: 5)
  --enhanced-idx    which enhanced prompt variant to use, 0-indexed (default: 0)
  --force-regenerate  ignore cached PNGs and regenerate every image
  --width / --height  override image size (auto-set from --model)
  --steps             inference steps (default: 50)
"""

from __future__ import annotations

import argparse
import ast
import csv
import random
import io
import json
import sys
import time
from pathlib import Path
from typing import Optional

import requests
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from logger import get_logger, log_section, log_subsection, log_timer

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Per-model configuration
# ─────────────────────────────────────────────────────────────────────────────

# Supported generation models.
# Each entry defines the server URL, the native 1:1 resolution and a human-
# readable folder slug used to separate outputs under image_output_dir.
MODEL_CONFIGS: dict[str, dict] = {
    "qwen": {
        "server": "http://localhost:8001",
        "width":  1328,   # Qwen-Image-2512 native 1:1
        "height": 1328,
    },
    "flux-dev": {
        "server": "http://localhost:8004",
        "width":  1024,   # FLUX.1-dev native 1:1
        "height": 1024,
    },
}
DEFAULT_MODEL: str = "qwen"

# ─────────────────────────────────────────────────────────────────────────────
# Shared defaults (model-independent)
# ─────────────────────────────────────────────────────────────────────────────
GENERATION_SERVER:   str  = MODEL_CONFIGS[DEFAULT_MODEL]["server"]
DEFAULT_NUM_IMAGES:  int  = 5
IMAGE_WIDTH:         int  = MODEL_CONFIGS[DEFAULT_MODEL]["width"]
IMAGE_HEIGHT:        int  = MODEL_CONFIGS[DEFAULT_MODEL]["height"]
NUM_INFERENCE_STEPS: int  = 50
IMAGE_OUTPUT_DIR:    Path = Path("outputs") / "directional_similarity"
INPUT_CSV:           Path = Path("inputs") / "prompts.csv"


def _generate_seeds(n: int) -> list[int]:
    """
    Deterministically generate *n* distinct random seeds.

    A fixed internal base seed is used so that the same ``--num-images`` value
    always produces the same seed list across runs — this makes it safe to
    resume an interrupted generation without re-generating already-cached images.
    """
    rng = random.Random(0)          # fixed base — purely internal, not user-facing
    seen: set[int] = set()
    seeds: list[int] = []
    while len(seeds) < n:
        s = rng.randint(0, 2**31 - 1)
        if s not in seen:
            seen.add(s)
            seeds.append(s)
    return seeds


# ─────────────────────────────────────────────────────────────────────────────
# Server helpers
# ─────────────────────────────────────────────────────────────────────────────

def wait_for_server(
    base_url: str,
    retries:  int   = 60,
    interval: float = 10.0,
) -> None:
    """Poll /health until the generation server reports ready (or raise)."""
    logger.info("Waiting for generation server at %s …", base_url)
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(f"{base_url}/health", timeout=10)
            if r.status_code == 200:
                data = r.json()
                logger.info(
                    "Server ready  |  model=%s",
                    data.get("model", "unknown"),
                )
                return
            logger.warning(
                "  [%d/%d] Not ready (HTTP %d) — retrying in %.0fs …",
                attempt, retries, r.status_code, interval,
            )
        except requests.exceptions.ConnectionError:
            logger.warning(
                "  [%d/%d] Unreachable — retrying in %.0fs …",
                attempt, retries, interval,
            )
        time.sleep(interval)
    raise RuntimeError(
        f"Generation server at {base_url} did not become ready "
        f"after {retries} attempts ({retries * interval:.0f} s)."
    )


def _generate_one_image(
    server_url: str,
    prompt:     str,
    seed:       int,
    width:      int = IMAGE_WIDTH,
    height:     int = IMAGE_HEIGHT,
    steps:      int = NUM_INFERENCE_STEPS,
) -> Image.Image:
    """POST to /generate and return the resulting PIL image."""
    payload = {
        "prompt":              prompt,
        "seed":                seed,
        "width":               width,
        "height":              height,
        "num_inference_steps": steps,
    }
    r = requests.post(f"{server_url}/generate", json=payload, timeout=600)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")


# ─────────────────────────────────────────────────────────────────────────────
# CSV helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_enhanced_prompts(raw: str) -> list[str]:
    """Parse the enhanced_prompts cell (JSON array or Python literal)."""
    raw = raw.strip()
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(raw)  # type: ignore[operator]
            if isinstance(parsed, list):
                return [str(p) for p in parsed]
            if isinstance(parsed, str):
                return [parsed]
        except (json.JSONDecodeError, ValueError, SyntaxError):
            continue
    logger.warning("Could not parse enhanced_prompts as list; using raw string.")
    return [raw]


def load_prompt_pairs(csv_path: Path) -> list[dict]:
    """
    Load prompt pairs from the CSV.

    Returns list of::
        { "row_id": str, "base_prompt": str, "enhanced_prompts": list[str] }
    """
    rows: list[dict] = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append(
                {
                    "row_id":           str(row.get("__row_id__", row.get("row_id", "?"))),
                    "base_prompt":      row["generic_prompt"].strip(),
                    "enhanced_prompts": _parse_enhanced_prompts(row["enhanced_prompts"]),
                }
            )
    logger.info("Loaded %d prompt pairs from %s", len(rows), csv_path)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Per-prompt generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_images_for_prompt(
    prompt:           str,
    output_dir:       Path,
    tag:              str,
    seeds:            list[int],
    server_url:       str,
    force_regenerate: bool = False,
    width:            int  = IMAGE_WIDTH,
    height:           int  = IMAGE_HEIGHT,
    steps:            int  = NUM_INFERENCE_STEPS,
) -> list[Path]:
    """
    Generate (or cache-load) one image per seed for *prompt*.

    Naming convention: <output_dir>/<tag>_seed<SEED:05d>.png

    If a PNG already exists and force_regenerate is False the request is
    skipped — fast path for resuming interrupted runs.

    Returns the ordered list of image paths (one per seed, matching seed order).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for seed in seeds:
        img_path = output_dir / f"{tag}_seed{seed:05d}.png"
        paths.append(img_path)

        if img_path.exists() and not force_regenerate:
            logger.debug("  [cache] %s  (skip)", img_path.name)
            continue

        logger.info("  Generating  tag=%-10s  seed=%-7d  →  %s", tag, seed, img_path.name)
        with log_timer(logger, f"{tag} seed={seed}"):
            img = _generate_one_image(server_url, prompt, seed, width, height, steps)
            img.save(img_path, format="PNG")
            logger.debug("  Saved %s  (%dx%d)", img_path.name, img.width, img.height)

    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Main generation pipeline
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_images(
    csv_path:         Path  = INPUT_CSV,
    image_output_dir: Path  = IMAGE_OUTPUT_DIR,
    model_name:       str   = DEFAULT_MODEL,
    server_url:       str   = GENERATION_SERVER,
    num_images:       int   = DEFAULT_NUM_IMAGES,
    enhanced_idx:     int   = 0,
    force_regenerate: bool  = False,
    width:            int   = IMAGE_WIDTH,
    height:           int   = IMAGE_HEIGHT,
    steps:            int   = NUM_INFERENCE_STEPS,
) -> dict[str, dict[str, list[Path]]]:
    """
    Generate ALL base and enhanced images for every prompt pair.

    Generation is completed for every row and both variants (base + enhanced)
    before this function returns — no metric calculation takes place here.

    Parameters
    ----------
    csv_path         : Path to the input prompts CSV.
    image_output_dir : Root directory for generated images  (model subfolder
                       is appended automatically: <image_output_dir>/<model_name>/).
    model_name       : Slug identifying the generation model ("qwen", "flux-dev").
                       Used as a subfolder so outputs from different models never
                       overwrite each other.
    server_url       : Generation server URL.
    num_images       : Number of images to generate per prompt variant.
                       Seeds are derived automatically and deterministically
                       so the same count always maps to the same seeds.
    enhanced_idx     : Which enhanced prompt variant to use (0-indexed).
    force_regenerate : Ignore disk cache and regenerate every image.
    width / height   : Pixel dimensions for generated images.
    steps            : Diffusion inference steps.

    Returns
    -------
    Nested dict keyed by row_id::
        {
            "<row_id>": {
                "base_paths":     [Path, ...],
                "enhanced_paths": [Path, ...],
            },
            ...
        }
    """
    seeds = _generate_seeds(num_images)

    # Outputs for this model live in their own subfolder
    model_output_dir = image_output_dir / model_name

    log_section(logger, "Image Generation Pipeline")

    logger.info("Model           : %s",  model_name)
    logger.info("Server          : %s",  server_url)
    logger.info("Output dir      : %s",  model_output_dir)
    logger.info("Images/variant  : %d",  num_images)
    logger.info("Seeds           : %s",  seeds)
    logger.info("Resolution      : %d×%d", width, height)
    logger.info("Steps           : %d",  steps)
    logger.info("Enhanced idx    : %d",  enhanced_idx)
    logger.info("Force regen     : %s",  force_regenerate)

    wait_for_server(server_url)
    prompt_pairs = load_prompt_pairs(csv_path)

    result: dict[str, dict[str, list[Path]]] = {}
    total_rows = len(prompt_pairs)

    for i, entry in enumerate(prompt_pairs, start=1):
        row_id       = entry["row_id"]
        base_prompt  = entry["base_prompt"]
        enh_list     = entry["enhanced_prompts"]

        if not enh_list:
            logger.warning("Row %s has no enhanced prompts — skipping.", row_id)
            continue

        idx              = min(enhanced_idx, len(enh_list) - 1)
        enhanced_prompt  = enh_list[idx]
        row_dir          = model_output_dir / f"row_{row_id}"

        log_subsection(
            logger,
            f"Row {row_id} ({i}/{total_rows})  ·  {num_images} images per variant",
        )
        logger.info("  Base     : %.100s", base_prompt)
        logger.info("  Enhanced : %.100s", enhanced_prompt)

        # ── Base images ────────────────────────────────────────────────────
        logger.info("  → BASE images …")
        base_paths = generate_images_for_prompt(
            prompt           = base_prompt,
            output_dir       = row_dir,
            tag              = "base",
            seeds            = seeds,
            server_url       = server_url,
            force_regenerate = force_regenerate,
            width=width, height=height, steps=steps,
        )

        # ── Enhanced images ────────────────────────────────────────────────
        logger.info("  → ENHANCED images …")
        enh_paths = generate_images_for_prompt(
            prompt           = enhanced_prompt,
            output_dir       = row_dir,
            tag              = "enhanced",
            seeds            = seeds,
            server_url       = server_url,
            force_regenerate = force_regenerate,
            width=width, height=height, steps=steps,
        )

        result[row_id] = {
            "base_paths":     base_paths,
            "enhanced_paths": enh_paths,
        }
        logger.info(
            "  ✔ Row %-5s  base=%d images  enhanced=%d images",
            row_id, len(base_paths), len(enh_paths),
        )

    total_imgs = sum(
        len(v["base_paths"]) + len(v["enhanced_paths"]) for v in result.values()
    )
    log_section(
        logger,
        f"Generation complete  |  model={model_name}  |  "
        f"{len(result)} rows  |  {total_imgs} images",
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate base + enhanced images for all prompt pairs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",            type=str,  default=DEFAULT_MODEL,
                   choices=list(MODEL_CONFIGS),
                   help="Generation model. Sets default server URL and resolution.")
    p.add_argument("--csv",              type=Path, default=INPUT_CSV,
                   metavar="PATH",       help="Input prompts CSV.")
    p.add_argument("--image-dir",        type=Path, default=IMAGE_OUTPUT_DIR,
                   metavar="DIR",        help="Root output dir. Images saved to <DIR>/<model>/row_<ID>/.")
    p.add_argument("--server",           type=str,  default=None,
                   metavar="URL",        help="Override generation server URL (auto-set from --model).")
    p.add_argument("--num-images",       type=int,  default=DEFAULT_NUM_IMAGES,
                   metavar="N",          help="Number of images to generate per prompt variant. Seeds are auto-derived.")
    p.add_argument("--enhanced-idx",     type=int,  default=0,
                   metavar="IDX",        help="Index into the enhanced_prompts list.")
    p.add_argument("--force-regenerate", action="store_true",
                   help="Re-generate even if PNG already exists on disk.")
    p.add_argument("--width",            type=int,  default=None, metavar="PX",
                   help="Override image width in pixels (auto-set from --model).")
    p.add_argument("--height",           type=int,  default=None, metavar="PX",
                   help="Override image height in pixels (auto-set from --model).")
    p.add_argument("--steps",            type=int,  default=NUM_INFERENCE_STEPS,
                   metavar="N",          help="Diffusion inference steps per image.")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    # Apply model-specific defaults for anything the user did not explicitly set
    cfg = MODEL_CONFIGS[args.model]
    server = args.server if args.server is not None else cfg["server"]
    width  = args.width  if args.width  is not None else cfg["width"]
    height = args.height if args.height is not None else cfg["height"]

    generate_all_images(
        csv_path         = args.csv,
        image_output_dir = args.image_dir,
        model_name       = args.model,
        server_url       = server,
        num_images       = args.num_images,
        enhanced_idx     = args.enhanced_idx,
        force_regenerate = args.force_regenerate,
        width            = width,
        height           = height,
        steps            = args.steps,
    )


if __name__ == "__main__":
    main()
