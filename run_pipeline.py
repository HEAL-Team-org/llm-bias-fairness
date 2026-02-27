#!/usr/bin/env python3
"""
Full Evaluation Pipeline — Generate Images → Calculate Directional Similarity
==============================================================================

Orchestrates the two-stage pipeline:

  Stage 1 — Image Generation (generate_images.py)
  ------------------------------------------------
  Generates ALL base and enhanced images for every prompt pair in the CSV,
  writing them to disk.  No metric calculation happens until every image is
  present on disk.

  Stage 2 — Metric Calculation (calculate_metric.py)
  ---------------------------------------------------
  Loads SigLIP 2 Giant, encodes all prompt pairs and their generated images,
  then computes the CLIP directional similarity score for each row.

This script is the single entry-point for a complete evaluation run.
You can also run each stage independently:

    python generate_images.py   # Stage 1 only
    python calculate_metric.py  # Stage 2 only (needs Stage 1 output)

Pipeline
--------
    CSV  ──►  generate_images()  ──►  outputs/directional_similarity/<model>/
                                            row_<ID>/base_seed*.png
                                            row_<ID>/enhanced_seed*.png
         ──►  calculate_directional_similarity()
                                      ──►  results/directional_similarity_<TS>.json
                                      ──►  results/directional_similarity_<TS>.csv

Usage
-----
    # default: 5 images per variant, server on localhost:8001, CLIP on cuda:0
    python run_pipeline.py

    # 10 images per variant, second GPU for metric, more steps
    python run_pipeline.py --num-images 10 --clip-device cuda:1 --steps 30

    # skip generation (images already on disk) and only run metric
    python run_pipeline.py --skip-generation

    # re-generate all images (ignore cache) then run metric
    python run_pipeline.py --force-regenerate

CLI flags
---------
  General
    --csv             prompts CSV (default: inputs/prompts.csv)
    --enhanced-idx    which enhanced prompt variant to use (default: 0)
    --results-dir     where to write results (default: results)

  Stage 1 — generation
    --model           which generation model to use: qwen, flux-dev (default: qwen)
    --image-dir       root image output dir (default: outputs/directional_similarity)
    --server          generation server URL (auto-set from --model if omitted)
    --num-images      number of images per prompt variant (default: 5)
    --width / --height  image dimensions in pixels (auto-set from --model if omitted)
    --steps           diffusion inference steps (default: 50)
    --force-regenerate  ignore cached PNGs and regenerate
    --skip-generation   skip Stage 1 entirely (use pre-existing images)

  Stage 2 — metric
    --clip-device     torch device for SigLIP 2 (default: cuda:0)
    --batch-size      images per SigLIP2 forward pass (default: 8)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from logger import get_logger, log_section
from generate_images import (
    generate_all_images,
    MODEL_CONFIGS,
    DEFAULT_MODEL,
    DEFAULT_NUM_IMAGES,
    GENERATION_SERVER,
    IMAGE_OUTPUT_DIR,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    NUM_INFERENCE_STEPS,
    INPUT_CSV,
)
from calculate_metric import (
    calculate_directional_similarity,
    RESULTS_DIR,
    DEFAULT_BATCH,
    SIGLIP2_MODEL_ID,
)

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    csv_path:         Path              = INPUT_CSV,
    image_output_dir: Path              = IMAGE_OUTPUT_DIR,
    results_dir:      Path              = RESULTS_DIR,
    server_url:       str               = GENERATION_SERVER,
    num_images:       int               = DEFAULT_NUM_IMAGES,
    enhanced_idx:     int               = 0,
    model_name:       str               = DEFAULT_MODEL,
    clip_device:      torch.device | None = None,
    batch_size:       int               = DEFAULT_BATCH,
    force_regenerate: bool              = False,
    skip_generation:  bool              = False,
    width:            int               = IMAGE_WIDTH,
    height:           int               = IMAGE_HEIGHT,
    steps:            int               = NUM_INFERENCE_STEPS,
) -> list[dict]:
    """
    Run the complete two-stage evaluation pipeline.

    Stage 1 (unless skip_generation=True):
        Generate all base + enhanced images and write them to disk.
        The generation server is polled until ready before any requests are sent.
        Cached images are reused unless force_regenerate=True.

    Stage 2:
        Load SigLIP 2 Giant, compute CLIP directional similarity for every
        prompt pair, and persist results to JSON + CSV.

    Parameters
    ----------
    csv_path         : Path to the input prompts CSV.
    image_output_dir : Root directory for generated / cached images.
    results_dir      : Directory where result files are written.
    server_url       : Base URL of the image generation server.
    num_images       : Number of images to generate per prompt variant.
                       Seeds are derived automatically and deterministically.
    enhanced_idx     : Which enhanced prompt variant to use (0-indexed).
    model_name       : Generation model key (e.g. 'qwen', 'flux-dev').
                       Controls the server, resolution, and output subfolder.
    clip_device      : Torch device for SigLIP 2 inference.
    batch_size       : Images per SigLIP 2 forward pass.
    force_regenerate : Ignore cached PNGs and regenerate every image.
    skip_generation  : Skip Stage 1 and only run Stage 2.
    width / height   : Pixel dimensions for generated images.
    steps            : Diffusion inference steps.

    Returns
    -------
    List of per-row result dicts (see calculate_metric.py for schema).
    """
    log_section(
        logger,
        "RAG Prompt Evaluation  ·  Directional Similarity Pipeline",
    )
    logger.info("Model (metric) : %s", SIGLIP2_MODEL_ID)
    logger.info("Gen model      : %s", model_name)
    logger.info("Server (gen)   : %s", server_url)
    logger.info("Images/variant : %d", num_images)
    logger.info("Skip generation: %s", skip_generation)
    logger.info("Force regen    : %s", force_regenerate)

    # ──────────────────────────────────────────────────────────────────────
    # Stage 1 — Generate ALL images
    # ──────────────────────────────────────────────────────────────────────
    if skip_generation:
        logger.info(
            "Stage 1 skipped — using pre-existing images in %s", image_output_dir
        )
    else:
        log_section(logger, "Stage 1  ·  Image Generation")
        generate_all_images(
            csv_path         = csv_path,
            image_output_dir = image_output_dir,
            server_url       = server_url,
            num_images       = num_images,
            enhanced_idx     = enhanced_idx,
            model_name       = model_name,
            force_regenerate = force_regenerate,
            width            = width,
            height           = height,
            steps            = steps,
        )
        log_section(
            logger,
            "Stage 1 complete  ·  all images written to disk",
        )

    # ──────────────────────────────────────────────────────────────────────
    # Stage 2 — Compute metric for ALL rows
    # ──────────────────────────────────────────────────────────────────────
    log_section(logger, "Stage 2  ·  Directional Similarity Calculation")
    results = calculate_directional_similarity(
        csv_path     = csv_path,
        image_dir    = image_output_dir,
        results_dir  = results_dir,
        device       = clip_device,
        model_name   = model_name,
        enhanced_idx = enhanced_idx,
        batch_size   = batch_size,
    )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Full pipeline: generate images then compute "
            "CLIP directional similarity with SigLIP 2 Giant."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── General ─────────────────────────────────────────────────────────────
    grp_general = p.add_argument_group("general")
    grp_general.add_argument(
        "--csv", type=Path, default=INPUT_CSV, metavar="PATH",
        help="Input prompts CSV (columns: generic_prompt, enhanced_prompts).",
    )
    grp_general.add_argument(
        "--enhanced-idx", type=int, default=0, metavar="IDX",
        help="Index into the enhanced_prompts array (0 = first variant).",
    )
    grp_general.add_argument(
        "--results-dir", type=Path, default=RESULTS_DIR, metavar="DIR",
        help="Directory for output JSON and CSV result files.",
    )

    # ── Stage 1 — generation ────────────────────────────────────────────────
    grp_gen = p.add_argument_group("stage 1 — image generation")
    grp_gen.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        choices=list(MODEL_CONFIGS),
        help="Generation model to use. Controls server URL, resolution, and output subfolder.",
    )
    grp_gen.add_argument(
        "--image-dir", type=Path, default=IMAGE_OUTPUT_DIR, metavar="DIR",
        help="Root directory for generated (or cached) images.",
    )
    grp_gen.add_argument(
        "--server", type=str, default=None, metavar="URL",
        help="Base URL of the generation server (auto-set from --model if omitted).",
    )
    grp_gen.add_argument(
        "--num-images", type=int, default=DEFAULT_NUM_IMAGES, metavar="N",
        help=(
            "Number of images to generate per prompt variant (base and enhanced). "
            "Seeds are derived automatically and deterministically from this count."
        ),
    )
    grp_gen.add_argument(
        "--width",  type=int, default=None,  metavar="PX",
        help="Generated image width in pixels (auto-set from --model if omitted).",
    )
    grp_gen.add_argument(
        "--height", type=int, default=None, metavar="PX",
        help="Generated image height in pixels (auto-set from --model if omitted).",
    )
    grp_gen.add_argument(
        "--steps", type=int, default=NUM_INFERENCE_STEPS, metavar="N",
        help="Number of diffusion inference steps per image.",
    )
    grp_gen.add_argument(
        "--force-regenerate", action="store_true",
        help="Re-generate all images even if they already exist on disk.",
    )
    grp_gen.add_argument(
        "--skip-generation", action="store_true",
        help="Skip Stage 1 entirely and run the metric on pre-existing images.",
    )

    # ── Stage 2 — metric ────────────────────────────────────────────────────
    grp_metric = p.add_argument_group("stage 2 — metric calculation")
    grp_metric.add_argument(
        "--clip-device", type=str, default=None, metavar="DEVICE",
        help=(
            "Torch device for SigLIP 2 (e.g. cuda:0, cuda:1, cpu). "
            "Defaults to the first available CUDA device."
        ),
    )
    grp_metric.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH, metavar="N",
        help="Number of images per SigLIP 2 forward pass.",
    )

    return p


def main() -> None:
    args        = _build_parser().parse_args()
    clip_device = torch.device(args.clip_device) if args.clip_device else None

    cfg    = MODEL_CONFIGS[args.model]
    server = args.server or cfg["server"]
    width  = args.width  or cfg["width"]
    height = args.height or cfg["height"]

    run_pipeline(
        csv_path         = args.csv,
        image_output_dir = args.image_dir,
        results_dir      = args.results_dir,
        server_url       = server,
        num_images       = args.num_images,
        enhanced_idx     = args.enhanced_idx,
        model_name       = args.model,
        clip_device      = clip_device,
        batch_size       = args.batch_size,
        force_regenerate = args.force_regenerate,
        skip_generation  = args.skip_generation,
        width            = width,
        height           = height,
        steps            = args.steps,
    )


if __name__ == "__main__":
    main()
