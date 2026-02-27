#!/usr/bin/env python3
"""
CLIP Directional Similarity — RAG Enhancement Evaluation Pipeline
==================================================================

Measures how well a RAG-enhanced prompt shifts image generation in the
same direction as the text embedding shift (base → enhanced).

Metric (StyleGAN-NADA / Gal et al., 2021):

    S_dir = (1/N) * Σᵢ  cosine( delta_I_i ,  delta_T )

    where
        delta_T   = clip_text(enhanced_prompt) − clip_text(base_prompt)
        delta_I_i = clip_image(enhanced_img_i) − clip_image(base_img_i)

    Both direction vectors are L2-normalised before the cosine is computed.
    A score of +1 means the visual shift perfectly mirrors the textual shift.
    A score of  0 means the two shifts are orthogonal (unrelated).
    A score of -1 means they point in opposite directions.

CLIP model used:
    laion/CLIP-ViT-H-14-laion2B-s32B-b79K
    → ViT-H/14 trained on LAION-2B; state-of-the-art CLIP available via the
      HuggingFace Transformers library. Embedding dim = 1024.

Image generation:
    Images are generated via the Qwen-Image-2512 FastAPI server (serve_qwen.py)
    running on localhost:8001.  Base and enhanced images are generated with the
    *same* set of random seeds so the paired comparison is valid.

Input CSV  (inputs/prompts.csv):
    Columns used:
        __row_id__       – row identifier
        generic_prompt   – the original / base prompt
        enhanced_prompts – JSON array of enhanced prompt strings

Output:
    outputs/directional_similarity/row_<ID>/base_seed<SEED>.png
    outputs/directional_similarity/row_<ID>/enhanced_seed<SEED>.png
    results/directional_similarity_<TIMESTAMP>.json
    results/directional_similarity_<TIMESTAMP>.csv

Usage examples:
    # full pipeline: generate images + evaluate
    python directional_similarity.py

    # evaluate only (skip generation, use cached images)
    python directional_similarity.py --eval-only

    # custom seeds, run CLIP on cuda:1, use Flux server on port 8000
    python directional_similarity.py --seeds 10 20 30 --clip-device cuda:1 --server http://localhost:8000

    # force re-generation even if images already exist
    python directional_similarity.py --force-regenerate
"""

from __future__ import annotations

import argparse
import ast
import csv
import io
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# Make the workspace root importable (logger.py lives at the root)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from logger import get_logger, log_section, log_subsection, log_timer

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration defaults  (all overridable via CLI)
# ─────────────────────────────────────────────────────────────────────────────

# Best CLIP model available through HuggingFace Transformers:
#   ViT-H/14 trained on LAION-2B (s32B-b79K).
#   Embedding dim: 1024. Significantly outperforms OpenAI ViT-L/14.
#   Fits comfortably in a single A100 40 GB at float16.
CLIP_MODEL_ID: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"

# Image generation server (serve_qwen.py via uvicorn)
GENERATION_SERVER: str = "http://localhost:8001"

# Seeds — base and enhanced images are generated with the *same* seeds so that
# the only variable is the prompt, making the paired delta_I well-defined.
DEFAULT_SEEDS: list[int] = [42, 123, 456, 789, 1024]

# Native Qwen-Image-2512 1:1 resolution (matches the server default)
IMAGE_WIDTH:  int = 1024
IMAGE_HEIGHT: int = 1024
NUM_INFERENCE_STEPS: int = 50

# Filesystem layout
IMAGE_OUTPUT_DIR: Path = Path("outputs") / "directional_similarity"
RESULTS_DIR:      Path = Path("results")
INPUT_CSV:        Path = Path("inputs") / "prompts.csv"


# ─────────────────────────────────────────────────────────────────────────────
# CLIP model — singleton loader
# ─────────────────────────────────────────────────────────────────────────────

_clip_model:     Optional[CLIPModel]     = None
_clip_processor: Optional[CLIPProcessor] = None
_clip_device:    Optional[torch.device]  = None


def load_clip_model(
    device: Optional[torch.device] = None,
) -> tuple[CLIPModel, CLIPProcessor, torch.device]:
    """
    Load CLIP once and cache the result in module-level globals.
    Subsequent calls return the cached objects immediately.
    """
    global _clip_model, _clip_processor, _clip_device

    if _clip_model is not None:
        assert _clip_processor is not None
        assert _clip_device is not None
        return _clip_model, _clip_processor, _clip_device

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_section(logger, f"Loading CLIP  ·  {CLIP_MODEL_ID}")
    logger.info("Target device : %s", device)
    logger.info(
        "Model         : ViT-H/14 trained on LAION-2B  "
        "(1024-dim embeddings, best available via HF Transformers)"
    )

    with log_timer(logger, "CLIP processor + model load"):
        processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
        model = CLIPModel.from_pretrained(
            CLIP_MODEL_ID,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        ).to(device)
        model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info("Model loaded  |  %.1f M parameters", n_params)

    if device.type == "cuda":
        used  = torch.cuda.memory_allocated(device) / 1024**3
        total = torch.cuda.get_device_properties(device).total_memory / 1024**3
        logger.info("VRAM after CLIP load : %.2f / %.2f GB  (%s)", used, total, device)

    _clip_model, _clip_processor, _clip_device = model, processor, device
    return model, processor, device


# ─────────────────────────────────────────────────────────────────────────────
# Embedding helpers
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_text_embedding(
    model: CLIPModel,
    processor: CLIPProcessor,
    text: str,
    device: torch.device,
) -> torch.Tensor:
    """
    Encode *text* → L2-normalised embedding of shape (D,).

    Automatically truncates to CLIP's 77-token context limit.
    """
    inputs = processor(
        text=[text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77,
    ).to(device)

    features = model.get_text_features(**inputs)       # (1, D)
    features = F.normalize(features, dim=-1)           # unit sphere
    return features.squeeze(0)                         # (D,)


@torch.no_grad()
def get_image_embedding(
    model: CLIPModel,
    processor: CLIPProcessor,
    image: Image.Image,
    device: torch.device,
) -> torch.Tensor:
    """
    Encode a PIL *image* → L2-normalised embedding of shape (D,).
    """
    inputs = processor(
        images=image,
        return_tensors="pt",
    ).to(device)

    features = model.get_image_features(**inputs)      # (1, D)
    features = F.normalize(features, dim=-1)           # unit sphere
    return features.squeeze(0)                         # (D,)


# ─────────────────────────────────────────────────────────────────────────────
# Core metric function
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_rag_enhancement_direction(
    base_prompt: str,
    enhanced_prompt: str,
    base_image_paths: list[str | Path],
    enhanced_image_paths: list[str | Path],
    device: Optional[torch.device] = None,
) -> tuple[list[float], float]:
    """
    Calculate the CLIP Directional Similarity metric.

    For each paired (base_image, enhanced_image) generated with the *same*
    random seed, the metric checks whether the visual shift mirrors the
    textual shift:

        delta_T      = clip_text(enhanced_prompt) − clip_text(base_prompt)
        delta_I_i    = clip_image(enhanced_img_i) − clip_image(base_img_i)
        score_i      = cosine_similarity(delta_T, delta_I_i)
        overall_score = mean(score_i)

    Parameters
    ----------
    base_prompt          : Original / generic prompt.
    enhanced_prompt      : RAG-enhanced prompt.
    base_image_paths     : Paths to images generated from *base_prompt*.
    enhanced_image_paths : Paths to images generated from *enhanced_prompt*
                           using the **same** random seeds.
    device               : Torch device. Auto-detected (preferring CUDA) if None.

    Returns
    -------
    (per_pair_scores, average_score)
        per_pair_scores : list of float  – one cosine score per seed pair.
        average_score   : float          – mean of per_pair_scores.
    """
    if len(base_image_paths) != len(enhanced_image_paths):
        raise ValueError(
            f"base_image_paths length ({len(base_image_paths)}) must equal "
            f"enhanced_image_paths length ({len(enhanced_image_paths)})."
        )
    if not base_image_paths:
        raise ValueError("Image path lists must not be empty.")

    model, processor, device = load_clip_model(device)

    # ── Text direction vector ──────────────────────────────────────────────
    logger.debug("Encoding text direction …")
    with log_timer(logger, "text direction encoding"):
        emb_base     = get_text_embedding(model, processor, base_prompt,     device)
        emb_enhanced = get_text_embedding(model, processor, enhanced_prompt, device)

        delta_T = emb_enhanced - emb_base                          # (D,)
        # Re-normalise the difference so cosine comparisons are scale-invariant
        delta_T_norm = F.normalize(delta_T.unsqueeze(0), dim=-1).squeeze(0)  # (D,)

    logger.debug(
        "delta_T  ‖raw‖=%.4f  (non-zero means prompts diverge in embedding space)",
        delta_T.norm().item(),
    )

    # ── Image direction vectors ────────────────────────────────────────────
    per_pair_scores: list[float] = []
    n = len(base_image_paths)

    with log_timer(logger, f"image direction encoding ({n} pairs)"):
        for i, (base_path, enh_path) in enumerate(
            zip(base_image_paths, enhanced_image_paths)
        ):
            base_img = Image.open(base_path).convert("RGB")
            enh_img  = Image.open(enh_path).convert("RGB")

            emb_base_img = get_image_embedding(model, processor, base_img,  device)
            emb_enh_img  = get_image_embedding(model, processor, enh_img,   device)

            delta_I = emb_enh_img - emb_base_img                   # (D,)
            delta_I_norm = F.normalize(delta_I.unsqueeze(0), dim=-1).squeeze(0)

            # Cosine similarity = dot product of two unit vectors
            score: float = torch.dot(delta_T_norm, delta_I_norm).item()
            per_pair_scores.append(score)

            logger.debug(
                "  pair %02d/%02d  score=%+.4f  "
                "base=%-30s  enh=%s",
                i + 1, n, score,
                Path(base_path).name, Path(enh_path).name,
            )

    avg_score = sum(per_pair_scores) / len(per_pair_scores)
    logger.info(
        "Directional similarity  avg=%+.4f  min=%+.4f  max=%+.4f  n=%d",
        avg_score,
        min(per_pair_scores),
        max(per_pair_scores),
        len(per_pair_scores),
    )
    return per_pair_scores, avg_score


# ─────────────────────────────────────────────────────────────────────────────
# Image generation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _wait_for_server(
    base_url: str,
    retries: int   = 60,
    interval: float = 10.0,
) -> None:
    """Poll /health until the generation server reports ready."""
    logger.info("Checking generation server at %s …", base_url)
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
        f"after {retries} attempts ({retries * interval:.0f}s)."
    )


def _generate_one_image(
    server_url: str,
    prompt: str,
    seed: int,
    width:  int = IMAGE_WIDTH,
    height: int = IMAGE_HEIGHT,
    num_inference_steps: int = NUM_INFERENCE_STEPS,
) -> Image.Image:
    """POST to /generate and return the resulting PIL image."""
    payload = {
        "prompt":  prompt,
        "seed":    seed,
        "width":   width,
        "height":  height,
        "num_inference_steps": num_inference_steps,
    }
    r = requests.post(f"{server_url}/generate", json=payload, timeout=600)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")


def generate_images_for_prompt(
    prompt:           str,
    output_dir:       Path,
    tag:              str,
    seeds:            list[int],
    server_url:       str,
    force_regenerate: bool = False,
) -> list[Path]:
    """
    Generate (or load from disk cache) one image per seed.

    File naming:  <output_dir>/<tag>_seed<SEED:05d>.png

    If the PNG already exists and force_regenerate is False, generation
    is skipped (fast path for resuming interrupted runs).

    Returns the ordered list of image paths (one per seed).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for seed in seeds:
        img_path = output_dir / f"{tag}_seed{seed:05d}.png"
        paths.append(img_path)

        if img_path.exists() and not force_regenerate:
            logger.debug("  [cache hit] %s", img_path.name)
            continue

        logger.info("  Generating  seed=%-7d →  %s", seed, img_path.name)
        with log_timer(logger, f"{tag} seed={seed}"):
            img = _generate_one_image(server_url, prompt, seed)
            img.save(img_path, format="PNG")

    return paths


# ─────────────────────────────────────────────────────────────────────────────
# CSV / prompt helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_enhanced_prompts(raw: str) -> list[str]:
    """
    Parse the `enhanced_prompts` cell which is stored as a JSON (or
    Python-literal) array of strings.

    Falls back gracefully to treating the raw value as a single prompt if
    parsing fails.
    """
    raw = raw.strip()

    # 1. Try JSON
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(p) for p in parsed]
        if isinstance(parsed, str):
            return [parsed]
    except json.JSONDecodeError:
        pass

    # 2. Try Python literal (handles single-quoted strings from Python's repr)
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list):
            return [str(p) for p in parsed]
        if isinstance(parsed, str):
            return [parsed]
    except (ValueError, SyntaxError):
        pass

    # 3. Last resort
    logger.warning(
        "Could not parse enhanced_prompts as a list; treating as raw string."
    )
    return [raw]


def load_prompt_pairs(csv_path: Path) -> list[dict]:
    """
    Load prompt pairs from the CSV.

    Returns a list of dicts::
        {
            "row_id":           str,
            "base_prompt":      str,
            "enhanced_prompts": list[str],
        }
    """
    rows: list[dict] = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            row_id        = str(row.get("__row_id__", row.get("row_id", "?")))
            base_prompt   = row["generic_prompt"].strip()
            enhanced_list = _parse_enhanced_prompts(row["enhanced_prompts"])
            rows.append(
                {
                    "row_id":           row_id,
                    "base_prompt":      base_prompt,
                    "enhanced_prompts": enhanced_list,
                }
            )

    logger.info("Loaded %d prompt pairs from %s", len(rows), csv_path)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation_pipeline(
    csv_path:         Path                  = INPUT_CSV,
    image_output_dir: Path                  = IMAGE_OUTPUT_DIR,
    results_dir:      Path                  = RESULTS_DIR,
    server_url:       str                   = GENERATION_SERVER,
    seeds:            list[int]             = DEFAULT_SEEDS,
    clip_device:      Optional[torch.device] = None,
    force_regenerate: bool                  = False,
    generate_images:  bool                  = True,
    enhanced_idx:     int                   = 0,
) -> list[dict]:
    """
    End-to-end evaluation pipeline.

    Steps
    -----
    1. Load prompt pairs from CSV.
    2. For each pair: generate base + enhanced images (different seeds).
    3. Compute CLIP directional similarity for each pair.
    4. Save per-row and aggregate results to JSON + CSV.

    Parameters
    ----------
    csv_path         : Path to the input prompts CSV.
    image_output_dir : Root folder for generated/cached images.
    results_dir      : Folder for result files.
    server_url       : Base URL of the image generation server.
    seeds            : Random seeds shared between base and enhanced generation.
    clip_device      : CUDA/CPU device for CLIP. Auto-selected if None.
    force_regenerate : Ignore cached images and re-generate everything.
    generate_images  : Set False to skip generation and evaluate cached images.
    enhanced_idx     : Which enhanced prompt to use (index into the list).

    Returns
    -------
    List of per-row result dicts.
    """
    log_section(logger, "CLIP Directional Similarity  ·  Evaluation Pipeline")
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("CLIP model      : %s",  CLIP_MODEL_ID)
    logger.info("Input CSV       : %s",  csv_path)
    logger.info("Image directory : %s",  image_output_dir)
    logger.info("Results dir     : %s",  results_dir)
    logger.info("Seeds           : %s",  seeds)
    logger.info("Generate images : %s",  generate_images)

    # ── 1. Server health check ─────────────────────────────────────────────
    if generate_images:
        _wait_for_server(server_url)

    # ── 2. Load prompts ────────────────────────────────────────────────────
    prompt_pairs = load_prompt_pairs(csv_path)

    # ── 3. Per-row: generate + evaluate ───────────────────────────────────
    all_results: list[dict] = []

    for entry in prompt_pairs:
        row_id          = entry["row_id"]
        base_prompt     = entry["base_prompt"]
        enh_list        = entry["enhanced_prompts"]

        if not enh_list:
            logger.warning("Row %s has no enhanced prompts — skipping.", row_id)
            continue

        # Select which enhanced prompt to use
        idx             = min(enhanced_idx, len(enh_list) - 1)
        enhanced_prompt = enh_list[idx]

        log_subsection(logger, f"Row {row_id}  /  {len(enh_list)} enhanced variant(s)")
        logger.info("  Base     : %.120s", base_prompt)
        logger.info("  Enhanced : %.120s", enhanced_prompt)

        row_dir = image_output_dir / f"row_{row_id}"

        # ── Image generation / cache ───────────────────────────────────────
        if generate_images:
            logger.info("  → Generating BASE images (%d seeds) …", len(seeds))
            base_paths = generate_images_for_prompt(
                prompt           = base_prompt,
                output_dir       = row_dir,
                tag              = "base",
                seeds            = seeds,
                server_url       = server_url,
                force_regenerate = force_regenerate,
            )
            logger.info("  → Generating ENHANCED images (%d seeds) …", len(seeds))
            enh_paths = generate_images_for_prompt(
                prompt           = enhanced_prompt,
                output_dir       = row_dir,
                tag              = "enhanced",
                seeds            = seeds,
                server_url       = server_url,
                force_regenerate = force_regenerate,
            )
        else:
            # Load whatever is already on disk
            base_paths = sorted(row_dir.glob("base_seed*.png"))
            enh_paths  = sorted(row_dir.glob("enhanced_seed*.png"))

            if not base_paths or not enh_paths:
                logger.warning(
                    "No cached images found for row %s at %s — skipping.",
                    row_id, row_dir,
                )
                continue

            if len(base_paths) != len(enh_paths):
                n = min(len(base_paths), len(enh_paths))
                logger.warning(
                    "Unequal image counts (base=%d, enh=%d) for row %s "
                    "— using first %d pairs.",
                    len(base_paths), len(enh_paths), row_id, n,
                )
                base_paths, enh_paths = base_paths[:n], enh_paths[:n]

            logger.info(
                "  Loaded %d cached image pairs for row %s.",
                len(base_paths), row_id,
            )

        # ── Metric ────────────────────────────────────────────────────────
        with log_timer(logger, f"directional similarity  row {row_id}"):
            per_pair_scores, avg_score = evaluate_rag_enhancement_direction(
                base_prompt          = base_prompt,
                enhanced_prompt      = enhanced_prompt,
                base_image_paths     = base_paths,
                enhanced_image_paths = enh_paths,
                device               = clip_device,
            )

        result = {
            "row_id":           row_id,
            "base_prompt":      base_prompt,
            "enhanced_prompt":  enhanced_prompt,
            "num_pairs":        len(per_pair_scores),
            "per_pair_scores":  [round(s, 6) for s in per_pair_scores],
            "avg_score":        round(avg_score, 6),
        }
        all_results.append(result)
        logger.info(
            "  ✔ Row %-5s  avg_directional_similarity = %+.4f",
            row_id, avg_score,
        )

    # ── 4. Save results ────────────────────────────────────────────────────
    _save_results(all_results, results_dir)

    if all_results:
        overall_avg = sum(r["avg_score"] for r in all_results) / len(all_results)
        log_section(
            logger,
            f"DONE  |  {len(all_results)} rows  "
            f"|  overall avg directional similarity = {overall_avg:+.4f}",
        )
    else:
        log_section(logger, "DONE  |  no rows evaluated")

    return all_results


def _save_results(results: list[dict], results_dir: Path) -> None:
    """Persist results to both a JSON file and a flat CSV file."""
    if not results:
        logger.warning("No results to save.")
        return

    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = results_dir / f"directional_similarity_{ts}.json"
    csv_path  = results_dir / f"directional_similarity_{ts}.csv"

    # JSON — preserves full nested structure
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    logger.info("Saved JSON results → %s", json_path)

    # CSV — per_pair_scores serialised as a JSON string in one column
    fieldnames = [
        "row_id",
        "base_prompt",
        "enhanced_prompt",
        "num_pairs",
        "avg_score",
        "per_pair_scores",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            out = dict(row)
            out["per_pair_scores"] = json.dumps(out["per_pair_scores"])
            writer.writerow(out)
    logger.info("Saved CSV results  → %s", csv_path)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "CLIP Directional Similarity evaluation pipeline for "
            "RAG-enhanced prompts."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=INPUT_CSV,
        metavar="PATH",
        help="Prompts CSV path (must contain generic_prompt + enhanced_prompts).",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=IMAGE_OUTPUT_DIR,
        metavar="DIR",
        help="Root directory for generated / cached images.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        metavar="DIR",
        help="Directory where JSON + CSV results are written.",
    )
    parser.add_argument(
        "--server",
        type=str,
        default=GENERATION_SERVER,
        metavar="URL",
        help="Base URL of the image generation server.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        metavar="N",
        help=(
            "Random seeds for image generation. "
            "The same seeds are used for both base and enhanced prompts."
        ),
    )
    parser.add_argument(
        "--clip-device",
        type=str,
        default=None,
        metavar="DEVICE",
        help=(
            "Torch device for CLIP inference "
            "(e.g. 'cuda:0', 'cuda:1', 'cpu'). "
            "Defaults to the first available CUDA device."
        ),
    )
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="Re-generate all images even when cached PNGs already exist.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help=(
            "Skip image generation entirely; evaluate previously cached images. "
            "Mutually exclusive with --force-regenerate."
        ),
    )
    parser.add_argument(
        "--enhanced-idx",
        type=int,
        default=0,
        metavar="IDX",
        help="Index into the enhanced_prompts array to use (0 = first variant).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=IMAGE_WIDTH,
        metavar="PX",
        help="Generated image width (pixels).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=IMAGE_HEIGHT,
        metavar="PX",
        help="Generated image height (pixels).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=NUM_INFERENCE_STEPS,
        metavar="N",
        help="Diffusion inference steps per image.",
    )
    return parser


def main() -> None:
    args   = _build_parser().parse_args()

    # Apply dimension / step overrides to module-level constants so that
    # _generate_one_image picks them up without needing to thread them through
    # every call signature.
    global IMAGE_WIDTH, IMAGE_HEIGHT, NUM_INFERENCE_STEPS
    IMAGE_WIDTH          = args.width
    IMAGE_HEIGHT         = args.height
    NUM_INFERENCE_STEPS  = args.steps

    clip_device = torch.device(args.clip_device) if args.clip_device else None

    run_evaluation_pipeline(
        csv_path         = args.csv,
        image_output_dir = args.image_dir,
        results_dir      = args.results_dir,
        server_url       = args.server,
        seeds            = args.seeds,
        clip_device      = clip_device,
        force_regenerate = args.force_regenerate,
        generate_images  = not args.eval_only,
        enhanced_idx     = args.enhanced_idx,
    )


if __name__ == "__main__":
    main()
