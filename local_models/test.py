#!/usr/bin/env python3
"""
Quick sanity test for all local generation servers:

  Port 8000 — FLUX.2-dev-Turbo          (serve_flux.py)
  Port 8001 — Qwen-Image-2512           (serve_qwen.py)
  Port 8002 — SD 3.5 Large              (serve_sd35.py)
  Port 8003 — Tongyi-MAI/Z-Image        (serve_zimage.py)
  Port 8004 — FLUX.1-dev                (serve_flux_dev.py)

Usage:
    # run all servers
    python local_models/test.py

    # run a specific server only
    python local_models/test.py --flux
    python local_models/test.py --qwen
    python local_models/test.py --sd35
    python local_models/test.py --zimage
    python local_models/test.py --flux-dev

Output:
    test/flux_turbo_output_YYYYMMDD/flux_turbo_0001.png
    test/qwen_output_YYYYMMDD/qwen_0001.png
    test/sd35_output_YYYYMMDD/sd35_0001.png
    test/zimage_output_YYYYMMDD/zimage_0001.png
    test/flux_dev_output_YYYYMMDD/flux_dev_0001.png
"""

import sys
from datetime import datetime
from pathlib import Path
import requests

# Make the parent evaluation/ directory importable so logger.py can be found
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from logger import get_logger, log_section

logger = get_logger(__name__)

FLUX_URL      = "http://localhost:8000"
QWEN_URL      = "http://localhost:8001"
SD35_URL      = "http://localhost:8002"
ZIMAGE_URL    = "http://localhost:8003"
FLUX_DEV_URL  = "http://localhost:8004"
PROMPT        = "a red fox sitting on a snowy hill at sunset, photorealistic"
SEED          = 42


def _next_indexed_png(model_dir: Path, stem: str) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(model_dir.glob(f"{stem}_*.png"))
    if not existing:
        return model_dir / f"{stem}_0001.png"

    max_idx = 0
    for path in existing:
        suffix = path.stem.rsplit("_", 1)[-1]
        if suffix.isdigit():
            max_idx = max(max_idx, int(suffix))
    return model_dir / f"{stem}_{max_idx + 1:04d}.png"


def _model_output_path(model_prefix: str) -> Path:
    date_tag = datetime.now().strftime("%Y%m%d")
    model_folder = Path("test") / f"{model_prefix}_output_{date_tag}"
    return _next_indexed_png(model_folder, model_prefix)


def check_health(name: str, base_url: str) -> bool:
    try:
        r = requests.get(f"{base_url}/health", timeout=10)
        if r.status_code == 200:
            logger.info("[%s] health: OK — %s", name, r.json())
            return True
        else:
            logger.warning("[%s] health: NOT READY (HTTP %d) — %s", name, r.status_code, r.text)
            return False
    except requests.exceptions.ConnectionError:
        logger.error("[%s] health: UNREACHABLE — is the server running?", name)
        return False


def generate_and_save(name: str, base_url: str, payload: dict, out_path: str) -> bool:
    logger.info("[%s] sending /generate request ...", name)
    try:
        r = requests.post(f"{base_url}/generate", json=payload, timeout=300)
    except requests.exceptions.ConnectionError:
        logger.error("[%s] /generate: UNREACHABLE", name)
        return False

    if r.status_code != 200:
        logger.error("[%s] /generate: FAILED (HTTP %d) — %s", name, r.status_code, r.text)
        return False

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(r.content)
    logger.info(
        "[%s] image saved → %s  (%.1f KB)", name, out_path, len(r.content) / 1024
    )
    return True


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run sanity tests for all local generation servers")
    parser.add_argument("--flux",      action="store_true", help="run only FLUX.2-dev-Turbo test (port 8000)")
    parser.add_argument("--qwen",      action="store_true", help="run only Qwen-Image-2512 test (port 8001)")
    parser.add_argument("--sd35",      action="store_true", help="run only SD 3.5 Large test (port 8002)")
    parser.add_argument("--zimage",    action="store_true", help="run only Z-Image test (port 8003)")
    parser.add_argument("--flux-dev",  action="store_true", help="run only FLUX.1-dev test (port 8004)")
    parser.add_argument("--prompt", type=str, default=PROMPT, help="override the prompt")
    parser.add_argument("--seed",   type=int, default=SEED,   help="override the seed")
    args = parser.parse_args()

    # If no specific flag is set, run all
    any_selected = args.flux or args.qwen or args.sd35 or args.zimage or args.flux_dev
    run_flux      = args.flux      or not any_selected
    run_qwen      = args.qwen      or not any_selected
    run_sd35      = args.sd35      or not any_selected
    run_zimage    = args.zimage    or not any_selected
    run_flux_dev  = args.flux_dev  or not any_selected

    all_ok = True

    # ── FLUX.2-dev-Turbo  (port 8000) ────────────────────────────────────────
    if run_flux:
        log_section(logger, "FLUX.2-dev-Turbo sanity test  (port 8000)")
        if check_health("FLUX", FLUX_URL):
            ok = generate_and_save("FLUX", FLUX_URL, {
                "prompt": args.prompt,
                "seed":   args.seed,
                "width":  1024,
                "height": 1024,
            }, str(_model_output_path("flux_turbo")))
            all_ok = all_ok and ok
        else:
            all_ok = False

    # ── Qwen-Image-2512  (port 8001) ─────────────────────────────────────────
    if run_qwen:
        log_section(logger, "Qwen-Image-2512 sanity test  (port 8001)")
        if check_health("Qwen", QWEN_URL):
            ok = generate_and_save("Qwen", QWEN_URL, {
                "prompt":              args.prompt,
                "seed":                args.seed,
                "width":               1328,
                "height":              1328,
                "negative_prompt":     "blurry, low quality, deformed",
                "num_inference_steps": 50,
                "true_cfg_scale":      4.0,
            }, str(_model_output_path("qwen")))
            all_ok = all_ok and ok
        else:
            all_ok = False

    # ── SD 3.5 Large  (port 8002) ────────────────────────────────────────────
    if run_sd35:
        log_section(logger, "Stable Diffusion 3.5 Large sanity test  (port 8002)")
        if check_health("SD35", SD35_URL):
            ok = generate_and_save("SD35", SD35_URL, {
                "prompt":              args.prompt,
                "seed":                args.seed,
                "width":               1024,
                "height":              1024,
                "negative_prompt":     "blurry, low quality, deformed",
                "num_inference_steps": 28,
                "guidance_scale":      3.5,
                "max_sequence_length": 512,
            }, str(_model_output_path("sd35")))
            all_ok = all_ok and ok
        else:
            all_ok = False

    # ── Tongyi-MAI/Z-Image  (port 8003) ──────────────────────────────────────
    if run_zimage:
        log_section(logger, "Tongyi-MAI/Z-Image sanity test  (port 8003)")
        if check_health("ZImage", ZIMAGE_URL):
            ok = generate_and_save("ZImage", ZIMAGE_URL, {
                "prompt":              args.prompt,
                "seed":                args.seed,
                "width":               1024,
                "height":              1024,
                "negative_prompt":     "blurry, low quality, deformed",
                "num_inference_steps": 50,
                "guidance_scale":      4.0,
                "cfg_normalization":   False,
            }, str(_model_output_path("zimage")))
            all_ok = all_ok and ok
        else:
            all_ok = False

    # ── FLUX.1-dev  (port 8004) ───────────────────────────────────────────────
    if run_flux_dev:
        log_section(logger, "FLUX.1-dev sanity test  (port 8004)")
        if check_health("FLUX_DEV", FLUX_DEV_URL):
            ok = generate_and_save("FLUX_DEV", FLUX_DEV_URL, {
                "prompt":              args.prompt,
                "seed":                args.seed,
                "width":               1024,
                "height":              1024,
                "num_inference_steps": 50,
                "guidance_scale":      3.5,
                "max_sequence_length": 512,
            }, str(_model_output_path("flux_dev")))
            all_ok = all_ok and ok
        else:
            all_ok = False

    # ── Summary ───────────────────────────────────────────────────────────────
    if all_ok:
        logger.info("✅ All tests PASSED. Open the output PNGs to inspect.")
    else:
        logger.error("❌ One or more tests FAILED. Check the log output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
