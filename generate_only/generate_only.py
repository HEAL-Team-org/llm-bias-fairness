#!/usr/bin/env python3
"""
Generate-only pipeline:
  1. Read a CSV of ORIGINAL prompts  → generate images via cloud API
  2. Read a CSV of ENHANCED prompts  → generate images via cloud API
  3. Save generation metadata (JSON) for each pass

No evaluation / metrics are performed here.
Run evaluate_only mode in concat_generate_and_evaluate.py afterwards,
pointing it at the folder produced by this script.
"""

import argparse
import csv
from datetime import datetime
import json
import logging
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import requests
import yaml
from dotenv import load_dotenv

# logger.py lives in the same generate/ folder — no parent dependency
from logger import get_logger, log_call, log_section, log_subsection, log_timer

logger = get_logger(__name__)

# All config/env files are resolved relative to this file's own directory
_HERE = Path(__file__).resolve().parent


# ============================================================
#  Config loader
# ============================================================

_DEFAULT_CONFIG_PATH = _HERE / "config.yaml"
_DEFAULT_ENV_PATH    = _HERE / ".env"


def _expand_env_vars(value: str) -> str:
    """Replace ${VAR_NAME} placeholders with environment-variable values."""
    return re.sub(
        r'\$\{([^}]+)\}',
        lambda m: os.environ.get(m.group(1), m.group(0)),
        value,
    )


def _walk_expand(obj):
    """Recursively expand env-var placeholders in every string of a nested dict/list."""
    if isinstance(obj, dict):
        return {k: _walk_expand(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_expand(v) for v in obj]
    if isinstance(obj, str):
        return _expand_env_vars(obj)
    return obj


def load_config(config_path: str = None) -> Dict:
    """
    Load YAML config, expanding ${VAR} placeholders from the .env file.

    Priority: .env file < shell environment variables.
    """
    env_path = _DEFAULT_ENV_PATH
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)
        logger.info(f"Loaded .env from: {env_path}")
    else:
        logger.warning(
            f".env not found at {env_path}. "
            "API key / base URL must be set as real environment variables."
        )

    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            "Please create config.yaml (copy the template in the project root)."
        )
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg = _walk_expand(cfg)
    logger.info(f"Loaded config from: {path}")
    return cfg


# ============================================================
#  CSV reading
# ============================================================

@log_call(logger)
def read_rows_from_csv(
    csv_path: str,
    prompt_col: str,
    is_json_list: bool = False,
    num_rows: Optional[int] = None,
    seed: Optional[int] = None,
    images_per_prompt: int = 10,
) -> List[Dict]:
    """
    Read prompts from *prompt_col* in the CSV.

    Args:
        csv_path:         Path to the CSV file.
        prompt_col:       Column containing the prompt(s).
        is_json_list:     True  → parse the column as a JSON list of strings
                                  (one image generated per variant in the list).
                          False → treat as a single string, duplicated
                                  *images_per_prompt* times.
        num_rows:         Optional cap on the number of rows to process.
        seed:             Random seed for row sampling.
        images_per_prompt: Number of images per prompt when is_json_list=False.
    """
    all_rows: List[Dict] = []

    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if prompt_col not in (reader.fieldnames or []):
                raise ValueError(
                    f"CSV {csv_path!r} does not have column {prompt_col!r}. "
                    f"Columns: {reader.fieldnames}"
                )

            for idx, row in enumerate(reader):
                raw_val = row[prompt_col]

                if is_json_list:
                    try:
                        prompt_variants = json.loads(raw_val)
                        if not isinstance(prompt_variants, list):
                            logger.warning(f"Row {idx}: parsed JSON is not a list; treating as single string.")
                            prompt_variants = [str(prompt_variants)]
                    except json.JSONDecodeError:
                        logger.warning(f"Row {idx}: JSON parse failed in {prompt_col!r}; treating as raw string.")
                        prompt_variants = [raw_val]
                else:
                    prompt_variants = [raw_val] * images_per_prompt

                base_seed = random.randint(0, 2**31 - 1)
                precised_prompts_dicts = [
                    {"prompt": p_text, "seed": base_seed + i}
                    for i, p_text in enumerate(prompt_variants)
                ]

                all_rows.append({
                    "csv_idx":           idx,
                    "row_id":            row.get("__row_id__", idx),
                    "original_prompt":   row.get("prompt", raw_val),
                    "target_prompt_col": prompt_col,
                    "precised_prompts":  precised_prompts_dicts,
                })

        logger.info(f"Read {len(all_rows)} total rows from {csv_path}")

        if num_rows is None:
            selected_rows = all_rows
        else:
            if seed is not None:
                random.seed(seed)
                selected_rows = (
                    all_rows if num_rows > len(all_rows)
                    else sorted(random.sample(all_rows, num_rows), key=lambda x: int(x["row_id"]))
                )
            else:
                selected_rows = all_rows[:num_rows]

        prompts_data: List[Dict] = []
        for idx, row in enumerate(selected_rows):
            row["row_idx"] = idx
            prompts_data.append(row)

        ids_preview = [r["row_id"] for r in prompts_data]
        logger.info(f"Selected row IDs: {ids_preview[:10]}{'...' if len(ids_preview) > 10 else ''}")
        return prompts_data

    except FileNotFoundError:
        logger.error(f"CSV file not found: {csv_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading CSV: {e}")
        raise


# ============================================================
#  Cloud image generation
# ============================================================

@log_call(logger)
def generate_image(cloud_cfg: Dict, prompt: str, seed: int, output_path: str) -> Dict:
    """
    Submit a text-to-image task, poll until it completes, download the image,
    and save it to *output_path*.

    Steps:
      1. POST  <base_url>/api/v2/generate       → task_id
      2. GET   <base_url>/api/v2/generate/<id>  (loop until COMPLETED / ERROR)
      3. Download image from generations[0]['url']
      4. Write raw bytes to output_path
    """
    api_key          = cloud_cfg["api_key"]
    base_url         = cloud_cfg["base_url"].rstrip("/")
    model_cfg        = cloud_cfg["model"]
    operation        = cloud_cfg.get("operation", "Imagine")
    poll_interval    = int(cloud_cfg.get("poll_interval", 5))
    poll_timeout     = int(cloud_cfg.get("poll_timeout", 300))
    download_timeout = int(cloud_cfg.get("download_timeout", 60))

    auth_headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model":     model_cfg,
        "operation": operation,
        "args":      {"prompt": prompt},
    }

    try:
        logger.info(
            f"Submitting  model={model_cfg.get('model', '?')}  "
            f"seed={seed}  prompt={prompt[:80]!r}"
        )

        # Step 1 – create task
        resp = requests.post(
            f"{base_url}/api/v2/generate",
            headers=auth_headers,
            json=payload,
            timeout=30,
        )
        if resp.status_code not in (200, 201):
            raise RuntimeError(
                f"API rejected request  status={resp.status_code}  body={resp.text[:300]}"
            )
        task_id = resp.json()["id"]
        logger.info(f"Task created: {task_id}")

        # Step 2 – poll
        elapsed, image_url = 0, None
        while elapsed < poll_timeout:
            time.sleep(poll_interval)
            elapsed += poll_interval

            status_resp = requests.get(
                f"{base_url}/api/v2/generate/{task_id}",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=30,
            )
            data   = status_resp.json()
            status = data.get("status")
            logger.debug(f"[{task_id}] status={status}  elapsed={elapsed}s")

            if status == "COMPLETED":
                image_url = data["generations"][0]["url"]
                cost = data.get("usage", {}).get("cost", "N/A")
                logger.info(f"✓ Completed  id={task_id}  cost={cost}c  url={image_url}")
                break
            elif status in ("ERROR", "CANCELLED"):
                raise RuntimeError(
                    f"Task {task_id} ended with status '{status}': "
                    f"{data.get('error', 'no detail')}"
                )

        if image_url is None:
            raise TimeoutError(f"Task {task_id} did not complete within {poll_timeout}s.")

        # Step 3 – download
        logger.info(f"Downloading from: {image_url}")
        img_resp = requests.get(image_url, timeout=download_timeout)
        img_resp.raise_for_status()

        # Step 4 – save
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(img_resp.content)
        logger.info(f"✓ Saved: {output_path}  ({len(img_resp.content) // 1024} KB)")

        return {
            "status":          "success",
            "output_path":     output_path,
            "original_prompt": prompt,
            "seed":            seed,
            "task_id":         task_id,
            "image_url":       image_url,
            "api_model":       model_cfg.get("model", ""),
        }

    except Exception as e:
        logger.error(f"✗ Failed: {e}")
        return {
            "status":          "error",
            "error":           str(e),
            "output_path":     output_path,
            "original_prompt": prompt,
            "seed":            seed,
        }


@log_call(logger, level=logging.DEBUG)
def generate_images_for_row(cloud_cfg: Dict, row_data: Dict, output_dir: Path) -> List[Dict]:
    """Generate all image variants for one CSV row."""
    results  = []
    row_idx  = row_data["row_idx"]
    row_id   = row_data["row_id"]

    row_dir = output_dir / f"row_{row_id}"
    row_dir.mkdir(parents=True, exist_ok=True)

    precised = row_data["precised_prompts"]
    log_subsection(logger, f"Row {row_idx}  (CSV ID: {row_id})  ·  {len(precised)} variants")

    for variant_idx, variant in enumerate(precised):
        prompt = variant["prompt"]
        seed   = variant["seed"]

        output_path = row_dir / f"row_{row_id}_variant_{variant_idx:02d}.png"
        result = generate_image(cloud_cfg, prompt, seed, str(output_path))
        result.update({"row_idx": row_idx, "row_id": row_id, "variant_idx": variant_idx})
        results.append(result)

    return results


def process_parallel(
    cloud_cfg: Dict,
    prompts_data: List[Dict],
    output_dir: Path,
    num_workers: int = 4,
) -> List[Dict]:
    """Process multiple CSV rows concurrently with a thread pool."""
    all_results: List[Dict] = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_row = {
            executor.submit(generate_images_for_row, cloud_cfg, row_data, output_dir): row_data
            for row_data in prompts_data
        }
        for future in as_completed(future_to_row):
            row_data = future_to_row[future]
            try:
                results = future.result()
                all_results.extend(results)
                logger.info(
                    f"✓ Row {row_data['row_idx']} (CSV ID: {row_data['row_id']}): "
                    f"{len(results)} images"
                )
            except Exception as e:
                logger.error(
                    f"✗ Row {row_data['row_idx']} (CSV ID: {row_data['row_id']}) failed: {e}"
                )
    return all_results


def save_results_metadata(metadata: Dict, output_path: str):
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadata saved: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save metadata: {e}")


def print_summary(results: List[Dict]):
    total      = len(results)
    successful = sum(1 for r in results if r["status"] == "success")
    failed     = total - successful
    logger.info("=" * 70)
    logger.info("GENERATION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  Total:      {total}")
    if total == 0:
        logger.warning("  No images attempted.")
        return
    logger.info(f"  Successful: {successful} ({successful/total*100:.1f}%)")
    logger.info(f"  Failed:     {failed} ({failed/total*100:.1f}%)")


def model_tag_from_config(cloud_cfg: Dict) -> str:
    """Derive a short snake_case tag from the model name in config."""
    model_name = cloud_cfg.get("model", {}).get("model", "model")
    tag = re.sub(r"[^a-z0-9]", "_", model_name.lower()).strip("_")
    return tag or "model"


@log_call(logger)
def generate_images_from_csv(
    cloud_cfg: Dict,
    csv_path: str,
    tag: str,
    output_root: Path,
    prompt_col: str,
    num_rows: Optional[int],
    seed: Optional[int],
    images_per_prompt: int,
    workers: int,
    sequential: bool,
    is_json_list: bool = False,
) -> List[Dict]:
    """
    Read prompts from a CSV, generate images for each row, save metadata JSON,
    and return all result dicts.
    """
    log_subsection(logger, f"[{tag}] Loading prompts  ·  csv={csv_path}  col={prompt_col}")
    prompts_data = read_rows_from_csv(
        csv_path=csv_path,
        prompt_col=prompt_col,
        is_json_list=is_json_list,
        num_rows=num_rows,
        seed=seed,
        images_per_prompt=images_per_prompt,
    )

    if not prompts_data:
        raise ValueError(f"No prompts loaded from {csv_path}")

    out_dir = output_root / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    mode = "sequential" if sequential else f"parallel  workers={workers}"
    log_subsection(logger, f"[{tag}] Generating images  ·  {len(prompts_data)} rows  ·  {mode}")

    start_time = time.time()
    with log_timer(logger, f"[{tag}] full generation pass  ({len(prompts_data)} rows)"):
        if sequential:
            all_results: List[Dict] = []
            for row_data in prompts_data:
                all_results.extend(generate_images_for_row(cloud_cfg, row_data, out_dir))
        else:
            all_results = process_parallel(cloud_cfg, prompts_data, out_dir, num_workers=workers)

    elapsed = time.time() - start_time

    metadata = {
        "configuration": {
            "api_base_url":      cloud_cfg.get("base_url", ""),
            "api_model":         cloud_cfg.get("model", {}).get("model", ""),
            "csv_file":          csv_path,
            "prompt_col":        prompt_col,
            "is_json_list":      is_json_list,
            "num_rows":          num_rows,
            "seed":              seed,
            "output_dir":        str(out_dir),
            "workers":           workers if not sequential else 1,
            "sequential":        sequential,
            "images_per_prompt": images_per_prompt,
            "tag":               tag,
        },
        "results": all_results,
        "summary": {
            "total_images":            len(all_results),
            "successful":              sum(1 for r in all_results if r["status"] == "success"),
            "failed":                  sum(1 for r in all_results if r["status"] == "error"),
            "total_time_seconds":      elapsed,
            "average_time_per_image":  elapsed / len(all_results) if all_results else 0,
        },
    }
    save_results_metadata(metadata, str(out_dir / "generation_results.json"))
    print_summary(all_results)
    return all_results


# ============================================================
#  Entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate images via the cloud API for ORIGINAL and/or ENHANCED "
            "prompt CSVs and save them to disk.  No metrics are computed here."
        )
    )

    # ── Config ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--config", type=str, default=None,
        help=(
            "Path to YAML config file.  "
            "Defaults to config.yaml in the project root.  "
            "CLI args below override individual config values when provided."
        ),
    )

    # ── Cloud API overrides ──────────────────────────────────────────────────
    parser.add_argument("--api-key",   type=str, default=None, help="Override cloud_api.api_key")
    parser.add_argument("--base-url",  type=str, default=None, help="Override cloud_api.base_url")
    parser.add_argument("--api-model", type=str, default=None, help="Override cloud_api.model.model")

    # ── CSV / prompt overrides ───────────────────────────────────────────────
    parser.add_argument("--csv-file",            type=str,   default=None, help="Single CSV for both passes")
    parser.add_argument("--original-csv",        type=str,   default=None, help="CSV with original prompts")
    parser.add_argument("--enhanced-csv",        type=str,   default=None, help="CSV with enhanced prompts")
    parser.add_argument("--original-prompt-col", type=str,   default=None, help="Column name for original prompts")
    parser.add_argument("--enhanced-prompt-col", type=str,   default=None, help="Column name for enhanced prompts")
    parser.add_argument(
        "--enhanced-is-json",
        action=argparse.BooleanOptionalAction, default=None,
        help="Parse enhanced prompt column as JSON list",
    )
    parser.add_argument("--num-rows", type=int,  default=None, help="Limit rows processed")
    parser.add_argument("--seed",     type=int,  default=None, help="Random seed for row sampling")

    # ── Which passes to run ──────────────────────────────────────────────────
    parser.add_argument(
        "--passes",
        type=str,
        default="original,enhanced",
        help=(
            "Comma-separated list of passes to run.  "
            "Use 'original' to only generate original images, "
            "'enhanced' for enhanced only, or 'original,enhanced' for both.  "
            "Default: original,enhanced"
        ),
    )

    # ── Generation overrides ────────────────────────────────────────────────
    parser.add_argument("--images-per-prompt", type=int,  default=None, help="Images per prompt")
    parser.add_argument("--output-dir",        type=str,  default=None, help="Output root folder")
    parser.add_argument("--workers",           type=int,  default=None, help="Parallel worker threads")
    parser.add_argument("--sequential",        action="store_true",     help="Force sequential processing")

    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────────────────────────
    cfg = load_config(args.config)

    cloud_cfg = cfg["cloud_api"]
    if args.api_key:
        cloud_cfg["api_key"] = args.api_key
    if args.base_url:
        cloud_cfg["base_url"] = args.base_url
    if args.api_model:
        cloud_cfg.setdefault("model", {})["model"] = args.api_model

    gen_cfg = cfg.get("generation", {})
    csv_cfg = cfg.get("csv", {})

    images_per_prompt = args.images_per_prompt or gen_cfg.get("images_per_prompt", 10)
    output_dir_str    = args.output_dir        or gen_cfg.get("output_dir", "generated_images")
    workers           = args.workers           or gen_cfg.get("workers", 4)
    sequential        = args.sequential        or gen_cfg.get("sequential", False)

    num_rows         = args.num_rows        if args.num_rows  is not None else csv_cfg.get("num_rows")
    seed             = args.seed            if args.seed      is not None else csv_cfg.get("seed")
    enhanced_is_json = args.enhanced_is_json if args.enhanced_is_json is not None else csv_cfg.get("enhanced_is_json", True)

    # ── Determine which passes to run ───────────────────────────────────────
    passes = {p.strip().lower() for p in args.passes.split(",") if p.strip()}
    valid_passes = {"original", "enhanced"}
    unknown = passes - valid_passes
    if unknown:
        parser.error(f"Unknown passes: {unknown}.  Valid values: {valid_passes}")

    # ── Resolve CSV paths ───────────────────────────────────────────────────
    single_csv   = args.csv_file     or csv_cfg.get("file")
    original_csv = args.original_csv or csv_cfg.get("original_csv")
    enhanced_csv = args.enhanced_csv or csv_cfg.get("enhanced_csv")
    orig_col     = args.original_prompt_col or csv_cfg.get("original_prompt_col", "generic_prompt")
    enh_col      = args.enhanced_prompt_col or csv_cfg.get("enhanced_prompt_col", "modified_prompts")

    if single_csv:
        original_csv = single_csv
        enhanced_csv = single_csv
        logger.info(f"Using single CSV for both passes: {single_csv}")

    if "original" in passes and not original_csv:
        parser.error("No original CSV provided. Set csv.original_csv in config or use --original-csv.")
    if "enhanced" in passes and not enhanced_csv:
        parser.error("No enhanced CSV provided. Set csv.enhanced_csv in config or use --enhanced-csv.")

    # ── Build run tag from model name + date ────────────────────────────────
    model_tag      = model_tag_from_config(cloud_cfg)
    date_tag       = datetime.now().strftime("%Y%m%d")
    run_tag        = f"{model_tag}_output_{date_tag}"

    output_root = Path(output_dir_str) / run_tag
    output_root.mkdir(parents=True, exist_ok=True)

    logger.info(f"Cloud API  base_url={cloud_cfg['base_url']}  model={cloud_cfg.get('model', {})}")
    logger.info(f"Run tag:    {run_tag}")
    logger.info(f"Output dir: {output_root.resolve()}")
    logger.info(f"Passes:     {sorted(passes)}")

    # ── Run generation passes ────────────────────────────────────────────────
    if "original" in passes:
        log_section(logger, "Original prompts  ·  cloud image generation")
        generate_images_from_csv(
            cloud_cfg=cloud_cfg,
            csv_path=original_csv,
            tag="original",
            output_root=output_root,
            prompt_col=orig_col,
            is_json_list=False,
            num_rows=num_rows,
            seed=seed,
            images_per_prompt=images_per_prompt,
            workers=workers,
            sequential=sequential,
        )

    if "enhanced" in passes:
        log_section(logger, "Enhanced prompts  ·  cloud image generation")
        generate_images_from_csv(
            cloud_cfg=cloud_cfg,
            csv_path=enhanced_csv,
            tag="enhanced",
            output_root=output_root,
            prompt_col=enh_col,
            is_json_list=enhanced_is_json,
            num_rows=num_rows,
            seed=seed,
            images_per_prompt=images_per_prompt,
            workers=workers,
            sequential=sequential,
        )

    logger.info("\n✅ Generation complete.")
    logger.info(f"Images saved under: {output_root.resolve()}")
    logger.info(
        "To evaluate, run:\n"
        f"  python concat_generate_and_evaluate.py --evaluate-only "
        f"--eval-folder {output_root.resolve()}"
    )


if __name__ == "__main__":
    main()
