"""Batch Prompt Enhancement with In-Pipeline Scoring.

Reads ``generic_prompt`` rows from inputs/prompts.csv and for each row:
    1. Scores the original prompt (keyword always; LLM judge when available).
    2. If original score >= threshold, finishes with no enhancement.
    3. Otherwise runs sequential enhancement, scoring each enhanced candidate in-loop.

Output layout
─────────────
  outputs/           (or test/ with --test flag)
    enhanced/        ← per-row JSON: enhanced prompt + full iteration history
    (no evaluation/ folder — scoring is embedded in enhancement outputs)
    summary.csv      ← flat scorecard: scores, delta, verdict for every row
    summary.json     ← complete structured data for all rows

Usage
─────
    python run_batch.py                        # all rows → outputs/
    python run_batch.py --test                 # all rows → test/
    python run_batch.py --no-llm               # keyword scoring only (skip LLM judge)
    python run_batch.py --no-enhance           # skip enhancement, use enhanced_prompts column
    python run_batch.py --row-ids 0 1          # only rows 0 and 1
    python run_batch.py --input inputs/my.csv  # different CSV file
    python run_batch.py --threshold 70 --max-iterations 5 --model qwen3.5
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from helper.logger import get_logger, log_timer
from enhance_prompt_sequential_cot import (
    load_bias_graph,
    load_cultural_graphs,
    retrieve_relevant_triples,
    sequential_enhance_prompt,
    llm_score,
    _log_score_result,
    write_output_csvs,
    save_run_json,
    # configure_llm as configure_enhancer_llm,
)
from src.generator import BaseLLM
from src.graphrag import GraphRAG

logger = get_logger(__name__)
_LLM_CLASSES = BaseLLM.registry

# ─────────────────────────────────────────────────────────────────────────────
# CSV loading
# ─────────────────────────────────────────────────────────────────────────────


def load_prompts(csv_path: Path, row_ids: Optional[List[int]] = None) -> List[Dict]:
    """Return rows from the CSV that have a non-empty ``generic_prompt`` column."""
    rows: List[Dict] = []
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gp = row.get("generic_prompt", "").strip()
            if not gp:
                logger.warning("Row missing generic_prompt — skipped: %s", dict(row))
                continue
            row_id = int(row.get("__row_id__", len(rows)))
            if row_ids is not None and row_id not in row_ids:
                continue
            rows.append(
                {
                    "row_id": row_id,
                    "generic_prompt": gp,
                    "enhanced_prompts_csv": _parse_enhanced_prompts(
                        row.get("enhanced_prompts", "")
                    ),
                }
            )
    return rows


def _parse_enhanced_prompts(value: str) -> List[str]:
    """Parse the ``enhanced_prompts`` JSON-array cell from the CSV (may be absent)."""
    value = value.strip()
    if not value:
        return []
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [str(p).strip() for p in parsed if str(p).strip()]
    except (json.JSONDecodeError, ValueError):
        pass
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────────────────


def _slug(text: str, max_words: int = 4) -> str:
    """Short filename-safe slug derived from the first few words of a prompt."""
    words = re.sub(r"[^a-z0-9 ]", "", text.lower()).split()[:max_words]
    return "_".join(words) or "prompt"


def _save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info("Saved: %s", path)


def _save_summary(results: List[Dict], out_dir: Path) -> None:
    """Write ``summary.json`` and ``summary.csv`` into *out_dir*."""
    # ── JSON ─────────────────────────────────────────────────────────────────
    _save_json(
        {"generated_at": datetime.now().isoformat(), "results": results},
        out_dir / "summary.json",
    )

    # ── CSV ──────────────────────────────────────────────────────────────────
    fieldnames = [
        "row_id",
        "original_prompt",
        "enhanced_prompt",
        "keyword_original",
        "keyword_enhanced",
        "keyword_delta",
        "llm_original",
        "llm_enhanced",
        "llm_delta",
        "verdict",
    ]
    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            ev = r.get("evaluation", {})
            orig = ev.get("original_score", {})
            enh = ev.get("enhanced_score", {})

            kw_orig = orig.get("keyword_score", orig.get("overall_score", ""))
            kw_enh = enh.get("keyword_score", enh.get("overall_score", ""))
            kw_delta = (
                (kw_enh - kw_orig)
                if isinstance(kw_orig, int) and isinstance(kw_enh, int)
                else ""
            )

            llm_orig = (
                orig.get("overall_score", "")
                if orig.get("score_method") == "llm_score"
                else ""
            )
            llm_enh = (
                enh.get("overall_score", "")
                if enh.get("score_method") == "llm_score"
                else ""
            )
            llm_delta = (
                (llm_enh - llm_orig)
                if isinstance(llm_orig, int) and isinstance(llm_enh, int)
                else ""
            )

            writer.writerow(
                {
                    "row_id": r["row_id"],
                    "original_prompt": r["original_prompt"],
                    "enhanced_prompt": r["enhanced_prompt"],
                    "keyword_original": kw_orig,
                    "keyword_enhanced": kw_enh,
                    "keyword_delta": kw_delta,
                    "llm_original": llm_orig,
                    "llm_enhanced": llm_enh,
                    "llm_delta": llm_delta,
                    "verdict": ev.get("verdict", ""),
                }
            )
    logger.info("Saved: %s", csv_path)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch enhance and score prompts from a CSV file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_batch.py                        # all rows → outputs/
  python run_batch.py --test                 # all rows → test/
  python run_batch.py --no-llm               # skip LLM judge (keyword only)
  python run_batch.py --no-enhance           # use CSV enhanced_prompts column
  python run_batch.py --row-ids 0 1          # only rows 0 and 1
  python run_batch.py --input inputs/my.csv  # custom CSV
        """,
    )
    parser.add_argument(
        "--input",
        default="inputs/prompts.csv",
        help="Path to input CSV (default: inputs/prompts.csv)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Save outputs to test/ instead of outputs/",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip the LLM judge — keyword metrics only",
    )
    parser.add_argument(
        "--no-enhance",
        action="store_true",
        help="Skip enhancement; use the enhanced_prompts column already in the CSV",
    )
    parser.add_argument(
        "--row-ids",
        nargs="+",
        type=int,
        default=None,
        help="Only process these row IDs (default: all rows)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=75,
        help="Diversity score threshold for sequential enhancement (default: 75)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Max enhancement iterations per prompt (default: 3)",
    )
    parser.add_argument("--bias-top-k", type=int, default=25)
    parser.add_argument("--cultural-top-k", type=int, default=25)
    parser.add_argument("--cache-file", default="embeddings.pkl")

    parser.add_argument(
        "--llm",
        type=str,
        default="OpenAILLM".lower(),
        choices=list(_LLM_CLASSES.keys()),
        help="Which LLM backend to use (default: openaillm)",
    )
    args, _ = parser.parse_known_args()
    llm = _LLM_CLASSES[args.llm].add_args(parser)
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    llm = _LLM_CLASSES[args.llm](args)
    # Route all LLM calls to the chosen model server
    # configure_enhancer_llm(args.model)

    # ── resolve output root ───────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # Use a model-specific prefix so outputs from different models don't mix
    model_slug = args.model.replace(".", "")
    if args.test:
        out_dir = Path("test") / f"{model_slug}-{timestamp}"
    else:
        out_dir = Path("outputs") / f"{model_slug}-{timestamp}"
    enh_dir = out_dir / "enhanced"
    enh_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", out_dir)

    # ── load CSV ──────────────────────────────────────────────────────────────
    csv_path = Path(args.input)
    if not csv_path.exists():
        logger.error("Input CSV not found: %s", csv_path)
        sys.exit(1)

    rows = load_prompts(csv_path, row_ids=args.row_ids)
    if not rows:
        logger.error("No matching rows found in %s (check --row-ids filter)", csv_path)
        sys.exit(1)

    logger.info("Loaded %d row(s) from %s", len(rows), csv_path)

    # ── initialise GraphRAG once (only when doing enhancement) ───────────────
    graphrag: Optional[GraphRAG] = None
    if not args.no_enhance:
        logger.info("Initialising GraphRAG …")
        graphrag = GraphRAG(cache_file=args.cache_file, embed_model=llm)
        bias_ok = load_bias_graph(graphrag)
        cultural_ok = load_cultural_graphs(graphrag)
        if not bias_ok and not cultural_ok:
            logger.error(
                "Failed to load any knowledge graphs — cannot enhance. Exiting."
            )
            sys.exit(1)

    # ── process rows ──────────────────────────────────────────────────────────
    all_results: List[Dict] = []
    all_rows_data: List[Dict] = []  # for write_output_csvs

    for row in rows:
        row_id = row["row_id"]
        original = row["generic_prompt"]
        slug = _slug(original)
        label = f"row_{row_id}_{slug}"

        logger.info("─" * 60)
        logger.info("Processing %s", label)
        logger.info("  prompt: %s", original)

        # ── STEP 1 : score original prompt (always) ─────────────────────────
        with log_timer(logger, f"Scoring original — {label}"):
            orig_score_data = llm_score(
                original, llm, out_dir=out_dir, use_llm=not args.no_llm
            )
        eval_orig_score = orig_score_data["overall_score"]
        eval_orig_method = orig_score_data["score_method"]
        _log_score_result(orig_score_data, label="original")

        # ── STEP 2 : threshold gate + enhancement loop if needed ────────────
        source = ""
        if args.no_enhance:
            csv_enhanced = row["enhanced_prompts_csv"]
            if csv_enhanced:
                enhanced_prompt = csv_enhanced[0]
                logger.info("  Using pre-existing enhanced prompt from CSV column")
                source = "csv_enhanced_prompts_column"
            else:
                logger.warning(
                    "  Row %d has no enhanced_prompts in CSV; skipping enhancement",
                    row_id,
                )
                enhanced_prompt = original
                source = "csv_missing_enhanced_prompt_fell_back_to_original"
            raw_iteration_history: list = []
            with log_timer(logger, f"Scoring final — {label}"):
                final_score_data = llm_score(
                    enhanced_prompt, llm, out_dir=out_dir, use_llm=not args.no_llm
                )
        else:
            if eval_orig_score >= args.threshold:
                logger.info(
                    "  Original score %d/100 (%s) already meets threshold %d — no enhancement needed",
                    eval_orig_score,
                    eval_orig_method,
                    args.threshold,
                )
                enhanced_prompt = original
                raw_iteration_history = []
                final_score_data = orig_score_data
                source = "original_meets_threshold_no_enhancement"
            else:
                triples = retrieve_relevant_triples(
                    graphrag,
                    original,
                    args.bias_top_k,
                    args.cultural_top_k,
                )
                with log_timer(logger, f"Sequential enhancement — {label}"):
                    enhanced_prompt, iteration_history = sequential_enhance_prompt(
                        graphrag,
                        llm,
                        original,
                        triples.get("bias", []),
                        triples.get("cultural", []),
                        args.threshold,
                        args.max_iterations,
                        out_dir=out_dir,
                    )
                raw_iteration_history = iteration_history
                final_score_data = iteration_history[-1]["diversity_score"]
                source = "sequential_enhancement"

        eval_enh_score = final_score_data["overall_score"]
        eval_enh_method = final_score_data["score_method"]
        score_delta = eval_enh_score - eval_orig_score

        if score_delta > 0:
            verdict = "improved"
        elif score_delta < 0:
            verdict = "degraded"
        else:
            verdict = "no_change"

        logger.info(
            "  scores | original=%d/100 (%s) | enhanced=%d/100 (%s) | delta=%+d | verdict=%s",
            eval_orig_score,
            eval_orig_method,
            eval_enh_score,
            eval_enh_method,
            score_delta,
            verdict,
        )
        _log_score_result(final_score_data, label="final")

        # ── build per-row output dicts ────────────────────────────────────────
        iterations_out = [
            {
                "iteration": it["iteration"],
                "enhanced_prompt": it["enhanced_prompt"],
                "overall_score": it["diversity_score"]["overall_score"],
                "keyword_score": it["diversity_score"].get("keyword_score"),
                "score_method": it["diversity_score"].get("score_method", ""),
                "breakdown": it["diversity_score"].get("breakdown"),
                "strengths": it["diversity_score"].get("strengths", []),
                "weaknesses": it["diversity_score"].get("weaknesses", []),
                "threshold_met": it["threshold_met"],
            }
            for it in raw_iteration_history
        ]

        enh_result = {
            "row_id": row_id,
            "original_prompt": original,
            "original_score": eval_orig_score,
            "original_score_method": eval_orig_method,
            "source": source,
            "final_enhanced_prompt": enhanced_prompt,
            "final_score": eval_enh_score,
            "final_score_method": eval_enh_method,
            "iterations": iterations_out,
        }

        _save_json(enh_result, enh_dir / f"{label}.json")

        # ── collect for summary and detailed CSVs ─────────────────────────────
        all_rows_data.append(
            {
                "row_id": row_id,
                "original_prompt": original,
                "original_score": eval_orig_score,
                "original_score_method": eval_orig_method,
                "eval_final_score": eval_enh_score,
                "eval_final_score_method": eval_enh_method,
                "iteration_history": raw_iteration_history,
            }
        )

        all_results.append(
            {
                "row_id": row_id,
                "original_prompt": original,
                "enhanced_prompt": enhanced_prompt,
                "evaluation": {
                    "original_score": orig_score_data,
                    "enhanced_score": final_score_data,
                    "score_delta": score_delta,
                    "verdict": verdict,
                },
            }
        )

        logger.info(
            "  done | score %d\u2192%d (%+d) | verdict=%s",
            eval_orig_score,
            eval_enh_score,
            score_delta,
            verdict,
        )

    # ── summary ───────────────────────────────────────────────────────────────
    _save_summary(all_results, out_dir)

    # ── detailed diversity CSVs ───────────────────────────────────────────────
    write_output_csvs(out_dir, all_rows_data, max_iters=args.max_iterations)
    save_run_json(
        out_dir,
        {
            "generated_at": datetime.now().isoformat(),
            "input_csv": str(args.input),
            "threshold": args.threshold,
            "max_iterations": args.max_iterations,
            "rows": all_rows_data,
        },
    )

    logger.info("═" * 60)
    logger.info("Batch complete. %d row(s) processed.", len(all_results))
    logger.info("Outputs written to:  %s/", out_dir)
    logger.info("  enhanced/           ← per-row enhanced prompt + iteration history")
    logger.info(
        "  (no evaluation/ folder; scores are in enhanced/*.json and summary.csv)"
    )
    logger.info(
        "  prompts_output.csv  ← mirrors inputs CSV with finalized_prompt column"
    )
    logger.info("  detailed_scores.csv ← per-iteration scores in wide format")
    logger.info("  summary.csv         ← flat evaluation scorecard")
    logger.info("  run_details.json    ← full structured run data")


if __name__ == "__main__":
    main()
