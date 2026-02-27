"""PreciseDebias-style batch prompt rewriting with GPT-5 (default gpt-5.2).

- Reads a CSV with a prompt column
- Strips existing demographic tokens (gender/ethnicity) to form a generic prompt
- Asks GPT-5 to generate one rewritten prompt per demographic combo
- Allocates counts to match a target distribution, producing a list of prompts ready for image generation

Dependencies:
  pip install "openai>=1.0.0" pydantic pandas tqdm

Auth:
  export OPENAI_API_KEY="..."

Example:
  python precise_debias_batch.py \
    --input prompts.csv --output prompts_precised.csv \
    --prompt-col prompt \
    --n-images 45 \
    --genders "female,male" \
    --ethnicities "white,black,asian,hispanic" \
    --gender-weights "female=0.5,male=0.5" \
    --ethnicity-weights "white=0.6,black=0.13,asian=0.07,hispanic=0.2" \
    --model gpt-5.2

"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError
from tqdm import tqdm


# -----------------------------
# Structured output schema
# -----------------------------
class PromptVariant(BaseModel):
    gender: str
    ethnicity: str
    prompt: str


class RewriteResult(BaseModel):
    has_person: bool = Field(
        ...,
        description="True if the prompt depicts at least one human subject; False otherwise.",
    )
    variants: List[PromptVariant] = Field(
        ...,
        description="One rewritten prompt per requested (gender, ethnicity) combo, in the same order.",
    )
    notes: List[str] = Field(default_factory=list)


# -----------------------------
# Defaults (edit as you like)
# -----------------------------
DEFAULT_GENDERS = ["female", "male"]
DEFAULT_ETHNICITIES = ["white", "black", "asian", "hispanic"]

# Regex vocab inspired by common usage; tune for your domain.
GENDER_SYNONYMS = {
    "female": [
        r"\bfemale\b",
        r"\bwoman\b",
        r"\bgirl\b",
        r"\blady\b",
        r"\bshe\b",
        r"\bher\b",
    ],
    "male": [
        r"\bmale\b",
        r"\bman\b",
        r"\bboy\b",
        r"\bgentleman\b",
        r"\bhe\b",
        r"\bhim\b",
        r"\bhis\b",
    ],
}

ETHNICITY_SYNONYMS = {
    "white": [r"\bwhite\b", r"\bcaucasian\b"],
    "black": [r"\bblack\b", r"\bafrican[- ]american\b"],
    "asian": [r"\basian\b", r"\beast asian\b", r"\bsouth asian\b"],
    "hispanic": [r"\bhispanic\b", r"\blatino\b", r"\blatina\b", r"\blatinx\b"],
}


# -----------------------------
# Helpers
# -----------------------------
def parse_kv_weights(spec: str) -> Dict[str, float]:
    """
    Parse "a=0.1,b=0.9" -> {"a":0.1,"b":0.9}
    """
    out: Dict[str, float] = {}
    if not spec:
        return out
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Invalid weight token: {part!r} (expected key=value)")
        k, v = part.split("=", 1)
        out[k.strip().lower()] = float(v.strip())
    return out


def normalize_weights(keys: List[str], weights: Dict[str, float]) -> Dict[str, float]:
    """
    Ensure all keys exist; if missing, fill uniformly among missing ones.
    Then normalize to sum to 1.
    """
    keys_l = [k.lower() for k in keys]
    w = {k.lower(): float(weights.get(k.lower(), 0.0)) for k in keys_l}

    missing = [k for k in keys_l if w[k] <= 0.0]
    provided_sum = sum(w[k] for k in keys_l if w[k] > 0.0)

    if provided_sum <= 0.0:
        # all uniform
        for k in keys_l:
            w[k] = 1.0 / len(keys_l)
        return w

    if missing:
        leftover = max(0.0, 1.0 - provided_sum)
        # distribute leftover uniformly among missing
        add = leftover / len(missing) if leftover > 0 else 0.0
        for k in missing:
            w[k] = add

    # normalize
    s = sum(w.values())
    if s <= 0:
        # fallback uniform
        for k in keys_l:
            w[k] = 1.0 / len(keys_l)
        return w
    for k in list(w.keys()):
        w[k] /= s
    return w


def allocate_counts(weights: List[float], total: int) -> List[int]:
    """
    Deterministically allocate integer counts that sum to total
    using largest remainder method.
    """
    raw = [w * total for w in weights]
    floor = [int(math.floor(x)) for x in raw]
    remainder = [x - f for x, f in zip(raw, floor)]
    diff = total - sum(floor)
    # distribute remaining counts to largest remainders
    idxs = sorted(range(len(weights)), key=lambda i: remainder[i], reverse=True)
    for i in idxs[:diff]:
        floor[i] += 1
    return floor


def build_strip_regex(genders: List[str], ethnicities: List[str]) -> re.Pattern:
    pats: List[str] = []

    for g in genders:
        pats.extend(GENDER_SYNONYMS.get(g.lower(), [rf"\b{re.escape(g)}\b"]))
    for e in ethnicities:
        pats.extend(ETHNICITY_SYNONYMS.get(e.lower(), [rf"\b{re.escape(e)}\b"]))

    # combine; case-insensitive; keep word boundaries in patterns
    combined = "(" + "|".join(pats) + ")"
    return re.compile(combined, flags=re.IGNORECASE)


def strip_demographic_tokens(prompt: str, strip_re: re.Pattern) -> str:
    # remove demographic tokens + collapse extra spaces
    s = strip_re.sub("", prompt)
    s = re.sub(r"\s+", " ", s).strip()
    # small cleanup for double articles like "a  doctor"
    s = re.sub(r"\b(a|an|the)\s+(?=(a|an|the)\b)", "", s, flags=re.IGNORECASE).strip()
    return s


def combos(genders: List[str], ethnicities: List[str]) -> List[Tuple[str, str]]:
    return [(g.lower(), e.lower()) for g in genders for e in ethnicities]


def compute_combo_counts(
    genders: List[str],
    ethnicities: List[str],
    gender_w: Dict[str, float],
    eth_w: Dict[str, float],
    n_images: int,
) -> List[int]:
    cmb = combos(genders, ethnicities)
    w = []
    for g, e in cmb:
        w.append(gender_w[g] * eth_w[e])
    # normalize combo weights
    s = sum(w)
    w = [x / s for x in w] if s > 0 else [1.0 / len(w) for _ in w]
    return allocate_counts(w, n_images)


SYSTEM_PROMPT = """You are a PreciseDebias-style prompt rewriter for text-to-image generation.

Goal:
Given a generic prompt and a list of requested (gender, ethnicity) combinations, produce one rewritten prompt per combination.
Each rewritten prompt MUST:
- Preserve the original meaning, style, setting, and all non-demographic details.
- Only add or replace demographic descriptors (gender + ethnicity) for the MAIN human subject.
- Avoid stereotypes or adding any other attributes (no religion, nationality, socioeconomic status, behavior, etc.).
- Stay natural and concise; minimal edits.

If the prompt does NOT depict a person, set has_person=false and return the original prompt for every variant (unchanged).

Return JSON that matches the provided schema exactly, with variants in the SAME ORDER as the input combos.
"""


def call_gpt_rewriter(
    client: OpenAI,
    model: str,
    generic_prompt: str,
    requested_combos: List[Tuple[str, str]],
    temperature: float = 0.0,
) -> RewriteResult:
    payload = {
        "generic_prompt": generic_prompt,
        "requested_combos": [{"gender": g, "ethnicity": e} for g, e in requested_combos],
        "rules": {
            "do_minimal_edits": True,
            "avoid_stereotypes": True,
            "only_gender_and_ethnicity": True,
        },
    }

    # Structured outputs parsing (Pydantic) via Responses API
    resp = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        text_format=RewriteResult,
        # temperature=temperature,
        # GPT-5 family knobs (safe defaults for batch rewriting)
        reasoning={"effort": "low"},
        text={"verbosity": "low"},
    )
    return resp.output_parsed


def process_single_row(
    row_data: Tuple[str, pd.Series, str, re.Pattern, List[Tuple[str, str]], List[int], OpenAI, str]
) -> Dict:
    """
    Process a single row in parallel.
    Returns a dictionary with the output row data.
    """
    row_id, row, prompt_col, strip_re, cmb, counts, client, model = row_data
    
    prompt = str(row.get(prompt_col, "")).strip()
    generic = strip_demographic_tokens(prompt, strip_re) if prompt else ""
    
    result_json = None
    precised_variants = []
    precised_prompts = []
    notes = []
    err = ""
    
    try:
        rr = call_gpt_rewriter(
            client=client,
            model=model,
            generic_prompt=generic if generic else prompt,
            requested_combos=cmb,
            temperature=0.0,
        )
        
        # Validate ordering/coverage
        got = [(v.gender.lower(), v.ethnicity.lower()) for v in rr.variants]
        if len(rr.variants) != len(cmb) or got != cmb:
            notes.append("Model output combos/order differed; results may need review.")
        
        # Build variants with deterministic counts
        for i, (g, e) in enumerate(cmb):
            p = rr.variants[i].prompt if i < len(rr.variants) else (generic or prompt)
            precised_variants.append(
                {"gender": g, "ethnicity": e, "count": int(counts[i]), "prompt": p}
            )
            precised_prompts.extend([p] * int(counts[i]))
        
        notes.extend(rr.notes)
        
    except ValidationError as ve:
        err = f"Structured parse error: {ve}"
    except Exception as ex:
        err = f"API/error: {type(ex).__name__}: {ex}"
    
    out_row = row.to_dict()
    out_row["__row_id__"] = row_id
    out_row["generic_prompt"] = generic
    out_row["precised_variants"] = json.dumps(precised_variants, ensure_ascii=False)
    out_row["precised_prompts"] = json.dumps(precised_prompts, ensure_ascii=False)
    out_row["notes"] = " | ".join(notes)
    out_row["error"] = err
    
    return out_row


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input CSV path")
    ap.add_argument("--output", required=True, help="Output CSV path")
    ap.add_argument("--prompt-col", default="prompt", help="Column name containing prompts")
    ap.add_argument("--id-col", default=None, help="Optional stable id column; else row index is used")

    ap.add_argument("--model", default="gpt-5.2", help="Model name (e.g., gpt-5.2 or gpt-5)")
    ap.add_argument("--n-images", type=int, default=45, help="Total prompts/images to allocate per row")

    ap.add_argument("--genders", default=",".join(DEFAULT_GENDERS), help="Comma-separated genders")
    ap.add_argument("--ethnicities", default=",".join(DEFAULT_ETHNICITIES), help="Comma-separated ethnicities")

    ap.add_argument("--gender-weights", default="", help='e.g. "female=0.5,male=0.5" (optional)')
    ap.add_argument("--ethnicity-weights", default="", help='e.g. "white=0.6,black=0.13,asian=0.07,hispanic=0.2" (optional)')

    ap.add_argument("--chunksize", type=int, default=200, help="CSV chunk size")
    ap.add_argument("--max-rows", type=int, default=None, help="Process only first N rows (debug)")
    ap.add_argument("--resume", action="store_true", help="Skip rows already present in output (by id)")
    ap.add_argument("--workers", type=int, default=10, help="Number of parallel workers for API calls")
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    genders = [x.strip().lower() for x in args.genders.split(",") if x.strip()]
    ethnicities = [x.strip().lower() for x in args.ethnicities.split(",") if x.strip()]

    if not genders or not ethnicities:
        raise ValueError("You must provide at least one gender and one ethnicity.")

    gender_w = normalize_weights(genders, parse_kv_weights(args.gender_weights))
    eth_w = normalize_weights(ethnicities, parse_kv_weights(args.ethnicity_weights))

    cmb = combos(genders, ethnicities)
    counts = compute_combo_counts(genders, ethnicities, gender_w, eth_w, args.n_images)

    strip_re = build_strip_regex(genders, ethnicities)

    # Resume support
    done_ids = set()
    if args.resume and os.path.exists(args.output):
        try:
            out_df = pd.read_csv(args.output)
            id_col = args.id_col if args.id_col else "__row_id__"
            if id_col in out_df.columns:
                done_ids = set(out_df[id_col].astype(str).tolist())
        except Exception:
            # If output can't be read, don't resume.
            done_ids = set()

    client = OpenAI()

    # Prepare output writing (append mode)
    first_write = not os.path.exists(args.output) or (not args.resume)
    out_cols = None

    # Get total row count for progress bar
    total_rows = sum(1 for _ in pd.read_csv(args.input, chunksize=args.chunksize))
    if args.max_rows:
        total_rows = min(total_rows, args.max_rows)
    
    processed = 0
    call_count = 0
    start_time = time.time()
    lock = threading.Lock()

    # Progress bar with detailed stats
    pbar = tqdm(
        total=total_rows,
        desc="Processing prompts",
        unit="row",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
    )

    for chunk in pd.read_csv(args.input, chunksize=args.chunksize):
        if args.max_rows is not None and processed >= args.max_rows:
            break

        # Prepare tasks for parallel processing
        tasks = []
        for idx, row in chunk.iterrows():
            if args.max_rows is not None and processed >= args.max_rows:
                break

            row_id = str(row.get(args.id_col)) if args.id_col else str(idx)

            if args.resume and row_id in done_ids:
                with lock:
                    processed += 1
                    pbar.update(1)
                continue

            tasks.append((row_id, row, args.prompt_col, strip_re, cmb, counts, client, args.model))

        if not tasks:
            continue

        # Process tasks in parallel
        rows_out = []
        errors = 0
        
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            # Submit all tasks
            future_to_task = {executor.submit(process_single_row, task): task for task in tasks}
            
            # Process completed tasks
            for future in as_completed(future_to_task):
                try:
                    result = future.result()
                    rows_out.append(result)
                    
                    with lock:
                        processed += 1
                        call_count += 1
                        if result["error"]:
                            errors += 1
                        
                        # Update progress bar with stats
                        elapsed = time.time() - start_time
                        rate = call_count / elapsed if elapsed > 0 else 0
                        pbar.set_postfix({
                            "req/s": f"{rate:.2f}",
                            "errors": errors,
                            "workers": args.workers
                        })
                        pbar.update(1)
                        
                except Exception as ex:
                    with lock:
                        processed += 1
                        errors += 1
                        pbar.update(1)
                    tqdm.write(f"Error processing task: {type(ex).__name__}: {ex}")

        if not rows_out:
            continue

        # Write batch results
        out_df = pd.DataFrame(rows_out)

        # Ensure stable column order
        if out_cols is None:
            out_cols = list(out_df.columns)
        out_df = out_df.reindex(columns=out_cols)

        # Write (thread-safe with lock)
        with lock:
            write_header = first_write and (not args.resume or not os.path.exists(args.output))
            mode = "w" if write_header else "a"
            out_df.to_csv(args.output, index=False, mode=mode, header=write_header)
            first_write = False

    pbar.close()
    
    elapsed_total = time.time() - start_time
    avg_rate = call_count / elapsed_total if elapsed_total > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"Processing Complete!")
    print(f"{'='*60}")
    print(f"Total rows processed: {processed}")
    print(f"API calls made: {call_count}")
    print(f"Total time: {elapsed_total:.2f}s")
    print(f"Average rate: {avg_rate:.2f} requests/second")
    print(f"Workers used: {args.workers}")
    print(f"Output saved to: {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
