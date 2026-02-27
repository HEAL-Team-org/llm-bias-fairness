# Directional Similarity Evaluation — Usage Guide

Measures how well a RAG-enhanced prompt shifts image generation in the same
direction as the text embedding shift (base → enhanced), using
**[SigLIP 2 Giant](https://huggingface.co/google/siglip2-giant-opt-patch16-384)**
(ViT-Giant/16, 2 B params, 1152-dim embeddings).

---

## Quick answer: controlling the number of images

> **Yes — use `--num-images`.**
>
> Pass the number of images you want per prompt variant.  Seeds are generated
> automatically and deterministically (so the same count always produces the
> same images, and interrupted runs can be safely resumed).
>
> `--num-images 3` produces **3 base + 3 enhanced = 6 images per row**.
>
> Default: `--num-images 5`  → **5 images per variant per row**

---

## Three scripts

| Script | Purpose |
|---|---|
| `generate_images.py` | **Stage 1** — generate all images, write to disk |
| `calculate_metric.py` | **Stage 2** — compute directional similarity from disk images |
| `run_pipeline.py` | **Full pipeline** — run Stage 1, then Stage 2 automatically |

---

## Prerequisites

```bash
pip install torch torchvision transformers Pillow requests accelerate
```

The **Qwen-Image-2512** generation server must already be running:

```bash
uvicorn serve_model.serve_qwen:app --host 0.0.0.0 --port 8001 --workers 1
```

Or for **FLUX.1-dev**:

```bash
uvicorn serve_model.serve_flux_dev:app --host 0.0.0.0 --port 8004 --workers 1
```

---

## `generate_images.py` — Stage 1: Image Generation

Generates **all** base and enhanced images for every row in the CSV and
writes them to disk. No metric computation takes place here.

**Output layout:**
```
outputs/directional_similarity/
  qwen/               ← model subfolder (set by --model)
    row_0/
      base_seed00042.png
      base_seed00123.png
      enhanced_seed00042.png
      enhanced_seed00123.png
    row_1/
      ...
  flux-dev/
    row_0/
      ...
```

### Minimal usage

```bash
python generate_images.py
```

### All options

```bash
python generate_images.py \
  --model            qwen \                         # ← which model to use (qwen or flux-dev)
  --csv              inputs/prompts.csv \        # input prompts CSV
  --image-dir        outputs/directional_similarity \  # where to save images
  --server           http://localhost:8001 \     # generation server URL (auto-set from --model)
  --num-images       5 \                         # ← number of images per variant
  --enhanced-idx     0 \                         # which enhanced prompt variant (0-indexed)
  --width            1328 \                      # image width in pixels  (auto-set from --model)
  --height           1328 \                      # image height in pixels (auto-set from --model)
  --steps            50 \                        # diffusion inference steps
  --force-regenerate                             # ignore cached PNGs, regenerate all
```

```bash
python generate_images.py \
  --csv              inputs/prompts.csv \
  --image-dir        outputs/directional_similarity \
  --server           http://localhost:8001 \
  --num-images       2
```

### Flag reference

| Flag | Default | Description |
|---|---|---|
| `--model` | `qwen` | Generation model: `qwen` (port 8001, 1328×1328) or `flux-dev` (port 8004, 1024×1024). Sets `--server`, `--width`, `--height` automatically |
| `--csv` | `inputs/prompts.csv` | Input CSV with `generic_prompt` and `enhanced_prompts` columns |
| `--image-dir` | `outputs/directional_similarity` | Root folder where images are saved |
| `--server` | *(auto from --model)* | Generation server URL — overrides the model default |
| `--num-images` | `5` | **Number of images per prompt variant.** Seeds are auto-generated deterministically |
| `--enhanced-idx` | `0` | Which enhanced prompt variant to use (the CSV may contain multiple) |
| `--width` | *(auto from --model)* | Generated image width (px) — overrides the model default |
| `--height` | *(auto from --model)* | Generated image height (px) — overrides the model default |
| `--steps` | `50` | Number of diffusion inference steps per image |
| `--force-regenerate` | off | Re-generate every image, even if a cached PNG already exists on disk |

### Examples

```bash
# Generate 3 images per variant per row (faster test run)
python generate_images.py --num-images 3

# 10 images per variant for higher-confidence scores
python generate_images.py --num-images 10

# Faster generation: fewer steps, smaller images
python generate_images.py --steps 20 --width 512 --height 512

# Re-run everything from scratch
python generate_images.py --force-regenerate

# Use the second enhanced prompt variant
python generate_images.py --enhanced-idx 1

# Use FLUX.1-dev instead of Qwen (1024×1024 native, port 8004)
python generate_images.py --model flux-dev

# FLUX.1-dev, 3 images, 30 steps
python generate_images.py --model flux-dev --num-images 3 --steps 30
```

---

## `calculate_metric.py` — Stage 2: Metric Calculation

Loads **SigLIP 2 Giant**, reads all pre-generated images from disk, and
computes the CLIP directional similarity score for every row.

Requires Stage 1 (or `run_pipeline.py`) to have run first.

**Output files** (written to `results/<model>/`):
```
results/
  qwen/
    directional_similarity_YYYYMMDD_HHMMSS.json
    directional_similarity_YYYYMMDD_HHMMSS.csv
  flux-dev/
    directional_similarity_YYYYMMDD_HHMMSS.json
    directional_similarity_YYYYMMDD_HHMMSS.csv
```

Each result row contains:
- `row_id` — matches the `__row_id__` from the CSV
- `base_prompt` / `enhanced_prompt` — the two prompts compared
- `num_pairs` — number of seed pairs evaluated
- `per_pair_scores` — individual cosine score per seed pair
- `avg_score` — mean directional similarity  ∈ [−1, +1]

### Minimal usage

```bash
python calculate_metric.py
```

### All options

```bash
python calculate_metric.py \
  --model            qwen \                           # which model's images to load
  --csv              inputs/prompts.csv \
  --image-dir        outputs/directional_similarity \  # where generated images live
  --results-dir      results \                         # where to write output files
  --device           cuda:0 \                          # GPU for SigLIP 2 (~4 GB VRAM in float16)
  --enhanced-idx     0 \                               # must match the index used during generation
  --batch-size       8                                 # images per SigLIP2 forward pass
```

### Flag reference

| Flag | Default | Description |
|---|---|---|
| `--model` | `qwen` | Which model's generated images to evaluate (`qwen` or `flux-dev`) |
| `--csv` | `inputs/prompts.csv` | Input CSV (used to read prompt text for text embeddings) |
| `--image-dir` | `outputs/directional_similarity` | Root folder where Stage 1 wrote images |
| `--results-dir` | `results` | Folder for output JSON and CSV files |
| `--device` | auto (`cuda:0`) | Torch device for SigLIP 2 inference: `cuda:0`, `cuda:1`, `cpu` |
| `--enhanced-idx` | `0` | Must match the `--enhanced-idx` used during image generation |
| `--batch-size` | `8` | Images processed per SigLIP 2 forward pass — increase for faster inference, decrease if OOM |

> **Number of images evaluated** is determined automatically by how many
> `base_seed*.png` / `enhanced_seed*.png` files exist in each row folder.
> No flag is needed here — the script discovers and pairs all available images
> by sort order.

### Examples

```bash
# Run metric on second GPU
python calculate_metric.py --device cuda:1

# Larger batch for faster throughput on A100
python calculate_metric.py --batch-size 32

# Custom image directory
python calculate_metric.py --image-dir /data/my_images --results-dir /data/results

# Evaluate FLUX.1-dev images
python calculate_metric.py --model flux-dev

# FLUX.1-dev images, metric on cuda:1
python calculate_metric.py --model flux-dev --device cuda:1
```

---

## `run_pipeline.py` — Full Pipeline (recommended)

Runs Stage 1 and Stage 2 back-to-back. All images are generated **before**
metric calculation begins.

### Minimal usage

```bash
python run_pipeline.py
```

### All options

```bash
python run_pipeline.py \
  --csv              inputs/prompts.csv \
  --enhanced-idx     0 \
  --results-dir      results \
  \
  --model            qwen \          # ← qwen or flux-dev (sets server, width, height)
  --image-dir        outputs/directional_similarity \
  --server           http://localhost:8001 \
  --num-images       5 \
  --width            1328 \
  --height           1328 \
  --steps            50 \
  --force-regenerate \        # Stage 1: ignore cache
  --skip-generation \         # skip Stage 1 entirely (use pre-existing images)
  \
  --clip-device      cuda:0 \
  --batch-size       8
```

### Flag reference

#### General

| Flag | Default | Description |
|---|---|---|
| `--csv` | `inputs/prompts.csv` | Input prompts CSV |
| `--enhanced-idx` | `0` | Enhanced prompt variant index (must be consistent across both stages) |
| `--results-dir` | `results` | Output folder for JSON and CSV results |

#### Stage 1 — Image Generation

| Flag | Default | Description |
|---|---|---|
| `--model` | `qwen` | Generation model: `qwen` (port 8001, 1328×1328) or `flux-dev` (port 8004, 1024×1024). Auto-sets `--server`, `--width`, `--height` |
| `--image-dir` | `outputs/directional_similarity` | Root folder for images |
| `--server` | *(auto from --model)* | Generation server URL — overrides model default |
| `--num-images` | `5` | **Number of images per variant** — seeds are auto-generated deterministically |
| `--width` | *(auto from --model)* | Image width (px) — overrides model default |
| `--height` | *(auto from --model)* | Image height (px) — overrides model default |
| `--steps` | `50` | Diffusion inference steps per image |
| `--force-regenerate` | off | Re-generate all images even if cached |
| `--skip-generation` | off | Skip Stage 1 entirely; evaluate pre-existing images |

#### Stage 2 — Metric Calculation

| Flag | Default | Description |
|---|---|---|
| `--clip-device` | auto (`cuda:0`) | GPU/CPU for SigLIP 2 inference |
| `--batch-size` | `8` | Images per SigLIP 2 forward pass |

### Common workflows

```bash
# ── Standard full run (Qwen) ───────────────────────────────────────────────
python run_pipeline.py

# ── Full run with FLUX.1-dev ──────────────────────────────────────────────
python run_pipeline.py --model flux-dev

# ── Quick test run: 3 images per variant, faster generation ───────────────
python run_pipeline.py --num-images 3 --steps 20

# ── High-confidence run: 10 images per variant ────────────────────────────
python run_pipeline.py --num-images 10

# ── Two A100s: generation & metric on separate GPUs ───────────────────────
# (Qwen occupies both GPUs while running; run metric after generation)
python generate_images.py --num-images 5
python calculate_metric.py --device cuda:1

# ── Two A100s: FLUX.1-dev images, metric on cuda:1 ────────────────────────
python generate_images.py --model flux-dev --num-images 5
python calculate_metric.py --model flux-dev --device cuda:1

# ── Re-run metric only after changing nothing about images ────────────────
python run_pipeline.py --skip-generation --clip-device cuda:0

# ── Force full re-run from scratch ────────────────────────────────────────
python run_pipeline.py --force-regenerate

# ── Use a non-default server port ─────────────────────────────────────────
python run_pipeline.py --server http://localhost:8000
```

---

## Interpreting results

| `avg_score` | Meaning |
|---|---|
| `+1.0` | Visual shift perfectly mirrors the textual shift |
| `+0.3` → `+0.7` | Moderate alignment — enhancement has measurable visual effect |
| `~0.0` | Visual shift is unrelated to the textual direction |
| `< 0` | Visual shift goes in the opposite direction from the text |

Results are saved to `results/directional_similarity_<TIMESTAMP>.json` and
`.csv` after every run, so multiple experiments do not overwrite each other.
