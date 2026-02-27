# Directional Similarity — Image Fairness Metric

Evaluates how well a **RAG-enhanced prompt** shifts image generation toward greater diversity and fairness, compared to a generic base prompt.

The metric is **CLIP Directional Similarity** (StyleGAN-NADA / Gal et al., 2021), measured using **[SigLIP 2 Giant](https://huggingface.co/google/siglip2-giant-opt-patch16-384)** (ViT-Giant/16, 2 B params, 1152-dim embeddings, float16).

---

## How the Metric Works

$$S_{\text{dir}} = \frac{1}{N} \sum_{i=1}^{N} \cos\!\left(\Delta I_i,\; \Delta T\right)$$

| Symbol | Meaning |
|---|---|
| $\Delta T$ | `norm(embed_text(enhanced) − embed_text(base))` |
| $\Delta I_i$ | `norm(embed_image(enhanced_i) − embed_image(base_i))` |
| $S_{\text{dir}}$ | Mean cosine similarity across all seed pairs |

| Score | Interpretation |
|---|---|
| **+1** | Visual shift perfectly mirrors the textual shift |
| **0** | Orthogonal — no measurable correlation |
| **−1** | Visual shift opposes the textual shift |

---

## Project Structure

```
metric/
├── run_pipeline.py          # Full pipeline: generate → evaluate (recommended entry point)
├── generate_images.py       # Stage 1 — generate images via inference server
├── calculate_metric.py      # Stage 2 — compute directional similarity with SigLIP2
├── directional_similarity.py# Legacy all-in-one script (CLIP-ViT-H-14)
├── logger.py                # Shared structured logger
├── inputs/
│   └── prompts.csv          # Input: generic_prompt + enhanced_prompts columns
├── outputs/
│   └── directional_similarity/
│       └── <model>/row_<ID>/
│           ├── base_seed00042.png
│           └── enhanced_seed00042.png
├── results/
│   └── <model>/
│       ├── directional_similarity_<TS>.json
│       └── directional_similarity_<TS>.csv
└── serve_model/
    ├── serve_qwen.py         # FastAPI server for Qwen-Image-2512 (port 8001)
    └── serve_flux_dev.py     # FastAPI server for FLUX.1-dev     (port 8004)
```

---

## Prerequisites

```bash
pip install torch torchvision transformers Pillow requests accelerate
```

Start the inference server for your chosen model **before** running the pipeline:

```bash
# Qwen-Image-2512  (default)
uvicorn serve_model.serve_qwen:app --host 0.0.0.0 --port 8001 --workers 1

# FLUX.1-dev
uvicorn serve_model.serve_flux_dev:app --host 0.0.0.0 --port 8004 --workers 1
```

---

## Quickstart

### Option A — Full pipeline (recommended)

```bash
# Default: Qwen model, 5 images per variant, SigLIP2 on cuda:0
python run_pipeline.py

# 10 images, FLUX.1-dev model, metric on second GPU
python run_pipeline.py --model flux-dev --num-images 10 --clip-device cuda:1

# Skip image generation (images already on disk) and only run metric
python run_pipeline.py --skip-generation

# Re-generate all images from scratch, ignoring cache
python run_pipeline.py --force-regenerate
```

### Option B — Stage by stage

```bash
# Stage 1: generate images
python generate_images.py --model qwen --num-images 5

# Stage 2: compute metric (uses images from Stage 1)
python calculate_metric.py --model qwen
```

---

## Input CSV Format

`inputs/prompts.csv` must contain at least these columns:

| Column | Description |
|---|---|
| `__row_id__` | Unique row identifier |
| `generic_prompt` | Base / original prompt |
| `enhanced_prompts` | JSON array of RAG-enhanced prompt strings |

---

## CLI Reference

### `run_pipeline.py`

| Flag | Default | Description |
|---|---|---|
| `--model` | `qwen` | Generation model: `qwen` or `flux-dev` |
| `--num-images` | `5` | Images per prompt variant per row |
| `--steps` | `50` | Diffusion inference steps |
| `--enhanced-idx` | `0` | Which enhanced prompt variant to use (0-indexed) |
| `--clip-device` | `cuda:0` | Torch device for SigLIP 2 |
| `--batch-size` | `8` | Images per SigLIP2 forward pass |
| `--skip-generation` | off | Skip Stage 1; use pre-existing images |
| `--force-regenerate` | off | Ignore cached PNGs and regenerate all |
| `--csv` | `inputs/prompts.csv` | Input CSV path |
| `--image-dir` | `outputs/directional_similarity` | Root image output directory |
| `--results-dir` | `results` | Directory for output JSON/CSV files |

### `generate_images.py`

| Flag | Default | Description |
|---|---|---|
| `--model` | `qwen` | `qwen` (1328×1328, port 8001) or `flux-dev` (1024×1024, port 8004) |
| `--num-images` | `5` | Images per prompt variant per row |
| `--steps` | `50` | Diffusion inference steps |
| `--enhanced-idx` | `0` | Enhanced prompt variant index |
| `--force-regenerate` | off | Re-generate even if PNG already exists |
| `--server` | *(auto)* | Override inference server URL |
| `--width` / `--height` | *(auto)* | Override image dimensions in pixels |

### `calculate_metric.py`

| Flag | Default | Description |
|---|---|---|
| `--model` | `qwen` | Which model's images to evaluate |
| `--device` | `cuda:0` | Torch device for SigLIP 2 (~4 GB VRAM in float16) |
| `--enhanced-idx` | `0` | Must match the value used during generation |
| `--batch-size` | `8` | Images per SigLIP2 forward pass |
| `--image-dir` | `outputs/directional_similarity` | Root folder of generated images |
| `--results-dir` | `results` | Output directory for JSON + CSV |

---

## Output

Results are written to `results/<model>/` as both JSON and CSV:

```json
{
  "row_id": "0",
  "base_prompt": "Generate a close-up image of a group of Doctors...",
  "enhanced_prompt": "Generate a vibrant, close-up image showcasing...",
  "num_pairs": 5,
  "per_pair_scores": [0.312, 0.278, 0.341, 0.290, 0.305],
  "avg_score": 0.305
}
```

---

## Hardware Requirements

| Component | Requirement |
|---|---|
| SigLIP 2 Giant (metric) | ~4 GB VRAM (float16), GPU recommended |
| Qwen-Image-2512 (generation) | ~16 GB VRAM recommended |
| FLUX.1-dev (generation) | ~24 GB VRAM recommended |

Running the generation server and metric on **separate GPUs** is supported via `--clip-device cuda:1`.

---

## References

- Gal et al. (2021). *StyleGAN-NADA: CLIP-Guided Domain Adaptation of Image Generators.* ([arXiv:2108.00946](https://arxiv.org/abs/2108.00946))
- Zhai et al. (2023). *Sigmoid Loss for Language-Image Pre-training (SigLIP).* ([arXiv:2303.15343](https://arxiv.org/abs/2303.15343))
