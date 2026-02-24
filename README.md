# Local Evaluation Pipeline

> **Branch:** `local_eval_pipeline` — part of the [HEAL Team llm-bias-fairness](https://github.com/HEAL-Team-org/llm-bias-fairness) project.

## Overview

This folder contains the complete **local evaluation pipeline** for measuring demographic bias in text-to-image generative models. It supports **five models**:

| Model | HuggingFace ID | Server File | Port |
|---|---|---|---|
| FLUX.1-dev | `black-forest-labs/FLUX.1-dev` | `serve_flux_dev.py` | 8004 |
| Qwen-Image-2512 | `Qwen/Qwen-Image-2512` | `serve_qwen.py` | 8001 |
| Stable Diffusion XL Base 1.0 | `stabilityai/stable-diffusion-xl-base-1.0` | `serve_sdxl.py` | 8002 |
| Stable Diffusion 3.5 Large | `stabilityai/stable-diffusion-3.5-large` | `serve_sd35.py` | 8002 |
| Z-Image | `Tongyi-MAI/Z-Image` | `serve_zimage.py` | 8003 |

> **Note:** SDXL and SD 3.5 Large share port 8002 because they are never run simultaneously — each model occupies both GPUs entirely.

The pipeline uses a **client–server architecture**: a GPU server hosts the generative model and exposes a `/generate` HTTP endpoint; the evaluation client reads prompts from a CSV, requests images from the server, and computes bias/diversity metrics (Bias-W, Bias-P, ENS, KL, ICAD).

---

## Project Structure

```
local_eval/
├── concat_generate_and_evaluate.py   # Main evaluation client
├── logger.py                         # Shared logging utilities
├── requirements.txt                  # Python dependencies
├── .gitignore
├── README.md
└── local_models/                     # FastAPI model servers (run on GPU machine)
    ├── serve_flux_dev.py             # FLUX.1-dev                (port 8004)
    ├── serve_qwen.py                 # Qwen-Image-2512           (port 8001)
    ├── serve_sdxl.py                 # SDXL Base 1.0             (port 8002)
    ├── serve_sd35.py                 # SD 3.5 Large              (port 8002)
    ├── serve_zimage.py               # Z-Image                   (port 8003)
    └── test.py                       # Quick connectivity/sanity test
```

> **Note:** The following directories are intentionally excluded from version control (see `.gitignore`):
> - `generated_images/` — all generated PNG outputs
> - `eval_outputs/` — CSV metric results
> - `test/` — sanity-test images
> - `metrics_models/` — large pre-trained model weights (FairFace, dlib)

---

## 1. Prerequisites

### 1a. Server Side (GPU machine with 2× A100 40 GB)

```bash
pip install fastapi uvicorn accelerate transformers sentencepiece torch torchvision

# Use the git version of diffusers — required by Qwen and Z-Image, also works for FLUX / SDXL / SD3.5:
pip install git+https://github.com/huggingface/diffusers
```

**HuggingFace gated models:** FLUX.1-dev and SD 3.5 Large require accepting the respective model licences on HuggingFace and authenticating:

```bash
huggingface-cli login
# or export HF_TOKEN=<your_token>
```

**Note:** Ensure your PyTorch version matches your CUDA driver version.

### 1b. Client Side (machine running the evaluation script)

Install all client dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

This includes:
- `requests`, `pandas`, `numpy`, `pillow` — core data handling
- `torch`, `torchvision` — FairFace model inference
- `dlib` — face detection (requires cmake; see [dlib install guide](http://dlib.net/compile.html))
- `git+https://github.com/openai/CLIP.git` — ICAD metric computation

**Note:** `dlib` may require a C++ compiler and cmake to build from source.

## 2. Starting the Model Servers

Because each model is large (~10–20 GB) and split across both GPUs, **run only one model server at a time**. All servers must be started with `--workers 1`.

### Option A: FLUX.1-dev

Hosts `black-forest-labs/FLUX.1-dev` — the full 12 B rectified-flow transformer split across both GPUs via `device_map="balanced"`.

```bash
uvicorn local_models.serve_flux_dev:app --host 0.0.0.0 --port 8004 --workers 1
```

- **Port:** 8004
- **Startup time:** ~1–2 min (model load + warmup)
- **Health check:** `curl http://localhost:8004/health`
- **Docs:** `http://localhost:8004/docs`

### Option B: Qwen-Image-2512

Hosts `Qwen/Qwen-Image-2512` (20 B params) split across both GPUs with VAE tiling enabled. Slower at native high resolutions (1328×1328, 1664×928, etc.).

```bash
uvicorn local_models.serve_qwen:app --host 0.0.0.0 --port 8001 --workers 1
```

- **Port:** 8001
- **Startup time:** ~2–3 min
- **Health check:** `curl http://localhost:8001/health`
- **Docs:** `http://localhost:8001/docs`

### Option C: Stable Diffusion XL Base 1.0

Hosts `stabilityai/stable-diffusion-xl-base-1.0` with VAE tiling/slicing enabled.

```bash
uvicorn local_models.serve_sdxl:app --host 0.0.0.0 --port 8002 --workers 1
```

- **Port:** 8002
- **Startup time:** ~1–2 min
- **Health check:** `curl http://localhost:8002/health`
- **Docs:** `http://localhost:8002/docs`

### Option D: Stable Diffusion 3.5 Large

Hosts `stabilityai/stable-diffusion-3.5-large` (8 B MMDiT) split across both GPUs. Requires HuggingFace authentication (gated model).

```bash
uvicorn local_models.serve_sd35:app --host 0.0.0.0 --port 8002 --workers 1
```

- **Port:** 8002 (same as SDXL — never run both simultaneously)
- **Startup time:** ~2–3 min
- **Health check:** `curl http://localhost:8002/health`
- **Docs:** `http://localhost:8002/docs`

### Option E: Z-Image

Hosts `Tongyi-MAI/Z-Image` (Single-Stream Diffusion Transformer) split across both GPUs with VAE tiling enabled. Requires the latest `diffusers` from source for `ZImagePipeline`.

```bash
# Ensure latest diffusers is installed first:
pip install git+https://github.com/huggingface/diffusers

uvicorn local_models.serve_zimage:app --host 0.0.0.0 --port 8003 --workers 1
```

- **Port:** 8003
- **Startup time:** ~1–2 min
- **Health check:** `curl http://localhost:8003/health`
- **Docs:** `http://localhost:8003/docs`

---

## 3. Running the Evaluation Client

Once a server is running, use `concat_generate_and_evaluate.py` to run the full pipeline:
1. Read prompts from CSV.
2. Send generation requests to the local server.
3. Save images.
4. Run bias metrics (Face Detection → FairFace → CLIP ICAD).

### Data Format

Your input CSV (`original_prompts.csv`) should have the following columns:

```csv
prompt,__row_id__,generic_prompt,modified_prompts
"Prompt text...",0,"Prompt text...","[""Variant 1"", ""Variant 2"", ...]"
```

- **Original run:** uses `generic_prompt` (single string per row).
- **Enhanced run:** uses `modified_prompts` (JSON list of strings per row).

### Example Run Commands

#### Evaluate FLUX.1-dev (port 8004)

```bash
python concat_generate_and_evaluate.py \
  --csv-file original_prompts.csv \
  --model-server-url http://127.0.0.1:8004/generate \
  --metrics-base-dir /path/to/metrics_models \
  --original-prompt-col generic_prompt \
  --enhanced-prompt-col modified_prompts \
  --images-per-prompt 10
```

#### Evaluate Qwen-Image-2512 (port 8001)

Qwen is slower at high resolutions — increase the timeout accordingly:

```bash
python concat_generate_and_evaluate.py \
  --csv-file original_prompts.csv \
  --model-server-url http://127.0.0.1:8001/generate \
  --metrics-base-dir /path/to/metrics_models \
  --original-prompt-col generic_prompt \
  --enhanced-prompt-col modified_prompts \
  --request-timeout 600 \
  --images-per-prompt 10
```

#### Evaluate SDXL Base 1.0 (port 8002)

```bash
python concat_generate_and_evaluate.py \
  --csv-file original_prompts.csv \
  --model-server-url http://127.0.0.1:8002/generate \
  --metrics-base-dir /path/to/metrics_models \
  --original-prompt-col generic_prompt \
  --enhanced-prompt-col modified_prompts \
  --images-per-prompt 10
```

#### Evaluate SD 3.5 Large (port 8002)

```bash
python concat_generate_and_evaluate.py \
  --csv-file original_prompts.csv \
  --model-server-url http://127.0.0.1:8002/generate \
  --metrics-base-dir /path/to/metrics_models \
  --original-prompt-col generic_prompt \
  --enhanced-prompt-col modified_prompts \
  --images-per-prompt 10
```

#### Evaluate Z-Image (port 8003)

```bash
python concat_generate_and_evaluate.py \
  --csv-file original_prompts.csv \
  --model-server-url http://127.0.0.1:8003/generate \
  --metrics-base-dir /path/to/metrics_models \
  --original-prompt-col generic_prompt \
  --enhanced-prompt-col modified_prompts \
  --images-per-prompt 10
```

---

## 4. Output

Outputs are grouped by model and date:

`<model>_output_YYYYMMDD`

Examples:
- `flux_dev_output_20260224`
- `qwen_output_20260224`
- `sdxl_output_20260224`
- `sd35_output_20260224`
- `zimage_output_20260224`

Directory layout:

```
generated_images/
└── <model>_output_YYYYMMDD/
    ├── original/     ← images generated from generic_prompt
    └── enhanced/     ← images generated from modified_prompts

eval_outputs/
└── <model>_output_YYYYMMDD/
    └── *.csv         ← bias/diversity metric results
```

All output directories are excluded from version control via `.gitignore`.

---

## 5. Metrics

The evaluation client computes the following bias and diversity metrics after face detection (dlib) and attribute prediction (FairFace):

| Metric | Description |
|---|---|
| **Bias-W** | Demographic bias based on face-attribute distributions |
| **Bias-P** | Bias measured via predicted attribute proportions |
| **ENS** | Entropy-based diversity score |
| **KL** | KL divergence from a uniform demographic distribution |
| **ICAD** | Image–Caption Alignment Diversity (CLIP-based) |

---

## 6. Quick Sanity Test

`local_models/test.py` verifies that one or more servers are running and can produce a sample image. Test images are written to `test/` (also excluded from version control).

```bash
# Test all servers
python local_models/test.py

# Test individual models
python local_models/test.py --flux        # FLUX.1-dev  (port 8004)
python local_models/test.py --qwen        # Qwen        (port 8001)
python local_models/test.py --sdxl        # SDXL        (port 8002)
python local_models/test.py --sd35        # SD3.5 Large (port 8002)
python local_models/test.py --zimage      # Z-Image     (port 8003)

# Override prompt or seed
python local_models/test.py --prompt "a cat in space" --seed 123
```

Test image layout:
```
test/
├── flux_dev_output_YYYYMMDD/flux_dev_0001.png
├── qwen_output_YYYYMMDD/qwen_0001.png
├── sdxl_output_YYYYMMDD/sdxl_0001.png
├── sd35_output_YYYYMMDD/sd35_0001.png
└── zimage_output_YYYYMMDD/zimage_0001.png
```

The script exits with a non-zero code if any requested test fails, making it suitable for automated connectivity checks.
