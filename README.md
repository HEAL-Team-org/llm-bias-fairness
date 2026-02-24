# LLM Bias & Fairness — Cloud Image Evaluation Toolkit

This repository contains three self-contained Python pipelines for **generating images with cloud text-to-image APIs** and **measuring demographic bias and diversity** in the outputs using FairFace and CLIP.

---

## Repository layout

```
cloud_eval/
├── generate_only/              # Step 1 – generate images, no evaluation
├── evaluate_only/              # Step 2 – evaluate existing images, no generation
└── generate_and_evaluate/      # All-in-one – generate then evaluate in one run
```

---

## Module overview

### `generate_only/`

Submits prompts from a CSV to a cloud image-generation API (Metis / OpenAI-compatible) and saves the images locally.

- Supports an **original** pass (generic prompts) and an **enhanced** pass (diversity-augmented prompts).
- Parallel generation with configurable worker threads.
- Outputs a timestamped folder of images plus generation metadata JSON.

→ See [generate_only/README.md](generate_only/README.md) for full usage.

---

### `evaluate_only/`

Runs a full bias and fairness evaluation on a folder of already-generated images.

- Face detection via **dlib**.
- Age / gender / race prediction via **FairFace** (ResNet-34).
- Computes **Bias-W**, **Bias-P**, **ENS**, **KL-divergence**, and **Stratified ICAD** (CLIP).
- Writes metric CSVs to `eval_outputs/`.

→ See [evaluate_only/README.md](evaluate_only/README.md) for full usage.

---

### `generate_and_evaluate/`

End-to-end pipeline that combines both steps above in a single script.

1. Generate images (original + enhanced prompts).
2. Immediately evaluate each set of images.

Useful when you want to go from prompts → metric CSVs without intermediate steps.  
Also includes a `test.py` script to verify API connectivity.

→ See [generate_and_evaluate/README.md](generate_and_evaluate/README.md) for full usage.

---

## Recommended workflow

```
┌──────────────────────────────────────────────────────────┐
│  Option A – quick, combined run                          │
│                                                          │
│  generate_and_evaluate/concat_generate_and_evaluate.py   │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│  Option B – separate steps (more control / reuse)        │
│                                                          │
│  1.  generate_only/generate_only.py                      │
│  2.  evaluate_only/evaluate_only.py  --eval-folder …     │
└──────────────────────────────────────────────────────────┘
```

---

## Shared requirements

The following resources are needed by **evaluate_only** and **generate_and_evaluate**:

| Resource | Purpose | Where to get it |
|----------|---------|-----------------|
| `res34_fair_align_multi_7_20190809.pt` | FairFace model weights | [FairFace GitHub](https://github.com/dchen236/FairFace) |
| `shape_predictor_5_face_landmarks.dat` | dlib face landmark model | [dlib model zoo](http://dlib.net/files/) |
| CLIP | ICAD metric | `pip install git+https://github.com/openai/CLIP.git` |

Place the two model files in a directory and set `metrics.base_dir` in the relevant `config.yaml`.

---

## Secrets / environment variables

Each module that calls the cloud API reads credentials from a `.env` file placed **in the same folder as the script**:

```dotenv
API_KEY=your_api_key_here
BASE_URL=https://api.metisai.ir
```

`.env` files are excluded from version control via `.gitignore`.

---

## Gitignored outputs

The following are excluded from this repository:

- Generated image folders (`generated_images/`)
- Evaluation output CSVs (`eval_outputs/`)
- Log files (`logs/`, `*.log`)
- Model weight files (`*.pt`, `*.dat`)
- Environment files (`.env`)
- Python cache (`__pycache__/`, `*.pyc`)
