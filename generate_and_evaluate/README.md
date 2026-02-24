# generate_and_evaluate

> **End-to-end pipeline** — generates images via a cloud API **and** immediately computes bias/fairness metrics in a single run.

---

## What it does

`concat_generate_and_evaluate.py` combines the generation and evaluation steps into one script:

1. **Original pass** — reads generic prompts from a CSV, generates images via the cloud API.
2. **Evaluate original** — runs FairFace + CLIP metrics on the original-pass images.
3. **Enhanced pass** — reads diversity-enhanced prompts from the same (or a different) CSV, generates images.
4. **Evaluate enhanced** — runs the same metrics on the enhanced-pass images.

This is the most convenient entry point if you want to go from prompts → metric CSVs in a single command. Use `generate_only` + `evaluate_only` separately if you want more control over each step.

A `test.py` helper is also included to verify your API key and connectivity before running the full pipeline.

---

## Folder structure

```
generate_and_evaluate/
├── concat_generate_and_evaluate.py   # main script
├── config.yaml                       # all settings (edit before running)
├── logger.py                         # shared logging helpers
├── requirements.txt                  # Python dependencies
├── test.py                           # quick API connectivity test
├── .env                              # secrets – API_KEY, BASE_URL (not committed)
├── inputs/                           # put your CSV prompt file(s) here
│   └── prompts.csv
├── generated_images/                 # output images (gitignored)
└── eval_outputs/                     # metric CSVs (gitignored)
```

You also need a **models/** directory accessible from the run path:

```
models/
├── res34_fair_align_multi_7_20190809.pt   # FairFace weights
└── shape_predictor_5_face_landmarks.dat   # dlib landmark model
```

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
# CLIP (required for ICAD metrics):
pip install git+https://github.com/openai/CLIP.git
```

### 2. Download model weights

| File | Source |
|------|--------|
| `res34_fair_align_multi_7_20190809.pt` | [FairFace GitHub](https://github.com/dchen236/FairFace) |
| `shape_predictor_5_face_landmarks.dat` | [dlib model zoo](http://dlib.net/files/) |

### 3. Create `.env`

```dotenv
API_KEY=your_api_key_here
BASE_URL=https://api.metisai.ir
```

### 4. Edit `config.yaml`

Key sections:

```yaml
cloud_api:
  model:
    name: "google"
    model: "nano-banana"

generation:
  images_per_prompt: 2
  workers: 4
  output_dir: "generated_images"

csv:
  file: "inputs/prompts.csv"
  original_prompt_col: "generic_prompt"
  enhanced_prompt_col: "enhanced_prompts"
  enhanced_is_json: true

metrics:
  base_dir: "/path/to/models"   # FairFace + dlib weights

# Evaluate-only shortcut (skip generation):
evaluate_only:
  enabled: false
  eval_folder: null
  eval_tags: "original,enhanced"
```

### 5. Test your API connection (optional)

```bash
python test.py
```

### 6. Run the full pipeline

```bash
python concat_generate_and_evaluate.py
```

---

## Evaluate-only mode

If you already have images from a previous run, set `evaluate_only.enabled: true` in `config.yaml` and point `eval_folder` at the images directory to skip generation and jump straight to metrics.

---

## CSV format

| Column | Description |
|--------|-------------|
| `generic_prompt` | Plain text prompt for the *original* pass |
| `enhanced_prompts` | JSON list of strings **or** plain string for the *enhanced* pass |

---

## Outputs

```
generated_images/<model>_output_<date>/
    original/   ← generated images (original pass)
    enhanced/   ← generated images (enhanced pass)

eval_outputs/
    original/
        fairface_predictions.csv
        bias_metrics.csv
        icad_metrics.csv
    enhanced/
        fairface_predictions.csv
        bias_metrics.csv
        icad_metrics.csv
```

---

## Metrics computed

| Metric | Description |
|--------|-------------|
| **Bias-W** | Worst-group representation gap |
| **Bias-P** | Pairwise demographic disparity |
| **ENS** | Effective Number of Species (diversity index) |
| **KL-divergence** | Distribution distance from reference |
| **Stratified ICAD** | Per-group Image-Caption Alignment Discrepancy (CLIP) |

---

## Configuration reference

| Key | Default | Description |
|-----|---------|-------------|
| `cloud_api.model.name` | `google` | Provider family |
| `cloud_api.model.model` | `nano-banana` | Model identifier |
| `cloud_api.poll_interval` | `5` | Seconds between status polls |
| `cloud_api.poll_timeout` | `300` | Max wait per task (seconds) |
| `generation.images_per_prompt` | `2` | Images per prompt |
| `generation.workers` | `4` | Parallel threads |
| `csv.file` | `inputs/prompts.csv` | Prompt CSV path |
| `csv.num_rows` | `null` | Limit rows processed |
| `csv.seed` | `null` | RNG seed |
| `metrics.base_dir` | *(required)* | Model weights directory |
| `evaluate_only.enabled` | `false` | Skip generation, evaluate existing images |
