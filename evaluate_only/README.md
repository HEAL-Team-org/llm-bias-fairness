# evaluate_only

> **Bias & fairness evaluation pipeline** — runs entirely on already-generated images. No image generation is performed.

---

## What it does

`evaluate_only.py` takes a folder of images (typically produced by `generate_only`) and computes a comprehensive set of demographic-bias and diversity metrics:

| Metric | Description |
|--------|-------------|
| **Bias-W** | Worst-group representation gap |
| **Bias-P** | Pairwise demographic disparity |
| **ENS** | Effective Number of Species (diversity index) |
| **KL-divergence** | Distance between the generated distribution and a reference/uniform distribution |
| **Stratified ICAD** | Image-Caption Alignment Discrepancy computed per demographic group via CLIP |

Faces are detected with **dlib** and age / gender / race are predicted by the **FairFace ResNet-34** model.

Results are written as CSV files under `eval_outputs/<run_tag>/`.

---

## Folder structure

```
evaluate_only/
├── evaluate_only.py      # main script
├── config.yaml           # settings (edit before running)
├── logger.py             # shared logging helpers
├── requirements.txt      # Python dependencies
└── eval_outputs/         # metric CSVs written here (gitignored)
    └── <run_tag>/
```

You also need a **models/** directory somewhere accessible containing:

```
models/
├── res34_fair_align_multi_7_20190809.pt   # FairFace weights
└── shape_predictor_5_face_landmarks.dat   # dlib face landmark model
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

Place both files in the directory set by `metrics.base_dir` in `config.yaml`.

### 3. Edit `config.yaml`

```yaml
metrics:
  base_dir: "/path/to/models"   # folder with FairFace + dlib weights
  output_dir: "eval_outputs"    # where to write metric CSVs

input:
  eval_folder: "generated_images/my_run_output_20260224"
  eval_tags: "original,enhanced"   # sub-folder names to evaluate
```

### 4. Run

```bash
# Minimal – reads eval_folder from config.yaml
python evaluate_only.py

# Point at a specific images folder
python evaluate_only.py --eval-folder generated_images/my_run_output_20260224

# Evaluate only one tag
python evaluate_only.py --eval-folder path/to/images --eval-tags original

# Override model and output directories
python evaluate_only.py \
    --eval-folder path/to/images \
    --metrics-base-dir /path/to/models \
    --output-dir my_eval_results
```

---

## Expected input layout

The script accepts **either** layout:

**Layout A** – tag sub-folders directly inside `eval_folder`:

```
eval_folder/
    original/
        image_001.png
        image_002.png
    enhanced/
        image_001.png
```

**Layout B** – flat folder (all images in `eval_folder` itself, single tag).

---

## Outputs

All metric files are written to `eval_outputs/<run_tag>/`:

```
eval_outputs/
└── original/
    ├── fairface_predictions.csv     # per-image age/gender/race predictions
    ├── bias_metrics.csv             # Bias-W, Bias-P, ENS, KL
    ├── icad_metrics.csv             # Stratified ICAD scores
    └── detected_faces/              # cropped face images (optional)
```

---

## Configuration reference

| Key | Default | Description |
|-----|---------|-------------|
| `metrics.base_dir` | *(required)* | Path to FairFace + dlib model files |
| `metrics.output_dir` | `eval_outputs` | Root directory for output CSVs |
| `input.eval_folder` | *(required)* | Folder containing the generated images |
| `input.eval_tags` | `original,enhanced` | Comma-separated list of sub-folder tags to evaluate |

---

## CLI flags

| Flag | Description |
|------|-------------|
| `--eval-folder PATH` | Override `input.eval_folder` from config |
| `--eval-tags TAGS` | Override `input.eval_tags` (comma-separated) |
| `--metrics-base-dir PATH` | Override `metrics.base_dir` |
| `--output-dir PATH` | Override `metrics.output_dir` |
| `--config PATH` | Use a custom config file (default: `config.yaml` next to the script) |
