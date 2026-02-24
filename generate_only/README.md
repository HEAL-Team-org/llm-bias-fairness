# generate_only

> **Image generation pipeline** — calls a cloud text-to-image API and saves the results locally. No evaluation is performed here.

---

## What it does

`generate_only.py` reads one or two CSV files containing prompts, submits them to a cloud image-generation API (Metis / compatible OpenAI-style endpoint), polls until each task is complete, and saves the output images into a timestamped folder under `generated_images/`.

Two passes are supported:

| Pass | Description |
|------|-------------|
| `original` | Runs the raw/generic prompts from the CSV |
| `enhanced` | Runs the diversity-enhanced prompts from the CSV |

Both passes can be run together or independently via the `passes` config key.

---

## Folder structure

```
generate_only/
├── generate_only.py      # main script
├── config.yaml           # all settings (edit before running)
├── logger.py             # shared logging helpers
├── requirements.txt      # Python dependencies
├── .env                  # secrets – API_KEY, BASE_URL  (not committed)
├── inputs/               # put your CSV prompt file(s) here
│   └── simple.csv
└── generated_images/     # output images saved here (gitignored)
    └── <model>_output_<date>/
        ├── original/
        └── enhanced/
```

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Create a `.env` file

```dotenv
API_KEY=your_api_key_here
BASE_URL=https://api.metisai.ir
```

### 3. Edit `config.yaml`

Key settings to review:

```yaml
cloud_api:
  model:
    name: "google"        # provider / model family
    model: "nano-banana"  # specific model identifier

generation:
  images_per_prompt: 2   # images generated per prompt row
  workers: 4             # parallel threads

csv:
  file: "inputs/simple.csv"           # prompt CSV
  original_prompt_col: "generic_prompt"
  enhanced_prompt_col: "enhanced_prompts"
  enhanced_is_json: true              # true if enhanced column is a JSON list

passes: "original,enhanced"          # which passes to run
```

### 4. Run

```bash
# Run both passes (reads config.yaml)
python generate_only.py

# Override the CSV at runtime
python generate_only.py --csv inputs/my_prompts.csv

# Run only the original pass
python generate_only.py --passes original
```

---

## CSV format

| Column | Description |
|--------|-------------|
| `generic_prompt` | Plain text prompt for the *original* pass |
| `enhanced_prompts` | JSON list of strings **or** plain string for the *enhanced* pass |

Example row:

```csv
generic_prompt,enhanced_prompts
"A doctor examining a patient","[\"A female doctor examining a patient\",\"An elderly doctor examining a patient\"]"
```

---

## Outputs

All images are saved under:

```
generated_images/<model>_output_<YYYYMMDD>/
    original/   ← images from the original-pass prompts
    enhanced/   ← images from the enhanced-pass prompts
```

A `metadata.json` file is written alongside each pass with generation details (model, prompts, timestamps, cost).

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
| `generation.output_dir` | `generated_images` | Root output folder |
| `csv.file` | `inputs/simple.csv` | Prompt CSV path |
| `csv.num_rows` | `null` | Limit rows processed (`null` = all) |
| `csv.seed` | `null` | RNG seed for reproducible sampling |
| `passes` | `original,enhanced` | Which passes to run |

---

## Logs

Runtime logs are written to `logs/` in the working directory. See `errors.log` and `evaluation.log`.
