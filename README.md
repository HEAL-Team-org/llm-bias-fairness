# Bias-Aware Image Prompt Enhancer

A GraphRAG-powered pipeline that rewrites generic image-generation prompts to be more diverse, inclusive, and free of demographic stereotypes. The system combines a bias knowledge graph, a multi-country cultural-values knowledge graph, and a local LLM server to iteratively refine prompts until they meet a configurable diversity threshold.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Starting the LLM Servers](#starting-the-llm-servers)
- [Quick Start](#quick-start)
  - [Single Prompt — Interactive Mode](#single-prompt--interactive-mode)
  - [Single Prompt — Direct Mode](#single-prompt--direct-mode)
  - [Batch Mode](#batch-mode)
- [CLI Reference](#cli-reference)
  - [`enhance_prompt_sequential.py`](#enhance_prompt_sequentialpy)
  - [`run_batch.py`](#run_batchpy)
- [Data Inputs](#data-inputs)
  - [Bias Graph CSV](#bias-graph-csv)
  - [Cultural Values Triples](#cultural-values-triples)
  - [Prompts CSV](#prompts-csv)
- [Output Files](#output-files)
- [Configuration](#configuration)
  - [Embedding Model](#embedding-model)
  - [LLM Model Selection](#llm-model-selection)
  - [Diversity Threshold and Iterations](#diversity-threshold-and-iterations)
- [Evaluation Metrics](#evaluation-metrics)
- [Module Reference](#module-reference)
- [Troubleshooting](#troubleshooting)

---

## Overview

Many text-to-image models reflect the biases present in their training data, producing outputs that over-represent certain demographics. This project addresses that by:

1. **Retrieving relevant bias triples** — facts that encode known stereotypes (e.g., "doctor → always male").
2. **Retrieving cultural-value triples** — facts from 14+ country knowledge graphs to encourage genuine cultural representation.
3. **Rewriting the prompt** with a local LLM instructed to avoid detected stereotypes and include authentic cultural and demographic diversity.
4. **Scoring the result** with a two-stage scorer (keyword lexicon + optional LLM judge) on a 0–100 scale.
5. **Iterating** until the score meets the configured threshold or the maximum number of iterations is reached.

---

## Architecture

```
                 ┌─────────────────────────────────────────────┐
                 │            enhance_prompt_sequential.py      │
                 │                                             │
  input prompt ──►  GraphRAG retrieval (bias + culture)        │
                 │           │                                 │
                 │           ▼                                 │
                 │     LLM rewrite  ◄──── serve_qwen3/35.py   │
                 │       (port 8001 / 8002)                    │
                 │           │                                 │
                 │           ▼                                 │
                 │   Diversity scorer (keyword + LLM judge)    │
                 │           │                                 │
                 │    score ≥ threshold?                       │
                 │    No → iterate  │  Yes → done              │
                 └─────────────────────────────────────────────┘
                                    │
                             enhanced prompt
                             + full history
                             + score JSON
```

### Knowledge Graphs

| Graph | Source file | Size |
|---|---|---|
| Bias / stereotypes | `data/biases/ADV_GRAPH_20240119.csv` | ~thousands of triples |
| Cultural values | `data/cultural_values/*.txt` (14 countries) | ~hundreds per country |

The `GraphRAG` class builds a `networkx` graph from these triples, embeds every node and edge with a `SentenceTransformer` model, and answers queries with cosine-similarity retrieval. Embeddings are persisted to a `.pkl` cache on first run.

---

## Project Structure

```
.
├── enhance_prompt_sequential.py   # Single-prompt enhancement pipeline
├── run_batch.py                   # Batch runner (reads prompts.csv)
├── data/
│   ├── biases/
│   │   └── ADV_GRAPH_20240119 - ADV_GRAPH_20240119.csv   # Stereotype triples
│   ├── cultural_values/
│   │   ├── iran triples.txt
│   │   ├── us triples.txt
│   │   ├── china triples.txt
│   │   └── ...                    # 14 country files total
│   └── prompts/
│       └── prompts.csv            # Batch input prompts
├── helper/
│   ├── logger.py                  # Coloured, file-backed logging utility
│   ├── serve_qwen3.py             # FastAPI server — Qwen3-32B  (port 8001)
│   ├── serve_qwen35.py            # FastAPI server — Qwen3.5-27B (port 8002)
│   └── test.py                    # Manual server smoke test
└── src/
    ├── __init__.py
    ├── graphrag.py                # GraphRAG core: graph build, embed, retrieve
    ├── parsers.py                 # CSV / text triple parsers
    └── evaluation_metrics.py     # DI and CMMD bias-evaluation metrics
```

---

## Requirements

- Python 3.10 or higher
- A CUDA-capable GPU (recommended) — see VRAM notes below
- Internet access for the first run (model weight downloads from Hugging Face)

### Python packages

```
torch
transformers
accelerate
sentence-transformers
networkx
numpy
pandas
requests
fastapi
uvicorn
pydantic
```

Optional (for CMMD evaluation):
```
clip                  # openai/clip-vit-base-patch32
Pillow
```

---

## Installation

```bash
# 1. Clone / copy the project
cd path/to/project

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

# 3. Install core dependencies
pip install torch transformers accelerate sentence-transformers \
            networkx numpy pandas requests fastapi uvicorn pydantic

# 4. (Optional) Install CLIP for full evaluation metrics
pip install git+https://github.com/openai/CLIP.git Pillow
```

---

## Starting the LLM Servers

The enhancement pipeline calls a locally-hosted LLM via HTTP. Two server scripts are provided. Start **at least one** before running any enhancement commands.

### Qwen3-32B — port 8001 (default)

```bash
cd helper
uvicorn serve_qwen3:app --host 0.0.0.0 --port 8001 --workers 1
```

- Requires ~70 GB VRAM (bfloat16) / falls back to CPU float32 (very slow).
- Downloads ~65 GB of weights on first run.

### Qwen3.5-27B — port 8002

```bash
cd helper
uvicorn serve_qwen35:app --host 0.0.0.0 --port 8002 --workers 1
```

- Requires ~55 GB VRAM.

### Health check

```bash
curl http://localhost:8001/health   # should return {"status": "ok"}
curl http://localhost:8002/health
```

> **Note:** Keep `--workers 1`. Each extra worker loads a full copy of the model into GPU memory.

---

## Quick Start

### Single Prompt — Interactive Mode

```bash
python enhance_prompt_sequential.py
```

You will be prompted to type your image-generation prompt, then the pipeline runs and prints the enhanced result.

### Single Prompt — Direct Mode

```bash
python enhance_prompt_sequential.py -p "a doctor in a hospital"
```

```bash
python enhance_prompt_sequential.py \
    --prompt "students studying in a library" \
    --threshold 80 \
    --max-iterations 5 \
    --model qwen3.5
```

### Batch Mode

Place your prompts in `data/prompts/prompts.csv` (see [Prompts CSV](#prompts-csv)) then run:

```bash
python run_batch.py
```

Results are written to `outputs/`.

---

## CLI Reference

### `enhance_prompt_sequential.py`

| Flag | Default | Description |
|---|---|---|
| `-p`, `--prompt` | *(interactive)* | The image-generation prompt to enhance |
| `--threshold` | `75` | Minimum diversity score (0–100) to accept an enhanced prompt |
| `--max-iterations` | `3` | Maximum enhancement iterations before returning the best result |
| `--model` | `qwen3` | LLM backend: `qwen3` (port 8001) or `qwen3.5` (port 8002) |

**Examples**

```bash
# Interactive mode
python enhance_prompt_sequential.py

# Provide prompt directly
python enhance_prompt_sequential.py -p "a nurse taking care of a patient"

# Use Qwen3.5 with a higher threshold and more iterations
python enhance_prompt_sequential.py \
    --prompt "a CEO giving a presentation" \
    --model qwen3.5 \
    --threshold 85 \
    --max-iterations 6
```

---

### `run_batch.py`

| Flag | Default | Description |
|---|---|---|
| `--input` | `data/prompts/prompts.csv` | Path to input CSV |
| `--test` | off | Write outputs to `test/` instead of `outputs/` |
| `--no-llm` | off | Skip LLM judge; use keyword scoring only |
| `--no-enhance` | off | Skip enhancement and score existing `enhanced_prompts` column |
| `--row-ids N [N …]` | *(all rows)* | Process only specified row indices |
| `--threshold` | `75` | Diversity score threshold passed to the enhancer |
| `--max-iterations` | `3` | Max enhancement iterations per prompt |
| `--model` | `qwen3` | LLM backend: `qwen3` or `qwen3.5` |

**Examples**

```bash
# Full batch run
python run_batch.py

# Test run — outputs to test/ folder
python run_batch.py --test

# Only rows 0 and 2, keyword scoring only
python run_batch.py --row-ids 0 2 --no-llm

# Different input file, use Qwen3.5
python run_batch.py --input inputs/my_prompts.csv --model qwen3.5

# Score pre-existing enhanced prompts without re-enhancing
python run_batch.py --no-enhance
```

---

## Data Inputs

### Bias Graph CSV

**Location:** `data/biases/ADV_GRAPH_20240119 - ADV_GRAPH_20240119.csv`

Expected columns:

| Column | Description |
|---|---|
| `targetMinority` | The demographic group targeted by the stereotype |
| `targetStereotype` | Free-text description of the stereotype |
| `Graph` | A triple in the format `` `(subject, predicate, object)` `` |

Example row:
```
black folks,are all well endowed,"Graph: `(black folks, are, well endowed)`"
```

---

### Cultural Values Triples

**Location:** `data/cultural_values/<country> triples.txt`

Available countries: Azerbaijan, China, East Africa, Greece, India, Indonesia, Iran, Mexico, North Africa, North Korea, South Korea, Spain, UK, USA.

Each file contains one RDF-style triple per line, for example:
```
(Iran, values, hospitality)
(Iran, celebrates, Nowruz)
```

---

### Prompts CSV

**Location:** `data/prompts/prompts.csv`

Required column:
- `generic_prompt` — the plain prompt to enhance

Optional columns:
- `__row_id__` — integer row identifier (auto-assigned if missing)
- `enhanced_prompts` — JSON array of pre-existing enhanced prompts (used with `--no-enhance`)

Example:
```csv
generic_prompt,__row_id__
"a doctor smiling at the camera",0
"a group of engineers in a lab",1
```

---

## Output Files

After a batch run the `outputs/` (or `test/`) directory contains:

```
outputs/
├── enhanced/
│   ├── row_0_enhanced.json    # Full enhancement history for row 0
│   ├── row_1_enhanced.json
│   └── ...
├── summary.csv                # Flat scorecard — one row per prompt
└── summary.json               # Complete structured data for all rows
```

### `row_N_enhanced.json` schema

```json
{
  "row_id": 0,
  "generic_prompt": "...",
  "final_enhanced_prompt": "...",
  "final_score": 82,
  "iterations": [
    {
      "iteration": 1,
      "enhanced_prompt": "...",
      "keyword_score": 65,
      "llm_score": 70,
      "combined_score": 67
    }
  ],
  "timestamp": "2026-02-27T12:00:00"
}
```

### `summary.csv` columns

| Column | Description |
|---|---|
| `row_id` | Row identifier |
| `generic_prompt` | Original input prompt |
| `final_enhanced_prompt` | Best enhanced prompt |
| `original_score` | Diversity score of the original prompt |
| `final_score` | Diversity score of the enhanced prompt |
| `score_delta` | `final_score − original_score` |
| `verdict` | `IMPROVED`, `ALREADY_GOOD`, or `NO_IMPROVEMENT` |

---

## Configuration

### Embedding Model

Defined at the top of `src/graphrag.py`:

```python
EMBED_MODEL = "Qwen/Qwen3-Embedding-8B"   # default (~15 GB)
```

Lighter alternatives:
```python
EMBED_MODEL = "BAAI/bge-large-en-v1.5"                 # ~1.3 GB — strong English
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # ~90 MB  — fastest
```

Change this constant before the first run, or delete the `.pkl` embedding cache to rebuild with a new model.

### LLM Model Selection

Pass `--model qwen3` (port 8001, default) or `--model qwen3.5` (port 8002) to either script. The URL mapping is in `enhance_prompt_sequential.py`:

```python
MODEL_URLS: dict[str, str] = {
    "qwen3":   "http://localhost:8001",
    "qwen3.5": "http://localhost:8002",
}
```

Change the host/port here if your servers run on a different machine or port.

### Diversity Threshold and Iterations

| Parameter | Default | Notes |
|---|---|---|
| `--threshold` | `75` | Score 0–100; higher values produce richer prompts at the cost of more LLM calls |
| `--max-iterations` | `3` | Hard cap on enhancement loops per prompt |

---

## Evaluation Metrics

`src/evaluation_metrics.py` implements two fairness metrics for analysing a set of generated images/prompts:

### Disparate Impact (DI)

Measures whether one demographic subgroup appears at a rate below 80 % of another.

$$DI = \frac{\text{rate}_{group_1}}{\text{rate}_{group_2}}$$

A result below 0.8 is flagged as biased.

### Conditional Maximum Mean Discrepancy (CMMD)

Uses CLIP embeddings to measure the distributional distance between two groups of generated images. A smaller distance indicates more equitable representation.

Both metrics are available via the `PromptAnalyzer`, `DisparateImpactCalculator`, and `CMMDCalculator` classes in `src/evaluation_metrics.py`.

---

## Module Reference

| Module | Key classes / functions |
|---|---|
| `enhance_prompt_sequential.py` | `sequential_enhance_prompt()`, `llm_score()`, `load_bias_graph()`, `load_cultural_graphs()`, `retrieve_relevant_triples()`, `configure_llm()` |
| `run_batch.py` | `load_prompts()`, `run_batch()` |
| `src/graphrag.py` | `GraphRAG`, `EmbeddingCache`, `KnowledgeGraph` |
| `src/parsers.py` | `BiasCSVParser`, `CulturalTriplesParser`, `DataParserFactory`, `Triple` |
| `src/evaluation_metrics.py` | `PromptAnalyzer`, `DisparateImpactCalculator`, `CMMDCalculator`, `DisparateImpactResult`, `CMMDResult` |
| `helper/logger.py` | `get_logger()`, `log_timer()` |
| `helper/serve_qwen3.py` | FastAPI app — `POST /generate`, `GET /health` |
| `helper/serve_qwen35.py` | FastAPI app — `POST /generate`, `GET /health` |

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ConnectionRefusedError` on port 8001/8002 | LLM server not running | Start the server with `uvicorn` (see [Starting the LLM Servers](#starting-the-llm-servers)) |
| `FileNotFoundError: embeddings.pkl` | First run — cache not created yet | Normal; the cache is built automatically on first run |
| Very slow embedding build | Large graphs + no GPU | Switch to a lighter embedding model (see [Embedding Model](#embedding-model)) |
| `CLIP not available` warning | `clip` package not installed | Install with `pip install git+https://github.com/openai/CLIP.git`; CMMD metric falls back to a mock implementation |
| JSON parsing errors in LLM output | Model producing non-standard JSON | The pipeline has a built-in JSON repair heuristic; persistent failures indicate the model is not following the system prompt — try lowering `temperature` in the server script |
| GPU OOM on LLM server start | Not enough VRAM | Use `qwen3.5` (27B, lighter) or reduce `max_new_tokens` in the server script |
| Low diversity scores despite enhancement | Threshold too high for the topic | Lower `--threshold` or increase `--max-iterations` |
