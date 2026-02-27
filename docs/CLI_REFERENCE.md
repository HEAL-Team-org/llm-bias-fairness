# Command-Line Interface Reference

**Complete reference for the LLM Bias & Fairness unified CLI**

## Table of Contents

1. [Overview](#overview)
2. [Global Options](#global-options)
3. [Enhance Command](#enhance-command)
4. [Batch Command](#batch-command)
5. [Query Command](#query-command)
6. [Test Command](#test-command)
7. [Quick Reference](#quick-reference)
8. [Migration Guide](#migration-guide)

---

## Overview

All project functionality is accessible through a single unified command-line interface:

```bash
python main.py <command> [options]
```

### Available Commands

| Command | Purpose | Common Use |
|---------|---------|------------|
| `enhance` | Enhance single prompts | Interactive or direct prompt improvement |
| `batch` | Process CSV files | Bulk processing with image generation |
| `query` | Query knowledge graphs | Research bias patterns and cultural values |
| `test` | Run system tests | Quick tests, batch tests, visual evaluation |

### Getting Help

```bash
# Show all commands
python main.py --help

# Show command-specific help
python main.py enhance --help
python main.py batch --help
python main.py query --help
python main.py test --help
```

---

## Global Options

Options that work with all commands:

### `--verbose` / `-v`

Enable detailed logging output for debugging.

```bash
python main.py --verbose enhance "a doctor"
python main.py -v batch prompts.csv
```

**Output Level:**
- Without flag: INFO level (standard output)
- With flag: DEBUG level (detailed diagnostics)

---

## Enhance Command

Enhance a single prompt for bias mitigation and diversity.

### Syntax

```bash
python main.py enhance [PROMPT] [OPTIONS]
```

### Arguments

#### Positional Arguments

**`PROMPT`** (optional)
- Type: String
- Description: Image prompt to enhance
- If omitted: Interactive mode (prompts user for input)

```bash
# Direct enhancement
python main.py enhance "a doctor examining a patient"

# Interactive mode
python main.py enhance
```

### Options

#### Enhancement Mode

**`--mode {basic,sequential,dual-pipeline}`**
- Default: `dual-pipeline`
- Description: Enhancement algorithm to use

| Mode | Speed | Quality | Iterations | Best For |
|------|-------|---------|------------|----------|
| `basic` | Fast | Good | 1 | Quick improvements |
| `sequential` | Medium | Very Good | 1-5 | Diversity-focused |
| `dual-pipeline` | Slower | Excellent | 1-5 | Production quality ⭐ |

```bash
python main.py enhance "prompt" --mode basic
python main.py enhance "prompt" --mode sequential
python main.py enhance "prompt" --mode dual-pipeline
```

#### Basic Mode Options

**`--bias-top-k INTEGER`**
- Default: `25`
- Range: 1-100
- Description: Number of bias-related knowledge triples to retrieve

```bash
python main.py enhance "prompt" --mode basic --bias-top-k 30
```

**`--cultural-top-k INTEGER`**
- Default: `25`
- Range: 1-100
- Description: Number of cultural knowledge triples to retrieve

```bash
python main.py enhance "prompt" --mode basic --cultural-top-k 30
```

#### Sequential Mode Options

**`--threshold INTEGER`**
- Default: `75`
- Range: 0-100
- Description: Minimum diversity score required (out of 100)

```bash
python main.py enhance "prompt" --mode sequential --threshold 85
```

**`--max-iterations INTEGER`**
- Default: `3`
- Range: 1-10
- Description: Maximum enhancement iterations

```bash
python main.py enhance "prompt" --mode sequential --max-iterations 5
```

#### Dual-Pipeline Mode Options

**`--bias-threshold INTEGER`**
- Default: `75`
- Range: 0-100
- Description: Minimum bias mitigation score required (out of 100)

```bash
python main.py enhance "prompt" --bias-threshold 80
```

**`--diversity-threshold INTEGER`**
- Default: `80`
- Range: 0-100
- Description: Minimum diversity enhancement score required (out of 100)

```bash
python main.py enhance "prompt" --diversity-threshold 85
```

**`--max-iterations INTEGER`**
- Default: `3`
- Range: 1-10
- Description: Maximum enhancement cycles

```bash
python main.py enhance "prompt" --max-iterations 5
```

**`--stereoset-top-k INTEGER`**
- Default: `10`
- Range: 1-50
- Description: Number of StereoSet bias examples to retrieve

```bash
python main.py enhance "prompt" --stereoset-top-k 15
```

**`--diversity-top-k INTEGER`**
- Default: `10`
- Range: 1-50
- Description: Number of CultureBank diversity examples to retrieve

```bash
python main.py enhance "prompt" --diversity-top-k 15
```

**`--graph-top-k INTEGER`**
- Default: `8`
- Range: 1-30
- Description: Number of GraphRAG cultural triples to retrieve

```bash
python main.py enhance "prompt" --graph-top-k 10
```

#### Knowledge Source Toggles

**`--use-stereoset`**
- Default: `True`
- Description: Enable StereoSet RAG for bias detection

**`--use-diversity-rag`**
- Default: `True`
- Description: Enable CultureBank RAG for diversity enhancement

**`--use-graphrag`**
- Default: `True`
- Description: Enable GraphRAG for cultural knowledge

```bash
# Disable specific sources (not recommended)
python main.py enhance "prompt" --no-use-stereoset
python main.py enhance "prompt" --no-use-graphrag
```

#### Cache Options

**`--cache-file FILENAME`**
- Default: `embeddings.pkl`
- Description: GraphRAG embedding cache file

**`--stereoset-cache FILENAME`**
- Default: `stereoset_embeddings.pkl`
- Description: StereoSet cache file

**`--diversity-cache FILENAME`**
- Default: `diversity_embeddings.pkl`
- Description: CultureBank cache file

```bash
python main.py enhance "prompt" --cache-file custom_cache.pkl
```

### Examples

```bash
# Interactive enhancement (easiest)
python main.py enhance

# Quick enhancement (fast, single-pass)
python main.py enhance "a teacher" --mode basic

# Diversity-focused (iterative)
python main.py enhance "engineers" --mode sequential --threshold 85

# Comprehensive quality (recommended)
python main.py enhance "a doctor" --bias-threshold 80 --diversity-threshold 85

# Strict quality control
python main.py enhance "business meeting" \
    --bias-threshold 90 \
    --diversity-threshold 90 \
    --max-iterations 5

# Custom knowledge retrieval
python main.py enhance "medical professionals" \
    --stereoset-top-k 20 \
    --diversity-top-k 15 \
    --graph-top-k 10
```

---

## Batch Command

Process multiple prompts from a CSV file with automatic image generation.

### Syntax

```bash
python main.py batch INPUT_CSV [OPTIONS]
```

### Arguments

#### Positional Arguments

**`INPUT_CSV`** (required)
- Type: File path
- Description: Path to input CSV file containing prompts

```bash
python main.py batch prompts.csv
python main.py batch data/marketing_prompts.csv
```

### Options

#### CSV Configuration

**`--prompt-column TEXT`**
- Default: `"prompt"`
- Description: Name of the CSV column containing prompts

```bash
python main.py batch data.csv --prompt-column "description"
```

#### Output Configuration

**`--output-dir DIRECTORY`**
- Default: `batch_results`
- Description: Directory for batch processing results

```bash
python main.py batch prompts.csv --output-dir my_results
```

**`--image-dir DIRECTORY`**
- Default: `generated_images`
- Description: Directory for generated images

```bash
python main.py batch prompts.csv --image-dir my_images
```

**`--output-csv FILEPATH`**
- Default: Auto-generated with timestamp
- Description: Custom path for output CSV file

```bash
python main.py batch prompts.csv --output-csv results.csv
```

#### Processing Control

**`--start-row INTEGER`**
- Default: `0`
- Description: Row number to start processing from (0-indexed)

```bash
# Skip first 10 rows
python main.py batch prompts.csv --start-row 10
```

**`--max-rows INTEGER`**
- Default: All rows
- Description: Maximum number of rows to process

```bash
# Process only first 5 rows
python main.py batch prompts.csv --max-rows 5

# Process rows 10-20
python main.py batch prompts.csv --start-row 10 --max-rows 10
```

#### Image Generation

**`--image-generator {dalle3,mock}`**
- Default: `dalle3`
- Description: Image generator type

| Generator | Cost | Speed | Quality | Use Case |
|-----------|------|-------|---------|----------|
| `dalle3` | $0.04-0.08/image | Real API | Real images | Production |
| `mock` | Free | Very fast | Placeholders | Testing |

```bash
# Production with real images
python main.py batch prompts.csv --image-generator dalle3

# Testing with mock images (free, fast)
python main.py batch prompts.csv --image-generator mock
```

#### Visual Bias Evaluation

**`--enable-visual-evaluation`**
- Default: `False`
- Description: Enable demographic analysis of generated images

```bash
python main.py batch prompts.csv --enable-visual-evaluation
```

**`--fairface-model FILEPATH`**
- Default: `None` (uses mock model)
- Description: Path to FairFace ResNet34 model file

```bash
python main.py batch prompts.csv \
    --enable-visual-evaluation \
    --fairface-model models/fairface.pt
```

**`--dlib-model FILEPATH`**
- Default: `None` (uses mock model)
- Description: Path to dlib shape predictor model

```bash
python main.py batch prompts.csv \
    --enable-visual-evaluation \
    --dlib-model models/shape_predictor.dat
```

#### Advanced Options

**`--no-save-intermediate`**
- Default: `False`
- Description: Don't save intermediate results during processing

```bash
python main.py batch prompts.csv --no-save-intermediate
```

### Examples

```bash
# Basic batch processing
python main.py batch prompts.csv

# Test with mock images (no cost)
python main.py batch prompts.csv --image-generator mock

# Process specific range
python main.py batch prompts.csv --start-row 10 --max-rows 20

# Custom output locations
python main.py batch prompts.csv \
    --output-dir campaign_results \
    --image-dir campaign_images

# With visual evaluation
python main.py batch prompts.csv \
    --enable-visual-evaluation \
    --fairface-model models/fairface.pt \
    --dlib-model models/dlib.dat

# Custom column name
python main.py batch data.csv --prompt-column "description"

# Complete example
python main.py batch marketing_prompts.csv \
    --prompt-column "text" \
    --output-dir results/marketing \
    --image-dir images/marketing \
    --max-rows 50 \
    --image-generator dalle3 \
    --enable-visual-evaluation
```

---

## Query Command

Query knowledge graphs for bias patterns and cultural information.

### Syntax

```bash
python main.py query [QUESTION] [OPTIONS]
```

### Arguments

#### Positional Arguments

**`QUESTION`** (optional)
- Type: String
- Description: Question to ask the knowledge graph
- If omitted: Interactive mode (prompts user for input)

```bash
# Direct query
python main.py query "What stereotypes exist about doctors?"

# Interactive mode
python main.py query
```

### Options

**`--top-k INTEGER`**
- Default: `8`
- Range: 1-50
- Description: Number of top knowledge triples to retrieve

```bash
python main.py query "Cultural practices in Asia" --top-k 15
```

**`--cache-file FILENAME`**
- Default: `embeddings.pkl`
- Description: Embedding cache file path

```bash
python main.py query "stereotypes" --cache-file custom_cache.pkl
```

### What Gets Queried

1. **Bias Data** (ADV_GRAPH)
   - 4,229 stereotype examples
   - Bias patterns and harmful assumptions

2. **Cultural Values** (15+ regions)
   - Cultural practices and traditions
   - Regional behaviors and values
   - Social norms and customs

3. **GPT-4 Synthesis**
   - Retrieved triples are synthesized into coherent answers
   - Context-aware responses with examples

### Examples

```bash
# Research stereotypes
python main.py query "What stereotypes exist about teachers?"

# Explore cultural practices
python main.py query "What are traditional greetings in different cultures?" --top-k 12

# Gender bias research
python main.py query "How are women stereotyped in tech?"

# Age-related biases
python main.py query "What age stereotypes exist in workplace?"

# Comprehensive research
python main.py query "Cultural values in East Asia" --top-k 20

# Interactive research session
python main.py query
```

---

## Test Command

Run various tests to verify system functionality.

### Syntax

```bash
python main.py test [OPTIONS]
```

### Options (Test Modes)

At least one test mode must be specified:

**`--quick`**
- Description: Quick test (1 prompt, DALL-E 3, ~30 seconds)
- Use case: Fast system verification

```bash
python main.py test --quick
```

**`--batch`**
- Description: Batch processing test (3 prompts, mock images)
- Use case: Verify batch processing pipeline

```bash
python main.py test --batch
```

**`--visual`**
- Description: Visual bias evaluation test (3 prompts with demographics)
- Use case: Test visual evaluation system

```bash
python main.py test --visual
```

**`--all`**
- Description: Run all tests sequentially
- Use case: Comprehensive system verification

```bash
python main.py test --all
```

### Test Outputs

Tests create temporary directories:
- `quick_test_results/` - Quick test results
- `test_results/` - Batch test results
- `visual_bias_test_results/` - Visual evaluation results

### Examples

```bash
# Quick system check (~30 seconds)
python main.py test --quick

# Test batch processing
python main.py test --batch

# Test visual evaluation
python main.py test --visual

# Run everything
python main.py test --all

# Multiple test modes
python main.py test --quick --batch
```

---

## Quick Reference

### Common Workflows

```bash
# 1. Interactive enhancement (best for beginners)
python main.py enhance

# 2. Quick enhancement
python main.py enhance "your prompt here"

# 3. High-quality enhancement
python main.py enhance "your prompt" \
    --bias-threshold 85 \
    --diversity-threshold 90

# 4. Batch processing (test first)
python main.py batch prompts.csv --image-generator mock --max-rows 3
python main.py batch prompts.csv  # then run for real

# 5. Research before enhancing
python main.py query "stereotypes about X"

# 6. Run quick test
python main.py test --quick
```

### Parameter Cheat Sheet

| Use Case | Command |
|----------|---------|
| Fast & simple | `--mode basic` |
| Diversity-focused | `--mode sequential --threshold 85` |
| Best quality | `--bias-threshold 80 --diversity-threshold 85` |
| Strict quality | `--bias-threshold 90 --diversity-threshold 90 --max-iterations 5` |
| Test batch | `--image-generator mock --max-rows 5` |
| Production batch | `--image-generator dalle3` |
| Visual analysis | `--enable-visual-evaluation` |
| More knowledge | `--stereoset-top-k 20 --diversity-top-k 15` |

---

## Migration Guide

### From Old Scripts to Unified CLI

| Old Command | New Command |
|-------------|-------------|
| `python enhance_prompt.py -p "X"` | `python main.py enhance "X" --mode basic` |
| `python enhance_prompt_sequential.py --prompt "X"` | `python main.py enhance "X" --mode sequential` |
| `python enhance_prompt_dual_pipeline.py -p "X"` | `python main.py enhance "X"` |
| `python batch_processor.py file.csv` | `python main.py batch file.csv` |
| `python main.py --query "X"` (old) | `python main.py query "X"` |
| `python quick_test.py` | `python main.py test --quick` |
| `python test_batch_pipeline.py` | `python main.py test --batch` |
| `python test_visual_bias_evaluation.py` | `python main.py test --visual` |

### Key Changes

1. **Single entry point:** All commands through `main.py`
2. **Subcommand structure:** `python main.py <command>`
3. **Consistent naming:** Arguments are uniform across commands
4. **Interactive mode:** All commands support interactive input
5. **Integrated help:** `--help` available for every command

### Backward Compatibility

Old scripts remain in `/scripts` directory for reference:

```bash
# Still works if needed
python scripts/enhance_prompt.py -p "a doctor"
python scripts/batch_processor.py prompts.csv
```

---

## Additional Resources

- **User Guide:** `docs/USER_GUIDE.md` - Complete usage guide with examples
- **Architecture:** `docs/ARCHITECTURE.md` - System design and technical details
- **Visual Evaluation:** `docs/VISUAL_BIAS_EVALUATION.md` - Demographic analysis guide
- **Migration Guide:** `scripts/README.md` - Detailed migration from old scripts
- **Changelog:** `CHANGELOG.md` - Version history and updates

---

**Version:** 3.1.0  
**Last Updated:** October 1, 2025  
**Status:** Production Ready ✅
