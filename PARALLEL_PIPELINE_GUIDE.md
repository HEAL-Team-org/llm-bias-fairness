# Parallel Pipeline Execution Guide

## Overview

`run_pipeline_parallel.py` is a high-performance parallelized version of the bias-aware image generation pipeline that can process large datasets efficiently using multiple worker processes.

## Features

- **Parallel Processing**: Uses 32 workers by default (configurable)
- **Dual Image Generation**: Creates both original and enhanced images for each prompt
- **Progress Tracking**: Real-time progress updates with ETA
- **Automatic Scaling**: Efficiently distributes work across CPU cores
- **Fault Tolerance**: Individual prompt failures don't stop the entire pipeline

## Usage

### Basic Usage

```bash
# Using proxychains (required for OpenAI API access from restricted regions)
proxychains4 python3 run_pipeline_parallel.py sample_prompts.csv
```

### With Custom Workers

```bash
# Use 16 workers instead of default 32
proxychains4 python3 run_pipeline_parallel.py sample_prompts.csv --workers 16

# Use 64 workers for maximum parallelization
proxychains4 python3 run_pipeline_parallel.py sample_prompts.csv --workers 64
```

### Command-Line Arguments

- `csv_file`: Path to CSV file with prompts (default: `sample_prompts.csv`)
- `--workers N`: Number of parallel workers (default: 32)
- `--output-dir DIR`: Output directory (default: `results`)

## Performance Comparison

### Sequential Pipeline (`run_complete_pipeline.py`)
- **Speed**: 1 prompt at a time
- **Time for 10 prompts**: ~5-10 minutes
- **Time for 1000 prompts**: ~8-16 hours
- **Best for**: Small datasets, testing, debugging

### Parallel Pipeline (`run_pipeline_parallel.py`)
- **Speed**: 32 prompts simultaneously (configurable)
- **Time for 10 prompts**: ~1-2 minutes
- **Time for 1000 prompts**: ~30-60 minutes
- **Best for**: Large datasets, production runs

## How It Works

1. **Main Process**:
   - Loads all prompts from CSV
   - Creates worker pool (default: 32 workers)
   - Distributes prompts across workers
   - Collects and aggregates results

2. **Each Worker Process**:
   - Initializes own RAG systems (GraphRAG, StereoSet, DiversityRAG)
   - Processes assigned prompts independently:
     * Generate original image
     * Enhance prompt using RAG systems
     * Generate enhanced image
   - Saves intermediate results

3. **Final Stage**:
   - Aggregates all results
   - Generates summary statistics
   - Saves complete results JSON

## Output Structure

```
results/
├── generated_images/
│   ├── sample_001_original.png
│   ├── sample_001_enhanced.png
│   ├── sample_002_original.png
│   ├── sample_002_enhanced.png
│   └── ... (2 images per prompt)
├── intermediate_result_001.json
├── intermediate_result_002.json
├── ...
└── complete_pipeline_results.json
```

## Example Output

```
======================================================================
PARALLEL PIPELINE TEST
======================================================================
Workers: 32
Input CSV: sample_prompts.csv
Output directory: results

Loading prompts from: sample_prompts.csv
Loaded 1000 prompts

======================================================================
PROCESSING 1000 PROMPTS WITH 32 WORKERS
======================================================================

Progress: 32/1000 (3.2%) | Rate: 1.2 prompts/sec | ETA: 13.4 min
Progress: 64/1000 (6.4%) | Rate: 1.3 prompts/sec | ETA: 12.0 min
...
Progress: 1000/1000 (100.0%) | Rate: 1.4 prompts/sec | ETA: 0.0 min

======================================================================
SUMMARY
======================================================================
Total prompts: 1000
Successful enhancements: 998/1000
Successful images (original): 1000/1000
Successful images (enhanced): 998/1000
Total time: 11.90 minutes
Average rate: 1.40 prompts/second
Results directory: results/
Images directory: results/generated_images/

✅ Parallel pipeline test complete!
```

## System Requirements

### Recommended Specifications

- **CPU**: 16+ cores (for 32 workers)
- **RAM**: 64GB+ (each worker loads ~2GB of RAG data)
- **Storage**: 10GB+ free space (for images and cache files)
- **Network**: Stable connection with proxy support

### Resource Calculation

- **Per Worker**: ~2GB RAM
- **32 Workers**: ~64GB RAM
- **16 Workers**: ~32GB RAM
- **8 Workers**: ~16GB RAM

## Optimizing Worker Count

Choose worker count based on:

1. **CPU Cores**: `workers ≈ number of CPU cores`
2. **Available RAM**: `workers ≤ RAM_GB / 2`
3. **API Rate Limits**: Consider OpenAI API rate limits
4. **Network Bandwidth**: More workers = more concurrent API calls

### Example Configurations

```bash
# Low-end system (8 cores, 16GB RAM)
python3 run_pipeline_parallel.py data.csv --workers 8

# Mid-range system (16 cores, 32GB RAM)
python3 run_pipeline_parallel.py data.csv --workers 16

# High-end system (32+ cores, 64GB+ RAM)
python3 run_pipeline_parallel.py data.csv --workers 32

# Maximum parallelization (64+ cores, 128GB+ RAM)
python3 run_pipeline_parallel.py data.csv --workers 64
```

## Troubleshooting

### Memory Issues

If you see "Out of Memory" errors:
```bash
# Reduce workers
python3 run_pipeline_parallel.py data.csv --workers 8
```

### API Rate Limit Issues

If you hit OpenAI rate limits:
```bash
# Reduce workers to slow down requests
python3 run_pipeline_parallel.py data.csv --workers 4
```

### Worker Initialization Failures

If workers fail to initialize:
- Check that all cache files exist
- Ensure sufficient disk space
- Verify RAG data files are accessible

## Comparison with Sequential Pipeline

| Feature | Sequential | Parallel |
|---------|-----------|----------|
| Speed | 1x | 32x (default) |
| Resource Usage | Low | High |
| Scalability | Poor | Excellent |
| Progress Tracking | Basic | Real-time with ETA |
| Fault Tolerance | Stops on error | Continues on error |
| Best Use Case | Testing | Production |

## Notes

- Each worker maintains its own copy of RAG systems in memory
- Cache files are shared (read-only) across all workers
- Image generation is I/O bound, so 32+ workers is beneficial
- Progress is logged in real-time with rate and ETA calculations
- Intermediate results are saved immediately after each prompt completes

---

**Created**: October 19, 2025
**Pipeline Version**: Parallel execution with 32 workers
**Supports**: Large-scale bias-aware image generation
