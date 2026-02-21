# Precised Image Generation - Quick Start

## Overview

I've created a complete system to generate images for the precised prompts in `prompts_precised.csv`. The system reads demographic variants (white/Black/Asian/Hispanic × male/female) and generates images using DALL-E 3.

## Files Created

1. **generate_precised_images.py** - Main generation script
2. **test_precised_generation.py** - Quick test with 2 rows
3. **run_precised_generation.sh** - Interactive menu for easy usage
4. **PRECISED_GENERATION_README.md** - Comprehensive documentation

## Quick Start

### 1. Setup Environment

```bash
export OPENAI_API_KEY="your-key-here"
# Or for proxychains:
export OPENAI_BASE_URL="http://127.0.0.1:8000/v1"
```

### 2. Test with 2 Rows

```bash
python test_precised_generation.py
```

This generates 10 images (2 rows × 5 variants) in `test_precised_images/`.

### 3. Run Default Batch (100 rows)

```bash
python generate_precised_images.py
```

This generates 500 images (100 rows × 5 variants) in `precised_generated_images/`.

### 4. Interactive Menu

```bash
./run_precised_generation.sh
```

Choose from preset configurations (10/50/100/500/1000 rows).

## Command Line Options

```bash
python generate_precised_images.py \
  --csv prompts_precised.csv \     # Input CSV file
  --num-rows 100 \                 # Number of rows to process
  --output-dir precised_images \   # Output directory
  --workers 4 \                    # Parallel workers (1-10)
  --sequential                     # Sequential mode (for testing)
```

## CSV Structure

Your `prompts_precised.csv` has:
- **1000 data rows** (1001 including header)
- **5 precised variants per row** (different demographics)
- **Total possible images: 5000** (1000 rows × 5 variants)

Example row structure:
```json
{
  "prompt": "Generate an image of doctors...",
  "__row_id__": 6,
  "generic_prompt": "Generate an image of doctors...",
  "precised_prompts": [
    "Generate an image of white female doctors...",
    "Generate an image of Black female doctors...",
    "Generate an image of Asian female doctors...",
    "Generate an image of Hispanic female doctors...",
    "Generate an image of white male doctors..."
  ]
}
```

## Output Structure

```
precised_generated_images/
├── row_000/
│   ├── row_000_variant_00.png  # white female
│   ├── row_000_variant_01.png  # Black female
│   ├── row_000_variant_02.png  # Asian female
│   ├── row_000_variant_03.png  # Hispanic female
│   └── row_000_variant_04.png  # white male
├── row_001/
│   └── ...
└── generation_results.json      # Metadata & statistics
```

## Features

✅ **Parallel Processing** - Use multiple workers for speed  
✅ **Sequential Mode** - For testing and debugging  
✅ **Progress Tracking** - Real-time logs and statistics  
✅ **Error Handling** - Continue on failures, log errors  
✅ **Metadata** - JSON file with all generation details  
✅ **Organized Output** - Images grouped by row  
✅ **Rate Limiting** - Built-in delays to avoid API limits  

## Performance

- **Sequential**: ~6-8 seconds per image
- **Parallel (4 workers)**: ~2-3 seconds per image
- **100 rows (500 images)**: ~15-20 minutes with 4 workers
- **1000 rows (5000 images)**: ~2-3 hours with 8 workers

## Cost Estimates

DALL-E 3 standard quality (1024x1024): **$0.040 per image**

| Rows | Images | Cost | Time (4 workers) |
|------|--------|------|------------------|
| 10   | 50     | $2   | ~2 minutes       |
| 100  | 500    | $20  | ~20 minutes      |
| 500  | 2500   | $100 | ~2 hours         |
| 1000 | 5000   | $200 | ~4 hours         |

## Recommended Workflow

```bash
# Step 1: Test with 2 rows
python test_precised_generation.py

# Step 2: Verify test results
ls test_precised_images/row_000/

# Step 3: Small production run (10 rows)
python generate_precised_images.py --num-rows 10

# Step 4: Check results
cat precised_generated_images/generation_results.json | grep '"status"'

# Step 5: Full production (100 rows default)
python generate_precised_images.py

# Step 6: Large batch if needed
python generate_precised_images.py --num-rows 500 --workers 8
```

## Examples

### Test Run (2 rows, 10 images)
```bash
python test_precised_generation.py
```

### Small Batch (10 rows, 50 images, ~$2)
```bash
python generate_precised_images.py --num-rows 10 --sequential
```

### Default (100 rows, 500 images, ~$20)
```bash
python generate_precised_images.py
```

### Large Batch (500 rows, 2500 images, ~$100)
```bash
python generate_precised_images.py --num-rows 500 --workers 8
```

### Full Dataset (1000 rows, 5000 images, ~$200)
```bash
python generate_precised_images.py --num-rows 1000 --workers 8
```

## Monitoring Progress

The script logs:
- Each image generation attempt
- Success/failure status
- Row and variant being processed
- Final statistics summary

Example output:
```
2024-01-15 10:30:45 - INFO - Processing Row 0 (ID: 6) - 5 variants
2024-01-15 10:30:51 - INFO - ✓ Image saved to: row_000/row_000_variant_00.png
2024-01-15 10:30:58 - INFO - ✓ Image saved to: row_000/row_000_variant_01.png
...
================================================================================
GENERATION SUMMARY
================================================================================
Total images attempted: 500
Successful: 498 (99.6%)
Failed: 2 (0.4%)

Total time: 1245.67 seconds
Average time per image: 2.49 seconds
```

## Troubleshooting

### API Key Issues
```bash
# Check if set
echo $OPENAI_API_KEY

# Set it
export OPENAI_API_KEY="sk-..."
```

### Rate Limiting
If you get rate limit errors, reduce workers:
```bash
python generate_precised_images.py --workers 2
```

### Memory Issues
Process fewer rows at a time:
```bash
python generate_precised_images.py --num-rows 50
```

### Network Issues
Use sequential mode for better error visibility:
```bash
python generate_precised_images.py --sequential
```

## Integration Notes

This script is designed as a lightweight alternative to the full pipeline:

- **deploy_and_run.py**: Full pipeline (enhancement + generation)
- **generate_precised_images.py**: Precised-only (just generation)

Key differences:
- No repository cloning
- No virtual environment creation
- No enhancement step
- Direct CSV reading
- Faster startup
- Lower overhead

Perfect for:
- Pre-generated precised prompts
- Batch image generation
- Testing PreciseDebias methodology
- Demographic analysis

## Next Steps

1. **Test**: Run `python test_precised_generation.py`
2. **Verify**: Check generated images in `test_precised_images/`
3. **Scale**: Run with `--num-rows 100` for production
4. **Analyze**: Use `generation_results.json` for statistics
5. **Expand**: Increase to 500 or 1000 rows as needed

## Support

For detailed documentation, see [PRECISED_GENERATION_README.md](PRECISED_GENERATION_README.md)

For issues:
1. Check logs for error messages
2. Verify API key and base URL
3. Test with sequential mode
4. Review `generation_results.json` for failures
