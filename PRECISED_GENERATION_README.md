# Precised Image Generation

Generate images for precised prompts from `prompts_precised.csv` using DALL-E 3.

## Overview

This script reads precised prompts (demographic variants) from a CSV file and generates images for each variant using OpenAI's DALL-E 3 model. Each row in the CSV contains 5 precised prompt variations (e.g., "white female doctor", "Hispanic male doctor", etc.).

## Features

- **Parallel Processing**: Use multiple workers for faster generation
- **Sequential Mode**: Generate images one at a time (useful for testing/debugging)
- **Configurable Rows**: Process any number of rows (default: 100)
- **Organized Output**: Images saved in row-specific subdirectories
- **Metadata Tracking**: JSON file with generation results and statistics
- **Error Handling**: Continue processing even if some images fail

## Requirements

```bash
pip install openai
```

## Environment Setup

Set your OpenAI API credentials:

```bash
export OPENAI_API_KEY="your-api-key-here"
export OPENAI_BASE_URL="https://api.openai.com/v1"  # Optional, defaults to standard API
```

For proxychains:
```bash
export OPENAI_BASE_URL="http://127.0.0.1:8000/v1"
```

## Usage

### Basic Usage (100 rows, parallel)

```bash
python generate_precised_images.py
```

### Custom Number of Rows

```bash
python generate_precised_images.py --num-rows 50
```

### Sequential Processing (for testing)

```bash
python generate_precised_images.py --num-rows 10 --sequential
```

### Full Options

```bash
python generate_precised_images.py \
  --csv prompts_precised.csv \
  --num-rows 100 \
  --output-dir precised_generated_images \
  --workers 4 \
  --sequential  # Optional: process sequentially
```

## Quick Test

Test with just 2 rows:

```bash
python test_precised_generation.py
```

This will generate images for the first 2 rows and save them to `test_precised_images/`.

## Output Structure

```
precised_generated_images/
├── row_000/
│   ├── row_000_variant_00.png
│   ├── row_000_variant_01.png
│   ├── row_000_variant_02.png
│   ├── row_000_variant_03.png
│   └── row_000_variant_04.png
├── row_001/
│   ├── row_001_variant_00.png
│   └── ...
└── generation_results.json
```

## CSV Format

The script expects `prompts_precised.csv` with these columns:
- `prompt`: Original prompt text
- `__row_id__`: Row identifier
- `generic_prompt`: Generic version of prompt
- `precised_variants`: JSON array with demographic details
- `precised_prompts`: JSON array of 5 precised prompt strings

Example:
```csv
prompt,__row_id__,generic_prompt,precised_variants,precised_prompts
"Generate an image of doctors...",0,"Generate an image...","[...]","[""white female doctors..."", ""Hispanic male doctors..."", ...]"
```

## Generation Statistics

After completion, the script displays:
- Total images attempted
- Success rate
- Failed prompts (if any)
- Total execution time
- Average time per image

Example output:
```
================================================================================
GENERATION SUMMARY
================================================================================
Total images attempted: 500
Successful: 498 (99.6%)
Failed: 2 (0.4%)

Total time: 3245.67 seconds
Average time per image: 6.49 seconds
```

## Rate Limiting

The script includes a 1-second delay between image generation requests to avoid rate limiting. For parallel processing, each worker has its own delay.

## Error Handling

- Failed image generations are logged but don't stop the process
- Metadata includes error messages for debugging
- Check `generation_results.json` for detailed failure information

## Parallel vs Sequential

### Parallel (default)
- Faster processing with multiple workers
- Recommended for production runs
- Default: 4 workers

### Sequential
- One image at a time
- Better for debugging
- Easier to track progress
- Use `--sequential` flag

## Troubleshooting

### Rate Limiting Errors
Reduce workers or add longer delays:
```python
# In generate_precised_images.py, increase sleep time
time.sleep(2)  # Increase from 1 to 2 seconds
```

### Memory Issues
Use fewer workers:
```bash
python generate_precised_images.py --workers 2
```

### API Connection Issues
Verify environment variables:
```bash
echo $OPENAI_API_KEY
echo $OPENAI_BASE_URL
```

## Performance Tips

1. **Start Small**: Test with 10 rows first
2. **Monitor Progress**: Watch logs for errors
3. **Check Results**: Review `generation_results.json` for failures
4. **Adjust Workers**: Find optimal worker count for your system
5. **Use Proxychains**: If behind firewall, configure proxy

## Cost Estimation

DALL-E 3 pricing (standard quality, 1024x1024):
- $0.040 per image

For 100 rows × 5 variants = 500 images:
- Estimated cost: $20.00

For full dataset (1000 rows):
- 1000 rows × 5 variants = 5000 images
- Estimated cost: $200.00

## Example Workflow

```bash
# 1. Test with 2 rows
python test_precised_generation.py

# 2. Run small batch (10 rows)
python generate_precised_images.py --num-rows 10

# 3. Run larger batch (100 rows)
python generate_precised_images.py --num-rows 100

# 4. Run full dataset (1000 rows)
python generate_precised_images.py --num-rows 1000 --workers 8
```

## Integration with Existing Pipeline

This script is designed to work alongside the existing `deploy_and_run.py` infrastructure:

- Uses same OpenAI configuration
- Compatible with proxychains setup
- Follows similar output structure
- Can be integrated into automated workflows

## Notes

- Each row generates 5 precised variants (based on PreciseDebias methodology)
- Images are saved as PNG files (1024x1024 resolution)
- DALL-E 3 may apply safety filters and refuse some prompts
- The model may revise prompts automatically (tracked in metadata)
- Generation is not deterministic - same prompt may produce different images

## Advanced Usage

### Process Specific Rows

Modify the script to read specific row ranges:

```python
# In read_precised_prompts(), add offset parameter
for idx, row in enumerate(reader):
    if idx < start_offset:
        continue
    if idx >= start_offset + num_rows:
        break
```

### Custom Image Sizes

Modify the `generate_image()` function:

```python
size="1792x1024"  # Landscape
size="1024x1792"  # Portrait
```

### Higher Quality

Change quality setting:

```python
quality="hd"  # Higher quality, more expensive
```
