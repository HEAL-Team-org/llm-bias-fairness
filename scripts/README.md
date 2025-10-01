# Legacy Scripts Directory

This directory contains the original individual scripts that have been consolidated into the unified `main.py` in the parent directory.

## Why These Files Are Here

These scripts are kept for:
- **Backward compatibility** - If you have existing workflows using these scripts
- **Reference** - Understanding the evolution of the project
- **Backup** - Safety net during transition to unified CLI

## Migration to New main.py

All functionality from these scripts is now available through the unified `main.py`:

### Script Migration Guide

| Old Script | New Command | Notes |
|------------|-------------|-------|
| `main.py` (old) | `python main.py query "question"` | Knowledge graph queries |
| `enhance_prompt.py` | `python main.py enhance "prompt" --mode basic` | Basic single-pass enhancement |
| `enhance_prompt_sequential.py` | `python main.py enhance "prompt" --mode sequential` | Sequential with diversity scoring |
| `enhance_prompt_with_stereoset.py` | `python main.py enhance "prompt" --mode sequential` | Use sequential mode |
| `enhance_prompt_dual_pipeline.py` | `python main.py enhance "prompt" --mode dual-pipeline` | Default mode (RECOMMENDED) |
| `batch_processor.py` | `python main.py batch input.csv` | Batch processing (also available as module) |
| `test_batch_pipeline.py` | `python main.py test --batch` | Batch processing test |
| `test_visual_bias_evaluation.py` | `python main.py test --visual` | Visual evaluation test |
| `quick_test.py` | `python main.py test --quick` | Quick demo test |
| `run_pipeline.py` | `python main.py batch input.csv` | Simple batch processing |
| `enhance_prompt_fixed.py` | `python main.py enhance "prompt" --mode basic` | Duplicate of enhance_prompt.py |

### Quick Examples

**Old Way:**
```bash
python enhance_prompt_dual_pipeline.py -p "a doctor" --bias-threshold 75
```

**New Way:**
```bash
python main.py enhance "a doctor" --bias-threshold 75
```

---

**Old Way:**
```bash
python test_batch_pipeline.py
```

**New Way:**
```bash
python main.py test --batch
```

---

**Old Way:**
```bash
python run_pipeline.py prompts.csv --real-images
```

**New Way:**
```bash
python main.py batch prompts.csv --image-generator dalle3
```

## Important Notes

### Still-Used Modules

Two files are **NOT** in this directory because they're still actively used as importable modules:

- **`batch_processor.py`** - Core batch processing module (imported by main.py)
- **`enhance_prompt_dual_pipeline.py`** - Dual-pipeline enhancement logic (imported by main.py)

These remain in the project root as they provide essential functionality.

### If You Need the Old Scripts

If you have existing workflows that depend on these old scripts:

1. **Update to new CLI** - Recommended for consistency
2. **Use from scripts/ directory** - Run with full path:
   ```bash
   python scripts/enhance_prompt.py -p "a doctor"
   ```
3. **Copy back to root** - Temporary solution (not recommended)

## Deprecation Timeline

- **Current** (v3.0): Old scripts moved to `scripts/` directory
- **v3.1-v3.5**: Maintained for compatibility
- **v4.0+**: May be removed entirely

We recommend updating to the new unified CLI as soon as possible.

## Getting Help

For the new unified interface:
```bash
python main.py --help              # General help
python main.py enhance --help      # Enhancement help
python main.py batch --help        # Batch processing help
python main.py query --help        # Query help
python main.py test --help         # Test help
```

For detailed documentation:
- See `docs/USER_GUIDE.md` for complete usage guide
- See `docs/ARCHITECTURE.md` for technical details
- See `README.md` for quick start

## Questions?

If you encounter issues or need help migrating:
1. Check the USER_GUIDE.md for detailed examples
2. Use `python main.py --help` for command-specific help
3. Review this migration guide for old → new command mapping
