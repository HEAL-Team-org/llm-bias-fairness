# User Guide

**LLM Bias & Fairness Project - Complete Usage Guide (v2.0)**

**Last Updated:** October 2025  
**Version:** 2.0 (Refactored Architecture)

## Table of Contents
1. [Quick Start](#quick-start)
2. [System Overview](#system-overview)
3. [CLI Commands](#cli-commands)
4. [Enhancement Guide](#enhancement-guide)
5. [Batch Processing](#batch-processing)
6. [Knowledge Queries](#knowledge-queries)
7. [Testing & Demos](#testing--demos)
8. [Configuration](#configuration)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Installation

```bash
# 1. Clone the repository
git clone <repository-url>
cd llm-bias-fairness

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set OpenAI API key
export OPENAI_API_KEY="sk-your-api-key-here"

# Optional: For visual bias evaluation
# Download FairFace and dlib models (see Visual Bias Evaluation section)
```

### 30-Second Start

```bash
# Enhance a single prompt (simplest)
python main.py enhance "a doctor examining a patient"

# Or interactive mode
python main.py enhance

# Process a CSV file
python main.py batch prompts.csv

# Query knowledge graphs
python main.py query "What stereotypes exist about doctors?"

# Run quick test
python main.py test --quick
```

---

## System Overview

### What This System Does

The LLM Bias & Fairness Project helps you:
1. **Detect Bias**: Identify stereotypical patterns in prompts
2. **Enhance Diversity**: Add inclusive representations across demographics
3. **Generate Images**: Create images with DALL-E 3
4. **Evaluate Fairness**: Measure demographic diversity in generated images
5. **Query Knowledge**: Explore bias and cultural data

### Architecture at a Glance

```
CLI Layer: User-friendly commands
    ↓
Enhancement: Bias mitigation + diversity
    ↓
Knowledge: 3 RAG systems (StereoSet, CultureBank, GraphRAG)
    ↓
Generation: DALL-E 3 image creation
    ↓
Evaluation: Visual bias metrics
```

### Key Features

✅ **Unified CLI**: All functionality through one command interface  
✅ **Multi-source Knowledge**: 78K+ data points from 3 knowledge systems  
✅ **Dual Scoring**: Separate bias and diversity metrics  
✅ **Iterative Enhancement**: Automatic refinement until thresholds met  
✅ **Visual Evaluation**: Demographic analysis of generated images  
✅ **Batch Processing**: Handle hundreds of prompts efficiently  

---

## CLI Commands

All functionality is accessed through the main CLI:

```bash
python main.py <command> [options]
```

### Command Overview

| Command | Purpose | Example |
|---------|---------|---------|
| `enhance` | Single prompt enhancement | `python main.py enhance "a doctor"` |
| `batch` | CSV batch processing | `python main.py batch prompts.csv` |
| `query` | Knowledge graph queries | `python main.py query "stereotypes"` |
| `test` | Run tests and demos | `python main.py test --quick` |

### Global Options

```bash
--verbose, -v    Enable verbose logging (DEBUG level)
--help, -h       Show help message
```

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

## Enhancement Guide

### The `enhance` Command

Enhance single prompts for bias mitigation and diversity.

```bash
python main.py enhance [PROMPT] [OPTIONS]
```

### Basic Usage

```bash
# Interactive mode (recommended for first use)
python main.py enhance

# Direct enhancement
python main.py enhance "a doctor examining a patient"

# With custom thresholds
python main.py enhance "students in classroom" \
    --bias-threshold 80 \
    --diversity-threshold 85

# Enable all RAG systems
python main.py enhance "engineers working" \
    --use-stereoset \
    --use-diversity-rag \
    --use-graphrag
```

### Enhancement Options

#### Thresholds

```bash
--bias-threshold INT          Bias mitigation threshold (0-100, default: 75)
--diversity-threshold INT     Diversity threshold (0-100, default: 80)
--max-iterations INT          Maximum enhancement iterations (default: 5)
```

**Recommendations**:
- **Strict**: `--bias-threshold 85 --diversity-threshold 90`
- **Balanced**: `--bias-threshold 75 --diversity-threshold 80` (default)
- **Lenient**: `--bias-threshold 65 --diversity-threshold 70`

#### RAG System Control

```bash
--use-stereoset              Enable StereoSet RAG (stereotype detection)
--use-diversity-rag          Enable Diversity RAG (cultural behaviors)
--use-graphrag               Enable GraphRAG (knowledge graph)
```

**When to Use Each**:
- **StereoSet**: Always recommended for bias detection
- **Diversity RAG**: When you want cultural diversity
- **GraphRAG**: For deeper knowledge integration

### How Enhancement Works

```
1. Parse your prompt
   ↓
2. Retrieve negative examples from StereoSet (if enabled)
   - Identifies stereotypical patterns to avoid
   ↓
3. Retrieve diversity recommendations from CultureBank (if enabled)
   - Suggests inclusive cultural behaviors
   ↓
4. Retrieve knowledge from GraphRAG (if enabled)
   - Provides bias and cultural context
   ↓
5. Generate enhanced prompt with GPT-4
   - Incorporates retrieved knowledge
   ↓
6. Score the enhancement
   - Bias score (0-100)
   - Diversity score (0-100)
   ↓
7. Check thresholds
   - If both met → Done!
   - If not and iterations remain → Refine and repeat
   ↓
8. Return best result
```

### Understanding Scores

#### Bias Score (0-100)
Measures how well the prompt avoids bias:
- **90-100**: Excellent bias mitigation
- **75-89**: Good bias mitigation (default target)
- **60-74**: Moderate bias mitigation
- **0-59**: Needs improvement

**Scoring Criteria**:
- Inclusive language (25 points)
- Stereotype avoidance (25 points)
- Neutral framing (25 points)
- Respectful descriptions (25 points)

#### Diversity Score (0-100)
Measures demographic inclusivity:
- **90-100**: Excellent diversity
- **80-89**: Good diversity (default target)
- **70-79**: Moderate diversity
- **0-69**: Limited diversity

**Scoring Criteria**:
- Age diversity (15 points): Multiple age groups
- Ethnic/racial diversity (20 points): Various backgrounds
- Gender diversity (15 points): Inclusive representation
- Cultural diversity (20 points): Multiple cultures
- Ability inclusion (10 points): Different abilities
- Socioeconomic diversity (10 points): Various backgrounds
- Specificity (10 points): Clear, detailed descriptions

### Example Enhancement Session

```bash
$ python main.py enhance "a doctor"

Original Prompt:
   a doctor

Enhanced Prompt:
   a diverse group of doctors from various ethnic backgrounds, ages, 
   and genders, including both male and female practitioners of Asian, 
   African, Latino, and European descent, ranging from young residents 
   to experienced senior physicians, some using mobility aids, working 
   collaboratively in a modern hospital setting

Scores:
   Bias Score:      50.0 → 85.0
   Diversity Score: 40.0 → 88.0

Improvements:
   Bias:      +35.0
   Diversity: +48.0

Iterations: 2

✅ Both thresholds met!
```

---

## Batch Processing

### The `batch` Command

Process multiple prompts from a CSV file with automatic image generation.

```bash
python main.py batch INPUT_CSV [OPTIONS]
```

### Basic Usage

```bash
# Process entire CSV
python main.py batch prompts.csv

# Process first 10 rows
python main.py batch prompts.csv --max-rows 10

# Start from row 5
python main.py batch prompts.csv --start-row 5 --max-rows 10

# Custom output directory
python main.py batch prompts.csv --output-dir my_results
```

### Batch Options

```bash
# Input/Output
--output-csv PATH           Output CSV path (auto-generated if not provided)
--prompt-column NAME        CSV column with prompts (default: "prompt")
--output-dir PATH           Results directory (default: "batch_results")
--image-output-dir PATH     Images directory (default: "generated_images")

# Processing Control
--start-row INT             Starting row index (default: 0)
--max-rows INT              Maximum rows to process (default: all)

# Image Generation
--image-generator TYPE      Generator type: "dalle3" or "mock" (default: "dalle3")

# Visual Evaluation
--enable-visual-bias        Enable visual bias evaluation on generated images
```

### Input CSV Format

Your CSV file should have at least a `prompt` column:

```csv
prompt
a doctor examining a patient
students in a classroom
engineers working on a project
a teacher giving a lecture
```

Additional columns are preserved in output.

### Output Files

Batch processing generates:

1. **Results CSV**: `batch_results/enhanced_prompts_[timestamp].csv`
   ```csv
   row_index,original_prompt,enhanced_prompt,bias_score,diversity_score,original_image_path,enhanced_image_path
   0,"a doctor","diverse doctors...",85.0,88.0,images/0000_original.png,images/0000_enhanced.png
   ```

2. **Results JSON**: `batch_results/enhanced_prompts_[timestamp].json`
   - Complete metadata
   - Enhancement details
   - Error information

3. **Generated Images**: `generated_images/`
   - `0000_original.png`: Original prompt image
   - `0000_enhanced.png`: Enhanced prompt image

### Image Generation

#### DALL-E 3 Generator (Default)

```bash
python main.py batch prompts.csv --image-generator dalle3
```

- **Size**: 1024x1024
- **Quality**: Standard
- **Style**: Natural
- **Cost**: ~$0.04 per image

#### Mock Generator (Testing)

```bash
python main.py batch prompts.csv --image-generator mock
```

- **Purpose**: Testing without API costs
- **Output**: Placeholder images
- **Speed**: Instant

### Visual Bias Evaluation

Enable demographic analysis of generated images:

```bash
python main.py batch prompts.csv --enable-visual-bias
```

**Requirements**:
- FairFace model: `models/fairface_alldata_4race_20191111.pt`
- dlib model: `models/shape_predictor_5_face_landmarks.dat`

**Metrics Provided**:
- **Bias-W**: Population-level bias (0.0-1.0, lower is better)
- **Bias-P**: Per-image bias
- **ENS**: Diversity score (higher is better)
- **KL Divergence**: Distribution difference
- **Demographics**: Race, gender, age distributions

### Example Batch Session

```bash
$ python main.py batch test_prompts.csv --max-rows 5

Processing CSV file: test_prompts.csv
Initializing enhancement systems...
Initializing dalle3 image generator...

Processing prompt 1/5: "a doctor"
  ✅ Enhanced (bias: 85.0, diversity: 88.0)
  ✅ Generated original image
  ✅ Generated enhanced image
  
Processing prompt 2/5: "students in classroom"
  ✅ Enhanced (bias: 82.0, diversity: 90.0)
  ✅ Generated original image
  ✅ Generated enhanced image
  
... (continues for all prompts)

Batch processing complete!
Results saved to: batch_results/enhanced_prompts_1698765432.csv
Successful: 5/5
```

---

## Knowledge Queries

### The `query` Command

Query knowledge graphs for bias patterns and cultural information.

```bash
python main.py query [QUESTION] [OPTIONS]
```

### Basic Usage

```bash
# Interactive mode
python main.py query

# Direct query
python main.py query "What stereotypes exist about doctors?"

# Query bias data only
python main.py query "racial stereotypes" --data-source bias

# Query cultural data only
python main.py query "food traditions" --data-source cultural

# Get more results
python main.py query "gender bias" --top-k 15
```

### Query Options

```bash
--data-source TYPE      Data source: "bias", "cultural", or "both" (default: "both")
--top-k INT            Number of top results (default: 8)
```

### Data Sources

#### Bias Data
- **Source**: ADV_GRAPH dataset
- **Content**: Bias patterns, stereotypes
- **Size**: ~1,300 triples
- **Format**: (subject, relation, object)

**Example Results**:
```
1. (Asian, are_perceived_as, math-oriented)
2. (Women, are_stereotyped_as, emotional)
3. (Elderly, are_portrayed_as, forgetful)
```

#### Cultural Data
- **Source**: Cultural value triples (15 regions)
- **Content**: Cultural practices, values, behaviors
- **Size**: 50,000+ triples
- **Regions**: US, UK, China, India, Iran, Korea, Japan, Mexico, etc.

**Example Results**:
```
1. (Japan, values, group harmony and consensus)
2. (India, celebrates, diverse festivals and traditions)
3. (Mexico, emphasizes, family connections and gatherings)
```

### Example Query Session

```bash
$ python main.py query "What are common gender stereotypes?"

BIAS DATA RESULTS
======================================================================

1. (women, are_stereotyped_as, emotional and sensitive)
2. (men, are_expected_to, show strength and hide emotions)
3. (women, are_portrayed_as, primary caregivers)
4. (men, are_associated_with, leadership roles)
5. (women, face_bias_in, technical and STEM fields)

CULTURAL DATA RESULTS
======================================================================

1. (various_cultures, have_different, gender role expectations)
2. (progressive_societies, promote, gender equality and inclusion)
3. (traditional_cultures, may_maintain, distinct gender roles)
```

---

## Testing & Demos

### The `test` Command

Run various tests and demos to validate the system.

```bash
python main.py test [OPTIONS]
```

### Test Options

```bash
--quick       Run quick enhancement test
--batch       Run batch processing test
--visual      Run visual bias evaluation test
--all         Run all tests
```

### Quick Test

Tests basic enhancement functionality:

```bash
$ python main.py test --quick

Testing enhancement system...
Original: a doctor
Enhanced: a diverse doctor from various backgrounds
Bias improvement: +35.0
Diversity improvement: +40.0
✅ Quick test passed
```

### Batch Test

Tests batch processing pipeline:

```bash
python main.py test --batch
```

### Visual Test

Tests visual bias evaluator:

```bash
python main.py test --visual

Testing visual bias evaluator...
Visual bias evaluator initialized successfully
✅ Visual test passed
```

### All Tests

```bash
python main.py test --all
```

---

## Configuration

### Environment Variables

```bash
# Required
export OPENAI_API_KEY="sk-..."

# Optional
export OPENAI_API_BASE="https://api.openai.com/v1"  # Custom endpoint
```

### Configuration File

Create `.env` file in project root:

```bash
OPENAI_API_KEY=sk-...
EMBEDDING_MODEL=text-embedding-3-small
CHAT_MODEL=gpt-4-turbo-preview
```

### Model Configuration

Edit `src/config/settings.py` to customize:

```python
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4-turbo-preview"
MAX_RETRIES = 3
TIMEOUT = 30
```

### Data Paths

Default data locations (configurable in `src/config/paths.py`):

```
data/
├── biases/           # Bias CSV files
├── cultural_values/  # Cultural triples
└── stereoset/        # StereoSet dataset

models/
├── fairface_alldata_4race_20191111.pt
└── shape_predictor_5_face_landmarks.dat

generated_images/     # Generated images
batch_results/        # Batch processing results
*.pkl                # Embedding caches
```

---

## Best Practices

### For Single Prompts

1. **Start with defaults**: The default thresholds (75/80) work well
2. **Use interactive mode**: Easier to experiment
3. **Enable RAG systems**: Better results with `--use-stereoset --use-diversity-rag`
4. **Iterate if needed**: Allow 3-5 iterations for best quality

### For Batch Processing

1. **Start small**: Test with `--max-rows 10` first
2. **Use mock generator**: Test pipeline with `--image-generator mock`
3. **Enable visual eval**: Add `--enable-visual-bias` for metrics
4. **Save intermediate**: Results saved after each prompt automatically
5. **Monitor costs**: DALL-E 3 costs ~$0.04 per image

### For Production

1. **Set higher thresholds**: Use 85/90 for production quality
2. **Enable all RAG systems**: Maximum knowledge integration
3. **Use visual evaluation**: Validate demographic diversity
4. **Batch processing**: More efficient for multiple prompts
5. **Review results**: Human oversight recommended

### Cost Management

**OpenAI API Costs** (approximate):
- Embeddings: $0.00002 per 1K tokens (~$0.02 per 1000 prompts)
- GPT-4 calls: $0.01-0.03 per enhancement (varies by length)
- DALL-E 3: $0.04 per image
- **Total for batch of 100**: ~$3-5 with images, ~$0.50 without

**Cost Reduction Tips**:
- Use embedding cache (automatically enabled)
- Test with mock generator
- Set `--max-iterations 3` instead of 5
- Batch processing reuses RAG systems

---

## Troubleshooting

### Common Issues

#### 1. API Key Not Found

```
Error: OpenAI API key not found
```

**Solution**:
```bash
export OPENAI_API_KEY="sk-..."
# Or add to .env file
```

#### 2. Import Errors

```
ModuleNotFoundError: No module named 'src'
```

**Solution**:
```bash
# Make sure you're in project root
cd /path/to/llm-bias-fairness

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### 3. Embedding Cache Errors

```
Error loading cache: pickle file corrupted
```

**Solution**:
```bash
# Delete cache and regenerate
rm embeddings.pkl stereoset_embeddings.pkl diversity_embeddings.pkl
```

#### 4. Visual Evaluation Errors

```
Error: FairFace model not found
```

**Solution**:
```bash
# Download required models
mkdir -p models
# Download FairFace model
wget https://... -O models/fairface_alldata_4race_20191111.pt
# Download dlib model
wget https://... -O models/shape_predictor_5_face_landmarks.dat
```

#### 5. Rate Limiting

```
Error: Rate limit exceeded
```

**Solution**:
- Wait a few seconds
- System automatically retries with backoff
- Reduce `--max-iterations` if frequent

#### 6. Out of Memory

```
MemoryError: Cannot allocate memory
```

**Solution**:
- Process smaller batches with `--max-rows`
- Reduce `--top-k` for retrieval
- Close other applications

### Debug Mode

Enable verbose logging for troubleshooting:

```bash
python main.py --verbose enhance "a doctor"
python main.py -v batch prompts.csv
```

### Getting Help

1. **Documentation**: Check `docs/` folder
2. **Examples**: See `examples/` folder
3. **Issues**: GitHub issues for bug reports
4. **Logs**: Check console output with `--verbose`

---

## Advanced Usage

### Custom Thresholds by Use Case

**Academic Research**:
```bash
python main.py enhance "a scientist" \
    --bias-threshold 90 \
    --diversity-threshold 95 \
    --max-iterations 10
```

**Marketing Content**:
```bash
python main.py enhance "customers shopping" \
    --bias-threshold 85 \
    --diversity-threshold 90 \
    --use-diversity-rag
```

**Quick Prototyping**:
```bash
python main.py enhance "a meeting" \
    --bias-threshold 65 \
    --diversity-threshold 70 \
    --max-iterations 2
```

### Programmatic Usage

Use the system in your Python code:

```python
from src.enhancement import EnhancementSystem, EnhancementConfig
from src.knowledge import StereoSetRAG, DiversityRAG

# Setup
config = EnhancementConfig(
    bias_threshold=75,
    diversity_threshold=80,
    max_iterations=5
)

stereoset = StereoSetRAG()
diversity = DiversityRAG()

system = EnhancementSystem(
    config=config,
    stereoset_rag=stereoset,
    diversity_rag=diversity
)

# Enhance
result = system.enhance("a doctor")

print(f"Enhanced: {result.final_prompt}")
print(f"Bias: {result.final_bias_score:.1f}")
print(f"Diversity: {result.final_diversity_score:.1f}")
```

---

## Appendix

### File Naming Conventions

**Generated Images**:
```
{row_index:04d}_{type}.png

Examples:
0000_original.png    # Row 0, original prompt
0000_enhanced.png    # Row 0, enhanced prompt
0042_original.png    # Row 42, original prompt
```

**Result Files**:
```
enhanced_prompts_{timestamp}.csv
enhanced_prompts_{timestamp}.json

Examples:
enhanced_prompts_1698765432.csv
enhanced_prompts_1698765432.json
```

### Keyboard Shortcuts

When in interactive mode:
- **Ctrl+C**: Cancel current operation
- **Ctrl+D**: Exit interactive mode (Unix)
- **Ctrl+Z then Enter**: Exit interactive mode (Windows)

### Further Reading

- **Architecture**: See `docs/ARCHITECTURE_V2.md`
- **Developer Guide**: See `docs/DEVELOPER_GUIDE.md`
- **CLI Reference**: See `docs/CLI_REFERENCE.md`
- **Visual Evaluation**: See `docs/VISUAL_BIAS_EVALUATION.md`
- **Phase Completions**: See `docs/PHASE*_COMPLETION.md`

---

**User Guide Version:** 2.0  
**Last Updated:** October 2025  
**System Version:** Production Ready  
**Questions?** Check the documentation or open an issue on GitHub
