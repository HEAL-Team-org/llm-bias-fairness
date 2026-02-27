# LLM Bias & Fairness Project

**Version 4.0 (Refactored)** - Production-ready system for bias mitigation, diversity enhancement, and visual bias evaluation in AI-generated content.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-green.svg)](https://openai.com/)
[![Tests](https://img.shields.io/badge/tests-66%20passing-brightgreen.svg)](#testing)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## 🎯 Overview

A comprehensive, enterprise-grade system for creating inclusive, culturally-aware AI-generated images. Combines knowledge-based prompt enhancement with visual bias evaluation to ensure diverse and unbiased representation across all AI-generated content.

**New in v4.0**: Complete architectural refactoring with 8-layer design, unified CLI, and comprehensive testing (66 integration tests).

### Architecture

Built on a clean, layered architecture with strict dependency management:

```
Layer 8: CLI           → Command-line interface
Layer 7: Enhancement   → Prompt enhancement system  
Layer 6: Evaluation    → Visual bias evaluation
Layer 5: Generation    → Image generation
Layer 4: Knowledge     → RAG systems (StereoSet, Diversity, GraphRAG)
Layer 3: Data          → Data access, embeddings, caching
Layer 2: Prompts       → Template management
Layer 1: Config        → Settings, configuration
```

**See**: [`docs/ARCHITECTURE_V2.md`](docs/ARCHITECTURE_V2.md) for complete architecture documentation.

### Core Components

- **🛡️ StereoSet RAG**: Bias detection using academic stereotype research (4,229 examples)
- **� Diversity RAG**: Diversity enhancement using social media cultural data (22,990 behaviors)  
- **📚 GraphRAG**: Cultural awareness using structured knowledge graphs (51,301+ triples)
- **🖼️ Image Generation**: Generic interface supporting DALL-E 3 and extensible to other models
- **🔍 Visual Bias Evaluation**: Demographic analysis using FairFace for objective measurement
- **⚡ Enhancement System**: Unified dual-pipeline enhancement with configurable thresholds
- **🎨 CLI Interface**: Professional command-line interface for all operations

## 🚀 Key Features

### ✅ Production-Ready Architecture
- **Layered Design**: Clean separation of concerns with unidirectional dependencies
- **Type Safety**: Full type hints with Pydantic models
- **Comprehensive Testing**: 66 integration tests across 8 architectural layers
- **Extensible**: Add new RAG systems, generators, or evaluators easily
- **Well-Documented**: Complete API reference, user guide, and developer documentation

### ✅ Complete End-to-End Pipeline
- **Batch Processing**: Process CSV files with automatic result tracking
- **Dual Scoring**: Independent bias mitigation (0-100) and diversity (0-100) metrics
- **Sequential Enhancement**: Iterative improvement until quality thresholds are met
- **Visual Evaluation**: Demographic analysis and bias metrics for generated images
- **Flexible Configuration**: Customizable thresholds, iterations, and quality settings

### ✅ Multi-Source Knowledge Integration
- **Academic Research**: McGill-NLP StereoSet dataset for stereotype identification
- **Social Media Insights**: Reddit/TikTok cultural discussions for authentic diversity
- **Structured Knowledge**: 15+ cultural datasets with comprehensive bias patterns
- **Weighted Retrieval**: Vector similarity + keyword matching + agreement scores

### ✅ Professional CLI Interface
- **Unified Commands**: All functionality through single `main.py` entry point
- **Interactive Mode**: User-friendly prompts for beginners
- **Batch Operations**: Process multiple prompts efficiently
- **Knowledge Queries**: Explore RAG systems interactively
- **Rich Output**: Clear, formatted results with progress indicators

## 🎨 Enhanced Image Prompt Generation

Transform basic prompts into inclusive, culturally-aware descriptions:

**Input**: "a doctor examining a patient"

**Output**: "In a global healthcare setting, Dr. Ji-Yeon Kim, a highly skilled physician of South Korean origin educated at the prestigious Seoul National University, offers compassionate medical services to a broad spectrum of patients. With a name common to all genders in Korea, Dr. Kim's practice is a beacon of inclusivity, shattering stereotypes and celebrating diversity in healthcare..."

**Scores**:
- Bias Score: 23 → 87 (+64 improvement)
- Diversity Score: 35 → 92 (+57 improvement)

## 💻 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repository-url>
cd llm-bias-fairness

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set OpenAI API key
export OPENAI_API_KEY="sk-your-api-key-here"
```

### 2. Basic Usage

#### Enhance a Single Prompt

```bash
# Quick enhancement (interactive)
python main.py enhance

# Direct enhancement
python main.py enhance "a doctor examining a patient"

# With custom thresholds
python main.py enhance "an engineer" --bias-threshold 85 --diversity-threshold 85
```

#### Process CSV Batch

```bash
# Basic batch processing
python main.py batch prompts.csv

# With image generation (DALL-E 3)
python main.py batch prompts.csv --generate-images

# With visual bias evaluation
python main.py batch prompts.csv --generate-images --evaluate-images
```

#### Query Knowledge Systems

```bash
# Interactive query
python main.py query

# Direct query
python main.py query "What stereotypes exist about doctors?"

# Query specific RAG system
python main.py query "diversity recommendations" --system diversity
```

#### Run Tests

```bash
# Quick test
python main.py test --quick

# Full test suite
python main.py test --comprehensive
```

### 3. View Results

```
batch_results/
├── enhanced_prompts_[timestamp].csv    # All results in CSV format
├── enhanced_prompts_[timestamp].json   # Detailed metadata
└── visual_bias_evaluation/             # Demographic analysis
    ├── bias_metrics.csv                # Aggregate metrics
    └── detailed_results.json           # Per-image analysis

generated_images/
├── 0001_original.png                   # Original prompt image
├── 0001_enhanced.png                   # Enhanced prompt image
├── 0002_original.png
└── 0002_enhanced.png
```


## 📖 CLI Command Reference

### `enhance` - Enhance a Single Prompt

```bash
# Interactive mode (recommended for beginners)
python main.py enhance

# Direct enhancement
python main.py enhance "a doctor"

# With custom thresholds
python main.py enhance "an engineer" --bias-threshold 85 --diversity-threshold 90

# Control RAG systems
python main.py enhance "a teacher" --no-stereoset --no-graphrag

# Adjust iterations
python main.py enhance "a scientist" --max-iterations 3
```

**Options**:
- `--bias-threshold INT`: Minimum bias score (0-100, default: 75)
- `--diversity-threshold INT`: Minimum diversity score (0-100, default: 80)
- `--max-iterations INT`: Maximum enhancement attempts (default: 5)
- `--no-stereoset`: Disable StereoSet RAG
- `--no-diversity-rag`: Disable Diversity RAG
- `--no-graphrag`: Disable GraphRAG

### `batch` - Process CSV File

```bash
# Basic batch processing
python main.py batch prompts.csv

# With image generation (DALL-E 3)
python main.py batch prompts.csv --generate-images

# With visual bias evaluation
python main.py batch prompts.csv --generate-images --evaluate-images

# Custom output locations
python main.py batch prompts.csv --output-dir results --image-dir images

# Limit rows (for testing)
python main.py batch prompts.csv --max-rows 5
```

**Input CSV Format**:
```csv
prompt
a doctor examining a patient
an engineer working on a project
a teacher in a classroom
```

**Options**:
- `--generate-images`: Generate images using DALL-E 3
- `--evaluate-images`: Run visual bias evaluation
- `--output-dir PATH`: Results directory (default: `batch_results/`)
- `--image-dir PATH`: Images directory (default: `generated_images/`)
- `--max-rows INT`: Limit number of prompts to process

### `query` - Query Knowledge Systems

```bash
# Interactive mode
python main.py query

# Direct query
python main.py query "What stereotypes exist about doctors?"

# More results
python main.py query "cultural practices in Asia" --top-k 15

# Query specific system
python main.py query "diversity recommendations" --system diversity
python main.py query "bias examples" --system stereoset
python main.py query "cultural knowledge" --system graphrag
```

**Options**:
- `--top-k INT`: Number of results to return (default: 10)
- `--system NAME`: Specific RAG system (stereoset/diversity/graphrag)

### `test` - Run Tests

```bash
# Quick test (single prompt, ~30 seconds)
python main.py test --quick

# Comprehensive test
python main.py test --comprehensive

# Demo mode (shows system capabilities)
python main.py test --demo
```

## 📊 Understanding the Scores

### Bias Mitigation Score (0-100)

Measures how well the prompt avoids harmful stereotypes:

- **0-40**: High bias - Contains explicit stereotypes
- **40-70**: Moderate bias - Some problematic patterns  
- **70-85**: Low bias - Mostly neutral
- **85-100**: Excellent - No detectable bias

### Diversity Enhancement Score (0-100)

Measures representation across 7 dimensions:

- **0-40**: Low diversity - Homogeneous
- **40-70**: Moderate diversity - Some variety
- **70-85**: Good diversity - Well-rounded
- **85-100**: Excellent - Comprehensive inclusion

**Dimensions evaluated**:
1. Gender representation
2. Ethnic/racial diversity
3. Age range
4. Geographic/cultural backgrounds
5. Socioeconomic diversity
6. Ability/disability representation
7. Professional/role diversity

## 🏗️ System Architecture

The system is built on a clean, 8-layer architecture:

```
┌─────────────────────────────────────────────────────────┐
│  Layer 8: CLI (src/cli/)                                │
│  ├── Commands: enhance, batch, query, test              │
│  ├── Argument parsing and validation                    │
│  └── User interface and output formatting               │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│  Layer 7: Enhancement (src/enhancement/)                │
│  ├── EnhancementSystem: Main orchestrator               │
│  ├── Dual scoring: Bias + Diversity                     │
│  ├── Iterative improvement                              │
│  └── Result aggregation                                 │
└──────────┬──────────────────────┬───────────────────────┘
           │                      │
┌──────────▼──────────┐  ┌───────▼─────────────────────┐
│  Layer 6:           │  │  Layer 5:                   │
│  Evaluation         │  │  Generation                 │
│  (src/evaluation/)  │  │  (src/generation/)          │
│                     │  │                             │
│  ├── Visual Bias    │  │  ├── DALLE3Generator       │
│  ├── FairFace       │  │  ├── MockGenerator         │
│  └── Demographics   │  │  └── Factory pattern        │
└─────────────────────┘  └─────────────────────────────┘
           │                      │
┌──────────▼──────────────────────▼───────────────────────┐
│  Layer 4: Knowledge (src/knowledge/)                    │
│  ├── StereoSetRAG: Bias detection                       │
│  ├── DiversityRAG: Diversity recommendations            │
│  ├── GraphRAG: Cultural knowledge                       │
│  └── Vector similarity search                           │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│  Layer 3: Data (src/data/)                              │
│  ├── EmbeddingCache: Persistent caching                 │
│  ├── OpenAIEmbedder: text-embedding-3-small             │
│  ├── BiasDataParser: Parse CSV bias data                │
│  └── CulturalDataParser: Parse cultural triples         │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│  Layer 2: Prompts (src/prompts/)                        │
│  ├── PromptManager: Template loading                    │
│  ├── Enhancement templates                              │
│  └── Query templates                                    │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│  Layer 1: Config (src/config/)                          │
│  ├── Settings: Environment variables                    │
│  ├── Paths: Data file locations                         │
│  └── Pydantic validation                                │
└─────────────────────────────────────────────────────────┘
```

**Key Principles**:
- **Unidirectional dependencies**: Higher layers depend on lower layers only
- **Separation of concerns**: Each layer has a single, well-defined responsibility
- **Extensibility**: Easy to add new components at any layer
- **Testability**: 66 integration tests across all 8 layers

**See**: [`docs/ARCHITECTURE_V2.md`](docs/ARCHITECTURE_V2.md) for detailed architecture documentation.

## 📚 Documentation

### User Documentation
- **[User Guide](docs/USER_GUIDE_V2.md)** - Complete usage guide with examples
- **[CLI Reference](docs/CLI_REFERENCE.md)** - All CLI commands and options
- **[Visual Bias Evaluation](docs/VISUAL_BIAS_EVALUATION.md)** - Demographic analysis guide

### Developer Documentation
- **[Architecture](docs/ARCHITECTURE_V2.md)** - System design and technical details
- **[Developer Guide](docs/DEVELOPER_GUIDE.md)** - Contributing and development setup
- **[Phase Completion Docs](docs/)** - Detailed refactoring process (PHASE1-8_COMPLETION.md)

### Other Resources
- **[Changelog](CHANGELOG.md)** - Version history and updates
- **[Scripts README](scripts/README.md)** - Legacy scripts reference

## 🧪 Testing

The project includes comprehensive testing:

```bash
# Run all phase tests (66 tests)
python test_phase1.py  # Config layer (5 tests)
python test_phase2.py  # Prompts layer (5 tests)
python test_phase3.py  # Data layer (7 tests)
python test_phase4.py  # Knowledge layer (10 tests)
python test_phase5.py  # Generation layer (8 tests)
python test_phase6.py  # Evaluation layer (7 tests)
python test_phase7.py  # Enhancement layer (15 tests)
python test_phase8.py  # CLI layer (9 tests)

# Or use the CLI
python main.py test --comprehensive
```

**Test Coverage**:
- ✅ Import validation
- ✅ Basic functionality
- ✅ No circular dependencies
- ✅ Integration between layers
- ✅ End-to-end workflows

## 🔬 Knowledge Base Details

### StereoSet RAG (Bias Detection)
- **Source**: McGill-NLP academic research
- **Size**: 4,229 stereotype examples
- **Coverage**: Professions, demographics, social groups
- **Purpose**: Identify and avoid harmful stereotypes

### Diversity RAG (Cultural Enhancement)
- **Source**: Reddit & TikTok (CultureBank)
- **Size**: 22,990 cultural behaviors
- **Coverage**: 15+ global regions
- **Purpose**: Authentic cultural representation

### GraphRAG (Knowledge Graphs)
- **Source**: Structured cultural knowledge
- **Size**: 51,301+ triples
- **Coverage**: Stereotypes, cultural norms, demographics
- **Purpose**: Comprehensive bias pattern detection

## 🎯 Use Cases

### Academic Research
```bash
# Generate dataset of enhanced prompts
python main.py batch research_prompts.csv --output-dir research_results

# Evaluate visual bias in generated images
python main.py batch prompts.csv --generate-images --evaluate-images
```

### Content Creation
```bash
# Enhance marketing prompts
python main.py enhance "a software developer"

# Batch process campaign concepts
python main.py batch campaign_ideas.csv
```

### Prototyping & Testing
```bash
# Quick test with mock images (no API costs)
python main.py test --quick

# Evaluate diversity improvements
python main.py query "diversity recommendations for healthcare"
```

## 📦 Installation

### Prerequisites
- Python 3.10+ (Python 3.13+ recommended)
- OpenAI API key for GPT-4 and embeddings
- Git for version control

### Standard Installation

```bash
# Clone the repository
git clone <repository-url>
cd llm-bias-fairness

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set OpenAI API key
export OPENAI_API_KEY="sk-your-api-key-here"
```

### Development Installation

```bash
# Clone repository
git clone <repository-url>
cd llm-bias-fairness

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install in editable mode (for development)
pip install -e .

# Run tests to verify installation
python test_phase1.py
python test_phase8.py
```

### Dependencies

Core dependencies:
```
openai>=1.82.0       # GPT-4 and embeddings
pandas>=2.2.3        # Data processing
numpy>=2.2.6         # Numerical operations
scikit-learn         # ML utilities
networkx==3.4.2      # Knowledge graphs
pydantic>=2.0        # Type-safe configuration
pyyaml               # Template management
```

## 💡 Examples

### Example 1: Simple Enhancement

```bash
$ python main.py enhance "a doctor"

=== Original Prompt ===
a doctor

=== Enhanced Prompt ===
In a global healthcare setting, Dr. Ji-Yeon Kim, a highly skilled physician
of South Korean origin educated at Seoul National University, offers
compassionate medical services to a broad spectrum of patients...

=== Scores ===
Initial Bias: 23/100
Final Bias: 87/100 (+64 improvement)
Initial Diversity: 35/100
Final Diversity: 92/100 (+57 improvement)
Iterations: 3
```

### Example 2: Batch Processing

```bash
$ python main.py batch prompts.csv --generate-images

Processing: 5 prompts
[1/5] Enhancing: "a doctor"... ✓ (87/92)
[2/5] Enhancing: "an engineer"... ✓ (89/91)
[3/5] Enhancing: "a teacher"... ✓ (85/88)
[4/5] Enhancing: "a scientist"... ✓ (90/93)
[5/5] Enhancing: "a lawyer"... ✓ (86/90)

Generating images...
[1/5] Original + Enhanced ✓
[2/5] Original + Enhanced ✓
...

Results saved to:
- batch_results/enhanced_prompts_1234567890.csv
- batch_results/enhanced_prompts_1234567890.json
- generated_images/ (10 images)
```

### Example 3: Knowledge Query

```bash
$ python main.py query "stereotypes about doctors"

=== Query Results ===
Found 10 relevant examples:

1. Context: Medical professionals
   Bias: "All doctors are rich"
   Type: Socioeconomic stereotype
   Source: StereoSet

2. Context: Healthcare workers
   Bias: "Doctors are always male"
   Type: Gender stereotype
   Source: StereoSet
...
```

## 🚀 Migration from v3.0

If you're upgrading from version 3.0, the new v4.0 architecture is fully backward compatible:

**Old CLI (v3.0)**:
```bash
python enhance_prompt_dual_pipeline.py -p "a doctor"
python batch_processor.py input.csv
```

**New CLI (v4.0)**:
```bash
python main.py enhance "a doctor"
python main.py batch input.csv
```

**Key Changes**:
- All functionality moved to unified `main.py` interface
- Layered architecture with 8 clean layers
- 66 integration tests for reliability
- Comprehensive documentation (ARCHITECTURE_V2.md, USER_GUIDE_V2.md, DEVELOPER_GUIDE.md)
- Old scripts preserved in `scripts/` directory

**See**: [`docs/ARCHITECTURE_V2.md`](docs/ARCHITECTURE_V2.md) for complete migration guide.

## 🤝 Contributing

We welcome contributions! Please see our developer guide:

```bash
# Read the developer guide
cat docs/DEVELOPER_GUIDE.md

# Run tests before submitting
python test_phase1.py
python test_phase2.py
# ... (run all 8 phase tests)

# Or use comprehensive test
python main.py test --comprehensive
```

**Contribution Areas**:
- New RAG systems (Layer 4)
- Additional image generators (Layer 5)
- Enhanced evaluation metrics (Layer 6)
- CLI improvements (Layer 8)
- Documentation updates
- Bug fixes and performance improvements

**See**: [`docs/DEVELOPER_GUIDE.md`](docs/DEVELOPER_GUIDE.md) for complete contribution guidelines.

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- **StereoSet Dataset**: McGill-NLP for academic stereotype research
- **CultureBank**: Cultural behavior data from social media
- **FairFace**: Demographic prediction model
- **OpenAI**: GPT-4 and embedding APIs

## 📞 Support

- **Issues**: Open an issue on GitHub
- **Documentation**: See `docs/` directory
- **Email**: Contact project maintainers

---

**Version**: 4.0 (Refactored)  
**Last Updated**: October 2025  
**Status**: Production Ready ✅  
**Tests**: 66/66 Passing ✅

## 🤝 Contributing

This project is part of academic research in AI bias mitigation and fairness. Contributions are welcome:

1. **Fork the repository**
2. **Create feature branch** (`git checkout -b feature/enhancement`)
3. **Commit changes** (`git commit -am 'Add new enhancement'`)
4. **Push to branch** (`git push origin feature/enhancement`)
5. **Create Pull Request**

## 📄 License

This project is part of academic research. Please cite appropriately if used in academic work.

## 🙏 Acknowledgments

- **McGill-NLP** for the StereoSet dataset
- **CultureBank Project** for cultural discussion data  
- **OpenAI** for embedding and language model APIs
- **Academic Research Community** for bias and fairness research foundations
```
