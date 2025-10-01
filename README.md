# LLM Bias & Fairness Project

**Version 3.0** - Complete pipeline for bias mitigation, diversity enhancement, and visual bias evaluation in AI-generated content.

## 🎯 Overview

This project implements a comprehensive system for creating inclusive, culturally-aware AI-generated images. It combines knowledge-based prompt enhancement with visual bias evaluation to ensure diverse and unbiased representation.

### Core Components

- **🛡️ StereoSet RAG**: Bias detection using academic stereotype research (4,229 examples)
- **🌍 CultureBank RAG**: Diversity enhancement using social media cultural data (22,990 behaviors)  
- **📚 GraphRAG**: Cultural awareness using structured knowledge graphs (51,301+ triples)
- **🖼️ Image Generation**: Generic interface supporting DALL-E 3 and extensible to other models
- **🔍 Visual Bias Evaluation**: Demographic analysis using FairFace for objective measurement

## 🚀 Key Features

### ✅ Complete End-to-End Pipeline
- **Batch Processing**: Process multiple prompts from CSV with automatic result tracking
- **Dual Scoring**: Independent bias mitigation (0-100) and diversity enhancement (0-100) metrics
- **Sequential Enhancement**: Iterative improvement until quality thresholds are met
- **Visual Evaluation**: Demographic analysis and bias metrics for generated images
- **Flexible Configuration**: Customizable thresholds, iterations, and quality settings

### ✅ Multi-Source Knowledge Integration
- **Academic Research**: McGill-NLP StereoSet dataset for stereotype identification
- **Social Media Insights**: Reddit/TikTok cultural discussions for authentic diversity
- **Structured Knowledge**: 15+ cultural datasets with comprehensive bias patterns
- **Weighted Retrieval**: Vector similarity + keyword matching + agreement scores

### ✅ Production-Ready Implementation
- **Generic Image Generation**: Extensible interface for DALL-E 3, Stable Diffusion, etc.
- **Visual Bias Metrics**: FairFace-based demographic analysis (race, gender, age)
- **Comprehensive Results**: CSV + JSON outputs with full metadata
- **Robust Error Handling**: Graceful fallbacks and detailed logging

## 🎨 Enhanced Image Prompt Generation

Transform basic prompts into inclusive, culturally-aware descriptions:

**Original**: "a doctor examining a patient"

**Enhanced**: "In a global healthcare setting, Dr. Ji-Yeon Kim, a highly skilled physician of South Korean origin educated at the prestigious Seoul National University, offers compassionate medical services to a broad spectrum of patients. With a name common to all genders in Korea, Dr. Kim's practice is a beacon of inclusivity, shattering stereotypes and celebrating diversity in healthcare..."

## 💻 Quick Start

### 1. Installation

```bash
# Clone and setup
git clone <repository-url>
cd llm-bias-fairness
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Set API key
export OPENAI_API_KEY="sk-your-api-key-here"
```

### 2. Unified CLI - All Commands in One Place

All functionality is now available through a single unified interface:

```bash
# Enhance a single prompt
python main.py enhance "a doctor examining a patient"

# Process a CSV file with images
python main.py batch prompts.csv

# Query knowledge graphs
python main.py query "What stereotypes exist about doctors?"

# Run tests
python main.py test --quick
```

### 3. Interactive Mode

```bash
# Interactive enhancement (easiest for beginners)
python main.py enhance

# Interactive query
python main.py query
```

### 4. View Results

```
batch_results/
├── enhanced_prompts_[timestamp].csv    # All results
├── enhanced_prompts_[timestamp].json   # Detailed metadata
└── visual_bias_evaluation/             # Demographic analysis

generated_images/
├── 0001_original.png                   # Original prompt image
├── 0001_enhanced.png                   # Enhanced prompt image
└── ...
```

## 📖 Command Reference

### Enhance Command

Three modes for different needs:

```bash
# Basic (fast, single-pass)
python main.py enhance "a teacher" --mode basic

# Sequential (iterative with diversity threshold)
python main.py enhance "engineers" --mode sequential --threshold 85

# Dual-pipeline (comprehensive, recommended) ⭐
python main.py enhance "a doctor" --bias-threshold 80 --diversity-threshold 85
```

### Batch Command

Process CSV files with image generation:

```bash
# Basic batch processing
python main.py batch prompts.csv

# With mock images (testing, no API cost)
python main.py batch prompts.csv --image-generator mock

# With visual bias evaluation
python main.py batch prompts.csv --enable-visual-evaluation

# Custom settings
python main.py batch prompts.csv \
    --max-rows 10 \
    --output-dir my_results \
    --image-dir my_images
```

### Query Command

Research bias and cultural information:

```bash
# Interactive
python main.py query

# Direct query
python main.py query "What stereotypes exist in media?"

# With more results
python main.py query "Cultural practices in Asia" --top-k 15
```

### Test Command

Run tests and demos:

```bash
# Quick test (1 prompt, ~30 seconds)
python main.py test --quick

# Batch test (3 prompts with mock images)
python main.py test --batch

# Visual evaluation test
python main.py test --visual

# All tests
python main.py test --all
```

## 🆕 What's New in v3.0

- ✨ **Unified CLI**: All functionality in one `main.py` command
- 🔍 **Visual Bias Evaluation**: Demographic analysis with FairFace
- 📦 **Better Organization**: Old scripts moved to `/scripts` for reference
- 📚 **Improved Documentation**: Complete user guide with CLI examples
- 🧪 **Integrated Testing**: Built-in test commands

### Migration from Old Scripts

```bash
# Old way
python enhance_prompt_dual_pipeline.py -p "a doctor"

# New way
python main.py enhance "a doctor"

# See scripts/README.md for complete migration guide
```

## 📊 System Performance

### Knowledge Base Scale
- **57,000+ Total Triples**: Comprehensive knowledge from multiple sources
- **4,229 Stereotype Examples**: Academic research on harmful biases
- **22,990 Cultural Behaviors**: Authentic social media cultural insights
- **15+ Cultural Regions**: Global representation across continents

### Enhancement Quality
- **Bias Reduction**: Systematic identification and avoidance of harmful patterns
- **Diversity Improvement**: Measurable increases across 7 key dimensions
- **Cultural Authenticity**: Real social media data ensures genuine representation
- **Processing Speed**: Complete enhancement cycle in under 30 seconds

## 📚 Documentation

- **[User Guide](docs/USER_GUIDE.md)** - Complete usage guide with examples
- **[Architecture](docs/ARCHITECTURE.md)** - System design and technical details
- **[Visual Bias Evaluation](docs/VISUAL_BIAS_EVALUATION.md)** - Demographic analysis guide
- **[Changelog](CHANGELOG.md)** - Version history and updates

## 🏗️ System Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│                     INPUT: CSV FILE                               │
│              (prompts, metadata, configurations)                  │
└──────────────────────────┬────────────────────────────────────────┘
                           │
                           ▼
┌───────────────────────────────────────────────────────────────────┐
│              ENHANCEMENT SYSTEM (Dual-Pipeline)                   │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐             │
│  │ StereoSet   │  │ CultureBank  │  │   GraphRAG   │             │
│  │ (Bias Det.) │  │ (Diversity)  │  │ (Cultural)   │             │
│  └─────────────┘  └──────────────┘  └──────────────┘             │
│          │                │                 │                     │
│          └────────────────┼─────────────────┘                     │
│                           │                                       │
│  ┌────────────────────────────────────────────────┐               │
│  │  GPT-4 Enhancement with Dual Scoring           │               │
│  │  • Bias Mitigation Score (0-100)               │               │
│  │  • Diversity Enhancement Score (0-100)         │               │
│  │  • Iterative improvement until thresholds met  │               │
│  └────────────────────────────────────────────────┘               │
└──────────────────────────┬────────────────────────────────────────┘
                           │
                           ▼
┌───────────────────────────────────────────────────────────────────┐
│                    IMAGE GENERATION                               │
│  • Generate from original prompt → original.png                   │
│  • Generate from enhanced prompt → enhanced.png                   │
│  • Extensible interface (DALL-E 3, Stable Diffusion, etc.)        │
└──────────────────────────┬────────────────────────────────────────┘
                           │
                           ▼
┌───────────────────────────────────────────────────────────────────┐
│              VISUAL BIAS EVALUATION (Optional)                    │
│  ┌────────────────────────────────────────────────┐               │
│  │  Face Detection → Demographic Prediction       │               │
│  │  (dlib)          (FairFace ResNet34)           │               │
│  │                                                │               │
│  │  Metrics:                                      │               │
│  │  • Bias-W (population-level bias)              │               │
│  │  • Bias-P (per-image bias)                     │               │
│  │  • ENS (diversity via entropy)                 │               │
│  │  • KL Divergence (distribution comparison)     │               │
│  └────────────────────────────────────────────────┘               │
└──────────────────────────┬────────────────────────────────────────┘
                           │
                           ▼
┌───────────────────────────────────────────────────────────────────┐
│                      OUTPUT                                       │
│  • CSV with results and scores                                    │
│  • JSON with detailed metadata                                    │
│  • Generated images (original + enhanced)                         │
│  • Visual bias evaluation reports                                 │
└───────────────────────────────────────────────────────────────────┘
```

## 📦 Installation

### Prerequisites
- Python 3.13+ (recommended)
- OpenAI API key for full functionality
- Git for repository cloning

### Quick Setup
```bash
# Clone the repository
git clone <repository-url>
cd llm-bias-fairness

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set OpenAI API key
export OPENAI_API_KEY="sk-your-api-key-here"
```

### Dependencies
```python
openai>=1.82.0      # LLM integration and embeddings
pandas>=2.2.3       # Data processing
numpy>=2.2.6        # Numerical computing
scikit-learn        # Machine learning utilities
nltk               # Natural language processing
networkx==3.4.2     # Graph data structures
```

## 🎯 Use Cases

### Content Creation
- **Image Generation**: Enhanced prompts for DALL-E, Midjourney, Stable Diffusion
- **Marketing Materials**: Inclusive advertising and promotional content
- **Educational Resources**: Diverse representation in learning materials
- **Media Production**: Culturally aware visual storytelling

### Research & Analysis
- **Bias Studies**: Systematic analysis of stereotypes and cultural patterns
- **Cultural Research**: Cross-cultural behavior and value exploration
- **AI Ethics**: Fairness and inclusion in generative AI systems
- **Social Impact**: Understanding diversity representation in AI

## 🔧 Advanced Usage

### Custom Configuration

```bash
# Dual-pipeline with custom parameters (default mode)
python main.py enhance "medical professionals" \
    --bias-threshold 80 \
    --diversity-threshold 85 \
    --max-iterations 7 \
    --stereoset-top-k 15 \
    --diversity-top-k 12

# Sequential enhancement with quality control
python main.py enhance "business executives" \
    --mode sequential \
    --threshold 90 \
    --max-iterations 5

# Basic enhancement for quick results
python main.py enhance "teachers in classroom" \
    --mode basic \
    --bias-top-k 10 \
    --cultural-top-k 15
```

### Batch Processing
```bash
# Process multiple prompts from CSV
python main.py batch prompts.csv \
    --max-rows 100 \
    --image-generator dall-e \
    --enable-visual-evaluation

# Process with custom output location
python main.py batch prompts.csv \
    --output-dir custom_results \
    --image-dir custom_images
```

### API Integration
```python
from src.stereoset_rag import StereoSetRAG
from src.diversity_rag import DiversityRAG
from src.graphrag import GraphRAG

# Initialize systems
stereoset = StereoSetRAG()
diversity = DiversityRAG() 
graphrag = GraphRAG()

# Load knowledge bases
stereoset.load_dataset()
diversity.load_datasets()
graphrag.add_graph("bias", parser)

# Get enhancement data
bias_examples = stereoset.get_negative_examples("doctor")
diversity_recs = diversity.get_diversity_recommendations("doctor")
cultural_triples = graphrag.query("bias", "doctor", top_k=10)
```

## 📊 Performance & Metrics

### Knowledge Base Statistics
- **Total Knowledge**: 57,000+ triples from 3 complementary sources
- **Bias Detection**: 4,229 stereotypical patterns identified
- **Cultural Insights**: 22,990 authentic cultural behaviors
- **Global Coverage**: 15+ countries and cultural regions

### System Performance
- **Processing Speed**: Complete dual-pipeline cycle in under 30 seconds
- **Cache Efficiency**: 100% hit rate after initial embedding generation
- **Memory Usage**: Optimized for large knowledge bases
- **API Efficiency**: Intelligent caching minimizes OpenAI API costs

### Enhancement Quality Examples

| Original Prompt | Diversity Score | Enhancement Quality |
|----------------|-----------------|-------------------|
| "a doctor" | 0/100 → 45/100 | Added age, ethnic, gender diversity |
| "students studying" | 5/100 → 73/100 | Included cultural, ability representation |
| "business meeting" | 10/100 → 67/100 | Enhanced socioeconomic, cultural elements |

## 📚 Documentation

### Complete Documentation Set
- **[PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md)** - Comprehensive system overview
- **[CHANGELOG.md](docs/CHANGELOG.md)** - Complete development history  
- **[DUAL_PIPELINE_COMPLETE.md](docs/DUAL_PIPELINE_COMPLETE.md)** - Latest implementation guide
- **[DOCUMENTATION_INDEX.md](docs/DOCUMENTATION_INDEX.md)** - Documentation organization

### Quick References
- **[SEQUENTIAL_ENHANCEMENT.md](docs/SEQUENTIAL_ENHANCEMENT.md)** - Iterative enhancement guide
- **[IMAGE_PROMPT_ENHANCEMENT.md](docs/IMAGE_PROMPT_ENHANCEMENT.md)** - Enhancement features
- **[REPORT.md](docs/REPORT.md)** - Technical analysis and research

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
