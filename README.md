# LLM Bias & Fairness Project - Dual-Pipeline Enhancement System

**Version 2.0** - Production-ready system for bias mitigation and diversity enhancement in AI-generated content.

## 🎯 Overview

This project implements a comprehensive **dual-pipeline enhancement system** that addresses bias and promotes diversity in image generation prompts through three integrated knowledge sources:

- **🛡️ StereoSet RAG**: Bias detection using academic stereotype research (4,229 examples)
- **🌍 CultureBank RAG**: Diversity enhancement using social media cultural data (22,990 behaviors)  
- **📚 GraphRAG**: Cultural awareness using structured knowledge graphs (51,301+ triples)

## 🚀 Key Features

### ✅ Dual-Pipeline Enhancement System
- **Separate Scoring**: Independent bias mitigation (0-100) and diversity enhancement (0-100) metrics
- **Sequential Enhancement**: Continues iterating until BOTH thresholds are met
- **Real-Time Assessment**: Detailed score breakdowns with actionable feedback
- **Configurable Quality**: Custom thresholds and maximum iterations

### ✅ Multi-Source Knowledge Integration
- **Academic Research**: McGill-NLP StereoSet dataset for stereotype identification
- **Social Media Insights**: Reddit/TikTok cultural discussions for authentic diversity
- **Structured Knowledge**: 15+ cultural datasets with comprehensive bias patterns
- **Weighted Retrieval**: Vector similarity + keyword matching + agreement scores

### ✅ Production-Ready Architecture
- **Modular Design**: Clean separation of concerns with extensible components
- **Performance Optimization**: Persistent caching and efficient processing
- **User-Friendly Interface**: Interactive CLI with progress tracking
- **Robust Error Handling**: Graceful fallbacks and comprehensive logging

## 🎨 Enhanced Image Prompt Generation

Transform basic prompts into inclusive, culturally-aware descriptions:

**Original**: "a doctor examining a patient"

**Enhanced**: "In a global healthcare setting, Dr. Ji-Yeon Kim, a highly skilled physician of South Korean origin educated at the prestigious Seoul National University, offers compassionate medical services to a broad spectrum of patients. With a name common to all genders in Korea, Dr. Kim's practice is a beacon of inclusivity, shattering stereotypes and celebrating diversity in healthcare..."

## 💻 Quick Start

### 1. Dual-Pipeline Enhancement (Recommended) ⭐
The latest and most advanced enhancement system:

```bash
# Interactive mode with guided prompts
python enhance_prompt_dual_pipeline.py

# Direct enhancement with custom thresholds
python enhance_prompt_dual_pipeline.py \
    -p "students studying in library" \
    --bias-threshold 75 \
    --diversity-threshold 80 \
    --max-iterations 5
```

### 2. Sequential Enhancement
Iterative improvement with diversity scoring:

```bash
# Basic sequential enhancement
python enhance_prompt_sequential.py -p "a business meeting"

# Custom parameters
python enhance_prompt_sequential.py \
    --prompt "engineers working" \
    --threshold 85 \
    --max-iterations 4
```

### 3. Basic Enhancement
Single-pass enhancement for quick improvements:

```bash
# Interactive mode
python enhance_prompt.py

# Direct enhancement
python enhance_prompt.py -p "students in a classroom"
```

### 4. Knowledge Graph Analysis
Explore bias patterns and cultural values:

```bash
# Analyze bias patterns
python main.py -q "What stereotypes exist about doctors?"

# Study cultural practices  
python main.py -q "How do different cultures celebrate achievements?"
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

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DUAL-PIPELINE SYSTEM                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  StereoSet   │  │ CultureBank  │  │   GraphRAG   │         │
│  │     RAG      │  │     RAG      │  │   Knowledge  │         │
│  │ (4,229 bias  │  │ (22,990 cult │  │ (51,301      │         │
│  │  examples)   │  │  behaviors)  │  │  triples)    │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│           │               │               │                   │
│           └───────────────┼───────────────┘                   │
│                           │                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │         WEIGHTED RETRIEVAL ENGINE                       │   │
│  │  • Vector Similarity (α=0.6)                          │   │
│  │  • Keyword Matching (β=0.3)                           │   │
│  │  • Agreement Scores (γ=0.1)                           │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              DUAL SCORING SYSTEM                        │   │
│  │  Bias Score (0-100)     │  Diversity Score (0-100)     │   │
│  │  • Inclusive language   │  • Age diversity (15pts)     │   │
│  │  • Anti-bias terms      │  • Ethnic diversity (20pts)  │   │
│  │  • Stereotype avoidance │  • Gender diversity (15pts)  │   │
│  │                         │  • Cultural diversity (20pts)│   │
│  │                         │  • Ability inclusion (10pts) │   │
│  │                         │  • Socioeconomic (10pts)     │   │
│  │                         │  • Specificity (10pts)       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │           SEQUENTIAL ENHANCEMENT LOOP                   │   │
│  │  1. Generate enhancement with GPT-4                     │   │
│  │  2. Score both bias and diversity                       │   │
│  │  3. Check if both thresholds met                        │   │
│  │  4. If not, repeat with focused improvement             │   │
│  │  5. Continue until success or max iterations            │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
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
# Dual-pipeline with custom parameters
python enhance_prompt_dual_pipeline.py \
    -p "medical professionals" \
    --bias-threshold 80 \
    --diversity-threshold 85 \
    --max-iterations 7 \
    --stereoset-top-k 15 \
    --diversity-top-k 12

# Sequential enhancement with quality control
python enhance_prompt_sequential.py \
    --prompt "business executives" \
    --threshold 90 \
    --max-iterations 5

# Basic enhancement for quick results
python enhance_prompt.py \
    -p "teachers in classroom" \
    --bias-top-k 10 \
    --cultural-top-k 15
```

### Batch Processing
```bash
# Process multiple prompts from file
cat prompts.txt | while read prompt; do
    python enhance_prompt_dual_pipeline.py -p "$prompt" --bias-threshold 75
done
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
    --top_k 25
```
