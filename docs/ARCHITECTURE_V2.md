# System Architecture

**LLM Bias & Fairness Project - Technical Architecture Overview (v2.0)**

**Last Updated:** October 2025  
**Architecture Version:** 2.0 (Refactored)

## Overview

The LLM Bias & Fairness Project is a comprehensive system for bias mitigation and diversity enhancement in AI-generated image prompts. The architecture is modular, layered, and built on a clean separation of concerns across eight distinct layers, from configuration to command-line interface.

## Architecture Philosophy

### Design Principles

1. **Separation of Concerns**: Each layer has a single, well-defined responsibility
2. **Dependency Direction**: Dependencies flow in one direction (no circular imports)
3. **Interface Segregation**: Clean public APIs with minimal exports
4. **Open/Closed Principle**: Open for extension, closed for modification
5. **Dependency Injection**: Components accept dependencies rather than creating them

### Layer Dependency Graph

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLI Layer                               │
│                    (src/cli/)                                   │
│  Command-line interface, user interaction                       │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Enhancement Layer                            │
│                  (src/enhancement/)                             │
│  Prompt enhancement, bias mitigation                            │
└────────────┬────────────────────────┬───────────────────────────┘
             │                        │
             ▼                        ▼
┌────────────────────────┐  ┌────────────────────────────────────┐
│   Evaluation Layer     │  │      Knowledge Layer               │
│   (src/evaluation/)    │  │      (src/knowledge/)              │
│  Visual bias metrics   │  │  RAG systems, GraphRAG             │
└────────┬───────────────┘  └────────┬───────────────────────────┘
         │                           │
         ▼                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Generation Layer                             │
│                   (src/generation/)                             │
│  Image generation, DALL-E integration                           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Data Layer                                 │
│                     (src/data/)                                 │
│  Embeddings, caching, parsing                                   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Prompts Layer                                │
│                    (src/prompts/)                               │
│  Prompt templates, YAML management                              │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Config Layer                                 │
│                    (src/config/)                                │
│  Settings, paths, API keys                                      │
└─────────────────────────────────────────────────────────────────┘
```

## System Layers

### Layer 1: Configuration Layer (`src/config/`)

**Purpose**: Centralized configuration management

**Components**:
- `settings.py`: Application settings with Pydantic validation
- `paths.py`: Path configuration and validation

**Key Features**:
- Type-safe configuration with Pydantic
- Environment variable support
- Validation on load
- No business logic

**Public API**:
```python
from src.config import settings, paths

# Access configuration
api_key = settings.OPENAI_API_KEY
data_dir = paths.DATA_DIR
```

**Configuration Structure**:
```python
class Settings(BaseSettings):
    # API Keys
    OPENAI_API_KEY: str
    
    # Model Settings
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    CHAT_MODEL: str = "gpt-4-turbo-preview"
    
    # Processing Settings
    MAX_RETRIES: int = 3
    BATCH_SIZE: int = 10
```

---

### Layer 2: Prompts Layer (`src/prompts/`)

**Purpose**: Prompt template management

**Components**:
- `prompt_manager.py`: Prompt loading and formatting
- `templates/`: YAML template files
  - `enhancement_prompts.yaml`: Enhancement system prompts
  - `query_prompts.yaml`: Knowledge query prompts

**Key Features**:
- YAML-based templates
- Variable substitution
- Organized by functionality
- Easy to modify and version

**Public API**:
```python
from src.prompts import PromptManager

pm = PromptManager()
prompt = pm.get_prompt("enhancement", "dual_pipeline", 
                       original="a doctor", context="...")
```

---

### Layer 3: Data Layer (`src/data/`)

**Purpose**: Data access, caching, and parsing

**Components**:
- `embeddings.py`: Embedding generation and caching
  - `OpenAIEmbedder`: OpenAI API integration
  - `EmbeddingCache`: Persistent pickle cache
- `parsers.py`: Data parsing utilities
  - `BiasCSVParser`: Bias dataset parser
  - `CulturalTriplesParser`: Cultural data parser
  - `StereoSetParser`: StereoSet JSON parser
  - `DataParserFactory`: Automatic parser selection

**Key Features**:
- Embedding cache with 85%+ hit rate
- Factory pattern for parser selection
- Type-safe data structures
- Efficient batch processing

**Public API**:
```python
from src.data import EmbeddingCache, OpenAIEmbedder, DataParserFactory

# Embeddings
cache = EmbeddingCache("embeddings.pkl")
embedder = OpenAIEmbedder(cache=cache)
vector = embedder.embed("text to embed")

# Parsing
parser = DataParserFactory.create_parser("data.csv")
triples = parser.parse()
```

**Cache Statistics**:
- Average hit rate: 85-90%
- Storage: ~50KB per 1000 embeddings
- Load time: <100ms for 10K embeddings

---

### Layer 4: Knowledge Layer (`src/knowledge/`)

**Purpose**: Knowledge retrieval and RAG systems

**Components**:

#### 4.1 StereoSet RAG (`stereoset.py`)
- **Purpose**: Stereotype detection and bias mitigation
- **Dataset**: McGill-NLP StereoSet (4,229 examples)
- **Method**: Vector similarity search

**Example Record**:
```python
{
    "context": "The Asian student was...",
    "sentence_type": "stereotype",
    "target": "Asian",
    "bias_type": "race"
}
```

#### 4.2 Diversity RAG (`diversity.py`)
- **Purpose**: Cultural diversity enhancement
- **Dataset**: CultureBank (22,990 cultural behaviors)
- **Method**: Weighted retrieval (vector + keyword + agreement)

**Weighting Formula**:
```
score = 0.6 × similarity + 0.3 × keyword + 0.1 × agreement
```

**Example Record**:
```python
{
    "cultural_group": "South Korea",
    "actor_behavior": "showing respect through formal language",
    "topic": "workplace interaction",
    "agreement": 0.85
}
```

#### 4.3 GraphRAG (`graphrag.py`)
- **Purpose**: Structured knowledge retrieval
- **Dataset**: 51,301+ knowledge triples
- **Components**:
  - `KnowledgeGraph`: Triple storage
  - `GraphRetriever`: Semantic search
  - `LLMAnswerer`: GPT-4 answering
  - `GraphRAG`: Unified interface

**Public API**:
```python
from src.knowledge import StereoSetRAG, DiversityRAG, GraphRAG

# StereoSet
stereoset = StereoSetRAG()
examples = stereoset.get_negative_examples("doctor", top_k=5)

# Diversity
diversity = DiversityRAG()
recommendations = diversity.get_recommendations("engineer", top_k=5)

# GraphRAG
graphrag = GraphRAG()
results = graphrag.query("What stereotypes exist?", top_k=10)
```

---

### Layer 5: Generation Layer (`src/generation/`)

**Purpose**: Image generation with multiple backends

**Components**:
- `base.py`: Abstract base generator
- `dalle.py`: DALL-E 3 implementation
- `mock.py`: Mock generator for testing
- `factory.py`: Generator factory

**Architecture**:
```python
BaseImageGenerator (Abstract)
├── DALLE3Generator (Production)
└── MockImageGenerator (Testing)
```

**Public API**:
```python
from src.generation import create_image_generator, DALLE3Generator

# Factory pattern
generator = create_image_generator("dalle3", output_dir="images/")

# Direct instantiation
dalle = DALLE3Generator(output_dir="images/", quality="hd")

# Generate image
result = generator.generate(
    prompt="a diverse group of doctors",
    filename="doctor_001.png"
)
```

**DALL-E 3 Configuration**:
- Size: 1024x1024 (configurable: 1024x1024, 1024x1792, 1792x1024)
- Quality: standard/hd
- Style: natural/vivid
- Automatic retry with exponential backoff

**Error Handling**:
- Rate limiting with backoff
- Invalid prompt detection
- Content policy violations
- Network errors

---

### Layer 6: Evaluation Layer (`src/evaluation/`)

**Purpose**: Visual bias and diversity evaluation

**Components**:
- `visual_bias_evaluator.py`: Comprehensive demographic analysis
  - Face detection (dlib)
  - Demographic prediction (FairFace ResNet34)
  - Bias metrics calculation

**Metrics**:

1. **Bias-W (Population-level Bias)**
   ```
   bias_w = Σ |predicted_i - reference_i| / 2
   Range: 0.0 (perfect match) to 1.0 (complete mismatch)
   ```

2. **Bias-P (Per-image Bias)**
   ```
   Average deviation from expected distribution per image
   ```

3. **ENS (Effective Number of Species)**
   ```
   ENS = exp(-Σ p_i * log(p_i))
   Diversity measure: higher is more diverse
   ```

4. **KL Divergence**
   ```
   KL(P||Q) = Σ P(i) * log(P(i)/Q(i))
   Distribution difference from reference
   ```

**Public API**:
```python
from src.evaluation import VisualBiasEvaluator

evaluator = VisualBiasEvaluator(
    fairface_model_path="models/fairface.pth",
    dlib_model_path="models/shape_predictor.dat"
)

# Evaluate single image
result = evaluator.evaluate_image("image.png")

# Evaluate batch
results = evaluator.evaluate_batch([
    "image1.png",
    "image2.png"
])

# Get metrics
metrics = evaluator.calculate_bias_metrics(results)
```

**Demographic Categories**:
- **Race**: 7 categories (White, Black, Asian, Indian, Middle Eastern, Latino, Southeast Asian)
- **Gender**: Male/Female
- **Age**: 9 groups (0-2, 3-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70+)

---

### Layer 7: Enhancement Layer (`src/enhancement/`)

**Purpose**: Unified prompt enhancement system

**Components**:
- `prompt_enhancer.py`: Main enhancement system
  - `EnhancementConfig`: Configuration dataclass
  - `EnhancementResult`: Result dataclass
  - `EnhancementSystem`: Main orchestrator

**Enhancement Flow**:
```
1. Load prompt and configuration
2. Retrieve negative examples (StereoSetRAG)
3. Retrieve diversity recommendations (DiversityRAG)
4. Retrieve knowledge context (GraphRAG)
5. Generate enhancement with GPT-4
6. Score bias mitigation (0-100)
7. Score diversity (0-100)
8. Check thresholds
9. If not met and iterations remain → iterate
10. Return best result
```

**Scoring Criteria**:

**Bias Score (0-100)**:
- Inclusive language: 25 points
- Stereotype avoidance: 25 points
- Neutral framing: 25 points
- Respectful descriptions: 25 points

**Diversity Score (0-100)**:
- Age diversity: 15 points
- Ethnic/racial diversity: 20 points
- Gender diversity: 15 points
- Cultural diversity: 20 points
- Ability inclusion: 10 points
- Socioeconomic diversity: 10 points
- Specificity: 10 points

**Public API**:
```python
from src.enhancement import EnhancementSystem, EnhancementConfig

# Create configuration
config = EnhancementConfig(
    bias_threshold=75,
    diversity_threshold=80,
    max_iterations=5
)

# Create system
system = EnhancementSystem(
    config=config,
    stereoset_rag=stereoset,
    diversity_rag=diversity,
    graphrag=graphrag
)

# Enhance prompt
result = system.enhance("a doctor")

# Access results
print(f"Original: {result.original_prompt}")
print(f"Enhanced: {result.final_prompt}")
print(f"Bias improvement: +{result.bias_improvement:.1f}")
print(f"Diversity improvement: +{result.diversity_improvement:.1f}")
```

**Configuration Options**:
```python
EnhancementConfig(
    # RAG system toggles
    use_stereoset: bool = True,
    use_diversity_rag: bool = True,
    use_graphrag: bool = True,
    
    # Thresholds
    bias_threshold: int = 75,
    diversity_threshold: int = 80,
    
    # Iteration settings
    max_iterations: int = 5,
    
    # Context settings
    num_negative_examples: int = 5,
    num_diversity_recommendations: int = 5,
    
    # Model settings
    model: str = "gpt-4-turbo-preview",
    temperature: float = 0.7,
    max_tokens: int = 500
)
```

---

### Layer 8: CLI Layer (`src/cli/`)

**Purpose**: Command-line interface for all functionality

**Components**:
- `runner.py`: CLI runner and dispatcher
- `parser.py`: Argument parser
- `commands/`: Command implementations
  - `__init__.py`: BaseCommand abstract class
  - `enhance.py`: EnhanceCommand
  - `batch.py`: BatchCommand
  - `query.py`: QueryCommand
  - `test.py`: TestCommand

**Command Architecture**:
```python
BaseCommand (Abstract)
├── EnhanceCommand: Single prompt enhancement
├── BatchCommand: CSV batch processing
├── QueryCommand: Knowledge graph queries
└── TestCommand: Testing and demos
```

**Public API**:
```python
from src.cli import CLIRunner, create_cli_parser

# Create parser and runner
parser = create_cli_parser()
runner = CLIRunner()

# Run CLI
exit_code = runner.main()
```

**Command Usage**:

```bash
# Enhance command
python main.py enhance "a doctor"
python main.py enhance "a doctor" --bias-threshold 80

# Batch command
python main.py batch prompts.csv
python main.py batch prompts.csv --enable-visual-bias

# Query command
python main.py query "What stereotypes exist?"
python main.py query --data-source cultural

# Test command
python main.py test --quick
python main.py test --all
```

---

## Data Flow

### Complete System Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    CLI LAYER (src/cli/)                         │
│  User Input → Command Parsing → Command Execution               │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              ENHANCEMENT LAYER (src/enhancement/)               │
│  Prompt → RAG Retrieval → GPT-4 Enhancement → Scoring           │
└────────┬──────────────────────────┬─────────────────────────────┘
         │                          │
         ▼                          ▼
┌──────────────────────┐   ┌────────────────────────────────────┐
│  EVALUATION LAYER    │   │    KNOWLEDGE LAYER                 │
│  (src/evaluation/)   │   │    (src/knowledge/)                │
│  Visual Analysis     │   │  StereoSet, Diversity, GraphRAG    │
└──────┬───────────────┘   └────────┬───────────────────────────┘
       │                            │
       ▼                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              GENERATION LAYER (src/generation/)                 │
│  Prompt → DALL-E 3 API → Image Download → Storage               │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                 DATA LAYER (src/data/)                          │
│  Embeddings, Caching, Parsing                                   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              PROMPTS & CONFIG LAYERS                            │
│  Templates, Settings, Paths                                     │
└─────────────────────────────────────────────────────────────────┘
```

### Example: Batch Processing Flow

```
1. CLI Layer: Parse batch command arguments
   ↓
2. Batch Command: Load CSV file
   ↓
3. For each prompt:
   ↓
   3a. Enhancement Layer: Enhance prompt
       ↓
       - Knowledge Layer: Retrieve from RAG systems
       - Prompts Layer: Load enhancement templates
       - Data Layer: Get embeddings from cache
       ↓
   3b. Generation Layer: Generate images
       ↓
       - Original prompt → DALL-E 3
       - Enhanced prompt → DALL-E 3
       ↓
   3c. Evaluation Layer: Analyze images (optional)
       ↓
       - Detect faces
       - Predict demographics
       - Calculate metrics
   ↓
4. Collect all results
   ↓
5. Save to CSV and JSON
```

## Technology Stack

### Core Technologies
- **Python 3.13+**: Modern Python with type hints
- **Pydantic**: Data validation and settings management
- **OpenAI GPT-4**: Enhancement and scoring
- **OpenAI Embeddings**: text-embedding-3-small

### Key Libraries
- **pandas**: CSV processing and data manipulation
- **numpy**: Numerical computing
- **networkx**: Graph data structures (GraphRAG)
- **scikit-learn**: Similarity calculations
- **PyYAML**: Configuration and templates
- **PIL (Pillow)**: Image processing

### Visual Evaluation Stack
- **PyTorch**: Deep learning framework
- **FairFace**: ResNet34 demographic classifier
- **dlib**: Face detection
- **OpenCV**: Image preprocessing

### Data Storage
- **Pickle**: Embedding cache (persistent)
- **CSV**: Input/output data
- **JSON**: Structured results and metadata
- **PNG**: Generated images

## Design Patterns

### Layered Architecture
- Clear separation of concerns
- Unidirectional dependencies
- Easy to test and maintain

### Factory Pattern
- `create_image_generator()`: Image generator creation
- `DataParserFactory`: Parser selection
- Benefits: Easy extension, loose coupling

### Strategy Pattern
- Enhancement strategies (EnhancementSystem)
- Scoring algorithms
- Benefits: Runtime algorithm selection

### Template Method Pattern
- `BaseCommand`: CLI command structure
- `BaseImageGenerator`: Image generator interface
- Benefits: Consistent interface, customizable behavior

### Dataclass Pattern
- `EnhancementConfig`: Configuration
- `EnhancementResult`: Results
- `ProcessingResult`: Batch results
- Benefits: Type safety, immutability, clarity

### Dependency Injection
- Components receive dependencies
- No hard-coded dependencies
- Benefits: Testability, flexibility

## Performance & Scalability

### Performance Optimizations

1. **Embedding Cache**
   - 85-90% hit rate
   - ~50KB per 1000 embeddings
   - <100ms load time for 10K embeddings

2. **Lazy Loading**
   - RAG systems loaded on demand
   - Models loaded only when needed

3. **Batch Processing**
   - Process multiple prompts in single session
   - Reuse initialized systems

4. **Top-K Retrieval**
   - Limit search space
   - Balance quality vs. performance

### Scalability Considerations

**Current Scale**:
- Prompts: 10-1000 per batch
- Knowledge: 51K+ triples
- Images: Hundreds per session

**Future Scale**:
- Database backend for millions of triples
- Distributed processing with Celery
- Caching layer with Redis
- API rate limiting and queuing

### Resource Management
- Graceful error handling
- API retry logic with exponential backoff
- Memory-efficient streaming
- Comprehensive logging

## Security & Ethics

### API Key Management
- Environment variables only
- No hardcoded credentials
- Secure initialization

### Data Privacy
- Local data storage
- No cloud storage of sensitive data
- User controls all generated content

### Bias Mitigation Philosophy
- Multi-source validation
- Transparent scoring
- Human oversight supported
- Continuous improvement

## Testing Strategy

### Test Coverage
- **Unit Tests**: Each layer independently
- **Integration Tests**: Layer interactions
- **End-to-End Tests**: Complete workflows
- **Phase Tests**: Each refactoring phase validated

### Test Structure
```
test_phase1.py: Config layer (16 tests)
test_phase2.py: Prompts layer (7 tests)
test_phase3.py: Data layer (5 tests)
test_phase4.py: Knowledge layer (7 tests)
test_phase5.py: Generation layer (8 tests)
test_phase6.py: Evaluation layer (7 tests)
test_phase7.py: Enhancement layer (7 tests)
test_phase8.py: CLI layer (9 tests)
```

**Total: 66 integration tests, all passing**

## Migration Guide

### Old Structure → New Structure

```
Old                              New
──────────────────────          ─────────────────────────
main.py                      →  src/cli/runner.py
enhance_prompt_*.py          →  src/enhancement/
batch_processor.py           →  src/cli/commands/batch.py
image_generator.py           →  src/generation/
visual_bias_evaluator.py     →  src/evaluation/
stereoset_rag.py            →  src/knowledge/stereoset.py
diversity_rag.py            →  src/knowledge/diversity.py
graphrag.py                 →  src/knowledge/graphrag.py
parsers.py                  →  src/data/parsers.py
```

### Import Updates

**Before**:
```python
from stereoset_rag import StereoSetRAG
from diversity_rag import DiversityRAG
from graphrag import GraphRAG
from image_generator import create_image_generator
```

**After**:
```python
from src.knowledge import StereoSetRAG, DiversityRAG, GraphRAG
from src.generation import create_image_generator
from src.enhancement import EnhancementSystem
from src.evaluation import VisualBiasEvaluator
```

## Future Enhancements

### Planned Features
1. **Web Interface**: REST API + React frontend
2. **Database Backend**: PostgreSQL for enterprise scale
3. **Real-time Monitoring**: Metrics dashboard
4. **Model Customization**: Domain-specific fine-tuning
5. **Multi-modal Support**: Video and audio analysis

### Integration Opportunities
- CI/CD pipelines for content validation
- CMS integration for production use
- Analytics platforms for insights
- Collaboration tools for team workflows

## Documentation

### Available Documentation
- `ARCHITECTURE.md`: This document
- `USER_GUIDE.md`: End-user documentation
- `DEVELOPER_GUIDE.md`: Development guide
- `CLI_REFERENCE.md`: CLI command reference
- `PHASE*_COMPLETION.md`: Refactoring documentation
- `VISUAL_BIAS_EVALUATION.md`: Evaluation details

### Code Documentation
- Comprehensive docstrings
- Type hints throughout
- Inline comments for complex logic
- Examples in docstrings

---

**Architecture Version:** 2.0  
**Last Updated:** October 2025  
**Status:** Production Ready  
**Test Coverage:** 66 integration tests passing
