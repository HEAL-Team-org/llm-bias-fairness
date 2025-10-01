# System Architecture

**LLM Bias & Fairness Project - Technical Architecture Overview**

## Overview

The LLM Bias & Fairness Project is a comprehensive system for bias mitigation and diversity enhancement in AI-generated image prompts. The architecture is modular, extensible, and built on three core knowledge systems combined through a dual-pipeline enhancement approach.

## System Components

### 1. Core Pipeline (`batch_processor.py`)

The central orchestration system that manages the complete workflow:

```
Input CSV → Enhancement → Image Generation → Visual Evaluation → Results
```

**Key Features:**
- Batch processing of multiple prompts
- Configurable enhancement methods
- Generic image generation interface
- Optional visual bias evaluation
- Comprehensive result tracking (CSV + JSON)

**ProcessingResult Structure:**
```python
@dataclass
class ProcessingResult:
    row_index: int
    original_prompt: str
    enhanced_prompt: Optional[str]
    bias_score: Optional[float]          # 0-100 scale
    diversity_score: Optional[float]     # 0-100 scale
    original_image_path: Optional[str]
    enhanced_image_path: Optional[str]
    visual_bias_evaluation: Optional[Dict]
    enhancement_metadata: Optional[Dict]
    processing_time: Optional[float]
    error: Optional[str]
```

### 2. Knowledge Systems (RAG Components)

#### StereoSet RAG (`src/stereoset_rag.py`)
- **Purpose**: Stereotype detection and bias mitigation
- **Dataset**: McGill-NLP StereoSet (4,229 stereotype examples)
- **Method**: Vector similarity search on stereotype contexts
- **Output**: Identified stereotypical patterns to avoid

**Example Record:**
```python
{
    "context": "The Asian student was...",
    "sentence_type": "stereotype",
    "target": "Asian",
    "bias_type": "race"
}
```

#### CultureBank RAG (`src/diversity_rag.py`)
- **Purpose**: Cultural diversity enhancement
- **Dataset**: CultureBank Reddit/TikTok (22,990 cultural behaviors)
- **Method**: Weighted retrieval (vector + keyword + agreement scores)
- **Output**: Culturally diverse behavior patterns

**Weighting Formula:**
```
score = α × similarity + β × keyword_match + γ × agreement
where α=0.6, β=0.3, γ=0.1
```

**Example Record:**
```python
{
    "cultural_group": "South Korea",
    "actor_behavior": "showing respect through formal language",
    "topic": "workplace interaction",
    "agreement": 0.85
}
```

#### GraphRAG (`src/graphrag.py`)
- **Purpose**: Structured knowledge retrieval for bias patterns and cultural values
- **Dataset**: 51,301+ knowledge triples from multiple sources
- **Method**: Graph-based semantic triple similarity
- **Output**: Relevant bias patterns and cultural insights

**Knowledge Sources:**
1. Bias CSV files (ADV_GRAPH dataset)
2. Cultural value triples (15 regions: US, UK, China, India, Iran, etc.)

**Components:**
- `EmbeddingCache`: Persistent caching for efficiency
- `OpenAIEmbedder`: Embedding generation
- `KnowledgeGraph`: Triple storage and graph structure
- `GraphRetriever`: Semantic triple similarity search
- `LLMAnswerer`: GPT-4 based question answering

### 3. Enhancement Systems

#### Dual-Pipeline Enhancement (`enhance_prompt_dual_pipeline.py`)
The **recommended** enhancement system with separate bias and diversity scoring.

**Flow:**
```
1. Load all three knowledge systems (StereoSet, CultureBank, GraphRAG)
2. Retrieve relevant context from each system
3. Combine context with weighted importance
4. Generate enhanced prompt with GPT-4
5. Score bias mitigation (0-100)
6. Score diversity enhancement (0-100)
7. If both thresholds met → SUCCESS
8. If not and iterations remain → Iterate with feedback
9. Return best enhancement
```

**Bias Scoring Criteria (0-100):**
- Inclusive language and terminology
- Anti-bias wording and framing
- Stereotype avoidance and counteraction
- Neutral, respectful descriptions

**Diversity Scoring Criteria (0-100):**
- Age diversity (15 points): Multiple age groups
- Ethnic/racial diversity (20 points): Various backgrounds
- Gender diversity (15 points): Inclusive representation
- Cultural diversity (20 points): Multiple cultures/regions
- Ability inclusion (10 points): Different abilities
- Socioeconomic diversity (10 points): Various backgrounds
- Specificity & actionability (10 points): Clear, detailed descriptions

**Configuration:**
```python
--bias-threshold 75        # Minimum bias mitigation score
--diversity-threshold 80   # Minimum diversity score
--max-iterations 5         # Maximum enhancement cycles
```

#### Sequential Enhancement (`enhance_prompt_sequential.py`)
Iterative improvement with diversity-focused scoring.

**Difference from Dual-Pipeline:**
- Single diversity score (no separate bias score)
- Simpler configuration
- Faster processing

#### Basic Enhancement (`enhance_prompt.py`)
Single-pass enhancement for quick improvements.

**Use When:**
- Speed is critical
- Simple prompts
- Basic diversity needs

### 4. Image Generation (`src/image_generator.py`)

Generic interface supporting multiple generators through factory pattern.

**Architecture:**
```python
BaseImageGenerator (Abstract)
├── DALLE3ImageGenerator (DALL-E 3 API)
└── MockImageGenerator (Testing/Development)
```

**DALL-E 3 Implementation:**
- Size: 1024x1024 (configurable)
- Quality: Standard/HD options
- Style: Natural/Vivid options
- Error handling and retries
- Automatic image download and storage

**File Naming Convention:**
```
{row_index:04d}_{original|enhanced}.png
Example: 0001_original.png, 0001_enhanced.png
```

### 5. Visual Bias Evaluation (`src/visual_bias_evaluator.py`)

**NEW** component for comprehensive demographic analysis of generated images.

**Architecture:**
```
Face Detection (dlib) → Demographic Prediction (FairFace) → Bias Metrics
```

**Components:**
- **Face Detection**: dlib's 5-point facial landmarks
- **Demographic Prediction**: FairFace ResNet34 model
  - Race: 7 categories (White, Black, Asian, Indian, Middle Eastern, Latino, Southeast Asian)
  - Gender: Male/Female
  - Age: 9 groups (0-2, 3-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70+)

**Bias Metrics:**

1. **Bias-W (Population-level Bias)**
   ```
   bias_w = Σ |p_i - r_i| / 2
   where p_i = predicted proportion, r_i = reference proportion
   ```

2. **Bias-P (Per-image Bias)**
   ```
   Measures average deviation from expected distribution per image
   ```

3. **ENS (Diversity via Shannon Entropy)**
   ```
   ENS = -Σ p_i * log(p_i)
   Higher values indicate greater diversity
   ```

4. **KL Divergence**
   ```
   KL(P||Q) = Σ P(i) * log(P(i)/Q(i))
   Measures distribution difference from reference
   ```

**Output Structure:**
```json
{
  "bias_w": 0.15,
  "bias_p": 0.23,
  "ens": 1.85,
  "kl_divergence": 0.12,
  "total_faces": 45,
  "images_with_faces": 38,
  "demographic_distribution": {
    "race": {...},
    "gender": {...},
    "age": {...}
  }
}
```

### 6. Data Parsers (`src/parsers.py`)

Extensible parser system with factory pattern.

**Parser Types:**
- `BiasCSVParser`: ADV_GRAPH bias dataset
- `CulturalTriplesParser`: Cultural value text files
- `StereoSetParser`: StereoSet JSON dataset
- `DataParserFactory`: Automatic format detection

**Factory Usage:**
```python
parser = DataParserFactory.create_parser(file_path)
triples = parser.parse()
```

## Data Flow

### Complete Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                      INPUT: CSV FILE                            │
│              (prompts, metadata, configurations)                │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  BATCH PROCESSOR INITIALIZATION                 │
│  • Load enhancement systems (StereoSet, CultureBank, GraphRAG)  │
│  • Initialize image generator (DALL-E 3 / Mock)                 │
│  • Initialize visual bias evaluator (optional)                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                FOR EACH PROMPT IN CSV:                          │
├─────────────────────────────────────────────────────────────────┤
│  Step 1: ENHANCEMENT                                            │
│  ┌────────────────────────────────────────────────┐             │
│  │ • Retrieve from StereoSet (bias patterns)      │             │
│  │ • Retrieve from CultureBank (diversity)        │             │
│  │ • Retrieve from GraphRAG (cultural knowledge)  │             │
│  │ • Combine weighted context                     │             │
│  │ • Generate enhancement with GPT-4              │             │
│  │ • Score bias (0-100) and diversity (0-100)     │             │
│  │ • Iterate until thresholds met                 │             │
│  └────────────────────────────────────────────────┘             │
│                           │                                     │
│  Step 2: IMAGE GENERATION                                       │
│  ┌────────────────────────────────────────────────┐             │
│  │ • Generate image from ORIGINAL prompt          │             │
│  │ • Generate image from ENHANCED prompt          │             │
│  │ • Save with trackable filenames                │             │
│  │ • Capture metadata (model, size, timing)       │             │
│  └────────────────────────────────────────────────┘             │
│                           │                                     │
│  Step 3: VISUAL BIAS EVALUATION (Optional)                      │
│  ┌────────────────────────────────────────────────┐             │
│  │ • Detect faces in both images                  │             │
│  │ • Predict demographics (race, gender, age)     │             │
│  │ • Calculate bias metrics (Bias-W, Bias-P, ENS) │             │
│  │ • Compare original vs enhanced                 │             │
│  └────────────────────────────────────────────────┘             │
│                           │                                     │
│  Step 4: RESULT COLLECTION                                      │
│  ┌────────────────────────────────────────────────┐             │
│  │ • Create ProcessingResult object               │             │
│  │ • Store all metadata and scores                │             │
│  │ • Handle errors gracefully                     │             │
│  └────────────────────────────────────────────────┘             │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    OUTPUT GENERATION                            │
│  • Save results to CSV (enhanced_prompts_[timestamp].csv)       │
│  • Save results to JSON (enhanced_prompts_[timestamp].json)     │
│  • Save visual evaluation results (if enabled)                  │
│  • Generate summary statistics                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Technology Stack

### Core Technologies
- **Python 3.13+**: Modern Python with type hints
- **OpenAI GPT-4**: Enhancement generation and scoring
- **OpenAI Embeddings**: text-embedding-3-small for vector similarity

### Key Libraries
- **pandas**: CSV processing and data manipulation
- **numpy**: Numerical computing for embeddings and metrics
- **networkx**: Graph data structures for GraphRAG
- **scikit-learn**: Cosine similarity calculations
- **PIL (Pillow)**: Image processing
- **requests**: API calls and image downloads

### Visual Bias Evaluation Stack
- **PyTorch**: Deep learning framework
- **FairFace Model**: ResNet34-based demographic classifier
- **dlib**: Face detection and alignment
- **OpenCV**: Image preprocessing

### Data Storage
- **Pickle**: Embedding cache persistence
- **CSV**: Input prompts and output results
- **JSON**: Detailed metadata and structured results
- **PNG**: Generated images

## Design Patterns

### Factory Pattern
Used for:
- Image generator creation (`create_image_generator()`)
- Data parser selection (`DataParserFactory`)

**Benefit**: Easy extension for new generators/parsers without modifying existing code.

### Strategy Pattern
Used for:
- Different enhancement methods (dual-pipeline, sequential, basic)
- Scoring algorithms (bias vs. diversity)

**Benefit**: Flexible algorithm selection at runtime.

### Template Method Pattern
Used for:
- Base classes with abstract methods (`BaseImageGenerator`, `BaseDataParser`)

**Benefit**: Consistent interface with customizable implementation.

### Cache Pattern
Used for:
- Embedding storage (`EmbeddingCache`)
- Dataset loading (pickle files)

**Benefit**: Performance optimization and cost reduction.

## Scalability Considerations

### Performance Optimizations
1. **Embedding Cache**: Persistent storage prevents redundant API calls
2. **Batch Processing**: Process multiple prompts in single session
3. **Lazy Loading**: Knowledge systems loaded only when needed
4. **Efficient Retrieval**: Top-K search limits processing overhead

### Extensibility Points
1. **New Image Generators**: Extend `BaseImageGenerator`
2. **New Knowledge Sources**: Add to GraphRAG datasets
3. **New Parsers**: Implement `BaseDataParser`
4. **Custom Metrics**: Extend bias/diversity scoring
5. **New Evaluation Methods**: Add to visual bias evaluator

### Resource Management
- **API Rate Limiting**: Built-in retry logic
- **Memory Management**: Streaming for large datasets
- **Error Handling**: Graceful degradation with fallbacks
- **Logging**: Comprehensive debugging information

## Security & Ethics

### API Key Management
- Environment variables for sensitive data
- No hardcoded credentials
- Secure API client initialization

### Data Privacy
- No personal data storage
- Generated images stored locally
- User control over all data

### Bias Mitigation Approach
- Multi-source validation
- Transparent scoring methodology
- Human-in-the-loop verification supported
- Continuous improvement through evaluation

## Future Architecture Considerations

### Potential Enhancements
1. **Database Backend**: PostgreSQL for large-scale deployments
2. **Distributed Processing**: Celery for async job processing
3. **Web Interface**: REST API and frontend
4. **Model Customization**: Fine-tuned models for specific domains
5. **Real-time Monitoring**: Metrics dashboard and alerting
6. **Multi-modal Evaluation**: Video and audio bias analysis

### Integration Points
- **CI/CD Pipelines**: Automated testing of generated content
- **Content Management Systems**: Direct integration
- **Analytics Platforms**: Export metrics and insights
- **Collaboration Tools**: Team-based review workflows
