# User Guide

**LLM Bias & Fairness Project - Complete Usage Guide**

## Table of Contents
1. [Quick Start](#quick-start)
2. [Unified CLI Overview](#unified-cli-overview)
3. [Enhancement Commands](#enhancement-commands)
4. [Batch Processing](#batch-processing)
5. [Knowledge Queries](#knowledge-queries)
6. [Testing Commands](#testing-commands)
7. [Visual Bias Evaluation](#visual-bias-evaluation)
8. [Configuration Options](#configuration-options)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

## Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd llm-bias-fairness

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set OpenAI API key
export OPENAI_API_KEY="sk-your-api-key-here"
```

### Your First Enhancement

```bash
# Interactive mode - easiest way to get started
python main.py enhance

# You'll be prompted to enter your image prompt
# Example: "a doctor examining a patient"

# Or directly enhance a prompt
python main.py enhance "a doctor examining a patient"
```

## Unified CLI Overview

All project functionality is now available through a single unified command-line interface:

```
python main.py <command> [options]
```

### Available Commands

| Command | Description | Use Case |
|---------|-------------|----------|
| `enhance` | Enhance single prompts | Interactive or direct prompt enhancement |
| `batch` | Process CSV files | Bulk processing with image generation |
| `query` | Query knowledge graphs | Research bias patterns and cultural values |
| `test` | Run tests | Quick tests, batch tests, visual evaluation |

### Getting Help

```bash
# General help
python main.py --help

# Command-specific help
python main.py enhance --help
python main.py batch --help
python main.py query --help
python main.py test --help
```

## Enhancement Commands

The system offers three enhancement modes for different needs.

### Enhancement Mode Overview

| Mode | Speed | Quality | Iterations | Best For |
|------|-------|---------|------------|----------|
| `basic` | ⚡⚡⚡ Fast | Good | 1 (single-pass) | Quick improvements, simple prompts |
| `sequential` | ⚡⚡ Medium | Very Good | 1-5 (diversity threshold) | Iterative improvement, diversity focus |
| `dual-pipeline` | ⚡ Slower | Excellent | 1-5 (bias + diversity thresholds) | Production use, comprehensive quality ⭐ |

### 1. Basic Enhancement (Fast Single-Pass)

**When to Use:**
- Speed is critical
- Simple prompts
- Quick prototyping

**Usage:**

```bash
# Interactive mode
python main.py enhance --mode basic

# Direct enhancement
python main.py enhance "students in classroom" --mode basic

# With custom retrieval settings
python main.py enhance "a teacher" --mode basic --bias-top-k 30 --cultural-top-k 30
```

**What It Does:**
- Retrieves relevant bias and cultural information
- Single-pass enhancement with GPT-4
- No quality scoring or iteration

**Example:**
```bash
python main.py enhance "a doctor" --mode basic
```

### 2. Sequential Enhancement (Iterative with Diversity)

**When to Use:**
- Need diversity guarantees
- Willing to wait for iterations
- Focus on diversity over bias mitigation

**Usage:**

```bash
# Interactive mode
python main.py enhance --mode sequential

# With custom diversity threshold
python main.py enhance "engineers working" --mode sequential --threshold 85

# With more iterations allowed
python main.py enhance "a business meeting" --mode sequential --threshold 80 --max-iterations 5
```

**What It Does:**
- Retrieves relevant bias and cultural information
- Iteratively enhances until diversity threshold met
- Scores diversity (0-100) on 7 dimensions:
  - Age diversity (15 points)
  - Ethnic diversity (20 points)
  - Gender diversity (15 points)
  - Cultural diversity (20 points)
  - Ability inclusion (10 points)
  - Socioeconomic diversity (10 points)
  - Specificity (10 points)

**Example:**
```bash
python main.py enhance "students in classroom" --mode sequential --threshold 85 --max-iterations 4
```

### 3. Dual-Pipeline Enhancement ⭐ **RECOMMENDED**

The most advanced method with separate bias mitigation and diversity enhancement.

**When to Use:**
- Production image generation
- Maximum quality needed
- Comprehensive diversity AND bias mitigation
- Research and analysis

**Usage:**

```bash
# Interactive mode (uses dual-pipeline by default)
python main.py enhance

# Direct enhancement with defaults
python main.py enhance "a doctor examining a patient"

# Explicit dual-pipeline mode
python main.py enhance "a teacher" --mode dual-pipeline

# Custom thresholds for strict quality
python main.py enhance "business meeting" \
    --bias-threshold 85 \
    --diversity-threshold 85 \
    --max-iterations 5

# Disable specific knowledge sources (not recommended)
python main.py enhance "engineers working" \
    --no-use-stereoset \
    --no-use-graphrag
```

**What It Does:**
- Uses THREE knowledge sources:
  - **StereoSet RAG**: Stereotype detection (4,229 examples)
  - **CultureBank RAG**: Diversity enhancement (22,990 behaviors)
  - **GraphRAG**: Cultural knowledge (51,301+ triples)
- Separate scoring for bias mitigation (0-100) AND diversity (0-100)
- Iterates until BOTH thresholds are met
- Provides detailed score breakdowns

**Key Parameters:**
- `--bias-threshold`: Minimum bias mitigation score (0-100, default: 75)
- `--diversity-threshold`: Minimum diversity score (0-100, default: 80)
- `--max-iterations`: Maximum enhancement cycles (default: 3)

**Output:**
```
🎯 Dual-Pipeline Enhancement Results
================================================================================
Original Prompt: students in a classroom

Enhanced Prompt: In a diverse learning environment, students of various ages,
ethnicities, and abilities collaborate together. The classroom includes
wheelchair-accessible spaces, and students use different learning methods...

📊 SCORING BREAKDOWN:
Bias Mitigation: 82/100 ✅ (threshold: 75)
  • Inclusive language: ✓
  • Stereotype avoidance: ✓
  • Neutral descriptions: ✓

Diversity Enhancement: 87/100 ✅ (threshold: 80)
  • Age diversity: 13/15
  • Ethnic diversity: 18/20
  • Gender diversity: 14/15
  • Cultural diversity: 17/20
  • Ability inclusion: 10/10
  • Socioeconomic: 8/10
  • Specificity: 7/10

Iterations: 2
Processing Time: 18.5 seconds
```

**Example:**
```bash
python main.py enhance "a doctor" --bias-threshold 80 --diversity-threshold 85
```

### Real-World Enhancement Examples

#### Example 1: Medical Professional

**Scenario:** Creating diverse medical imagery for healthcare campaign

```bash
python main.py enhance "a doctor examining a patient" \
    --bias-threshold 80 \
    --diversity-threshold 85
```

**Original Prompt:**
```
a doctor examining a patient
```

**Enhanced Prompt:**
```
In a modern medical facility, Dr. Ji-Yeon Kim, a compassionate physician 
of South Korean descent, attentively examines an elderly African American 
patient using state-of-the-art diagnostic equipment. The clinic serves 
a diverse community including Middle Eastern refugees, young Latinx families, 
and patients of various socioeconomic backgrounds. Dr. Kim works alongside 
Dr. Aisha Patel, a wheelchair-using specialist from India, ensuring their 
practice embraces true inclusivity across age, ethnicity, gender expression, 
and physical abilities.
```

**Scores:**
- Bias Mitigation: 88/100 ✅
- Diversity: 92/100 ✅
- Iterations: 2

#### Example 2: Educational Setting

**Scenario:** Diverse classroom imagery for textbook publisher

```bash
python main.py enhance "students studying in a classroom" \
    --mode sequential \
    --threshold 85
```

**Original Prompt:**
```
students studying in a classroom
```

**Enhanced Prompt:**
```
A vibrant classroom filled with learners spanning elementary to high school 
age, representing diverse ethnic backgrounds including East Asian, African, 
Middle Eastern, and Indigenous students. Some use wheelchairs while others 
sit at traditional desks. Students of all gender expressions collaborate 
on projects, with some wearing cultural attire including hijabs and turbans. 
The learning space accommodates various learning styles with visual aids, 
audio equipment, and hands-on materials, reflecting both urban and rural 
educational experiences.
```

**Scores:**
- Diversity: 89/100 ✅
- Iterations: 3

#### Example 3: Professional Workplace

**Scenario:** Corporate diversity imagery for annual report

```bash
python main.py enhance "business executives in a meeting"
```

**Original Prompt:**
```
business executives in a meeting
```

**Enhanced Prompt:**
```
A boardroom meeting brings together executives from varied professional 
backgrounds: a young Black woman in her 30s presenting data, a Middle Eastern 
man in his 50s contributing insights, a South Asian person using sign language 
with an interpreter present, and a white senior executive in a wheelchair 
reviewing documents. The diverse leadership team represents multiple 
generations (millennials to senior executives), cultural perspectives 
(wearing both Western business attire and cultural professional wear), 
and career trajectories from various socioeconomic origins.
```

**Scores:**
- Bias Mitigation: 85/100 ✅
- Diversity: 88/100 ✅
- Iterations: 2

#### Example 4: Tech Industry

**Scenario:** Diverse tech workforce for recruitment campaign

```bash
python main.py enhance "software engineers coding" \
    --bias-threshold 85 \
    --diversity-threshold 90 \
    --max-iterations 5
```

**Original Prompt:**
```
software engineers coding
```

**Enhanced Prompt:**
```
A collaborative tech workspace where software engineers of diverse backgrounds 
contribute their expertise: a neurodivergent developer from East Asia wearing 
noise-canceling headphones, a Black woman engineer explaining code architecture, 
a Latina developer in her 40s mentoring younger colleagues, a non-binary 
programmer from the Middle East, and a white male engineer who is hard of 
hearing using assistive technology. The team represents varied educational 
paths including traditional computer science degrees, coding bootcamps, and 
self-taught programmers from different socioeconomic backgrounds.
```

**Scores:**
- Bias Mitigation: 91/100 ✅
- Diversity: 94/100 ✅
- Iterations: 4

#### Example 5: Food & Culinary

**Scenario:** Diverse chef imagery for cooking show

```bash
python main.py enhance "a chef preparing food" --mode basic
```

**Original Prompt:**
```
a chef preparing food
```

**Enhanced Prompt:**
```
A skilled chef from Southeast Asia demonstrating culinary techniques, 
incorporating ingredients and cooking methods from multiple cultural 
traditions including South American, African, and Asian cuisines. 
The kitchen environment welcomes diverse culinary professionals of 
various ages and backgrounds, showcasing inclusive food culture.
```

**Scores:**
- Single-pass completion
- Processing time: 8 seconds

### Command Quick Reference

```bash
# Quick & simple
python main.py enhance "prompt" --mode basic

# Diversity-focused with iterations
python main.py enhance "prompt" --mode sequential --threshold 85

# Comprehensive quality (recommended)
python main.py enhance "prompt"  # uses dual-pipeline by default

# Strict quality control
python main.py enhance "prompt" \
    --bias-threshold 85 \
    --diversity-threshold 90 \
    --max-iterations 5

# Interactive (guided)
python main.py enhance
```

## Batch Processing

Process multiple prompts from a CSV file with automatic image generation and optional visual evaluation.

### Basic Batch Processing

**1. Prepare Your CSV File**

Create a CSV file with prompts (example: `prompts.csv`):

```csv
prompt
a doctor examining a patient
students in a classroom
engineers working on a project
a chef preparing food
artists creating artwork
```

**2. Run Batch Processing**

```bash
# Basic batch processing
python main.py batch prompts.csv

# Process with mock images (no API cost, for testing)
python main.py batch prompts.csv --image-generator mock

# Process specific rows
python main.py batch prompts.csv --start-row 10 --max-rows 5

# Custom output directories
python main.py batch prompts.csv \
    --output-dir my_results \
    --image-dir my_images

# Specify prompt column name
python main.py batch prompts.csv --prompt-column "description"

# With visual bias evaluation
python main.py batch prompts.csv \
    --enable-visual-evaluation \
    --fairface-model /path/to/fairface.pt \
    --dlib-model /path/to/dlib_model.dat
```

**3. Understanding Results**

After processing, you'll find:

```
my_results/
├── enhanced_prompts_1234567890.csv    # Results with all columns
├── enhanced_prompts_1234567890.json   # Detailed metadata
└── visual_bias_evaluation/            # If enabled
    ├── original_images/
    │   ├── demographic_analysis.json
    │   ├── bias_metrics.json
    │   └── face_predictions.json
    └── enhanced_images/
        ├── demographic_analysis.json
        ├── bias_metrics.json
        └── face_predictions.json

my_images/
├── 0001_original.png
├── 0001_enhanced.png
├── 0002_original.png
├── 0002_enhanced.png
└── ...
```

### Batch Processing Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `input_csv` | Path to input CSV file | Required |
| `--prompt-column` | Column name containing prompts | `"prompt"` |
| `--output-dir` | Directory for results | `"batch_results"` |
| `--image-dir` | Directory for images | `"generated_images"` |
| `--output-csv` | Output CSV path | Auto-generated with timestamp |
| `--start-row` | Row to start from | `0` |
| `--max-rows` | Maximum rows to process | All rows |
| `--image-generator` | Generator type (`dalle3` or `mock`) | `"dalle3"` |
| `--enable-visual-evaluation` | Enable demographic analysis | `False` |
| `--fairface-model` | Path to FairFace model | `None` (uses mock) |
| `--dlib-model` | Path to dlib face detector | `None` (uses mock) |

### Batch Processing Real-World Examples

#### Example 1: Marketing Campaign (10 Diverse Images)

**Scenario:** Generate 10 diverse marketing images for inclusive brand campaign

**1. Create Input CSV** (`marketing_campaign.csv`):
```csv
prompt,campaign_theme
healthcare professionals treating patients,Medical Excellence
students learning together,Education for All
small business owners,Entrepreneurship
families enjoying meals,Food & Family
athletes training,Sports & Wellness
artists creating,Creative Expression
scientists researching,Innovation
teachers instructing,Learning Leaders
community volunteers,Community Care
tech workers collaborating,Digital Future
```

**2. Run Batch Processing:**
```bash
python main.py batch marketing_campaign.csv \
    --output-dir campaign_results \
    --image-dir campaign_images \
    --max-rows 10
```

**3. Results:**
- **Processing Time:** ~5 minutes (10 prompts × ~30 seconds each)
- **Generated Files:**
  - 10 original images
  - 10 enhanced images
  - CSV with all enhancement details
  - JSON with metadata and scores

**4. Sample Output (from CSV):**
```csv
prompt,enhanced_prompt,bias_score,diversity_score,iterations
"healthcare professionals","In a modern medical facility, healthcare professionals...",85,88,2
"students learning together","A vibrant classroom with students of diverse ages...",82,91,3
...
```

#### Example 2: Educational Textbook (Testing with Mock Images)

**Scenario:** Test enhancement quality without API costs before production

**1. Create Test CSV** (`textbook_test.csv`):
```csv
prompt
a scientist in a laboratory
an engineer designing
a mathematician solving equations
a historian researching
a writer composing
```

**2. Run with Mock Images:**
```bash
python main.py batch textbook_test.csv \
    --image-generator mock \
    --max-rows 5 \
    --output-dir test_results
```

**Benefits:**
- ✅ No DALL-E API costs
- ✅ Fast processing (~5 seconds per prompt)
- ✅ Perfect for testing enhancement quality
- ✅ Validate CSV format and column names
- ⚠️ Images are random placeholders (not real)

**3. Review Enhanced Prompts:**
After validation, run with real images:
```bash
python main.py batch textbook_test.csv \
    --image-generator dalle3 \
    --max-rows 5
```

#### Example 3: Large Dataset (Processing in Batches)

**Scenario:** Process 100 prompts in manageable batches

**1. Process First 20:**
```bash
python main.py batch large_dataset.csv \
    --start-row 0 \
    --max-rows 20 \
    --output-dir batch_01
```

**2. Process Next 20:**
```bash
python main.py batch large_dataset.csv \
    --start-row 20 \
    --max-rows 20 \
    --output-dir batch_02
```

**3. Continue in Batches:**
```bash
# Batch 3
python main.py batch large_dataset.csv --start-row 40 --max-rows 20 --output-dir batch_03

# Batch 4
python main.py batch large_dataset.csv --start-row 60 --max-rows 20 --output-dir batch_04

# Final batch
python main.py batch large_dataset.csv --start-row 80 --max-rows 20 --output-dir batch_05
```

**Benefits:**
- ✅ Manageable API costs
- ✅ Can stop/resume anytime
- ✅ Easier to review in smaller groups
- ✅ Parallel processing across machines (different batches)

#### Example 4: With Visual Bias Evaluation

**Scenario:** Measure demographic diversity in generated images

**1. Create CSV** (`diversity_analysis.csv`):
```csv
prompt
medical professionals
teachers and educators
business leaders
tech industry workers
```

**2. Run with Visual Evaluation:**
```bash
python main.py batch diversity_analysis.csv \
    --enable-visual-evaluation \
    --fairface-model models/fairface_model.pt \
    --dlib-model models/shape_predictor.dat \
    --output-dir diversity_eval
```

**3. Results Include:**
```
diversity_eval/
├── enhanced_prompts_xxx.csv
├── enhanced_prompts_xxx.json
└── visual_bias_evaluation/
    ├── original_images/
    │   ├── demographic_analysis.json  ← Demographics of original images
    │   ├── bias_metrics.json          ← Bias-W, Bias-P, ENS scores
    │   └── face_predictions.json      ← Per-face predictions
    └── enhanced_images/
        ├── demographic_analysis.json  ← Demographics of enhanced images
        ├── bias_metrics.json          ← Bias-W, Bias-P, ENS scores
        └── face_predictions.json      ← Per-face predictions
```

**4. Sample Demographic Analysis:**
```json
{
  "total_faces": 42,
  "race_distribution": {
    "White": 8,
    "Black": 9,
    "Asian": 11,
    "Indian": 7,
    "Latino": 5,
    "Middle Eastern": 2
  },
  "gender_distribution": {
    "Male": 19,
    "Female": 23
  },
  "age_distribution": {
    "0-2": 0,
    "3-9": 3,
    "10-19": 5,
    "20-29": 12,
    "30-39": 10,
    "40-49": 7,
    "50-59": 3,
    "60-69": 2,
    "70+": 0
  }
}
```

**5. Metrics:**
- **Bias-W** (population-level): Lower is more balanced
- **Bias-P** (per-image): Lower is more diverse per image
- **ENS** (effective number): Higher is more diverse

#### Example 5: Custom Column Names

**Scenario:** Your CSV has different column names

**1. Your CSV** (`custom_format.csv`):
```csv
id,description,category,notes
1,"a doctor","healthcare","for brochure"
2,"students","education","textbook chapter 3"
3,"chef cooking","culinary","recipe book"
```

**2. Specify Column:**
```bash
python main.py batch custom_format.csv \
    --prompt-column description \
    --output-dir custom_results
```

**3. Output Preserves All Columns:**
```csv
id,description,category,notes,enhanced_prompt,bias_score,diversity_score,iterations
1,"a doctor","healthcare","for brochure","In a modern medical...",85,88,2
2,"students","education","textbook chapter 3","A vibrant classroom...",82,91,3
3,"chef cooking","culinary","recipe book","A skilled chef...",80,85,2
```

### Batch Processing Best Practices

1. **Start Small:** Test with 3-5 prompts using mock images first
2. **Use Batches:** Process large datasets in groups of 20-50
3. **Monitor Costs:** Each DALL-E 3 image costs $0.04-0.08
4. **Save Regularly:** Results are saved incrementally during processing
5. **Check Errors:** Review JSON output for any failed prompts
6. **Visual Evaluation:** Only use when you have the model files (large downloads)

### Batch Processing Troubleshooting

**Problem:** Out of memory
```bash
# Solution: Process smaller batches
python main.py batch large.csv --max-rows 10
```

**Problem:** API rate limits
```bash
# Solution: OpenAI limits are usually 50-100 req/min
# Process will automatically handle rate limits
# Just let it run, it will retry
```

**Problem:** Want to resume after failure
```bash
# Solution: Use --start-row to skip completed rows
# If you processed 30 rows before failure:
python main.py batch prompts.csv --start-row 30
```

**Problem:** Testing before production
```bash
# Solution: Always test with mock first
python main.py batch prompts.csv --image-generator mock --max-rows 3
```
| `--enable-visual-evaluation` | Enable visual bias evaluation | `False` |
| `--fairface-model` | Path to FairFace model | `None` (uses mock) |
| `--dlib-model` | Path to dlib model | `None` (uses mock) |
| `--no-save-intermediate` | Don't save during processing | Save enabled |

### CSV Output Columns

The output CSV includes all input columns plus:

| Column | Description | Example |
|--------|-------------|---------|
| `enhanced_prompt` | Improved prompt | "In a diverse medical setting..." |
| `bias_score` | Bias mitigation score | 82.5 |
| `diversity_score` | Diversity enhancement score | 87.0 |
| `original_image_path` | Path to original image | "images/0001_original.png" |
| `enhanced_image_path` | Path to enhanced image | "images/0001_enhanced.png" |
| `processing_time_seconds` | Processing duration | 23.4 |
| `enhancement_iterations` | Number of cycles | 2 |
| `error` | Error message if any | "" or "Error: ..." |

### Batch Processing Examples

**Example 1: Quick Test with Mock Images**
```bash
python main.py batch test_prompts.csv \
    --image-generator mock \
    --max-rows 3
```

**Example 2: Production Run with DALL-E 3**
```bash
python main.py batch prompts.csv \
    --image-generator dalle3 \
    --output-dir production_results \
    --image-dir production_images
```

**Example 3: With Visual Bias Evaluation**
```bash
python main.py batch prompts.csv \
    --enable-visual-evaluation \
    --fairface-model models/fairface.pt \
    --dlib-model models/dlib_predictor.dat
```

## Knowledge Queries

Query the knowledge graphs to research bias patterns and cultural values.

### Basic Queries

```bash
# Interactive mode
python main.py query

# Direct query
python main.py query "What stereotypes exist about doctors?"

# With custom retrieval settings
python main.py query "Cultural practices in Asia" --top-k 10
```

### Query Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `question` | Question to ask (optional) | Interactive prompt |
| `--top-k` | Number of results to retrieve | `8` |
| `--cache-file` | Embedding cache file | `"embeddings.pkl"` |

### Query Examples

**Example 1: Research Stereotypes**
```bash
python main.py query "What racial stereotypes exist in media?"
```

**Example 2: Cultural Practices**
```bash
python main.py query "What are common greetings in different cultures?" --top-k 15
```

**Example 3: Bias Patterns**
```bash
python main.py query "How are women stereotyped in professional settings?"
```

### What Gets Queried

The query command searches through:
1. **Bias Data**: ADV_GRAPH dataset with 4,229 stereotype examples
2. **Cultural Values**: 15+ cultural regions with practices and values  
3. Uses GPT-4 to synthesize answers from retrieved knowledge triples

### Knowledge Query Examples

#### Example 1: Understanding Stereotypes Before Enhancement

**Use Case:** Research existing stereotypes before enhancing prompts

```bash
python main.py query "What stereotypes exist about doctors?"
```

**Sample Output:**
```
📚 Query Results (Top 8 triples)
================================================================================

Triple 1:
Subject: doctor | Relation: stereotype | Object: always_male
Context: Medical professional stereotypes in media representation

Triple 2:
Subject: doctor | Relation: stereotype | Object: white_coat_authority
Context: Professional appearance expectations

Triple 3:
Subject: physician | Relation: bias | Object: assumes_patient_english
Context: Language assumptions in medical settings

[... more triples ...]

💡 GPT-4 Synthesis:
Common stereotypes about doctors include the assumption that they are
predominantly male, wear white coats as a symbol of authority, and that
patients speak English fluently. These stereotypes can perpetuate
exclusionary practices and fail to represent the diversity of medical
professionals and patients in real healthcare settings.
```

#### Example 2: Exploring Cultural Practices

**Use Case:** Research authentic cultural details for prompt enhancement

```bash
python main.py query "What are traditional greetings in different cultures?" --top-k 15
```

**Sample Output:**
```
📚 Query Results (Top 15 triples)
================================================================================

Triple 1:
Subject: Japan | Relation: greeting | Object: bow_respect
Context: Traditional Japanese greeting customs

Triple 2:
Subject: India | Relation: greeting | Object: namaste_gesture
Context: Indian greeting combining gesture and respect

Triple 3:
Subject: Middle_East | Relation: greeting | Object: cheek_kiss_custom
Context: Traditional greeting among friends and family

Triple 4:
Subject: Maori_New_Zealand | Relation: greeting | Object: hongi_nose_touch
Context: Traditional Māori greeting ceremony

[... more triples ...]

💡 GPT-4 Synthesis:
Greetings vary significantly across cultures: Japanese culture emphasizes
bowing to show respect, Indians use the namaste gesture combining spiritual
acknowledgment, Middle Eastern cultures often include cheek kisses among
close relations, and Māori people perform the hongi by pressing noses together
to share breath. Understanding these diverse practices helps create culturally
authentic and respectful representations.
```

#### Example 3: Gender Bias Research

**Use Case:** Identify gender biases to avoid in professional imagery

```bash
python main.py query "How are women stereotyped in professional settings?"
```

**Sample Output:**
```
📚 Query Results (Top 8 triples)
================================================================================

Triple 1:
Subject: woman_professional | Relation: stereotype | Object: secretary_assistant
Context: Career limitation stereotypes

Triple 2:
Subject: female_executive | Relation: bias | Object: questioned_authority
Context: Leadership credibility assumptions

Triple 3:
Subject: woman_engineer | Relation: stereotype | Object: token_diversity
Context: Underrepresentation in STEM fields

[... more triples ...]

💡 GPT-4 Synthesis:
Women in professional settings often face stereotypes that limit their
perceived roles to support positions like secretaries, have their authority
questioned more than male counterparts, or are viewed as token diversity
hires rather than qualified professionals. These biases can be actively
countered in image generation by depicting women in leadership positions,
technical roles, and positions of genuine authority.
```

#### Example 4: Age Diversity Research

**Use Case:** Understand age-related stereotypes for inclusive imagery

```bash
python main.py query "What age-related stereotypes exist in workplace?"
```

**Sample Output:**
```
📚 Query Results (Top 8 triples)
================================================================================

Triple 1:
Subject: older_worker | Relation: stereotype | Object: technology_resistant
Context: Age and technology adoption assumptions

Triple 2:
Subject: young_professional | Relation: bias | Object: inexperienced_immature
Context: Youth and competence assumptions

Triple 3:
Subject: senior_employee | Relation: stereotype | Object: near_retirement
Context: Career trajectory assumptions

[... more triples ...]

💡 GPT-4 Synthesis:
Age-related workplace stereotypes include assuming older workers resist
technology, young professionals lack experience and maturity, and senior
employees are preparing for retirement rather than advancement. Inclusive
imagery should show people of all ages engaged in learning, leadership,
and technological competence.
```

#### Example 5: Cultural Values for Authentic Representation

**Use Case:** Research cultural values for specific regions

```bash
python main.py query "What values are important in East Asian cultures?" --top-k 12
```

**Sample Output:**
```
📚 Query Results (Top 12 triples)
================================================================================

Triple 1:
Subject: China | Relation: value | Object: family_harmony
Context: Core Confucian principle

Triple 2:
Subject: Japan | Relation: value | Object: group_harmony_wa
Context: Social cohesion emphasis

Triple 3:
Subject: Korea | Relation: value | Object: respect_elders
Context: Hierarchical social structure

Triple 4:
Subject: China | Relation: value | Object: education_achievement
Context: Academic success importance

[... more triples ...]

💡 GPT-4 Synthesis:
East Asian cultures emphasize family and group harmony, respect for elders
and hierarchical relationships, educational achievement, and collective
well-being over individual pursuits. These values influence social
interactions, professional environments, and family dynamics. Authentic
cultural representation should reflect these priorities rather than
Western-centric individualism.
```

#### Example 6: Interactive Research Session

**Use Case:** Explore multiple topics interactively

```bash
python main.py query
```

**Interactive Session:**
```
📚 Knowledge Graph Query System
======================================================================
Query the knowledge base for bias patterns and cultural information.

Examples:
  • What stereotypes exist about teachers?
  • What foods are popular in different cultures?
  • What cultural practices exist in various countries?
----------------------------------------------------------------------
🔍 Your question: What stereotypes exist about Muslims?

[System retrieves and displays 8 relevant triples]

🔍 Your question: What are common Middle Eastern cultural practices?

[System retrieves and displays 8 relevant triples]

🔍 Your question: How are older adults stereotyped?

[System retrieves and displays 8 relevant triples]

🔍 Your question: (press Ctrl+C to exit)
```

### Query Use Cases

1. **Pre-Enhancement Research**
   - Understand biases before generating prompts
   - Research cultural authenticity
   - Identify stereotypes to avoid

2. **Educational Purposes**
   - Study bias patterns in data
   - Learn about cultural diversity
   - Understand representation issues

3. **Content Validation**
   - Verify cultural accuracy
   - Check bias awareness
   - Research authentic practices

4. **Academic Research**
   - Analyze bias patterns
   - Study cultural representations
   - Collect examples for papers

### Query Tips

```bash
# Be specific for better results
python main.py query "gender stereotypes in tech industry"

# Use broader queries to explore
python main.py query "cultural food traditions"

# Increase results for comprehensive research
python main.py query "age discrimination patterns" --top-k 20

# Interactive mode for multiple questions
python main.py query
```

## Testing Commands

Run various tests to verify system functionality.

### Test Modes

| Test Type | What It Does | Duration |
|-----------|--------------|----------|
| `--quick` | Process 1 prompt with DALL-E 3 | ~30 seconds |
| `--batch` | Process 3 prompts with mock images | ~1 minute |
| `--visual` | Process 3 prompts with visual evaluation | ~2-3 minutes |
| `--all` | Run all tests | ~4-5 minutes |

### Running Tests

```bash
# Quick test (fastest, 1 prompt)
python main.py test --quick

# Batch processing test (3 prompts, mock images)
python main.py test --batch

# Visual evaluation test (3 prompts with bias evaluation)
python main.py test --visual

# Run all tests
python main.py test --all
```

### Test Examples

**Example 1: Quick Demo**
```bash
# Perfect for demos or verifying setup
python main.py test --quick
```

**Example 2: Test Before Production**
```bash
# Test batch processing before running on real data
python main.py test --batch
```

**Example 3: Validate Visual Evaluation**
```bash
# Ensure visual bias evaluation is working
python main.py test --visual
```

### Test Outputs

Tests create temporary directories:
- `quick_test_results/` - Quick test results
- `test_results/` - Batch test results  
- `visual_bias_test_results/` - Visual evaluation test results

You can safely delete these after reviewing.

---

## 6. Advanced Configuration

### 6.1 Custom Enhancement Parameters

Fine-tune the dual-pipeline enhancement with custom parameters:

```bash
# Strict quality thresholds
python main.py enhance "business meeting" \
    --bias-threshold 85 \
    --diversity-threshold 85 \
    --max-iterations 5

# Adjust knowledge base retrieval
python main.py enhance "engineers working" \
    --stereoset-top-k 20 \
    --diversity-top-k 15 \
    --graph-top-k 10
```

**Key Parameters:**
- `--bias-threshold`: Minimum bias mitigation score (0-100, default: 75)
- `--diversity-threshold`: Minimum diversity score (0-100, default: 80)
- `--max-iterations`: Maximum enhancement cycles (default: 3)
- `--stereoset-top-k`: Number of bias examples to retrieve (default: 10)
- `--diversity-top-k`: Number of diversity examples to retrieve (default: 10)
- `--graph-top-k`: Number of cultural triples to retrieve (default: 8)

**Output Example:**
```
🎯 Dual-Pipeline Enhancement Results
================================================================================
Original Prompt: students in a classroom

Enhanced Prompt: In a diverse learning environment, students of various ages,
ethnicities, and abilities collaborate together. The classroom includes
wheelchair-accessible spaces, and students use different learning methods...

📊 SCORING BREAKDOWN:
Bias Mitigation: 82/100 ✅ (threshold: 75)
  • Inclusive language: ✓
  • Stereotype avoidance: ✓
  • Neutral descriptions: ✓

Diversity Enhancement: 87/100 ✅ (threshold: 80)
  • Age diversity: 13/15
  • Ethnic diversity: 18/20
  • Gender diversity: 14/15
  • Cultural diversity: 17/20
  • Ability inclusion: 10/10
  • Socioeconomic: 8/10
  • Specificity: 7/10

Iterations: 2
Processing Time: 18.5 seconds
```

### 2. Sequential Enhancement

Iterative improvement focusing on diversity scoring.

**When to Use:**
- Simpler configuration needed
- Diversity-focused content
- Faster processing acceptable

**Usage:**

```bash
# Basic usage
python enhance_prompt_sequential.py -p "a business meeting"

# Custom diversity threshold
python enhance_prompt_sequential.py \
    --prompt "engineers in office" \
    --threshold 85 \
    --max-iterations 4

# Interactive mode
python enhance_prompt_sequential.py
```

**Key Parameters:**
- `--threshold`: Minimum diversity score (default: 75)
- `--max-iterations`: Maximum cycles (default: 3)

**Difference from Dual-Pipeline:**
- Single diversity score (no separate bias score)
- Same 7-dimension diversity assessment
- Slightly faster processing

### 3. Basic Enhancement

Single-pass enhancement for quick improvements.

**When to Use:**
- Speed is critical
- Simple prompts
- Basic diversity needs
- Quick prototyping

**Usage:**

```bash
# Interactive mode
python enhance_prompt.py

# Direct enhancement
python enhance_prompt.py -p "a scientist in lab"

# Disable specific sources
python enhance_prompt.py -p "teacher" --no-graphrag
```

**Key Parameters:**
- `--use-stereoset`: Enable bias detection (default: True)
- `--use-graphrag`: Enable cultural knowledge (default: True)

**Trade-offs:**
- ✅ Fastest processing (~10 seconds)
- ✅ Simplest to use
- ❌ No quality guarantees
- ❌ No iterative improvement
- ❌ No diversity scoring

## Batch Processing

Process multiple prompts from a CSV file with automatic image generation and evaluation.

### Basic Batch Processing

**1. Prepare Your CSV File**

Create a CSV file with prompts (example: `prompts.csv`):

```csv
prompt
a doctor examining a patient
students in a classroom
engineers working on a project
a chef preparing food
artists creating artwork
```

**2. Run Batch Processing**

```bash
# Basic batch processing (uses dual-pipeline by default)
python main.py batch prompts.csv

# Custom output directories
python main.py batch prompts.csv \
    --output-dir my_results \
    --image-dir my_images

# Process specific rows
python main.py batch prompts.csv \
    --start-row 10 \
    --max-rows 5

# Use different image generator
python main.py batch prompts.csv \
    --image-generator mock  # For testing without API costs
```

**3. Understanding Results**

After processing, you'll find:

```
my_results/
├── enhanced_prompts_1234567890.csv    # Results with all columns
├── enhanced_prompts_1234567890.json   # Detailed metadata
└── visual_bias_evaluation/            # If enabled
    ├── original_images/
    │   ├── demographic_analysis.json
    │   ├── bias_metrics.json
    │   └── face_predictions.json
    └── enhanced_images/
        ├── demographic_analysis.json
        ├── bias_metrics.json
        └── face_predictions.json

my_images/
├── 0001_original.png
├── 0001_enhanced.png
├── 0002_original.png
├── 0002_enhanced.png
└── ...
```

### CSV Output Columns

The output CSV includes all input columns plus:

| Column | Description | Example |
|--------|-------------|---------|
| `enhanced_prompt` | Improved prompt | "In a diverse medical setting..." |
| `bias_score` | Bias mitigation score | 82.5 |
| `diversity_score` | Diversity enhancement score | 87.0 |
| `original_image_path` | Path to original image | "images/0001_original.png" |
| `enhanced_image_path` | Path to enhanced image | "images/0001_enhanced.png" |
| `processing_time_seconds` | Processing duration | 23.4 |
| `enhancement_iterations` | Number of cycles | 2 |
| `error` | Error message if any | "" or "Error: ..." |

### JSON Output Structure

```json
{
  "metadata": {
    "total_processed": 10,
    "successful": 9,
    "failed": 1,
    "processing_time": 245.6,
    "timestamp": "2025-10-01T10:30:00",
    "configuration": {...}
  },
  "results": [
    {
      "row_index": 0,
      "original_prompt": "a doctor examining a patient",
      "enhanced_prompt": "In a global healthcare setting...",
      "scores": {
        "bias": 82.5,
        "diversity": 87.0
      },
      "images": {
        "original": "images/0001_original.png",
        "enhanced": "images/0001_enhanced.png"
      },
      "metadata": {
        "iterations": 2,
        "processing_time": 18.5,
        "model": "gpt-4",
        "image_generator": "dalle3"
      }
    }
  ]
}
```

## Visual Bias Evaluation

Analyze demographic representation in generated images using FairFace and bias metrics.

### Setup

**1. Install Required Models**

Download the following models:

- **FairFace Model**: `res34_fair_align_multi_7_20190809.pt`
  - Source: https://github.com/joojs/fairface
  - Size: ~84 MB
  
- **dlib Shape Predictor**: `shape_predictor_5_face_landmarks.dat`
  - Source: http://dlib.net/files/
  - Size: ~9.2 MB

**2. Install Dependencies**

```bash
pip install torch torchvision dlib opencv-python
```

### Usage with Batch Processing

```bash
# Enable visual bias evaluation
python main.py batch prompts.csv \
    --enable-visual-evaluation \
    --fairface-model /path/to/res34_fair_align_multi_7_20190809.pt \
    --dlib-model /path/to/shape_predictor_5_face_landmarks.dat
```

### Testing Without Models

The system includes mock models for testing:

```bash
# Test without actual models (uses random predictions)
python test_visual_bias_evaluation.py
```

### Understanding Evaluation Metrics

**1. Bias-W (Population-level Bias)**
- Measures overall demographic bias across all images
- Range: 0.0 (no bias) to 1.0 (maximum bias)
- Lower is better
- Formula: `Σ |predicted_proportion - reference_proportion| / 2`

**2. Bias-P (Per-image Bias)**
- Average bias per individual image
- Identifies systematic over/under-representation
- Range: 0.0 to ∞
- Lower is better

**3. ENS (Diversity via Shannon Entropy)**
- Measures demographic diversity using information theory
- Range: 0 (no diversity) to log(N) (maximum diversity)
- Higher is better
- Formula: `-Σ p_i * log(p_i)`

**4. KL Divergence**
- Compares distribution to reference demographics
- Range: 0.0 (perfect match) to ∞
- Lower is better
- Formula: `Σ P(i) * log(P(i)/Q(i))`

### Demographic Categories

**Race (7 categories):**
- White
- Black
- Asian
- Indian
- Middle Eastern
- Latino
- Southeast Asian

**Gender (2 categories):**
- Male
- Female

**Age (9 groups):**
- 0-2 years
- 3-9 years
- 10-19 years
- 20-29 years
- 30-39 years
- 40-49 years
- 50-59 years
- 60-69 years
- 70+ years

### Evaluation Results Format

```json
{
  "bias_metrics": {
    "bias_w": 0.15,
    "bias_p": 0.23,
    "ens": 1.85,
    "kl_divergence": 0.12
  },
  "summary": {
    "total_faces": 45,
    "images_with_faces": 38,
    "avg_faces_per_image": 1.18
  },
  "demographic_distribution": {
    "race": {
      "White": 0.35,
      "Black": 0.20,
      "Asian": 0.18,
      "Indian": 0.12,
      "Middle Eastern": 0.08,
      "Latino": 0.05,
      "Southeast Asian": 0.02
    },
    "gender": {
      "Male": 0.55,
      "Female": 0.45
    },
    "age": {
      "20-29": 0.30,
      "30-39": 0.25,
      "40-49": 0.20,
      "50-59": 0.15,
      "60-69": 0.10
    }
  }
}
```

## Configuration Options

### Environment Variables

```bash
# Required for full functionality
export OPENAI_API_KEY="sk-your-api-key-here"

# Optional: Use alternative API endpoint
export OPENAI_API_BASE="https://your-proxy.com/v1"
```

### Image Generation Options

```bash
# DALL-E 3 size options
--image-size 1024x1024  # Default
--image-size 1792x1024  # Wide
--image-size 1024x1792  # Tall

# Quality options
--image-quality standard  # Default, faster
--image-quality hd        # Higher quality, slower

# Style options
--image-style natural     # Default, more realistic
--image-style vivid       # More dramatic
```

### Enhancement Weights

Advanced: Modify source weights in the `enhance_prompt_dual_pipeline.py` module (imported by `main.py`):

```python
# Default weights for CultureBank retrieval
ALPHA = 0.6  # Vector similarity weight
BETA = 0.3   # Keyword matching weight
GAMMA = 0.1  # Agreement score weight
```

## 7. Best Practices

### For Maximum Quality

1. **Use Dual-Pipeline Enhancement** (default mode)
   - Set high thresholds (bias: 85, diversity: 85)
   - Allow more iterations (5+)
   - Enable all knowledge sources

2. **Enable Visual Bias Evaluation**
   - Provides objective measurement
   - Validates enhancement effectiveness
   - Identifies systematic issues

3. **Iterative Refinement**
   - Review results
   - Adjust thresholds based on domain
   - Fine-tune for specific use cases

### For Speed

1. **Use Basic Enhancement**
   - Single-pass processing
   - Faster but less comprehensive

2. **Batch Processing**
   - Process multiple prompts together
   - Reuses loaded models and caches

3. **Mock Image Generator**
   - For testing workflows
   - No API costs

### For Research

1. **Keep All Metadata**
   - Save JSON outputs
   - Enable visual evaluation
   - Track all metrics

2. **Controlled Experiments**
   - Use consistent thresholds
   - Same datasets
   - Document configurations

3. **Statistical Analysis**
   - Compare original vs enhanced
   - Aggregate metrics across sets
   - Identify patterns

## Troubleshooting

### Common Issues

**1. OpenAI API Key Error**
```
Error: OpenAI API key not found
```

**Solution:**
```bash
export OPENAI_API_KEY="sk-your-key"
# Verify it's set
echo $OPENAI_API_KEY
```

**2. Rate Limiting**
```
Error: Rate limit exceeded
```

**Solution:**
- Use smaller batches
- Add delays between requests
- Upgrade OpenAI plan

**3. Image Generation Fails**
```
ImageGenerationError: Content policy violation
```

**Solution:**
- Review prompt content
- Ensure appropriate language
- Check OpenAI content policies

**4. Visual Evaluation Missing Dependencies**
```
ImportError: No module named 'dlib'
```

**Solution:**
```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt-get install cmake libboost-all-dev

# Install Python packages
pip install dlib torch torchvision opencv-python
```

**5. Out of Memory**
```
RuntimeError: CUDA out of memory
```

**Solution:**
- Process smaller batches
- Use CPU for visual evaluation
- Reduce image sizes

### Performance Optimization

**Slow Processing:**
1. Check internet connection
2. Use embedding cache (automatic)
3. Reduce max_iterations
4. Process in smaller batches

**High API Costs:**
1. Use mock generator for testing
2. Cache embeddings (automatic)
3. Optimize batch sizes
4. Review iteration settings

### Getting Help

1. **Check Logs**: Look for detailed error messages
2. **Review Documentation**: Architecture and API reference
3. **Test Components**: Use test scripts to isolate issues
4. **Verify Setup**: Ensure all dependencies installed

### Debug Mode

Enable detailed logging:

```bash
# Set logging level
export LOG_LEVEL=DEBUG

# Run with verbose output (batch processing)
python main.py batch prompts.csv --verbose
```

## 8. Examples Gallery

### Example 1: Medical Professional

**Original:** "a doctor"

**Enhanced:** "In a global healthcare setting, Dr. Ji-Yeon Kim, a highly skilled physician of South Korean origin educated at the prestigious Seoul National University, offers compassionate medical services to a broad spectrum of patients from diverse cultural backgrounds including elderly African American patients, young Latinx families, and Middle Eastern refugees. Dr. Kim, using a name common to all genders in Korea, works alongside Dr. Aisha Patel, a wheelchair-using Indian specialist, ensuring their practice is a beacon of inclusivity..."

**Scores:**
- Bias Mitigation: 88/100
- Diversity: 92/100
### Example 2: Educational Setting

**Original:** "students studying"

**Enhanced:** "In a multicultural learning environment, students of various ages ranging from young adults to mature learners, representing diverse ethnic backgrounds including Asian, African, Latin American, and Indigenous communities, collaborate using multiple learning approaches. Some students use assistive technologies, while others engage through hands-on experimentation, reflecting different abilities and socioeconomic backgrounds..."

**Scores:**
- Bias Mitigation: 85/100
- Diversity: 89/100

### Example 3: Professional Workspace

**Original:** "engineers working"

**Enhanced:** "A diverse team of engineers from various cultural backgrounds - including a senior Chinese engineer mentoring junior colleagues, an African American woman leading the technical design, and engineers from Middle Eastern and South American communities collaborating on sustainable solutions - work together in an accessible workspace. The team spans multiple age groups and includes engineers with different abilities using adaptive technologies..."

**Scores:**
- Bias Mitigation: 90/100
- Diversity: 94/100
