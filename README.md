# LLM Bias & Fairness Project (GraphRAG)

A Python-based system for analyzing bias and fairness in language models using Graph Retrieval-Augmented Generation (GraphRAG). This project combines knowledge graph processing with vector embeddings to explore racial stereotypes, bias patterns, and cultural values in structured data.

## 🎯 Overview

This project implements a comprehensive GraphRAG system that:
- **Processes multiple knowledge graphs** containing bias/stereotype and cultural values data
- **Uses OpenAI embeddings** for semantic similarity search  
- **Retrieves relevant graph triples** based on user queries
- **Generates LLM answers** about bias and fairness questions
- **Enhances image prompts** for diversity and bias mitigation
- **Caches embeddings** for efficient repeated queries

## 🎨 New Feature: Image Prompt Enhancement

The system now includes **`enhance_prompt.py`** - a specialized tool for enhancing image generation prompts to promote diversity and avoid biases:

### How It Works
1. **Input**: Takes an image generation prompt (e.g., "a doctor examining a patient")
2. **Retrieval**: Finds relevant bias/stereotype triples and cultural value triples  
3. **Enhancement**: Uses LLM to enhance the prompt with diversity elements
4. **Output**: Provides an inclusive, bias-aware image prompt

### Key Benefits
- **🚫 Bias Mitigation**: Identifies and avoids harmful stereotypes
- **🌍 Cultural Awareness**: Incorporates global cultural values and practices
- **👥 Diversity Promotion**: Explicitly includes age, ethnicity, gender, disability representation
- **🎨 Creative Enhancement**: Maintains original prompt concept while adding inclusive elements

### Usage Examples
```bash
# Interactive mode
python enhance_prompt.py

# Direct prompt enhancement  
python enhance_prompt.py -p "students in a classroom"

# Customize retrieval parameters
python enhance_prompt.py -p "engineers working" --bias-top-k 10 --cultural-top-k 15
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   CSV Data      │───▶│   Knowledge      │───▶│   NetworkX      │
│   (Stereotypes) │    │   Graph Builder  │    │   Graph         │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                       │
┌─────────────────┐    ┌──────────────────┐            ▼
│   Embedding     │◀───│   OpenAI API     │    ┌─────────────────┐
│   Cache (.pkl)  │    │   Embeddings     │◀───│   Vector        │
└─────────────────┘    └──────────────────┘    │   Embedder      │
                                               └─────────────────┘
                                                       │
┌─────────────────┐    ┌──────────────────┐            ▼
│   LLM Answer    │◀───│   OpenAI Chat    │    ┌─────────────────┐
│   Generation    │    │   Completion     │◀───│   Similarity    │
└─────────────────┘    └──────────────────┘    │   Search        │
                                               └─────────────────┘
```

## 🚀 Features

### Core Functionality
- **📊 Knowledge Graph Processing**: Converts CSV stereotype data into NetworkX graphs
- **🔍 Vector Similarity Search**: Uses OpenAI `text-embedding-3-large` for semantic matching
- **💾 Intelligent Caching**: Persistent embedding storage to minimize API costs
- **🎯 Top-K Retrieval**: Configurable number of most relevant graph nodes
- **🤖 LLM Integration**: GPT-powered answer generation from retrieved context

### Technical Features
- **⚡ Chunked Processing**: Handles large datasets with API rate limiting
- **🛡️ Error Handling**: Robust error recovery and logging
- **🔧 Fallback Mode**: Works without API key using substring search
- **📝 Structured Logging**: Detailed operation tracking and debugging
- **🧹 Code Quality**: Passes Ruff linting with proper type hints

## 📦 Installation

### Prerequisites
- Python 3.8+
- OpenAI API key (optional, for full functionality)

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd llm-bias-fairness

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set OpenAI API key (optional)
export OPENAI_API_KEY="sk-your-api-key-here"
```

## 🔧 Usage

### Image Prompt Enhancement (New!)

Enhance image generation prompts for diversity and bias mitigation:

```bash
# Interactive mode - get prompted for input
python enhance_prompt.py

# Direct prompt enhancement
python enhance_prompt.py -p "a doctor examining a patient"

# With custom retrieval parameters
python enhance_prompt.py -p "students in classroom" --bias-top-k 10 --cultural-top-k 15

# Using different cache file
python enhance_prompt.py -p "engineers working" --cache-file my_embeddings.pkl
```

**Command Line Options for enhance_prompt.py:**
```bash
python enhance_prompt.py [OPTIONS]

Options:
  -p, --prompt TEXT           Image prompt to enhance (optional - interactive if not provided)
  --bias-top-k INTEGER       Number of bias triples to retrieve [default: 25]
  --cultural-top-k INTEGER   Number of cultural triples to retrieve [default: 25]  
  --cache-file TEXT          Embedding cache file [default: embeddings.pkl]
  -h, --help                Show help message
```

### Multi-Source Graph Testing  

Test the GraphRAG system with multiple data sources:

```bash
# Interactive mode - get prompted for question
python main.py

# Direct question to all graphs
python main.py -q "What are common stereotypes about different groups?"

# With custom parameters
python main.py -q "food culture" --top-k 10 --cache-file test_embeddings.pkl
```

**Command Line Options for main.py:**
```bash
python main.py [OPTIONS]

Options:
  -q, --question TEXT     Question to ask all loaded graphs (optional - interactive if not provided)
  --top-k INTEGER        Number of top results to retrieve [default: 8]
  --cache-file TEXT      Embedding cache file [default: test_embeddings.pkl]
  -h, --help            Show help message
```

### Legacy Single-Source Usage

Use the original CLI interface:

```bash
python -m src.utils.graph_rag "data/biases/ADV_GRAPH_20240119 - ADV_GRAPH_20240119.csv" \
    --question "What racial stereotypes link dark skin to criminality?" \
    --top_k 25
```
