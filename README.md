 # LLM Bias & Fairness Project (GraphRAG)

A Python-based system for analyzing bias and fairness in language models using Graph Retrieval-Augmented Generation (GraphRAG). This project combines knowledge graph processing with vector embeddings to explore racial stereotypes and bias patterns in structured data.

## 🎯 Overview

This project implements a GraphRAG system that:
- **Processes knowledge graphs** containing stereotype and bias data
- **Uses OpenAI embeddings** for semantic similarity search
- **Retrieves relevant graph triples** based on user queries
- **Generates LLM answers** about bias and fairness questions
- **Caches embeddings** for efficient repeated queries

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

### Basic Usage
```bash
python -m src.utils.graph_rag "data/ADV_GRAPH_20240119 - ADV_GRAPH_20240119.csv" \
    --question "What racial stereotypes link dark skin to criminality?" \
    --top_k 25
```

### Command Line Options
```bash
python -m src.utils.graph_rag <csv_path> [OPTIONS]

Arguments:
  csv_path              Path to the knowledge graph CSV file

Options:
  -q, --question TEXT   Question to ask the system [default: "What stereotypes exist about black people?"]
  -k, --top_k INTEGER   Number of top similar nodes to retrieve [default: 15]
  --cache TEXT          Embedding cache file path [default: "embeddings.pkl"]
  -h, --help           Show help message
```

### Example Commands

#### Analyze Criminal Stereotypes
```bash
python -m src.utils.graph_rag "data/ADV_GRAPH_20240119 - ADV_GRAPH_20240119.csv" \
    --question "What racial stereotypes link dark skin to criminality?" \
    --top_k 25
```

#### Explore Physical Stereotypes
```bash
python -m src.utils.graph_rag "data/ADV_GRAPH_20240119 - ADV_GRAPH_20240119.csv" \
    --question "What stereotypes exist about physical appearance?" \
    --top_k 15
```

#### Use with Network Proxy
```bash
proxychains python -m src.utils.graph_rag "data/ADV_GRAPH_20240119 - ADV_GRAPH_20240119.csv" \
    --question "How are minority groups stereotyped in media?" \
    --top_k 10
```

## 📊 Sample Output

```
INFO: Embedding 1 new node strings …
INFO: Cache now holds 13690 embeddings
INFO: HTTP Request: POST https://api.openai.com/v1/embeddings "HTTP/1.1 200 OK"
INFO: Top 5 nodes by similarity:
INFO:   0.625 → skin color stereotypes
INFO:   0.565 → stereotype about skin
INFO:   0.516 → cultural stereotypes on color
INFO:   0.513 → dark-skinned people
INFO:   0.510 → black folks commit crimes
INFO: Retrieved 5 triples:
INFO:   dark-skinned people --are perceived as--> subhuman
INFO:   black folks --have--> stereotype about skin
INFO:   asian folks --associated with--> cultural stereotypes on color
INFO:   whites --saying--> black folks commit crimes
INFO:   black folks --associated with--> cultural stereotypes on color
INFO: 
---------- LLM ANSWER ----------
INFO: The stereotype that dark-skinned people are perceived as subhuman, along with 
the association of black folks with cultural stereotypes on color, links dark skin 
to criminality. Additionally, the statement that whites say black folks commit 
crimes further reinforces this connection.
```

## 📁 Project Structure

```
llm-bias-fairness/
├── src/
│   ├── __init__.py
│   └── utils/
│       ├── __init__.py
│       └── graph_rag.py          # Main GraphRAG implementation
├── data/
│   └── ADV_GRAPH_20240119 - ADV_GRAPH_20240119.csv  # Knowledge graph data
├── embeddings.pkl                # Cached embeddings (auto-generated)
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── CHANGELOG.md                  # Version history
├── ruff.toml                     # Code quality configuration
└── main.py                       # Entry point (if needed)
```

## 🔍 Data Format

The system expects CSV files with the following structure:
- **targetMinority**: The minority group being stereotyped
- **targetStereotype**: Description of the stereotype
- **Graph**: Knowledge graph triples in the format `(subject, predicate, object)`

Example:
```csv
,targetMinority,targetStereotype,Graph
0,black folks,are all well endowed,"Graph: `(black folks, are, well endowed)`"
1,black folks,there are good blacks and bad blacks,"Graph: `(good blacks, belong to, black people)` `(bad blacks, belong to, black people)`"
```

## ⚙️ Configuration

### Environment Variables
- `OPENAI_API_KEY`: Required for embedding generation and LLM answers
- If not set, the system falls back to substring-based retrieval

### Cache Management
- Embeddings are automatically cached in `embeddings.pkl`
- Cache persists across runs to minimize API costs
- Delete the cache file to force re-embedding

### API Limits
- Processes embeddings in chunks of 2000 texts to respect OpenAI rate limits
- Implements error handling and retry logic for API failures

## 🧪 Testing

Run the system with different parameters to test functionality:

```bash
# Test with small top_k for quick results
python -m src.utils.graph_rag "data/ADV_GRAPH_20240119 - ADV_GRAPH_20240119.csv" --top_k 5

# Test without API key (substring fallback)
unset OPENAI_API_KEY
python -m src.utils.graph_rag "data/ADV_GRAPH_20240119 - ADV_GRAPH_20240119.csv" --top_k 10

# Test with custom cache location
python -m src.utils.graph_rag "data/ADV_GRAPH_20240119 - ADV_GRAPH_20240119.csv" --cache "custom_embeddings.pkl"
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Ensure code passes Ruff linting: `ruff check .`
5. Submit a pull request

## 📄 License

This project is intended for academic research on bias and fairness in AI systems.

## 🔗 Dependencies

- **networkx**: Graph data structure and algorithms
- **pandas**: Data manipulation and CSV processing
- **numpy**: Numerical computing and array operations
- **openai**: OpenAI API client for embeddings and chat completions

## 🐛 Troubleshooting

### Common Issues

**API Key Error**: Ensure `OPENAI_API_KEY` is set correctly
```bash
export OPENAI_API_KEY="sk-your-key-here"
```

**Memory Issues**: For large datasets, consider reducing `chunk_size` in the code

**Network Issues**: Use proxychains if you live in Iran like me :))
```bash
proxychains python -m src.utils.graph_rag ...
```

**Cache Corruption**: Delete `embeddings.pkl` to regenerate embeddings

## 📚 Research Context

This tool is designed for academic research into:
- Bias detection in language models
- Stereotype analysis in structured data
- Fairness evaluation in AI systems
- Graph-based knowledge representation
- Retrieval-augmented generation techniques
