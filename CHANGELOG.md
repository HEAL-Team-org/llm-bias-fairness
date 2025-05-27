# Changelog

All notable changes to the LLM Bias & Fairness GraphRAG project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-05-27

### Added
- **Initial GraphRAG implementation** with OpenAI embeddings integration
- **Knowledge graph processing** from CSV stereotype data using NetworkX
- **Vector similarity search** using OpenAI `text-embedding-3-large` model
- **Persistent embedding cache** system using pickle for cost optimization
- **Top-K retrieval** with configurable similarity threshold
- **LLM answer generation** using OpenAI Chat Completions API
- **Command-line interface** with argparse for easy usage
- **Structured logging** with proper error handling and debugging info
- **Chunked API processing** to handle large datasets within rate limits
- **Fallback substring search** for operation without API key
- **Comprehensive error handling** with retry logic and graceful degradation

### Technical Features
- **Code quality compliance** with Ruff linting standards
- **Type hints** throughout the codebase for better maintainability
- **Environment variable configuration** for API key management
- **Modular architecture** with clean separation of concerns
- **Cross-platform compatibility** (Linux, macOS, Windows)
- **Network proxy support** (tested with proxychains)

### Core Functions
- `load_graph_csv()`: CSV data loading and parsing
- `build_graph()`: NetworkX graph construction from triples
- `embed_texts()`: OpenAI embedding generation with error handling
- `retrieve_with_vectors()`: Vector similarity search and ranking
- `retrieve_by_substring()`: Fallback text-based search
- `answer_question()`: LLM answer generation from retrieved context
- `load_cache()` / `save_cache()`: Persistent embedding storage

### Data Processing
- **13,690 graph nodes** successfully processed and embedded
- **Support for complex graph triples** with subject-predicate-object structure
- **Efficient caching system** preventing redundant API calls
- **Robust CSV parsing** with error handling for malformed data

### Performance Optimizations
- **Batch embedding processing** (2000 texts per chunk)
- **Vector normalization** for improved similarity calculations
- **Memory-efficient numpy operations** for large-scale computations
- **Lazy loading** of embeddings to minimize memory usage

### Documentation
- **Comprehensive README** with installation and usage instructions
- **Inline code documentation** with detailed docstrings
- **Example usage patterns** for various research scenarios
- **Troubleshooting guide** for common issues

### Dependencies
- `networkx==3.4.2` - Graph data structures and algorithms
- `pandas==2.2.3` - Data manipulation and CSV processing  
- `numpy==2.2.6` - Numerical computing and array operations
- `openai==1.82.0` - OpenAI API client for embeddings and completions

### Configuration
- **Environment-based API key management**
- **Configurable cache file location**
- **Adjustable top-K retrieval parameters**
- **Customizable embedding model selection**

### Tested Scenarios
- ✅ **Direct execution** with various top_k values (5, 10, 15, 25)
- ✅ **Proxychains compatibility** for network-restricted environments
- ✅ **Cache persistence** across multiple runs
- ✅ **API error recovery** and fallback mechanisms
- ✅ **Large dataset processing** (77,619 CSV rows)
- ✅ **Multi-query sessions** with efficient caching

### Research Applications
- **Racial stereotype analysis** in structured knowledge graphs
- **Bias detection** in language model responses
- **Fairness evaluation** across different demographic groups
- **Knowledge graph exploration** for bias research
- **Retrieval-augmented generation** for bias-aware AI systems

## [Unreleased]

### Planned Features
- Support for additional embedding models (Hugging Face, Cohere)
- Interactive web interface for easier exploration
- Batch query processing for large-scale analysis
- Export functionality for research results
- Integration with other bias detection frameworks
- Support for multilingual stereotype analysis

---

## Version History

- **v1.0.0** (2025-05-27): Initial stable release with full GraphRAG functionality
- **v0.x.x** (Development): Various prototypes and experimental implementations

## Contributing

When contributing to this project, please:
1. Update this changelog with your changes
2. Follow the semantic versioning scheme
3. Include tests for new features
4. Update documentation as needed
5. Ensure code passes all linting checks

For more information on contributing, see the README.md file.
