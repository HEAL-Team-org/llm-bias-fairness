# Changelog

All notable changes to the LLM Bias & Fairness GraphRAG project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.1.0] - 2025-01-27 - Script Consolidation & Unified CLI

### 🎯 Major Improvement: Unified Command-Line Interface

This release consolidates all functionality into a single, unified CLI for better user experience and maintainability.

### Added

- **Unified CLI** (`main.py`)
  - Single entry point for all functionality
  - Subcommand architecture: `enhance`, `batch`, `query`, `test`
  - Consistent argument handling across all commands
  - Interactive mode support for all operations
  - Comprehensive help system with examples
  
- **Enhanced Testing Integration**
  - `test --quick`: Fast single-prompt test (~30 seconds)
  - `test --batch`: 3-prompt batch test with mock images
  - `test --visual`: Visual bias evaluation test
  - `test --all`: Complete test suite
  
- **Migration Support**
  - `scripts/README.md`: Complete migration guide from old scripts
  - Backward compatibility: old scripts preserved in `/scripts` directory
  - Command mapping table for easy transition

### Changed

- **Command Structure**
  - Old: Multiple scripts (`enhance_prompt_dual_pipeline.py`, `batch_processor.py`, etc.)
  - New: Single CLI (`python main.py <command>`)
  - Simplified argument names for consistency
  - Default mode is now dual-pipeline (best quality)
  
- **Documentation Updates**
  - `README.md`: Updated with unified CLI examples
  - `docs/USER_GUIDE.md`: Complete rewrite for new CLI
  - All examples now use `main.py` commands
  
- **Project Organization**
  - Moved old scripts to `/scripts` for reference
  - Core modules (`enhance_prompt_dual_pipeline.py`, `batch_processor.py`) remain as importable libraries
  - Cleaner root directory structure

### Deprecated

- Individual script entry points (moved to `/scripts` for backward compatibility):
  - `enhance_prompt.py` → `python main.py enhance --mode basic`
  - `enhance_prompt_sequential.py` → `python main.py enhance --mode sequential`
  - `enhance_prompt_dual_pipeline.py` → `python main.py enhance`
  - `batch_processor.py` (as script) → `python main.py batch`
  - Test scripts → `python main.py test --<mode>`

### Migration Guide

```bash
# Old commands → New commands
python enhance_prompt_dual_pipeline.py -p "a doctor"
→ python main.py enhance "a doctor"

python batch_processor.py prompts.csv
→ python main.py batch prompts.csv

python quick_test.py
→ python main.py test --quick

python main.py --query "stereotypes"  # (old main.py)
→ python main.py query "stereotypes"
```

See `scripts/README.md` for complete migration documentation.

## [3.0.0] - 2025-10-01 - Complete Pipeline with Visual Bias Evaluation

### 🎉 Major Release: Production-Ready Complete Pipeline

This release represents the completion of the full end-to-end pipeline with visual bias evaluation capabilities, making the system production-ready for comprehensive bias mitigation and diversity enhancement.

### Added

- **Visual Bias Evaluation System** (`src/visual_bias_evaluator.py`)
  - FairFace ResNet34 integration for demographic prediction
  - dlib face detection and alignment
  - Comprehensive bias metrics: Bias-W, Bias-P, ENS, KL Divergence
  - Demographic analysis: 7 race categories, 2 genders, 9 age groups
  - Mock model support for testing without actual model files
  
- **Batch Processing Pipeline** (`batch_processor.py`)
  - Complete CSV-to-results workflow
  - Automatic image generation for original and enhanced prompts
  - Optional visual bias evaluation integration
  - Comprehensive result tracking (CSV + JSON)
  - Error handling and recovery for robust processing
  
- **Generic Image Generation Interface** (`src/image_generator.py`)
  - Extensible base class for multiple generators
  - DALL-E 3 implementation with full API support
  - Mock generator for testing without API costs
  - Trackable filename generation
  - Metadata capture for all generations

- **Complete Documentation Suite**
  - `docs/ARCHITECTURE.md` - Comprehensive system architecture
  - `docs/USER_GUIDE.md` - Complete usage guide with examples
  - `docs/VISUAL_BIAS_EVALUATION.md` - Visual evaluation guide
  - Updated README with quick start and architecture diagram

### Changed

- **Enhanced ProcessingResult Dataclass**
  - Added `visual_bias_evaluation` field
  - Expanded metadata capture
  - Better error tracking

- **Improved Enhancement Integration**
  - Seamless integration with batch processor
  - Configurable enhancement methods
  - Better progress tracking and logging

- **Documentation Reorganization**
  - Consolidated 11 outdated docs into 4 current ones
  - Removed redundant index files
  - Clear separation of user guide vs technical architecture

### Improved

- **Error Handling**: Graceful fallbacks for missing dependencies
- **Logging**: Comprehensive debug information throughout pipeline
- **Performance**: Efficient batch processing with caching
- **Extensibility**: Clear patterns for adding new components

### Technical Debt Addressed

- Removed outdated documentation files
- Consolidated overlapping content
- Updated all examples to current API
- Cleaned up legacy code references

## [2.0.0] - 2025-08-23 - Dual-Pipeline Enhancement System

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
