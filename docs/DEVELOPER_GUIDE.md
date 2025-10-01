# Developer Guide

**LLM Bias & Fairness Project - Developer Documentation**

**Last Updated:** October 2025  
**Version:** 2.0 (Refactored Architecture)

## Table of Contents
1. [Getting Started](#getting-started)
2. [Architecture Overview](#architecture-overview)
3. [Development Setup](#development-setup)
4. [Code Organization](#code-organization)
5. [Adding New Features](#adding-new-features)
6. [Testing](#testing)
7. [Code Style](#code-style)
8. [API Reference](#api-reference)
9. [Common Patterns](#common-patterns)
10. [Debugging](#debugging)

---

## Getting Started

### Prerequisites

- Python 3.13+ (or 3.10+)
- OpenAI API key
- Git
- Virtual environment tool (venv recommended)

### Development Installation

```bash
# Clone repository
git clone <repository-url>
cd llm-bias-fairness

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If exists

# Install in editable mode
pip install -e .

# Set up pre-commit hooks (if used)
pre-commit install
```

### Project Structure

```
llm-bias-fairness/
├── src/                    # Main source code
│   ├── config/            # Configuration layer
│   ├── prompts/           # Prompt templates layer
│   ├── data/              # Data access layer
│   ├── knowledge/         # Knowledge/RAG layer
│   ├── generation/        # Image generation layer
│   ├── evaluation/        # Evaluation layer
│   ├── enhancement/       # Enhancement layer
│   └── cli/               # CLI layer
├── data/                  # Data files
│   ├── biases/           # Bias CSV files
│   └── cultural_values/  # Cultural triples
├── docs/                  # Documentation
├── scripts/               # Utility scripts
├── tests/                 # Test files
├── main.py               # Entry point
└── requirements.txt      # Dependencies
```

---

## Architecture Overview

### Layered Architecture

The system follows a strict layered architecture with unidirectional dependencies:

```
Layer 8: CLI           → User interface
Layer 7: Enhancement   → Prompt enhancement
Layer 6: Evaluation    → Visual bias metrics
Layer 5: Generation    → Image generation
Layer 4: Knowledge     → RAG systems
Layer 3: Data          → Data access, caching
Layer 2: Prompts       → Template management
Layer 1: Config        → Settings, configuration
```

**Key Principle**: Higher layers can depend on lower layers, but NOT vice versa.

### Dependency Rules

✅ **Allowed**:
- CLI → Enhancement
- Enhancement → Knowledge
- Knowledge → Data
- Any layer → Config

❌ **Not Allowed**:
- Config → Any layer
- Data → Knowledge
- Knowledge → Enhancement
- Circular dependencies

---

## Development Setup

### Environment Variables

Create `.env` file in project root:

```bash
# Required
OPENAI_API_KEY=sk-...

# Optional
OPENAI_API_BASE=https://api.openai.com/v1
EMBEDDING_MODEL=text-embedding-3-small
CHAT_MODEL=gpt-4-turbo-preview
LOG_LEVEL=INFO
```

### IDE Configuration

#### VS Code

Create `.vscode/settings.json`:

```json
{
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.linting.flake8Enabled": true,
  "python.formatting.provider": "black",
  "python.testing.pytestEnabled": true,
  "python.testing.unittestEnabled": false,
  "python.analysis.typeCheckingMode": "basic",
  "files.exclude": {
    "**/__pycache__": true,
    "**/*.pyc": true,
    "**/.pytest_cache": true
  }
}
```

#### PyCharm

1. Mark `src/` as "Sources Root"
2. Enable Python type checking
3. Configure Black formatter
4. Set up pytest as test runner

### Running Tests

```bash
# Run all phase tests
python test_phase1.py
python test_phase2.py
# ... etc

# Run specific test file
python test_phase8.py

# Run with verbose output
python test_phase8.py -v

# Run all tests at once (if test runner configured)
pytest tests/
```

---

## Code Organization

### Layer 1: Config (`src/config/`)

**Purpose**: Centralized configuration management

```python
# src/config/settings.py
from pydantic import BaseSettings

class Settings(BaseSettings):
    OPENAI_API_KEY: str
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    
    class Config:
        env_file = ".env"

settings = Settings()
```

**Usage**:
```python
from src.config import settings

api_key = settings.OPENAI_API_KEY
```

### Layer 2: Prompts (`src/prompts/`)

**Purpose**: Template management

```python
# src/prompts/prompt_manager.py
class PromptManager:
    def get_prompt(self, category: str, name: str, **kwargs) -> str:
        """Load and format prompt template."""
        template = self._load_template(category, name)
        return template.format(**kwargs)
```

**Usage**:
```python
from src.prompts import PromptManager

pm = PromptManager()
prompt = pm.get_prompt("enhancement", "dual_pipeline", 
                       original="...", context="...")
```

### Layer 3: Data (`src/data/`)

**Purpose**: Data access and caching

```python
# src/data/embeddings.py
class EmbeddingCache:
    """Persistent embedding cache."""
    
    def __init__(self, cache_file: str):
        self.cache_file = cache_file
        self.cache = self._load()
    
    def get(self, text: str) -> Optional[list[float]]:
        """Get cached embedding."""
        return self.cache.get(text)
    
    def set(self, text: str, embedding: list[float]) -> None:
        """Cache embedding."""
        self.cache[text] = embedding
        self._save()
```

**Usage**:
```python
from src.data import EmbeddingCache, OpenAIEmbedder

cache = EmbeddingCache("embeddings.pkl")
embedder = OpenAIEmbedder(cache=cache)
vector = embedder.embed("text")
```

### Layer 4: Knowledge (`src/knowledge/`)

**Purpose**: RAG systems and knowledge retrieval

```python
# src/knowledge/stereoset.py
class StereoSetRAG:
    """Stereotype detection RAG system."""
    
    def get_negative_examples(
        self, 
        prompt: str, 
        top_k: int = 5
    ) -> list[dict]:
        """Retrieve negative stereotype examples."""
        embedding = self.embedder.embed(prompt)
        similarities = self._compute_similarities(embedding)
        return self._get_top_k(similarities, top_k)
```

**Usage**:
```python
from src.knowledge import StereoSetRAG, DiversityRAG, GraphRAG

stereoset = StereoSetRAG()
examples = stereoset.get_negative_examples("doctor", top_k=5)

diversity = DiversityRAG()
recs = diversity.get_recommendations("engineer", top_k=5)

graphrag = GraphRAG()
results = graphrag.query("stereotypes", top_k=10)
```

### Layer 5: Generation (`src/generation/`)

**Purpose**: Image generation

```python
# src/generation/base.py
class BaseImageGenerator(ABC):
    """Abstract base class for image generators."""
    
    @abstractmethod
    def generate(
        self, 
        prompt: str, 
        filename: str
    ) -> ImageGenerationResult:
        """Generate image from prompt."""
        pass
```

**Usage**:
```python
from src.generation import create_image_generator

generator = create_image_generator("dalle3")
result = generator.generate("a doctor", "doctor.png")
```

### Layer 6: Evaluation (`src/evaluation/`)

**Purpose**: Visual bias evaluation

```python
# src/evaluation/visual_bias_evaluator.py
class VisualBiasEvaluator:
    """Visual bias and diversity evaluation."""
    
    def evaluate_image(self, image_path: str) -> EvaluationResult:
        """Evaluate demographics in image."""
        faces = self._detect_faces(image_path)
        predictions = self._predict_demographics(faces)
        return self._create_result(predictions)
```

**Usage**:
```python
from src.evaluation import VisualBiasEvaluator

evaluator = VisualBiasEvaluator()
result = evaluator.evaluate_image("image.png")
metrics = result.bias_metrics
```

### Layer 7: Enhancement (`src/enhancement/`)

**Purpose**: Prompt enhancement

```python
# src/enhancement/prompt_enhancer.py
@dataclass
class EnhancementConfig:
    """Enhancement configuration."""
    bias_threshold: int = 75
    diversity_threshold: int = 80
    max_iterations: int = 5

class EnhancementSystem:
    """Main enhancement orchestrator."""
    
    def enhance(self, prompt: str) -> EnhancementResult:
        """Enhance prompt with bias mitigation and diversity."""
        for iteration in range(self.config.max_iterations):
            enhanced = self._enhance_once(prompt)
            if self._meets_thresholds(enhanced):
                return enhanced
        return enhanced
```

**Usage**:
```python
from src.enhancement import EnhancementSystem, EnhancementConfig

config = EnhancementConfig(bias_threshold=80)
system = EnhancementSystem(config=config)
result = system.enhance("a doctor")
```

### Layer 8: CLI (`src/cli/`)

**Purpose**: Command-line interface

```python
# src/cli/commands/enhance.py
class EnhanceCommand(BaseCommand):
    """Enhance command implementation."""
    
    def add_arguments(self, parser: ArgumentParser) -> None:
        """Add command-specific arguments."""
        parser.add_argument("prompt", nargs="?")
        parser.add_argument("--bias-threshold", type=int)
    
    def execute(self, args: Namespace) -> int:
        """Execute the command."""
        # Implementation
        return 0
```

**Usage**:
```python
from src.cli import CLIRunner

runner = CLIRunner()
exit_code = runner.main()
```

---

## Adding New Features

### Adding a New Command

1. **Create command file** in `src/cli/commands/`:

```python
# src/cli/commands/analyze.py
from src.cli.commands import BaseCommand

class AnalyzeCommand(BaseCommand):
    """Analyze prompts without enhancement."""
    
    def add_arguments(self, parser):
        parser.add_argument("prompt")
        parser.add_argument("--detailed", action="store_true")
    
    def execute(self, args):
        # Implementation
        return 0
```

2. **Register command** in `src/cli/__init__.py`:

```python
from src.cli.commands.analyze import AnalyzeCommand

__all__ = [
    # ... existing exports
    "AnalyzeCommand",
]
```

3. **Add to parser** in `src/cli/parser.py`:

```python
# Add analyze command
analyze_parser = subparsers.add_parser("analyze", help="...")
AnalyzeCommand().add_arguments(analyze_parser)
```

4. **Register in runner** in `src/cli/runner.py`:

```python
self.commands = {
    # ... existing commands
    "analyze": AnalyzeCommand,
}
```

### Adding a New RAG System

1. **Create RAG class** in `src/knowledge/`:

```python
# src/knowledge/my_rag.py
class MyRAG:
    """My custom RAG system."""
    
    def __init__(self):
        self.embedder = OpenAIEmbedder()
        self.data = self._load_data()
    
    def query(self, prompt: str, top_k: int = 5) -> list[dict]:
        """Query the RAG system."""
        embedding = self.embedder.embed(prompt)
        # Implementation
        return results
```

2. **Export in** `src/knowledge/__init__.py`:

```python
from src.knowledge.my_rag import MyRAG

__all__ = [
    # ... existing exports
    "MyRAG",
]
```

3. **Integrate with enhancement**:

```python
# In EnhancementSystem
def __init__(self, my_rag: Optional[MyRAG] = None):
    self.my_rag = my_rag

def enhance(self, prompt: str):
    if self.my_rag:
        context = self.my_rag.query(prompt)
        # Use context in enhancement
```

### Adding a New Image Generator

1. **Create generator class**:

```python
# src/generation/my_generator.py
from src.generation.base import BaseImageGenerator

class MyGenerator(BaseImageGenerator):
    """My custom image generator."""
    
    def generate(self, prompt: str, filename: str):
        # Implementation
        return ImageGenerationResult(...)
```

2. **Register in factory**:

```python
# src/generation/factory.py
def create_image_generator(generator_type: str):
    if generator_type == "my_generator":
        return MyGenerator()
    # ... existing generators
```

---

## Testing

### Test Structure

Each phase has a dedicated test file:

```python
# test_phase8.py
def test_cli_layer_imports():
    """TEST 1: Verify all CLI layer classes can be imported."""
    from src.cli import CLIRunner, create_cli_parser
    assert CLIRunner is not None

def test_enhance_command():
    """TEST 2: Verify EnhanceCommand exists."""
    from src.cli import EnhanceCommand
    command = EnhanceCommand()
    assert command is not None
```

### Running Tests

```bash
# Run single test file
python test_phase8.py

# Run all phase tests
for test in test_phase*.py; do
    python "$test"
done

# With pytest (if configured)
pytest tests/
pytest tests/test_phase8.py -v
pytest tests/test_phase8.py::test_cli_layer_imports
```

### Writing Tests

**Guidelines**:
1. One test file per layer/phase
2. Test imports first
3. Test basic functionality
4. Test no circular imports
5. Use descriptive test names
6. Print clear success/failure messages

**Example**:
```python
def test_my_feature():
    """TEST X: Description of what is being tested."""
    print("=" * 70)
    print("TEST X: My Feature")
    print("=" * 70)
    
    try:
        # Test code
        assert condition, "Error message"
        
        print("✅ Test passed")
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
```

### Integration Testing

Test layer interactions:

```python
def test_enhancement_with_knowledge():
    """Test enhancement layer uses knowledge layer correctly."""
    from src.enhancement import EnhancementSystem
    from src.knowledge import StereoSetRAG
    
    stereoset = StereoSetRAG()
    system = EnhancementSystem(stereoset_rag=stereoset)
    result = system.enhance("test")
    
    assert result is not None
    assert len(result.negative_examples) > 0
```

---

## Code Style

### Python Style Guide

Follow PEP 8 with these specifics:

```python
# Line length: 88 characters (Black default)
# Indentation: 4 spaces
# Quotes: Double quotes for strings
# Imports: Absolute imports from src/

# Good
from src.knowledge import StereoSetRAG
from src.config import settings

# Bad
from ..knowledge import StereoSetRAG
import sys
sys.path.append("..")
```

### Type Hints

Always use type hints:

```python
from typing import Optional, list, dict

def enhance(
    prompt: str,
    threshold: int = 75,
    max_iter: int = 5
) -> EnhancementResult:
    """Enhance prompt with type hints."""
    pass

# Use modern syntax (Python 3.10+)
def process(items: list[str]) -> dict[str, int]:
    pass
```

### Docstrings

Use Google-style docstrings:

```python
def enhance(prompt: str, threshold: int = 75) -> EnhancementResult:
    """Enhance prompt for bias mitigation.
    
    Args:
        prompt: Original prompt to enhance
        threshold: Minimum quality threshold (0-100)
        
    Returns:
        EnhancementResult with enhanced prompt and scores
        
    Raises:
        ValueError: If prompt is empty
        APIError: If OpenAI API fails
        
    Example:
        >>> result = enhance("a doctor", threshold=80)
        >>> print(result.final_prompt)
        "a diverse doctor from various backgrounds"
    """
    pass
```

### Naming Conventions

```python
# Classes: PascalCase
class EnhancementSystem:
    pass

# Functions/methods: snake_case
def enhance_prompt():
    pass

# Constants: UPPER_SNAKE_CASE
MAX_ITERATIONS = 5

# Private: Leading underscore
def _internal_method():
    pass

# Protected: Single leading underscore
def _compute_score():
    pass
```

### File Organization

```python
"""Module docstring describing purpose.

Detailed description if needed.
"""

# Standard library imports
import logging
from pathlib import Path

# Third-party imports
import numpy as np
from openai import OpenAI

# Local imports
from src.config import settings
from src.data import EmbeddingCache

# Constants
MAX_RETRIES = 3
DEFAULT_THRESHOLD = 75

# Module-level variables
logger = logging.getLogger(__name__)

# Classes and functions
class MyClass:
    pass

def my_function():
    pass
```

---

## API Reference

### Core APIs

#### Enhancement API

```python
from src.enhancement import (
    EnhancementSystem,
    EnhancementConfig,
    EnhancementResult
)

# Configuration
config = EnhancementConfig(
    bias_threshold=75,
    diversity_threshold=80,
    max_iterations=5,
    use_stereoset=True,
    use_diversity_rag=True,
    use_graphrag=True
)

# System
system = EnhancementSystem(config=config)
result = system.enhance("a doctor")

# Result attributes
result.original_prompt: str
result.final_prompt: str
result.initial_bias_score: float
result.final_bias_score: float
result.initial_diversity_score: float
result.final_diversity_score: float
result.iterations: int
result.bias_improvement: float  # Property
result.diversity_improvement: float  # Property
result.both_thresholds_met: bool  # Property
```

#### Knowledge API

```python
from src.knowledge import StereoSetRAG, DiversityRAG, GraphRAG

# StereoSet
stereoset = StereoSetRAG()
examples = stereoset.get_negative_examples(
    prompt="a doctor",
    top_k=5
)

# Diversity
diversity = DiversityRAG()
recommendations = diversity.get_recommendations(
    prompt="an engineer",
    top_k=5
)

# GraphRAG
graphrag = GraphRAG()
results = graphrag.query(
    question="What stereotypes exist?",
    top_k=10
)
```

#### Generation API

```python
from src.generation import create_image_generator

generator = create_image_generator(
    generator_type="dalle3",
    output_dir="images/",
    quality="hd",
    size="1024x1024"
)

result = generator.generate(
    prompt="a diverse doctor",
    filename="doctor.png"
)

# Result attributes
result.success: bool
result.image_path: str
result.metadata: dict
result.error: Optional[str]
```

#### Evaluation API

```python
from src.evaluation import VisualBiasEvaluator

evaluator = VisualBiasEvaluator(
    fairface_model_path="models/fairface.pth",
    dlib_model_path="models/dlib.dat"
)

result = evaluator.evaluate_image("image.png")

# Metrics
metrics = result.bias_metrics
metrics.bias_w: float
metrics.bias_p: float
metrics.ens: float
metrics.kl_divergence: float
```

---

## Common Patterns

### Dependency Injection

```python
# Good: Accept dependencies
class EnhancementSystem:
    def __init__(
        self,
        stereoset_rag: Optional[StereoSetRAG] = None,
        diversity_rag: Optional[DiversityRAG] = None
    ):
        self.stereoset_rag = stereoset_rag
        self.diversity_rag = diversity_rag

# Usage
stereoset = StereoSetRAG()
system = EnhancementSystem(stereoset_rag=stereoset)

# Bad: Hard-coded dependencies
class EnhancementSystem:
    def __init__(self):
        self.stereoset_rag = StereoSetRAG()  # Hard-coded!
```

### Factory Pattern

```python
# Good: Factory function
def create_image_generator(
    generator_type: str,
    **kwargs
) -> BaseImageGenerator:
    if generator_type == "dalle3":
        return DALLE3Generator(**kwargs)
    elif generator_type == "mock":
        return MockImageGenerator(**kwargs)
    raise ValueError(f"Unknown generator: {generator_type}")

# Usage
generator = create_image_generator("dalle3")
```

### Dataclass Pattern

```python
from dataclasses import dataclass

@dataclass
class EnhancementResult:
    """Enhancement result with computed properties."""
    original_prompt: str
    final_prompt: str
    initial_bias_score: float
    final_bias_score: float
    
    @property
    def bias_improvement(self) -> float:
        """Compute bias improvement."""
        return self.final_bias_score - self.initial_bias_score
```

### Context Manager

```python
# Good: Resource cleanup
class EmbeddingCache:
    def __enter__(self):
        self._load()
        return self
    
    def __exit__(self, *args):
        self._save()

# Usage
with EmbeddingCache("cache.pkl") as cache:
    cache.set("text", embedding)
```

---

## Debugging

### Logging

```python
import logging

logger = logging.getLogger(__name__)

# Use appropriate levels
logger.debug("Detailed debugging information")
logger.info("General information")
logger.warning("Warning messages")
logger.error("Error messages")
logger.exception("Error with traceback")

# Enable verbose logging
python main.py --verbose enhance "a doctor"
```

### Debugging Tools

```python
# IPython debugger
import ipdb; ipdb.set_trace()

# Standard debugger
import pdb; pdb.set_trace()

# VS Code breakpoints
# Set breakpoint in IDE, run with debugger

# Print debugging (temporary)
print(f"DEBUG: variable={variable}")
```

### Common Debug Scenarios

**Import Errors**:
```python
# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
```

**API Errors**:
```python
try:
    result = api_call()
except Exception as e:
    logger.exception("API call failed")
    raise
```

**Cache Issues**:
```bash
# Clear all caches
rm *.pkl

# Clear specific cache
rm embeddings.pkl
```

---

## Contributing

### Git Workflow

```bash
# Create feature branch
git checkout -b feature/my-feature

# Make changes and commit
git add .
git commit -m "Add my feature"

# Push to remote
git push origin feature/my-feature

# Create pull request on GitHub
```

### Commit Messages

```
feat: Add new RAG system for X
fix: Correct bias scoring calculation
docs: Update developer guide
test: Add tests for enhancement layer
refactor: Simplify CLI command structure
chore: Update dependencies
```

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] All tests pass
- [ ] Added new tests
- [ ] Manual testing done

## Checklist
- [ ] Code follows style guide
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings
```

---

## Resources

### Internal Documentation
- `docs/ARCHITECTURE_V2.md`: System architecture
- `docs/USER_GUIDE_V2.md`: User documentation
- `docs/CLI_REFERENCE.md`: CLI commands
- `docs/PHASE*_COMPLETION.md`: Refactoring docs

### External Resources
- [OpenAI API Docs](https://platform.openai.com/docs)
- [Python Type Hints](https://docs.python.org/3/library/typing.html)
- [Pydantic](https://docs.pydantic.dev/)
- [pytest](https://docs.pytest.org/)

---

**Developer Guide Version:** 2.0  
**Last Updated:** October 2025  
**For Questions**: Open an issue on GitHub
