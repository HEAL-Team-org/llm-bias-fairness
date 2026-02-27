"""Enhancement Layer - Prompt enhancement and bias mitigation.

This module provides a unified interface for prompt enhancement using multiple
strategies including bias mitigation, diversity enhancement, and cultural awareness.

The EnhancementSystem class orchestrates:
- StereoSet RAG for stereotype detection
- DiversityRAG for cultural diversity
- GraphRAG for additional context
- Dual scoring for bias and diversity metrics

Exports:
    - EnhancementSystem: Main enhancement orchestrator
    - EnhancementResult: Result dataclass
    - EnhancementConfig: Configuration dataclass
"""

from src.enhancement.prompt_enhancer import (
    EnhancementConfig,
    EnhancementResult,
    EnhancementSystem,
)

__all__ = [
    "EnhancementConfig",
    "EnhancementResult",
    "EnhancementSystem",
]
