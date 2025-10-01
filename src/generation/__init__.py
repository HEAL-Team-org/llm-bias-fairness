"""Generation Layer - Image generation and synthesis.

This module provides interfaces and implementations for generating images
from text prompts using various AI models like DALL-E 3.

Exports:
    - BaseImageGenerator: Abstract base class for image generators
    - DALLE3Generator: DALL-E 3 implementation
    - MockImageGenerator: Testing/mock implementation
    - ImageGenerationError: Custom exception for generation errors
    - create_image_generator: Factory function for creating generators
"""

from src.generation.image_generator import (
    BaseImageGenerator,
    DALLE3Generator,
    ImageGenerationError,
    MockImageGenerator,
    create_image_generator,
)

__all__ = [
    "BaseImageGenerator",
    "DALLE3Generator",
    "ImageGenerationError",
    "MockImageGenerator",
    "create_image_generator",
]
