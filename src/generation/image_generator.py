"""Generic Image Generation Interface.

This module provides a generic interface for image generation that can be extended
to support different image generation models like DALL-E, Midjourney, Stable Diffusion, etc.
"""

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import requests
from PIL import Image

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

logger = logging.getLogger(__name__)


class ImageGenerationError(Exception):
    """Custom exception for image generation errors."""


class BaseImageGenerator(ABC):
    """Abstract base class for image generators."""

    def __init__(self, output_dir: Union[str, Path] = "generated_images"):
        """Initialize the image generator.
        
        Args:
            output_dir: Directory to save generated images

        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.generation_count = 0

    @abstractmethod
    def generate_image(
        self,
        prompt: str,
        filename: Optional[str] = None,
        **kwargs
    ) -> Tuple[str, Dict]:
        """Generate an image from a text prompt.
        
        Args:
            prompt: Text prompt for image generation
            filename: Optional filename for saving (auto-generated if None)
            **kwargs: Additional generation parameters
            
        Returns:
            Tuple of (saved_filepath, generation_metadata)
            
        Raises:
            ImageGenerationError: If generation fails

        """
        ...

    def _generate_filename(self, prompt: str, suffix: str = "") -> str:
        """Generate a safe filename from prompt.
        
        Args:
            prompt: Original prompt text
            suffix: Optional suffix to add
            
        Returns:
            Safe filename string

        """
        # Clean prompt for filename
        safe_prompt = "".join(c for c in prompt[:50] if c.isalnum() or c in (" ", "-", "_")).rstrip()
        safe_prompt = safe_prompt.replace(" ", "_")

        # Add generation counter and suffix
        self.generation_count += 1
        filename = f"{self.generation_count:04d}_{safe_prompt}"
        if suffix:
            filename += f"_{suffix}"

        return f"{filename}.png"

    def _save_image(self, image_data: Union[bytes, Image.Image, str], filepath: Path) -> str:
        """Save image data to file.
        
        Args:
            image_data: Image data (bytes, PIL Image, or URL)
            filepath: Path to save the image
            
        Returns:
            Saved filepath as string

        """
        try:
            if isinstance(image_data, str) and image_data.startswith("http"):
                # Download from URL
                response = requests.get(image_data, timeout=30)
                response.raise_for_status()
                image_data = response.content

            if isinstance(image_data, bytes):
                # Save bytes directly
                with filepath.open("wb") as f:
                    f.write(image_data)
            elif isinstance(image_data, Image.Image):
                # Save PIL Image
                image_data.save(filepath, "PNG")
            else:
                msg = f"Unsupported image data type: {type(image_data)}"
                raise ImageGenerationError(msg)

            logger.info(f"Image saved: {filepath}")
            return str(filepath)

        except Exception as e:
            msg = f"Failed to save image to {filepath}: {e}"
            raise ImageGenerationError(msg) from e


class DALLE3Generator(BaseImageGenerator):
    """DALL-E 3 image generator implementation."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        output_dir: Union[str, Path] = "generated_images",
        model: str = "dall-e-3",
        size: str = "1024x1024",
        quality: str = "standard"
    ):
        """Initialize DALL-E 3 generator.
        
        Args:
            api_key: OpenAI API key (will try env var if None)
            output_dir: Directory to save images
            model: DALL-E model version
            size: Image size (1024x1024, 1792x1024, 1024x1792)
            quality: Image quality (standard, hd)

        """
        super().__init__(output_dir)

        try:
            from openai import OpenAI
        except ImportError:
            raise ImageGenerationError("OpenAI package not installed. Run: pip install openai")

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.size = size
        self.quality = quality

        # Test API connection
        try:
            self.client.models.list()
            logger.info("DALL-E 3 generator initialized successfully")
        except Exception as e:
            logger.warning(f"Could not verify OpenAI connection: {e}")

    def generate_image(
        self,
        prompt: str,
        filename: Optional[str] = None,
        size: Optional[str] = None,
        quality: Optional[str] = None,
        style: str = "vivid",
        **kwargs
    ) -> Tuple[str, Dict]:
        """Generate image using DALL-E 3.
        
        Args:
            prompt: Text prompt for image generation
            filename: Optional filename for saving
            size: Image size override
            quality: Quality override
            style: Image style (vivid, natural)
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (filepath, metadata)

        """
        try:
            # Use provided parameters or defaults
            image_size = size or self.size
            image_quality = quality or self.quality

            logger.info(f"Generating image with DALL-E 3: '{prompt[:100]}...'")

            # Make API call
            response = self.client.images.generate(
                model=self.model,
                prompt=prompt,
                size=image_size,
                quality=image_quality,
                style=style,
                n=1,
                response_format="url"
            )

            # Get image URL
            image_url = response.data[0].url
            revised_prompt = getattr(response.data[0], "revised_prompt", prompt)

            # Generate filename if not provided
            if not filename:
                filename = self._generate_filename(prompt)

            filepath = self.output_dir / filename

            # Download and save image
            saved_path = self._save_image(image_url, filepath)

            # Prepare metadata
            metadata = {
                "model": self.model,
                "original_prompt": prompt,
                "revised_prompt": revised_prompt,
                "size": image_size,
                "quality": image_quality,
                "style": style,
                "image_url": image_url,
                "generation_time": time.time(),
                "filename": filename
            }

            logger.info(f"Successfully generated and saved image: {saved_path}")
            return saved_path, metadata

        except Exception as e:
            raise ImageGenerationError(f"DALL-E 3 generation failed: {e}")


class MockImageGenerator(BaseImageGenerator):
    """Mock image generator for testing without API calls."""

    def __init__(self, output_dir: Union[str, Path] = "generated_images"):
        """Initialize mock generator."""
        super().__init__(output_dir)
        logger.info("Mock image generator initialized (for testing)")

    def generate_image(
        self,
        prompt: str,
        filename: Optional[str] = None,
        **kwargs
    ) -> Tuple[str, Dict]:
        """Generate a simple test image.
        
        Args:
            prompt: Text prompt (used for metadata only)
            filename: Optional filename
            **kwargs: Additional parameters (ignored)
            
        Returns:
            Tuple of (filepath, metadata)

        """
        try:
            # Generate filename if not provided
            if not filename:
                filename = self._generate_filename(prompt)

            filepath = self.output_dir / filename

            # Create a simple colored rectangle with text
            image = Image.new("RGB", (512, 512), color="lightblue")

            # Save the mock image
            saved_path = self._save_image(image, filepath)

            # Prepare metadata
            metadata = {
                "model": "mock_generator",
                "original_prompt": prompt,
                "revised_prompt": prompt,
                "size": "512x512",
                "quality": "mock",
                "generation_time": time.time(),
                "filename": filename
            }

            logger.info(f"Mock image generated: {saved_path}")
            return saved_path, metadata

        except Exception as e:
            raise ImageGenerationError(f"Mock generation failed: {e}")


def create_image_generator(
    generator_type: str = "dalle3",
    **kwargs
) -> BaseImageGenerator:
    """Factory function to create image generators.
    
    Args:
        generator_type: Type of generator ("dalle3", "mock")
        **kwargs: Arguments passed to generator constructor
        
    Returns:
        Image generator instance
        
    Raises:
        ValueError: If generator type not supported

    """
    generators = {
        "dalle3": DALLE3Generator,
        "mock": MockImageGenerator
    }

    if generator_type not in generators:
        raise ValueError(f"Unsupported generator type: {generator_type}. "
                        f"Available: {list(generators.keys())}")

    return generators[generator_type](**kwargs)
