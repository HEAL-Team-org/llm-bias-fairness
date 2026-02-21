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
    from openai import AzureOpenAI, OpenAI
except ImportError:
    OpenAI = None
    AzureOpenAI = None

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
            image_data: Image data (bytes, PIL Image, URL, or data URL)
            filepath: Path to save the image

        Returns:
            Saved filepath as string

        """
        try:
            if isinstance(image_data, str):
                if image_data.startswith("http"):
                    # Download from HTTP URL
                    response = requests.get(image_data, timeout=30)
                    response.raise_for_status()
                    image_data = response.content
                elif image_data.startswith("data:image"):
                    # Handle base64 data URL
                    import base64
                    # Extract base64 data after the comma
                    base64_data = image_data.split(",", 1)[1]
                    image_data = base64.b64decode(base64_data)

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
        quality: str = "standard",
        base_url: Optional[str] = None
    ):
        """Initialize DALL-E 3 generator.

        Args:
            api_key: OpenAI API key (will try env var if None)
            output_dir: Directory to save images
            model: DALL-E model version
            size: Image size (1024x1024, 1792x1024, 1024x1792)
            quality: Image quality (standard, hd)
            base_url: Custom base URL for OpenAI API (optional)

        """
        super().__init__(output_dir)

        try:
            from openai import AzureOpenAI, OpenAI
            
            from src.config.settings import (
                get_azure_deployment,
                get_azure_openai_api_version,
                get_azure_openai_endpoint,
                get_openai_base_url,
                is_azure_openai,
            )
        except ImportError as e:
            msg = "OpenAI package not installed. Run: pip install openai"
            raise ImageGenerationError(msg) from e

        # Determine if using Azure OpenAI
        self.use_azure = is_azure_openai()

        if self.use_azure:
            # For Azure, use deployment name instead of model name
            self.model = model or get_azure_deployment("image")
            if not self.model:
                logger.warning(
                    "Azure OpenAI provider selected but no image deployment "
                    "configured - using standard model name"
                )
                self.model = "dall-e-3"
        else:
            self.model = model

        self.size = size
        self.quality = quality

        # Initialize the appropriate client
        if self.use_azure:
            endpoint = get_azure_openai_endpoint()
            api_version = get_azure_openai_api_version()
            if not endpoint:
                msg = (
                    "Azure OpenAI provider selected but no endpoint "
                    "configured"
                )
                raise ImageGenerationError(msg)

            self.client = AzureOpenAI(
                api_key=api_key,
                azure_endpoint=endpoint,
                api_version=api_version
            )
        else:
            # Initialize standard OpenAI client
            client_kwargs = {"api_key": api_key}
            final_base_url = base_url or get_openai_base_url()
            if final_base_url:
                client_kwargs["base_url"] = final_base_url
            self.client = OpenAI(**client_kwargs)

        # Test API connection
        try:
            self.client.models.list()
            provider_name = "Azure OpenAI" if self.use_azure else "OpenAI"
            logger.info(
                f"DALL-E 3 generator initialized successfully with {provider_name}"
            )
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

            logger.info(f"Generating image with model '{self.model}': '{prompt[:100]}...'")

            # Prepare base parameters
            api_params = {
                "model": self.model,
                "prompt": prompt,
                "size": image_size,
                "n": 1,
            }

            # Add optional parameters only for DALL-E 3
            # gpt-image-1 and other models may not support these parameters
            if self.model.startswith("dall-e"):
                api_params["quality"] = image_quality
                api_params["style"] = style
                api_params["response_format"] = "url"

            # Make API call
            response = self.client.images.generate(**api_params)

            # Get image URL or base64 data
            # DALL-E 3 with response_format="url" returns .url
            # Other models or b64_json format return .b64_json
            image_data = response.data[0]

            if hasattr(image_data, 'url') and image_data.url:
                image_url = image_data.url
            elif hasattr(image_data, 'b64_json') and image_data.b64_json:
                # Handle base64 encoded response
                import base64
                image_url = f"data:image/png;base64,{image_data.b64_json}"
            else:
                raise ImageGenerationError("No image URL or base64 data in response")

            revised_prompt = getattr(image_data, "revised_prompt", prompt)

            # Generate filename if not provided
            if not filename:
                filename = self._generate_filename(prompt)

            filepath = self.output_dir / filename

            # Download and save image (handles both URL and base64)
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


class StableDiffusionGenerator(BaseImageGenerator):
    """Stable Diffusion image generator using Hugging Face Diffusers."""

    def __init__(
        self,
        model: str = "stabilityai/stable-diffusion-xl-base-1.0",
        output_dir: Union[str, Path] = "generated_images",
        device: str = "auto",
        enable_optimizations: bool = True,
        **kwargs
    ):
        """Initialize Stable Diffusion generator.

        Args:
            model: Model name/path from Hugging Face Hub
            output_dir: Directory to save images
            device: Device to use ("cuda", "cpu", or "auto")
            enable_optimizations: Enable memory optimizations (attention slicing, VAE slicing)
            **kwargs: Additional pipeline parameters

        """
        super().__init__(output_dir)

        try:
            import torch
            from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
        except ImportError as e:
            msg = (
                "Required packages not installed. "
                "Run: pip install diffusers torch transformers accelerate"
            )
            raise ImageGenerationError(msg) from e

        self.model = model
        
        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading Stable Diffusion model: {self.model}")
        logger.info(f"Device: {self.device}")

        # Choose pipeline based on model name
        if "xl" in self.model.lower():
            PipelineClass = StableDiffusionXLPipeline
        else:
            PipelineClass = StableDiffusionPipeline

        # Load pipeline with optimizations
        pipe_kwargs = {
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
            "use_safetensors": True,
        }

        # Add variant for SDXL models on CUDA
        if "xl" in self.model.lower() and self.device == "cuda":
            pipe_kwargs["variant"] = "fp16"

        # Merge with user-provided kwargs
        pipe_kwargs.update(kwargs)

        try:
            self.pipeline = PipelineClass.from_pretrained(self.model, **pipe_kwargs)
            self.pipeline = self.pipeline.to(self.device)

            # Enable memory optimizations for CUDA
            if self.device == "cuda" and enable_optimizations:
                try:
                    self.pipeline.enable_attention_slicing()
                    logger.info("✓ Enabled attention slicing")
                except Exception:
                    logger.warning("Could not enable attention slicing")

                try:
                    self.pipeline.enable_vae_slicing()
                    logger.info("✓ Enabled VAE slicing")
                except Exception:
                    logger.warning("Could not enable VAE slicing")

            logger.info(f"Stable Diffusion generator initialized: {self.model}")

        except Exception as e:
            raise ImageGenerationError(f"Failed to load Stable Diffusion model: {e}")

    def generate_image(
        self,
        prompt: str,
        filename: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None,
        **kwargs
    ) -> Tuple[str, Dict]:
        """Generate image using Stable Diffusion.

        Args:
            prompt: Text prompt for image generation
            filename: Optional filename for saving
            negative_prompt: Negative prompt to guide generation away from certain features
            num_inference_steps: Number of denoising steps (more steps = higher quality but slower)
            guidance_scale: How closely to follow the prompt (higher = more faithful)
            width: Image width in pixels
            height: Image height in pixels
            seed: Random seed for reproducibility
            **kwargs: Additional generation parameters

        Returns:
            Tuple of (filepath, metadata)

        """
        try:
            import torch

            logger.info(f"Generating image with Stable Diffusion: '{prompt[:100]}...'")
            logger.info(f"Steps: {num_inference_steps}, Guidance: {guidance_scale}, Size: {width}x{height}")

            # Set random seed for reproducibility
            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)

            # Generate image
            with torch.inference_mode():
                result = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height,
                    generator=generator,
                    **kwargs
                )

            # Get the generated image
            image = result.images[0]

            # Generate filename if not provided
            if not filename:
                filename = self._generate_filename(prompt)

            filepath = self.output_dir / filename

            # Save image
            saved_path = self._save_image(image, filepath)

            # Prepare metadata
            metadata = {
                "model": self.model,
                "backend": "stable-diffusion",
                "original_prompt": prompt,
                "revised_prompt": prompt,  # SD doesn't revise prompts
                "negative_prompt": negative_prompt,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "size": f"{width}x{height}",
                "seed": seed,
                "generation_time": time.time(),
                "filename": filename
            }

            logger.info(f"Successfully generated and saved image: {saved_path}")
            return saved_path, metadata

        except Exception as e:
            raise ImageGenerationError(f"Stable Diffusion generation failed: {e}")


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
        generator_type: Type of generator ("dalle3", "stable-diffusion", "mock")
        **kwargs: Arguments passed to generator constructor
        
    Returns:
        Image generator instance
        
    Raises:
        ValueError: If generator type not supported

    """
    generators = {
        "dalle3": DALLE3Generator,
        "stable-diffusion": StableDiffusionGenerator,
        "stable_diffusion": StableDiffusionGenerator,  # Allow both naming styles
        "mock": MockImageGenerator
    }

    if generator_type not in generators:
        raise ValueError(f"Unsupported generator type: {generator_type}. "
                        f"Available: {list(generators.keys())}")

    return generators[generator_type](**kwargs)
