"""Simple test of dual-pipeline enhancement with Stable Diffusion.

This is a minimal version for testing the complete workflow.
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Test the dual pipeline with minimal setup."""
    
    print("=" * 80)
    print("Dual-Pipeline Enhancement with Stable Diffusion - Simple Test")
    print("=" * 80)
    
    # Test prompts
    prompts = [
        {
            "id": "prompt_001_doctor",
            "text": "Generate an image of a doctor who is smiling at the camera"
        },
        {
            "id": "prompt_002_nurse",
            "text": "Generate an image of a nurse who is smiling at the camera"
        }
    ]
    
    # Output directory
    output_dir = Path(__file__).parent.parent / "results" / "simple_test" / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    print(f"Number of prompts: {len(prompts)}")
    
    # Test 1: Check if we can import the enhancement module
    print("\n[1/4] Testing enhancement module import...")
    try:
        from enhance_prompt_dual_pipeline import (
            load_enhancement_systems,
            run_dual_pipeline_enhancement,
            setup_logging,
        )
        print("✓ Enhancement module imported successfully")
    except Exception as e:
        print(f"✗ Failed to import enhancement module: {e}")
        return 1
    
    # Test 2: Check if we can import image generator
    print("\n[2/4] Testing image generator import...")
    try:
        from src.generation.image_generator import create_image_generator, ImageGenerationError
        print("✓ Image generator module imported successfully")
    except Exception as e:
        print(f"✗ Failed to import image generator: {e}")
        return 1
    
    # Test 3: Try to initialize Stable Diffusion generator
    print("\n[3/4] Testing Stable Diffusion initialization...")
    try:
        generator = create_image_generator(
            generator_type="stable-diffusion",
            model="stabilityai/stable-diffusion-xl-base-1.0",
            output_dir=output_dir / "images"
        )
        print(f"✓ Stable Diffusion generator initialized: {generator}")
        
        # Test a single image generation
        print("\n[4/4] Testing single image generation...")
        test_prompt = "a simple test image of a cat"
        filepath, metadata = generator.generate_image(
            prompt=test_prompt,
            num_inference_steps=20,  # Fewer steps for faster test
            width=512,  # Smaller size for faster test
            height=512,
            seed=42
        )
        print(f"✓ Test image generated: {filepath}")
        print(f"  Model: {metadata['model']}")
        print(f"  Size: {metadata['size']}")
        
    except ImageGenerationError as e:
        print(f"✗ Image generator error: {e}")
        print("\nNote: Stable Diffusion requires:")
        print("  - pip install diffusers torch transformers accelerate")
        print("  - ~13GB disk space for model download")
        print("  - GPU recommended (or patient CPU)")
        return 1
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)
    print("\nYou can now run the full batch script:")
    print("  python scripts/run_dual_pipeline_batch.py")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
