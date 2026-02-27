"""Test script to demonstrate both DALL-E and Stable Diffusion image generators.

This script shows how to use both image generation backends.

Usage:
    # Test DALL-E
    python test_image_generators.py --backend dalle --prompt "a cat"
    
    # Test Stable Diffusion
    python test_image_generators.py --backend stable-diffusion --prompt "a cat"
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generation.image_generator import create_image_generator, ImageGenerationError

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Test image generators")
    parser.add_argument(
        "--backend",
        choices=["dalle", "stable-diffusion"],
        default="dalle",
        help="Image generation backend"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (optional, uses defaults if not specified)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="a professional photograph of a doctor in a modern hospital",
        help="Text prompt for image generation"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="test_generated_images",
        help="Output directory for generated images"
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=1,
        help="Number of images to generate"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Number of inference steps (Stable Diffusion only)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print(f"Image Generator Test - {args.backend.upper()}")
    print("=" * 80)
    print(f"Prompt: {args.prompt}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of images: {args.num_images}")
    if args.backend == "stable-diffusion":
        print(f"Inference steps: {args.steps}")
    if args.seed:
        print(f"Seed: {args.seed}")
    print("=" * 80)
    
    try:
        # Create generator
        print(f"\n🔧 Setting up {args.backend} generator...")
        
        if args.backend == "dalle":
            generator = create_image_generator(
                generator_type="dalle3",
                output_dir=args.output_dir
            )
        else:  # stable-diffusion
            generator = create_image_generator(
                generator_type="stable-diffusion",
                model=args.model or "stabilityai/stable-diffusion-xl-base-1.0",
                output_dir=args.output_dir
            )
        
        print(f"✓ Generator initialized: {generator}")
        
        # Generate images
        print(f"\n🎨 Generating {args.num_images} image(s)...")
        results = []
        
        for i in range(args.num_images):
            print(f"\n--- Image {i+1}/{args.num_images} ---")
            
            # Use different seed for each image
            seed = args.seed + i if args.seed is not None else None
            
            try:
                filepath, metadata = generator.generate_image(
                    prompt=args.prompt,
                    num_inference_steps=args.steps,
                    seed=seed
                )
                
                results.append({
                    'status': 'success',
                    'filepath': filepath,
                    'metadata': metadata
                })
                
                print(f"✓ Success! Image saved to: {filepath}")
                
            except ImageGenerationError as e:
                results.append({
                    'status': 'error',
                    'error': str(e)
                })
                print(f"✗ Failed: {e}")
        
        # Summary
        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)
        successful = sum(1 for r in results if r['status'] == 'success')
        failed = sum(1 for r in results if r['status'] == 'error')
        print(f"Successful: {successful}/{args.num_images}")
        print(f"Failed: {failed}/{args.num_images}")
        
        if successful > 0:
            print(f"\nGenerated images saved in: {args.output_dir}")
        
        print("=" * 80)
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
