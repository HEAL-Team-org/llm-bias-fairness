#!/usr/bin/env python3
"""Quick test script with limited processing for faster demo."""

import logging
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from scripts.batch_processor import BatchProcessor


def main():
    """Run a quick test of the batch processing pipeline."""
    # Set up minimal logging
    logging.basicConfig(level=logging.WARNING)

    print("🚀 Quick Pipeline Test")
    print("=" * 50)

    # Test configuration
    input_csv = "test_prompts.csv"
    output_dir = "quick_test_results"
    image_dir = "quick_test_images"

    try:
        print("Initializing batch processor...")

        # Initialize processor with DALL-E 3 image generator
        processor = BatchProcessor(
            image_generator_type="dalle3",
            output_dir=output_dir,
            image_output_dir=image_dir
        )

        print("Processing 1 prompt from test CSV...")

        # Process only 1 row for quick demo
        output_csv_path, results = processor.process_csv(
            input_csv_path=input_csv,
            prompt_column="prompt",
            max_rows=1,  # Just 1 for quick test
            save_intermediate=True
        )

        # Print results
        print("\n" + "=" * 50)
        print("QUICK TEST RESULTS")
        print("=" * 50)

        result = results[0]
        print("\n📝 Original Prompt:")
        print(f"   {result.original_prompt}")

        if result.enhanced_prompt:
            print("\n✨ Enhanced Prompt:")
            print(f"   {result.enhanced_prompt[:200]}...")
            print("\n📊 Scores:")
            print(f"   Bias Score: {result.bias_score}/100")
            print(f"   Diversity Score: {result.diversity_score}/100")

        if result.original_image_path:
            print("\n🖼️  Images Generated:")
            print(f"   Original: {result.original_image_path}")
            print(f"   Enhanced: {result.enhanced_image_path}")

        if result.error:
            print(f"\n❌ Error: {result.error}")

        print("\n💾 Full Results:")
        print(f"   CSV: {output_csv_path}")
        print(f"   JSON: {output_csv_path.replace('.csv', '.json')}")

        return 0

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
