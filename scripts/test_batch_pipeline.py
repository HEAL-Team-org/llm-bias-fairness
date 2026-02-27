#!/usr/bin/env python3
"""Test script for the complete batch processing pipeline.

This script demonstrates the end-to-end pipeline:
1. Reading prompts from CSV
2. Running dual-pipeline enhancement
3. Generating images for original and enhanced prompts
4. Saving results with trackable filenames
"""

import logging
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from scripts.batch_processor import BatchProcessor


def main():
    """Run a simple test of the batch processing pipeline."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("batch_test.log")
        ]
    )

    logger = logging.getLogger(__name__)

    # Test configuration
    input_csv = "test_prompts.csv"
    output_dir = "test_results"
    image_dir = "test_images"

    # Check if test CSV exists
    if not Path(input_csv).exists():
        logger.error(f"Test CSV not found: {input_csv}")
        logger.info("Please run this script from the project root directory")
        return 1

    try:
        logger.info("Starting batch processing test...")

        # Initialize processor with mock image generator for testing
        # Change to "dalle3" for real DALL-E 3 generation (requires OpenAI API key)
        processor = BatchProcessor(
            image_generator_type="mock",  # Use "dalle3" for real image generation
            output_dir=output_dir,
            image_output_dir=image_dir
        )

        # Process first 3 rows as a test
        logger.info("Processing first 3 rows from test_prompts.csv...")
        output_csv_path, results = processor.process_csv(
            input_csv_path=input_csv,
            prompt_column="prompt",
            max_rows=3,  # Process only first 3 for testing
            save_intermediate=True
        )

        # Print results summary
        logger.info("="*60)
        logger.info("BATCH PROCESSING TEST RESULTS")
        logger.info("="*60)

        successful = len([r for r in results if r.error is None])
        total = len(results)

        print(f"\nProcessed: {successful}/{total} prompts successfully")
        print(f"Results saved to: {output_csv_path}")
        print(f"Images saved to: {image_dir}/")

        # Show individual results
        for i, result in enumerate(results):
            print(f"\n--- Prompt {i+1} ---")
            print(f"Original: {result.original_prompt[:80]}...")
            if result.enhanced_prompt:
                print(f"Enhanced: {result.enhanced_prompt[:80]}...")
                print(f"Bias Score: {result.bias_score}")
                print(f"Diversity Score: {result.diversity_score}")

            if result.original_image_path:
                print(f"Original Image: {result.original_image_path}")
            if result.enhanced_image_path:
                print(f"Enhanced Image: {result.enhanced_image_path}")

            if result.error:
                print(f"Error: {result.error}")

            if result.processing_time:
                print(f"Processing Time: {result.processing_time:.2f}s")

        logger.info("Test completed successfully!")

        # Instructions for next steps
        print("\n" + "="*60)
        print("NEXT STEPS:")
        print("="*60)
        print("1. To use real DALL-E 3 image generation:")
        print("   - Set OPENAI_API_KEY environment variable")
        print("   - Change image_generator_type to 'dalle3' in the script")
        print("   - Or use command line: python batch_processor.py test_prompts.csv --image-generator dalle3")
        print()
        print("2. To process your own CSV file:")
        print("   - Ensure it has a column with prompts")
        print("   - Run: python batch_processor.py your_file.csv --prompt-column your_column_name")
        print()
        print("3. Check the results:")
        print(f"   - CSV results: {output_csv_path}")
        print(f"   - JSON details: {output_csv_path.replace('.csv', '.json')}")
        print(f"   - Generated images: {image_dir}/")

        return 0

    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit(main())
