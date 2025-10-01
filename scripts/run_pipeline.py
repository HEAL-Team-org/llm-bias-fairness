#!/usr/bin/env python3
"""Simple wrapper script for the complete prompt enhancement and image generation pipeline.

This script provides an easy interface to run the complete pipeline with sensible defaults.
"""

import argparse
import os
import sys
from pathlib import Path


def main():
    """Main function with simple interface."""
    parser = argparse.ArgumentParser(
        description="Complete Prompt Enhancement and Image Generation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with sample data (mock images)
  python run_pipeline.py test

  # Process your CSV file with mock images
  python run_pipeline.py your_file.csv

  # Process with real DALL-E 3 images (requires API key)
  python run_pipeline.py your_file.csv --real-images

  # Process first 5 rows only
  python run_pipeline.py your_file.csv --max-rows 5

  # Specify prompt column name
  python run_pipeline.py your_file.csv --prompt-column "description"
        """
    )

    parser.add_argument(
        "input",
        help="CSV file to process, or 'test' to run with sample data"
    )
    parser.add_argument(
        "--prompt-column",
        default="prompt",
        help="Name of column containing prompts (default: prompt)"
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        help="Maximum number of rows to process (default: all)"
    )
    parser.add_argument(
        "--real-images",
        action="store_true",
        help="Use DALL-E 3 for real image generation (requires OPENAI_API_KEY)"
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory for output files (default: results)"
    )

    args = parser.parse_args()

    # Handle test mode
    if args.input.lower() == "test":
        print("Running test with sample data...")
        return run_test()

    # Check if input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        return 1

    # Check for API key if using real images
    if args.real_images and not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set it with: export OPENAI_API_KEY='your-api-key-here'")
        print("Or use mock images by removing --real-images flag")
        return 1

    # Run the pipeline
    print(f"Processing {args.input}...")
    return run_pipeline(args)

def run_test():
    """Run test with sample data."""
    try:
        from test_batch_pipeline import main as test_main
        return test_main()
    except ImportError as e:
        print(f"Error importing test module: {e}")
        return 1

def run_pipeline(args):
    """Run the main pipeline."""
    try:
        import logging

        from batch_processor import BatchProcessor

        # Set up basic logging
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        image_dir = output_dir / "images"

        # Choose image generator
        image_generator = "dalle3" if args.real_images else "mock"

        print(f"Using {image_generator} image generator")
        print(f"Output directory: {output_dir}")

        # Initialize processor
        processor = BatchProcessor(
            image_generator_type=image_generator,
            output_dir=output_dir,
            image_output_dir=image_dir
        )

        # Process the file
        output_csv_path, results = processor.process_csv(
            input_csv_path=args.input,
            prompt_column=args.prompt_column,
            max_rows=args.max_rows,
            save_intermediate=True
        )

        # Print summary
        successful = len([r for r in results if r.error is None])
        total = len(results)

        print(f"\n{'='*60}")
        print("PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Processed: {successful}/{total} prompts successfully")
        print(f"Results saved to: {output_csv_path}")
        print(f"Images saved to: {image_dir}")

        if successful < total:
            print(f"\nWarning: {total - successful} prompts failed to process")
            print("Check the JSON output file for detailed error information")

        return 0

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
