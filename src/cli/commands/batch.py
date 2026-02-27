"""Batch command - Process CSV files with enhancement and image generation."""

import argparse
import logging
from pathlib import Path

from src.cli.commands import BaseCommand

logger = logging.getLogger(__name__)


class BatchCommand(BaseCommand):
    """Command for batch processing CSV files with enhancement and generation."""

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add batch command arguments."""
        parser.add_argument(
            "input_csv",
            type=str,
            help="Input CSV file with prompts",
        )

        parser.add_argument(
            "--output-csv",
            type=str,
            help="Output CSV file (auto-generated if not provided)",
        )

        parser.add_argument(
            "--prompt-column",
            type=str,
            default="prompt",
            help="CSV column containing prompts (default: prompt)",
        )

        parser.add_argument(
            "--start-row",
            type=int,
            default=0,
            help="Starting row index (default: 0)",
        )

        parser.add_argument(
            "--max-rows",
            type=int,
            help="Maximum rows to process (default: all)",
        )

        parser.add_argument(
            "--image-generator",
            choices=["dalle3", "mock"],
            default="dalle3",
            help="Image generator to use (default: dalle3)",
        )

        parser.add_argument(
            "--output-dir",
            type=str,
            default="batch_results",
            help="Output directory for results (default: batch_results)",
        )

        parser.add_argument(
            "--image-output-dir",
            type=str,
            default="generated_images",
            help="Output directory for images (default: generated_images)",
        )

        parser.add_argument(
            "--enable-visual-bias",
            action="store_true",
            help="Enable visual bias evaluation on generated images",
        )

    def execute(self, args: argparse.Namespace) -> int:
        """Execute the batch command."""
        try:
            # Import here to avoid circular dependency
            from scripts.batch_processor import BatchProcessor

            # Validate input file
            input_path = Path(args.input_csv)
            if not input_path.exists():
                logger.error(f"Input CSV not found: {input_path}")
                return 1

            logger.info(f"Processing CSV file: {input_path}")

            # Create batch processor
            processor = BatchProcessor(
                image_generator_type=args.image_generator,
                output_dir=args.output_dir,
                image_output_dir=args.image_output_dir,
                enable_visual_bias_evaluation=args.enable_visual_bias,
            )

            # Process the CSV
            output_csv, results = processor.process_csv(
                input_csv_path=input_path,
                prompt_column=args.prompt_column,
                output_csv_path=args.output_csv,
                start_row=args.start_row,
                max_rows=args.max_rows,
                save_intermediate=True,
            )

            # Display summary
            successful = sum(1 for r in results if r.error is None)
            total = len(results)

            logger.info("\nBatch processing complete!")
            logger.info(f"Results saved to: {output_csv}")
            logger.info(f"Successful: {successful}/{total}")

            return 0 if successful == total else 1

        except KeyboardInterrupt:
            logger.info("\nBatch processing cancelled by user")
            return 130
        except Exception as e:
            logger.exception(f"Batch processing failed: {e}")
            return 1
