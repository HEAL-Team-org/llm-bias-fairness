"""Test command - Run various tests and demos."""

import argparse
import logging

from src.cli.commands import BaseCommand

logger = logging.getLogger(__name__)


class TestCommand(BaseCommand):
    """Command for running tests and demos."""

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add test command arguments."""
        parser.add_argument(
            "--quick",
            action="store_true",
            help="Run quick enhancement test",
        )

        parser.add_argument(
            "--batch",
            action="store_true",
            help="Run batch processing test",
        )

        parser.add_argument(
            "--visual",
            action="store_true",
            help="Run visual bias evaluation test",
        )

        parser.add_argument(
            "--all",
            action="store_true",
            help="Run all tests",
        )

    def execute(self, args: argparse.Namespace) -> int:
        """Execute the test command."""
        try:
            if not any([args.quick, args.batch, args.visual, args.all]):
                logger.error("No test selected. Use --quick, --batch, --visual, or --all")
                return 1

            exit_code = 0

            if args.quick or args.all:
                logger.info("Running quick enhancement test...")
                result = self._test_quick()
                if result != 0:
                    exit_code = result

            if args.batch or args.all:
                logger.info("Running batch processing test...")
                result = self._test_batch()
                if result != 0:
                    exit_code = result

            if args.visual or args.all:
                logger.info("Running visual bias evaluation test...")
                result = self._test_visual()
                if result != 0:
                    exit_code = result

            return exit_code

        except KeyboardInterrupt:
            logger.info("\nTests cancelled by user")
            return 130
        except Exception as e:
            logger.exception(f"Tests failed: {e}")
            return 1

    def _test_quick(self) -> int:
        """Run quick enhancement test."""
        try:
            from src.enhancement import EnhancementSystem

            logger.info("Testing enhancement system...")
            system = EnhancementSystem()
            result = system.enhance("a doctor")

            logger.info(f"Original: {result.original_prompt}")
            logger.info(f"Enhanced: {result.final_prompt}")
            logger.info(f"Bias improvement: +{result.bias_improvement:.1f}")
            logger.info(f"Diversity improvement: +{result.diversity_improvement:.1f}")

            return 0
        except Exception as e:
            logger.error(f"Quick test failed: {e}")
            return 1

    def _test_batch(self) -> int:
        """Run batch processing test."""
        try:
            logger.info("Batch test not yet implemented")
            return 0
        except Exception as e:
            logger.error(f"Batch test failed: {e}")
            return 1

    def _test_visual(self) -> int:
        """Run visual bias evaluation test."""
        try:
            from src.evaluation import VisualBiasEvaluator

            logger.info("Testing visual bias evaluator...")
            evaluator = VisualBiasEvaluator()
            logger.info("Visual bias evaluator initialized successfully")

            return 0
        except Exception as e:
            logger.error(f"Visual test failed: {e}")
            return 1
