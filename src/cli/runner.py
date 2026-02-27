"""CLI runner for executing commands."""

import argparse
import logging
import sys
from typing import Dict, Type

from src.cli.commands import BaseCommand
from src.cli.commands.batch import BatchCommand
from src.cli.commands.enhance import EnhanceCommand
from src.cli.commands.query import QueryCommand
from src.cli.commands.test import TestCommand

logger = logging.getLogger(__name__)


class CLIRunner:
    """Main CLI runner that routes commands to appropriate handlers."""

    def __init__(self):
        """Initialize the CLI runner."""
        self.commands: Dict[str, Type[BaseCommand]] = {
            "enhance": EnhanceCommand,
            "batch": BatchCommand,
            "query": QueryCommand,
            "test": TestCommand,
        }

    def setup_logging(self, verbose: bool = False) -> None:
        """Set up logging configuration.

        Args:
            verbose: Enable verbose (DEBUG) logging

        """
        level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format="%(levelname)s: %(message)s",
            handlers=[logging.StreamHandler()],
        )

    def run(self, args: argparse.Namespace) -> int:
        """Run the CLI with parsed arguments.

        Args:
            args: Parsed command-line arguments

        Returns:
            Exit code (0 for success, non-zero for failure)

        """
        # Setup logging
        self.setup_logging(args.verbose if hasattr(args, "verbose") else False)

        # Get command name
        command_name = getattr(args, "command", None)

        if not command_name:
            logger.error("No command specified. Use --help for usage information.")
            return 1

        # Get command class
        command_class = self.commands.get(command_name)
        if not command_class:
            logger.error(f"Unknown command: {command_name}")
            return 1

        # Create and execute command
        try:
            command = command_class()
            command.setup()
            exit_code = command.execute(args)
            command.cleanup()
            return exit_code

        except KeyboardInterrupt:
            logger.info("\nOperation cancelled by user")
            return 130

        except Exception as e:
            logger.exception(f"Command failed: {e}")
            return 1

    def main(self, argv=None) -> int:
        """Main entry point with argument parsing.

        Args:
            argv: Command-line arguments (defaults to sys.argv)

        Returns:
            Exit code (0 for success, non-zero for failure)

        """
        from src.cli.parser import create_cli_parser

        parser = create_cli_parser()
        args = parser.parse_args(argv)

        return self.run(args)


def main() -> int:
    """Convenience function for running the CLI."""
    runner = CLIRunner()
    return runner.main()


if __name__ == "__main__":
    sys.exit(main())
