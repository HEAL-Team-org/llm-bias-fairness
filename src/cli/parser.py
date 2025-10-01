"""CLI argument parser for the LLM Bias & Fairness project."""

import argparse
from typing import Optional

from src.cli.commands.batch import BatchCommand
from src.cli.commands.enhance import EnhanceCommand
from src.cli.commands.query import QueryCommand
from src.cli.commands.test import TestCommand


def create_cli_parser() -> argparse.ArgumentParser:
    """Create the main CLI argument parser with all subcommands.

    Returns:
        Configured argument parser with all commands
    """
    parser = argparse.ArgumentParser(
        prog="llm-bias-fairness",
        description="LLM Bias & Fairness Project - Unified CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Enhance a single prompt
  python main.py enhance "a doctor examining a patient"

  # Process CSV file
  python main.py batch prompts.csv

  # Query knowledge graphs
  python main.py query "What stereotypes exist?"

  # Run tests
  python main.py test --quick

For more help on a specific command:
  python main.py <command> --help
        """,
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    # Create subparsers for commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Add enhance command
    enhance_parser = subparsers.add_parser(
        "enhance",
        help="Enhance a single prompt for bias mitigation and diversity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python main.py enhance

  # Direct enhancement
  python main.py enhance "a doctor examining a patient"

  # With custom thresholds
  python main.py enhance "students in classroom" --bias-threshold 80 --diversity-threshold 85

  # With RAG systems
  python main.py enhance "engineers working" --use-stereoset --use-diversity-rag
        """,
    )
    EnhanceCommand().add_arguments(enhance_parser)

    # Add batch command
    batch_parser = subparsers.add_parser(
        "batch",
        help="Process CSV files with enhancement and image generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process entire CSV
  python main.py batch prompts.csv

  # Process with specific range
  python main.py batch prompts.csv --start-row 0 --max-rows 10

  # Use mock generator for testing
  python main.py batch prompts.csv --image-generator mock

  # Enable visual bias evaluation
  python main.py batch prompts.csv --enable-visual-bias
        """,
    )
    BatchCommand().add_arguments(batch_parser)

    # Add query command
    query_parser = subparsers.add_parser(
        "query",
        help="Query knowledge graphs for bias and cultural information",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python main.py query

  # Direct query
  python main.py query "What stereotypes exist?"

  # Query specific data source
  python main.py query "Cultural practices" --data-source cultural

  # Get more results
  python main.py query "Bias information" --top-k 15
        """,
    )
    QueryCommand().add_arguments(query_parser)

    # Add test command
    test_parser = subparsers.add_parser(
        "test",
        help="Run various tests and demos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run quick test
  python main.py test --quick

  # Run all tests
  python main.py test --all

  # Run specific test
  python main.py test --visual
        """,
    )
    TestCommand().add_arguments(test_parser)

    return parser
