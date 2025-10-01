"""Command-Line Interface Layer for LLM Bias & Fairness Project.

This package provides a clean, modular CLI structure for all project functionality:
- enhance: Single prompt enhancement with multiple modes
- batch: Batch processing of CSV files  
- query: Knowledge graph queries
- test: Various testing and demo functions

The CLI layer integrates all other layers (config, data, knowledge, generation,
evaluation, enhancement) into user-friendly command-line tools.
"""

from src.cli.commands.batch import BatchCommand
from src.cli.commands.enhance import EnhanceCommand
from src.cli.commands.query import QueryCommand
from src.cli.commands.test import TestCommand
from src.cli.parser import create_cli_parser
from src.cli.runner import CLIRunner

__all__ = [
    "CLIRunner",
    "create_cli_parser",
    "EnhanceCommand",
    "BatchCommand",
    "QueryCommand",
    "TestCommand",
]
