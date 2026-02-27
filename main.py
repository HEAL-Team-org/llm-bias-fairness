#!/usr/bin/env python3
"""LLM Bias & Fairness Project - Unified Command-Line Interface.

This script provides a unified interface for all project functionality:
- enhance: Enhance single prompts with bias mitigation and diversity
- batch: Process CSV files with enhancement and image generation
- query: Query knowledge graphs for bias and cultural information  
- test: Run various tests and demos

Usage:
    python main.py enhance "a doctor"                    # Enhance a prompt
    python main.py batch prompts.csv                     # Process CSV file
    python main.py query "What stereotypes exist?"       # Query knowledge
    python main.py test --quick                          # Run quick test

For more information:
    python main.py --help
    python main.py enhance --help
    python main.py batch --help
"""

import sys

from src.cli import CLIRunner


def main() -> int:
    """Main entry point for the CLI."""
    runner = CLIRunner()
    return runner.main()


if __name__ == "__main__":
    sys.exit(main())
