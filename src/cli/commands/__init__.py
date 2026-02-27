"""Base command class for CLI commands."""

import argparse
import logging
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)


class BaseCommand(ABC):
    """Base class for all CLI commands.
    
    Each command should implement:
    - add_arguments(): Add command-specific arguments to parser
    - execute(): Execute the command with parsed arguments
    """

    def __init__(self):
        """Initialize the command."""
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add command-specific arguments to the argument parser.
        
        Args:
            parser: The argument parser for this command

        """

    @abstractmethod
    def execute(self, args: argparse.Namespace) -> int:
        """Execute the command with parsed arguments.
        
        Args:
            args: Parsed command-line arguments
            
        Returns:
            Exit code (0 for success, non-zero for failure)

        """

    def setup(self) -> None:
        """Optional setup before command execution.
        
        Override this method if your command needs setup logic.
        """

    def cleanup(self) -> None:
        """Optional cleanup after command execution.
        
        Override this method if your command needs cleanup logic.
        """
