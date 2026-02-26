"""Data parsers for different knowledge graph formats.

This module contains parsers for various data formats used to build knowledge graphs
from different sources in the data/ directory.
"""

from __future__ import annotations

import csv
import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# Type definitions
Triple = Tuple[str, str, str]

# Constants
TRIPLE_PARTS_COUNT = 3


class BaseDataParser(ABC):
    """Abstract base class for data parsers."""
    
    def __init__(self, file_path: str | Path):
        """Initialize parser with file path.
        
        Args:
            file_path: Path to the data file
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            msg = f"Data file not found: {self.file_path}"
            raise FileNotFoundError(msg)
    
    @abstractmethod
    def parse(self) -> List[Triple]:
        """Parse the data file and return list of triples.
        
        Returns:
            List of (subject, predicate, object) tuples
        """
        ...
    
    @property
    def name(self) -> str:
        """Get parser name for logging."""
        return self.__class__.__name__


class BiasCSVParser(BaseDataParser):
    """Parser for bias CSV files with graph column format."""
    
    def parse(self) -> List[Triple]:
        """Parse CSV file with targetMinority, targetStereotype, and Graph columns.
        
        Expected format:
        ,targetMinority,targetStereotype,Graph
        0,black folks,are all well endowed,"Graph: `(black folks, are, well endowed)`"
        
        Returns:
            List of (subject, predicate, object) tuples
        """
        triples = []
        
        try:
            with self.file_path.open(encoding="utf-8") as f:
                reader = csv.DictReader(f)
                
                for row_idx, row in enumerate(reader):
                    if "Graph" not in row:
                        logger.warning(f"Row {row_idx} missing 'Graph' column")
                        continue
                    
                    graph_text = row["Graph"]
                    if not graph_text or pd.isna(graph_text):
                        continue
                    
                    # Extract triples from graph text
                    row_triples = self._extract_triples_from_graph_text(graph_text)
                    triples.extend(row_triples)
                    
        except Exception:
            logger.exception(f"Error parsing CSV file {self.file_path}")
            raise
        
        logger.info(f"Parsed {len(triples)} triples from {self.file_path.name}")
        return triples
    
    def _extract_triples_from_graph_text(self, graph_text: str) -> List[Triple]:
        """Extract triples from graph text format.
        
        Args:
            graph_text: Text containing graph triples like "`(subject, predicate, object)`"
            
        Returns:
            List of extracted triples
        """
        triples = []
        # Pattern to match triples in backticks: `(subject, predicate, object)`
        pattern = r"`\(([^,]+),\s*([^,]+),\s*([^)]+)\)`"
        
        matches = re.findall(pattern, graph_text)
        for match in matches:
            subject, predicate, obj = [s.strip() for s in match]
            if subject and predicate and obj:
                triples.append((subject, predicate, obj))
        
        return triples


class CulturalTriplesParser(BaseDataParser):
    """Parser for cultural values triple files."""
    
    def parse(self) -> List[Triple]:
        """Parse text files with comma-separated triples.
        
        Expected formats:
        1. CSV format with headers: subject,relation,object
        2. Simple format: subject,predicate,object (no headers)
        
        Returns:
            List of (subject, predicate, object) tuples
        """
        triples = []
        
        try:
            with self.file_path.open(encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            
            if not lines:
                logger.warning(f"Empty file: {self.file_path.name}")
                return triples
            
            # Check if first line is a header
            first_line = lines[0]
            has_header = self._looks_like_header(first_line)
            
            start_idx = 1 if has_header else 0
            
            for line_idx, line in enumerate(lines[start_idx:], start_idx):
                parts = [part.strip() for part in line.split(",")]
                if len(parts) >= TRIPLE_PARTS_COUNT:
                    subject, predicate, obj = parts[0], parts[1], parts[2]
                    if subject and predicate and obj:
                        triples.append((subject, predicate, obj))
                else:
                    logger.warning(f"Line {line_idx + 1} has insufficient parts: {line}")
                    
        except Exception:
            logger.exception(f"Error parsing file {self.file_path}")
            raise
        
        logger.info(f"Parsed {len(triples)} triples from {self.file_path.name}")
        return triples
    
    def _looks_like_header(self, line: str) -> bool:
        """Check if line looks like a CSV header.
        
        Args:
            line: First line of the file
            
        Returns:
            True if line appears to be a header
        """
        # Common header patterns
        header_patterns = ["subject", "relation", "object", "predicate"]
        line_lower = line.lower()
        return any(pattern in line_lower for pattern in header_patterns)


class DataParserFactory:
    """Factory for creating appropriate data parsers based on file type and location."""
    
    @staticmethod
    def create_parser(file_path: str | Path) -> BaseDataParser:
        """Create appropriate parser based on file path and format.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            Appropriate parser instance
            
        Raises:
            ValueError: If no suitable parser found
            
        """
        file_path = Path(file_path)
        
        # Determine parser based on file location and extension
        if "biases" in file_path.parts and file_path.suffix == ".csv":
            return BiasCSVParser(file_path)
        if "cultural_values" in file_path.parts and file_path.suffix == ".txt":
            return CulturalTriplesParser(file_path)
        
        # Try to determine by file extension
        if file_path.suffix == ".csv":
            return BiasCSVParser(file_path)
        if file_path.suffix == ".txt":
            return CulturalTriplesParser(file_path)
        
        msg = f"No suitable parser found for file: {file_path}"
        raise ValueError(msg)
