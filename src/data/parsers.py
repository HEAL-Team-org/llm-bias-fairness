"""Data parsers for different knowledge graph formats.

This module contains parsers for various data formats used to build knowledge graphs
from different sources in the data/ directory.
"""

from __future__ import annotations

import csv
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pandas as pd

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("datasets library not available. StereoSet functionality will be limited.")

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


@dataclass
class StereoSetRecord:
    """Data structure for StereoSet dataset records."""

    id: str
    target: str
    bias_type: str
    context: str
    sentences: list[str]
    gold_labels: list[int]
    config: str = "unknown"  # intersentence or intrasentence

    @property
    def stereotypical_sentences(self) -> list[str]:
        """Get sentences labeled as stereotypical (gold_label=1)."""
        return [
            sentence for sentence, label in zip(self.sentences, self.gold_labels)
            if label == 1
        ]

    @property
    def anti_stereotypical_sentences(self) -> list[str]:
        """Get sentences labeled as anti-stereotypical (gold_label=0)."""
        return [
            sentence for sentence, label in zip(self.sentences, self.gold_labels)
            if label == 0
        ]


class StereoSetParser:
    """Parser for StereoSet dataset from Hugging Face."""

    def __init__(self, cache_file: str | Path = "stereoset_cache.pkl"):
        """Initialize the StereoSet parser.
        
        Args:
            cache_file: Path to cache the dataset locally

        """
        self.cache_file = Path(cache_file)
        self.dataset = None
        self.records: list[StereoSetRecord] = []
        self.logger = logging.getLogger(__name__)

    def load_dataset(self, force_reload: bool = False) -> bool:
        """Load the StereoSet dataset from Hugging Face.
        
        Args:
            force_reload: Whether to force reload even if cache exists
            
        Returns:
            True if successfully loaded, False otherwise

        """
        if not DATASETS_AVAILABLE:
            self.logger.error("datasets library not available. Cannot load StereoSet.")
            return False

        # Check cache first
        if not force_reload and self.cache_file.exists():
            try:
                import pickle
                with self.cache_file.open("rb") as f:
                    self.records = pickle.load(f)
                self.logger.info(f"Loaded {len(self.records)} StereoSet records from cache")
                return True
            except Exception as e:
                self.logger.warning(f"Failed to load from cache: {e}")

        try:
            from datasets import load_dataset
            self.logger.info("Loading StereoSet dataset from Hugging Face...")

            # Load both configurations
            intersentence_data = load_dataset("McGill-NLP/stereoset", "intersentence", split="validation")
            intrasentence_data = load_dataset("McGill-NLP/stereoset", "intrasentence", split="validation")

            self.records = []

            # Process intersentence data
            for item in intersentence_data:
                sentences_data = item["sentences"]
                sentences = sentences_data["sentence"]
                gold_labels = sentences_data["gold_label"]

                record = StereoSetRecord(
                    id=item["id"],
                    target=item["target"],
                    bias_type=item["bias_type"],
                    context=item["context"],
                    sentences=sentences,
                    gold_labels=gold_labels,
                    config="intersentence"
                )
                self.records.append(record)

            # Process intrasentence data
            for item in intrasentence_data:
                sentences_data = item["sentences"]
                sentences = sentences_data["sentence"]
                gold_labels = sentences_data["gold_label"]

                record = StereoSetRecord(
                    id=item["id"],
                    target=item["target"],
                    bias_type=item["bias_type"],
                    context=item["context"],
                    sentences=sentences,
                    gold_labels=gold_labels,
                    config="intrasentence"
                )
                self.records.append(record)

            # Cache the results
            try:
                import pickle
                with self.cache_file.open("wb") as f:
                    pickle.dump(self.records, f)
                self.logger.info(f"Cached {len(self.records)} records to {self.cache_file}")
            except Exception as e:
                self.logger.warning(f"Failed to cache records: {e}")

            self.logger.info(f"Successfully loaded {len(self.records)} StereoSet records")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load StereoSet dataset: {e}")
            return False

    def get_contexts(self) -> list[str]:
        """Get all context strings for embedding.
        
        Returns:
            List of context strings

        """
        return [record.context for record in self.records]

    def get_records_by_bias_type(self, bias_type: str) -> list[StereoSetRecord]:
        """Get records filtered by bias type.
        
        Args:
            bias_type: Type of bias (Gender, Profession, Race, Religion)
            
        Returns:
            List of matching records

        """
        return [record for record in self.records if record.bias_type.lower() == bias_type.lower()]

    def get_stereotypical_sentences(self) -> dict[str, list[str]]:
        """Get all stereotypical sentences grouped by bias type.
        
        Returns:
            Dictionary mapping bias types to lists of stereotypical sentences

        """
        result = {}
        for record in self.records:
            bias_type = record.bias_type
            if bias_type not in result:
                result[bias_type] = []
            result[bias_type].extend(record.stereotypical_sentences)
        return result

    def __len__(self) -> int:
        """Return number of records."""
        return len(self.records)

    def __getitem__(self, index: int) -> StereoSetRecord:
        """Get record by index."""
        return self.records[index]
