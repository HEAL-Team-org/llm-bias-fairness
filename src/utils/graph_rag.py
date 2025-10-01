"""GraphRAG demo - **embedding-aware** version.
==========================================
This file provides backward compatibility for the original CLI interface
while using the new structured GraphRAG classes.

* 🔍 **Vector search** using **OpenAI `text-embedding-3-large`**.
* 💾 **Persistent cache** (`.pkl`) for every text string we embed - so we
  only pay once.
* 🪵 **Structured logging** of the retrieval step (both nodes & triples).

Quick start
-----------
```bash
pip install networkx pandas numpy openai  # plus faiss if you wish

export OPENAI_API_KEY="sk-…"
python graph_rag.py \
    "ADV_GRAPH_20240119 - ADV_GRAPH_20240119.csv" \
    --question "Why are black people stereotyped as criminals?" \
    --top_k 20
```

If the env-var is missing, the script falls back to the *old* substring
retriever so you can still test without an API key.

"""  # noqa: D205

import argparse
import logging
from pathlib import Path

from src.knowledge import GraphRAG
from src.parsers import DataParserFactory

# Set up logger
logger = logging.getLogger(__name__)

# Constants
CACHE_FILENAME = "embeddings.pkl"


def main() -> None:
    """Provide backward-compatible CLI interface."""
    ap = argparse.ArgumentParser(description="GraphRAG embedding demo")
    ap.add_argument("csv_path", help="Path to the ADV_GRAPH CSV")
    ap.add_argument(
        "--question", "-q", default="What stereotypes exist about black people?"
    )
    ap.add_argument("--top_k", "-k", type=int, default=15)
    ap.add_argument("--cache", default=CACHE_FILENAME, help="Embedding cache file (pkl)")
    args = ap.parse_args()

    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

    # Initialize GraphRAG system
    graphrag = GraphRAG(cache_file=args.cache)

    # Create parser and load graph
    try:
        parser = DataParserFactory.create_parser(args.csv_path)
        graph_name = Path(args.csv_path).stem
        graphrag.add_graph(graph_name, parser)

        # Query the graph
        graphrag.query(
            graph_name=graph_name,
            question=args.question,
            top_k=args.top_k,
            include_answer=True
        )

        logger.info(f"Query completed successfully for graph '{graph_name}'")

    except Exception:
        logger.exception("Error processing GraphRAG query")


if __name__ == "__main__":
    main()
