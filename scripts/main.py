"""Test script for GraphRAG system with multiple data sources.

This script demonstrates the structured GraphRAG system by:
1. Loading bias data from CSV files 
2. Loading cultural values data from text files
3. Building separate knowledge graphs for each dataset
4. Performing retrieval queries on both graphs

Usage:
    python main.py                              # Interactive mode
    python main.py -q "Your question here"     # Command line question
    python main.py --question "Your question"  # Command line question (long form)
"""

import argparse
import logging
from pathlib import Path

from src.knowledge import GraphRAG
from src.parsers import DataParserFactory


def setup_logging() -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
        handlers=[logging.StreamHandler()]
    )


def get_question_input() -> str:
    """Get question from user input with a nice interface."""
    print("\n" + "🤔 " + "=" * 50)
    print("   INTERACTIVE QUESTION INPUT")
    print("=" * 54)
    print("Enter your question about bias, stereotypes, or cultural practices.")
    print("Examples:")
    print("  • What racial stereotypes exist about black people?")
    print("  • How are minorities stereotyped in media?")
    print("  • What foods are popular in different cultures?")
    print("  • What cultural practices exist in various countries?")
    print("-" * 54)

    while True:
        question = input("❓ Your question: ").strip()
        if question:
            return question
        print("⚠️  Please enter a non-empty question.")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="GraphRAG System Test with Multiple Data Sources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Interactive mode
  python main.py -q "What stereotypes exist?"      # Direct question
  python main.py --question "Cultural practices?"  # Long form
        """
    )

    parser.add_argument(
        "-q", "--question",
        type=str,
        help="Question to ask all loaded graphs (optional - will prompt if not provided)"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Number of top results to retrieve (default: 8)"
    )

    parser.add_argument(
        "--cache-file",
        type=str,
        default="test_embeddings.pkl",
        help="Embedding cache file to use (default: test_embeddings.pkl)"
    )

    return parser.parse_args()


def test_bias_data(graphrag: GraphRAG, question: str, top_k: int = 8) -> None:
    """Test GraphRAG with bias CSV data."""
    print("\n" + "=" * 60)
    print("TESTING BIAS DATA (CSV FORMAT)")
    print("=" * 60)

    # Load bias data
    bias_file = "data/biases/ADV_GRAPH_20240119 - ADV_GRAPH_20240119.csv"
    if not Path(bias_file).exists():
        print(f"❌ Bias file not found: {bias_file}")
        return

    try:
        parser = DataParserFactory.create_parser(bias_file)
        graph = graphrag.add_graph("bias_graph", parser)
        print(f"✅ Loaded bias graph: {graph}")

        # Test with the provided question
        print(f"\n--- Querying: {question} ---")
        try:
            result = graphrag.query(
                graph_name="bias_graph",
                question=question,
                top_k=top_k,
                include_answer=True
            )
            print(f"Retrieved {result['num_triples']} relevant triples")

        except Exception as e:
            print(f"❌ Query failed: {e}")

    except Exception as e:
        print(f"❌ Error loading bias data: {e}")


def test_cultural_data(graphrag: GraphRAG, question: str, top_k: int = 5) -> None:
    """Test GraphRAG with cultural values text data."""
    print("\n" + "=" * 60)
    print("TESTING CULTURAL VALUES DATA (TEXT FORMAT)")
    print("=" * 60)

    # Find cultural values files
    cultural_dir = Path("data/cultural_values")
    if not cultural_dir.exists():
        print(f"❌ Cultural values directory not found: {cultural_dir}")
        return

    txt_files = list(cultural_dir.glob("*.txt"))
    if not txt_files:
        print("❌ No text files found in cultural values directory")
        return

    # Load a few cultural datasets
    test_files = txt_files[:3]  # Test first 3 files

    for file_path in test_files:
        country = file_path.stem.replace(" triples", "").replace("_", " ").title()
        graph_name = f"cultural_{file_path.stem.replace(' ', '_').lower()}"

        print(f"\n--- Loading {country} Cultural Data ---")

        try:
            parser = DataParserFactory.create_parser(file_path)
            graph = graphrag.add_graph(graph_name, parser)
            print(f"✅ Loaded {country} graph: {graph}")

            # Test with the provided question
            print(f"\n--- Querying {country}: {question} ---")
            try:
                result = graphrag.query(
                    graph_name=graph_name,
                    question=question,
                    top_k=top_k,
                    include_answer=True
                )
                print(f"Retrieved {result['num_triples']} relevant triples")

            except Exception as e:
                print(f"❌ Query failed: {e}")

        except Exception as e:
            print(f"❌ Error loading {country} data: {e}")


def main() -> None:
    """Run the GraphRAG test script."""
    args = parse_arguments()
    setup_logging()

    print("🚀 GraphRAG System Test")
    print("Testing structured GraphRAG with multiple data sources")

    # Get question from args or user input
    if args.question:
        question = args.question
        print(f"\n📝 Using provided question: {question}")
    else:
        question = get_question_input()

    # Initialize GraphRAG system
    graphrag = GraphRAG(cache_file=args.cache_file)

    try:
        # Test bias data
        test_bias_data(graphrag, question, args.top_k)

        # Test cultural data
        test_cultural_data(graphrag, question, args.top_k)

        # Summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)

        graphs = graphrag.list_graphs()
        print(f"✅ Successfully loaded {len(graphs)} knowledge graphs:")
        for name, description in graphs.items():
            print(f"  • {name}: {description}")

        print(f"\n✅ Embedding cache contains {len(graphrag.cache)} embeddings")
        print(f"📊 Question tested: {question}")
        print(f"🔢 Top-K results: {args.top_k}")
        print("🎉 GraphRAG system test completed!")

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        logging.exception("Full error details")


if __name__ == "__main__":
    main()
