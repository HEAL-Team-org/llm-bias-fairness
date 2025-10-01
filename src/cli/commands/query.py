"""Query command - Query knowledge graphs for bias and cultural information."""

import argparse
import logging

from src.cli.commands import BaseCommand
from src.knowledge import GraphRAG

logger = logging.getLogger(__name__)


class QueryCommand(BaseCommand):
    """Command for querying knowledge graphs."""

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add query command arguments."""
        parser.add_argument(
            "question",
            nargs="?",
            help="Question to ask (omit for interactive mode)",
        )

        parser.add_argument(
            "--top-k",
            type=int,
            default=8,
            help="Number of top results to retrieve (default: 8)",
        )

        parser.add_argument(
            "--data-source",
            choices=["bias", "cultural", "both"],
            default="both",
            help="Data source to query (default: both)",
        )

    def execute(self, args: argparse.Namespace) -> int:
        """Execute the query command."""
        try:
            # Get question (interactive or from args)
            question = args.question
            if not question:
                question = self._get_question_input()
                if not question:
                    logger.error("No question provided")
                    return 1

            logger.info(f"Querying knowledge graphs: {question}")

            # Initialize GraphRAG
            graphrag = GraphRAG()

            # Query based on data source
            if args.data_source in ["bias", "both"]:
                logger.info("\nQuerying bias data...")
                self._query_bias_data(graphrag, question, args.top_k)

            if args.data_source in ["cultural", "both"]:
                logger.info("\nQuerying cultural data...")
                self._query_cultural_data(graphrag, question, args.top_k)

            return 0

        except KeyboardInterrupt:
            logger.info("\nQuery cancelled by user")
            return 130
        except Exception as e:
            logger.exception(f"Query failed: {e}")
            return 1

    def _get_question_input(self) -> str:
        """Get question from user interactively."""
        print("\n" + "🤔 " + "=" * 68)
        print("   INTERACTIVE KNOWLEDGE GRAPH QUERY")
        print("=" * 72)
        print("Enter your question about bias, stereotypes, or cultural practices.")
        print("\nExamples:")
        print("  • What racial stereotypes exist about black people?")
        print("  • How are minorities stereotyped in media?")
        print("  • What foods are popular in different cultures?")
        print("  • What cultural practices exist in various countries?")
        print("-" * 72)

        question = input("\n❓ Your question: ").strip()
        return question

    def _query_bias_data(self, graphrag: GraphRAG, question: str, top_k: int) -> None:
        """Query bias data from GraphRAG."""
        print("\n" + "=" * 72)
        print("BIAS DATA RESULTS")
        print("=" * 72)

        try:
            results = graphrag.query(question, top_k=top_k, graph_type="bias")
            if results:
                for i, result in enumerate(results, 1):
                    print(f"\n{i}. {result}")
            else:
                print("No results found")
        except Exception as e:
            logger.error(f"Bias query failed: {e}")

    def _query_cultural_data(self, graphrag: GraphRAG, question: str, top_k: int) -> None:
        """Query cultural data from GraphRAG."""
        print("\n" + "=" * 72)
        print("CULTURAL DATA RESULTS")
        print("=" * 72)

        try:
            results = graphrag.query(question, top_k=top_k, graph_type="cultural")
            if results:
                for i, result in enumerate(results, 1):
                    print(f"\n{i}. {result}")
            else:
                print("No results found")
        except Exception as e:
            logger.error(f"Cultural query failed: {e}")
