"""Enhance command - Single prompt enhancement with multiple modes."""

import argparse
import logging

from src.cli.commands import BaseCommand
from src.enhancement import EnhancementConfig, EnhancementSystem
from src.knowledge import DiversityRAG, GraphRAG, StereoSetRAG

logger = logging.getLogger(__name__)


class EnhanceCommand(BaseCommand):
    """Command for enhancing single prompts with bias mitigation and diversity."""

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add enhance command arguments."""
        parser.add_argument(
            "prompt",
            nargs="?",
            help="Image prompt to enhance (omit for interactive mode)",
        )

        parser.add_argument(
            "--bias-threshold",
            type=int,
            default=75,
            help="Bias mitigation threshold (default: 75)",
        )

        parser.add_argument(
            "--diversity-threshold",
            type=int,
            default=80,
            help="Diversity threshold (default: 80)",
        )

        parser.add_argument(
            "--max-iterations",
            type=int,
            default=5,
            help="Maximum enhancement iterations (default: 5)",
        )

        parser.add_argument(
            "--use-stereoset",
            action="store_true",
            help="Enable StereoSet RAG for bias examples",
        )

        parser.add_argument(
            "--use-diversity-rag",
            action="store_true",
            help="Enable Diversity RAG for recommendations",
        )

        parser.add_argument(
            "--use-graphrag",
            action="store_true",
            help="Enable GraphRAG for knowledge retrieval",
        )

    def execute(self, args: argparse.Namespace) -> int:
        """Execute the enhance command."""
        try:
            # Get prompt (interactive or from args)
            prompt = args.prompt
            if not prompt:
                prompt = self._get_prompt_input()
                if not prompt:
                    logger.error("No prompt provided")
                    return 1

            # Create enhancement config
            config = EnhancementConfig(
                bias_threshold=args.bias_threshold,
                diversity_threshold=args.diversity_threshold,
                max_iterations=args.max_iterations,
                use_stereoset=args.use_stereoset,
                use_diversity_rag=args.use_diversity_rag,
                use_graphrag=args.use_graphrag,
            )

            # Initialize RAG systems if requested
            stereoset_rag = None
            diversity_rag = None
            graphrag = None

            if config.use_stereoset:
                logger.info("Initializing StereoSet RAG...")
                stereoset_rag = StereoSetRAG()

            if config.use_diversity_rag:
                logger.info("Initializing Diversity RAG...")
                diversity_rag = DiversityRAG()

            if config.use_graphrag:
                logger.info("Initializing GraphRAG...")
                graphrag = GraphRAG()

            # Create enhancement system
            system = EnhancementSystem(
                config=config,
                stereoset_rag=stereoset_rag,
                diversity_rag=diversity_rag,
                graphrag=graphrag,
            )

            # Enhance the prompt
            logger.info(f"Enhancing prompt: {prompt}")
            result = system.enhance(prompt)

            # Display results
            self._display_results(result)

            return 0

        except KeyboardInterrupt:
            logger.info("\nEnhancement cancelled by user")
            return 130
        except Exception as e:
            logger.error(f"Enhancement failed: {e}", exc_info=True)
            return 1

    def _get_prompt_input(self) -> str:
        """Get prompt from user interactively."""
        print("\n" + "🎨 " + "=" * 68)
        print("   INTERACTIVE PROMPT ENHANCEMENT")
        print("=" * 72)
        print("Enter an image generation prompt to enhance for bias mitigation and diversity.")
        print("\nExamples:")
        print("  • a doctor examining a patient")
        print("  • students in a classroom")
        print("  • engineers working on a project")
        print("  • a teacher giving a lecture")
        print("-" * 72)

        prompt = input("\n📝 Your prompt: ").strip()
        return prompt

    def _display_results(self, result) -> None:
        """Display enhancement results."""
        print("\n" + "=" * 72)
        print("ENHANCEMENT RESULTS")
        print("=" * 72)

        print(f"\n📝 Original Prompt:")
        print(f"   {result.original_prompt}")

        print(f"\n✨ Enhanced Prompt:")
        print(f"   {result.final_prompt}")

        print(f"\n📊 Scores:")
        print(f"   Bias Score:      {result.initial_bias_score:.1f} → {result.final_bias_score:.1f}")
        print(f"   Diversity Score: {result.initial_diversity_score:.1f} → {result.final_diversity_score:.1f}")

        print(f"\n📈 Improvements:")
        print(f"   Bias:      +{result.bias_improvement:.1f}")
        print(f"   Diversity: +{result.diversity_improvement:.1f}")

        print(f"\n🔄 Iterations: {result.iterations}")

        if result.both_thresholds_met:
            print("\n✅ Both thresholds met!")
        else:
            print("\n⚠️  Thresholds not fully met")
            if not result.bias_threshold_met:
                print(f"   - Bias threshold not met (need {result.final_bias_score:.1f} >= threshold)")
            if not result.diversity_threshold_met:
                print(f"   - Diversity threshold not met (need {result.final_diversity_score:.1f} >= threshold)")

        print()
