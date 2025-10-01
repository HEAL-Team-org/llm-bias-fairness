"""Enhanced Sequential Image Prompt Enhancement with StereoSet RAG Integration.

This script combines the original GraphRAG system with the new StereoSet RAG system to provide
comprehensive bias mitigation. It uses:
1. StereoSet RAG for identifying stereotypes to avoid (negative examples)
2. GraphRAG bias data for additional bias awareness
3. GraphRAG cultural values for promoting inclusivity
4. Sequential improvement with diversity scoring

Usage:
    python enhance_prompt_with_stereoset.py                           # Interactive mode
    python enhance_prompt_with_stereoset.py -p "a doctor"            # Direct prompt
    python enhance_prompt_with_stereoset.py --use-stereoset          # Enable StereoSet RAG
    python enhance_prompt_with_stereoset.py --threshold 85 --max-iterations 4  # Custom settings
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.knowledge import GraphRAG, StereoSetRAG

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
        handlers=[logging.StreamHandler()]
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Enhanced Image Prompt Enhancement with StereoSet RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "-p", "--prompt",
        type=str,
        help="Image prompt to enhance"
    )

    parser.add_argument(
        "--use-stereoset",
        action="store_true",
        default=True,
        help="Use StereoSet RAG for stereotype detection (default: True)"
    )

    parser.add_argument(
        "--use-graphrag-bias",
        action="store_true",
        default=True,
        help="Use GraphRAG bias data (default: True)"
    )

    parser.add_argument(
        "--bias-top-k",
        type=int,
        default=15,
        help="Number of bias triples to retrieve (default: 15)"
    )

    parser.add_argument(
        "--cultural-top-k",
        type=int,
        default=15,
        help="Number of cultural triples to retrieve (default: 15)"
    )

    parser.add_argument(
        "--stereoset-top-k",
        type=int,
        default=10,
        help="Number of StereoSet records to retrieve (default: 10)"
    )

    parser.add_argument(
        "--threshold",
        type=int,
        default=80,
        help="Diversity threshold to achieve (default: 80)"
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum enhancement iterations (default: 3)"
    )

    parser.add_argument(
        "--cache-file",
        type=str,
        default="embeddings.pkl",
        help="Embedding cache file (default: embeddings.pkl)"
    )

    parser.add_argument(
        "--stereoset-cache",
        type=str,
        default="stereoset_embeddings.pkl",
        help="StereoSet embedding cache file (default: stereoset_embeddings.pkl)"
    )

    return parser.parse_args()


def get_prompt_input() -> str:
    """Get image prompt from user input with a nice interface."""
    print("\n" + "🎨 " + "=" * 75)
    print("   ENHANCED IMAGE PROMPT ENHANCEMENT WITH STEREOSET INTEGRATION")
    print("=" * 79)
    print("Enter an image generation prompt that you want to enhance for diversity")
    print("and bias mitigation. The system will use both GraphRAG and StereoSet")
    print("to provide comprehensive stereotype detection and cultural awareness.")
    print("\nExamples:")
    print("  • a doctor examining a patient")
    print("  • students studying in a classroom")
    print("  • a family having dinner")
    print("  • engineers working on a project")
    print("  • people celebrating a festival")
    print("-" * 79)

    while True:
        prompt = input("🎨 Your image prompt: ").strip()
        if prompt:
            return prompt
        print("Please enter a non-empty prompt.")


def load_systems(
    use_stereoset: bool,
    use_graphrag_bias: bool,
    cache_file: str,
    stereoset_cache: str
) -> Tuple[Optional[StereoSetRAG], Optional[GraphRAG]]:
    """Load and initialize the enhancement systems.
    
    Args:
        use_stereoset: Whether to load StereoSet RAG
        use_graphrag_bias: Whether to load GraphRAG with bias data
        cache_file: GraphRAG cache file
        stereoset_cache: StereoSet cache file
        
    Returns:
        Tuple of (stereoset_rag, graphrag) systems

    """
    stereoset_rag = None
    graphrag = None

    # Initialize StereoSet RAG if requested
    if use_stereoset:
        print("\n⚙️  Initializing StereoSet RAG system...")
        stereoset_rag = StereoSetRAG(cache_file=stereoset_cache)

        if stereoset_rag.load_dataset():
            stats = stereoset_rag.get_dataset_stats()
            print(f"✅ StereoSet loaded: {stats['total_records']} records, "
                  f"{stats['total_stereotypical_sentences']} stereotypical sentences")
        else:
            print("❌ Failed to load StereoSet dataset")
            stereoset_rag = None

    # Initialize GraphRAG if requested
    if use_graphrag_bias:
        print("\n⚙️  Initializing GraphRAG system...")
        graphrag = GraphRAG(cache_file=cache_file)

        # Load bias data
        bias_loaded = load_bias_graph(graphrag)

        # Load cultural data
        cultural_loaded = load_cultural_graphs(graphrag)

        if not bias_loaded and not cultural_loaded:
            print("❌ Failed to load any GraphRAG datasets")
            graphrag = None
        else:
            print("✅ GraphRAG system loaded successfully")

    return stereoset_rag, graphrag


def load_bias_graph(graphrag: GraphRAG) -> bool:
    """Load bias graph data into GraphRAG system."""
    bias_file = Path("data/biases/ADV_GRAPH_20240119 - ADV_GRAPH_20240119.csv")

    if not bias_file.exists():
        print(f"⚠️  Bias file not found: {bias_file}")
        return False

    try:
        parser = DataParserFactory.create_parser(bias_file)
        bias_graph = graphrag.add_graph("bias", parser)
        print(f"📊 Loaded bias graph: {len(bias_graph)} triples")
        return True
    except Exception as e:
        print(f"❌ Error loading bias graph: {e}")
        return False


def load_cultural_graphs(graphrag: GraphRAG) -> bool:
    """Load cultural values graphs into GraphRAG system."""
    cultural_dir = Path("data/cultural_values")

    if not cultural_dir.exists():
        print(f"⚠️  Cultural values directory not found: {cultural_dir}")
        return False

    loaded_count = 0
    for file_path in cultural_dir.glob("*.txt"):
        try:
            parser = DataParserFactory.create_parser(file_path)
            graph_name = f"cultural_{file_path.stem.replace(' ', '_')}"
            cultural_graph = graphrag.add_graph(graph_name, parser)
            loaded_count += 1
            logger.debug(f"Loaded {graph_name}: {len(cultural_graph)} triples")
        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e}")

    if loaded_count > 0:
        print(f"🌍 Loaded {loaded_count} cultural value datasets")
        return True
    print("❌ No cultural value datasets loaded")
    return False


def retrieve_graphrag_triples(
    graphrag: GraphRAG,
    prompt: str,
    bias_top_k: int,
    cultural_top_k: int
) -> Dict[str, List[Triple]]:
    """Retrieve relevant triples from GraphRAG system."""
    relevant_triples = {"bias": [], "cultural": []}

    if "bias" in graphrag.graphs:
        try:
            bias_result = graphrag.query("bias", prompt, top_k=bias_top_k, include_answer=False)
            relevant_triples["bias"] = bias_result["triples"]
            logger.info(f"Retrieved {len(relevant_triples['bias'])} bias triples")
        except Exception as e:
            logger.warning(f"Failed to retrieve bias triples: {e}")

    # Retrieve from all cultural graphs
    cultural_triples = []
    for graph_name in graphrag.graphs:
        if graph_name.startswith("cultural_"):
            try:
                cultural_result = graphrag.query(graph_name, prompt, top_k=cultural_top_k//2, include_answer=False)
                cultural_triples.extend(cultural_result["triples"])
            except Exception as e:
                logger.warning(f"Failed to retrieve from {graph_name}: {e}")

    # Sort by similarity and take top-k
    if cultural_triples:
        cultural_triples.sort(key=lambda x: float(x[0].split(":")[-1]) if ":" in str(x[0]) else 0, reverse=True)
        relevant_triples["cultural"] = cultural_triples[:cultural_top_k]
        logger.info(f"Retrieved {len(relevant_triples['cultural'])} cultural triples")

    return relevant_triples


def create_enhancement_prompt(
    original_prompt: str,
    bias_triples: List[Triple],
    cultural_triples: List[Triple],
    negative_examples: List[str],
    previous_attempts: List[str] = None,
    iteration: int = 1
) -> str:
    """Create comprehensive enhancement prompt including StereoSet negative examples."""
    # Base enhancement prompt
    enhancement_prompt = f"""You are an expert AI assistant specializing in creating inclusive, bias-free image generation prompts. Your task is to enhance the given prompt to promote diversity, cultural awareness, and avoid stereotypes.

ORIGINAL PROMPT: "{original_prompt}"

ITERATION: {iteration}"""

    if previous_attempts:
        enhancement_prompt += f"""

PREVIOUS ATTEMPTS:
{chr(10).join(f"{i+1}. {attempt}" for i, attempt in enumerate(previous_attempts))}

Please improve upon these previous attempts while avoiding repetition."""

    if negative_examples:
        enhancement_prompt += f"""

STEREOTYPES TO AVOID (from StereoSet dataset):
The following examples contain stereotypical content that should be actively avoided:
{chr(10).join(f"- {example}" for example in negative_examples[:10])}

Make sure your enhanced prompt does NOT reinforce these types of stereotypes."""

    if bias_triples:
        enhancement_prompt += f"""

BIAS AWARENESS (from bias knowledge graph):
Consider these bias-related patterns to avoid:
{chr(10).join(f"- {subj} {pred} {obj}" for subj, pred, obj in bias_triples[:10])}"""

    if cultural_triples:
        enhancement_prompt += f"""

CULTURAL VALUES TO INCORPORATE (from cultural knowledge graph):
Draw inspiration from these diverse cultural elements:
{chr(10).join(f"- {subj} {pred} {obj}" for subj, pred, obj in cultural_triples[:10])}"""

    enhancement_prompt += """

ENHANCEMENT REQUIREMENTS:
1. Age Diversity: Include multiple age groups (young, middle-aged, elderly)
2. Ethnic/Racial Diversity: Represent various ethnicities and races
3. Gender Diversity: Include diverse gender identities 
4. Cultural Diversity: Incorporate elements from different cultures
5. Ability Inclusion: Consider people with different abilities
6. Socioeconomic Diversity: Represent varied backgrounds respectfully
7. Specificity: Be concrete and actionable, avoid vague terms

OUTPUT INSTRUCTIONS:
- Provide ONLY the enhanced prompt text
- Be specific and descriptive
- Maintain the original creative intent
- Ensure natural language flow
- Avoid tokenistic or forced diversity
- Make diversity feel authentic and meaningful

Enhanced Prompt:"""

    return enhancement_prompt


def score_diversity(prompt: str) -> Dict[str, any]:
    """Score the diversity of a prompt across multiple dimensions."""
    age_terms = ["young", "elderly", "senior", "teenager", "child", "adult", "middle-aged", "aged"]
    ethnic_terms = ["asian", "african", "latino", "hispanic", "european", "middle eastern", "indigenous", "native", "black", "white", "diverse ethnicities", "various backgrounds"]
    gender_terms = ["women", "men", "female", "male", "non-binary", "transgender", "gender diverse", "different genders"]
    cultural_terms = ["cultural", "traditional", "global", "international", "multicultural", "cross-cultural", "heritage", "customs"]
    ability_terms = ["disabilities", "accessible", "inclusive", "different abilities", "mobility", "neurodivergent"]
    socioeconomic_terms = ["backgrounds", "economic", "class", "working", "professional", "varied", "different walks"]

    prompt_lower = prompt.lower()

    # Score each dimension
    age_score = min(15, sum(3 for term in age_terms if term in prompt_lower))
    ethnic_score = min(20, sum(4 for term in ethnic_terms if term in prompt_lower))
    gender_score = min(15, sum(5 for term in gender_terms if term in prompt_lower))
    cultural_score = min(20, sum(4 for term in cultural_terms if term in prompt_lower))
    ability_score = min(10, sum(5 for term in ability_terms if term in prompt_lower))
    socioeconomic_score = min(10, sum(5 for term in socioeconomic_terms if term in prompt_lower))

    # Specificity score (bonus for concrete details)
    specificity_score = min(10, len(prompt.split()) // 8)  # Longer, more detailed prompts get higher scores

    total_score = age_score + ethnic_score + gender_score + cultural_score + ability_score + socioeconomic_score + specificity_score

    return {
        "total_score": total_score,
        "max_score": 100,
        "breakdown": {
            "age_diversity": age_score,
            "ethnic_diversity": ethnic_score,
            "gender_diversity": gender_score,
            "cultural_diversity": cultural_score,
            "ability_inclusion": ability_score,
            "socioeconomic_diversity": socioeconomic_score,
            "specificity": specificity_score
        }
    }


def get_llm_enhancement(enhancement_prompt: str, graphrag: GraphRAG) -> str:
    """Get LLM enhancement using GraphRAG system."""
    if not graphrag or not graphrag.answerer.embedder.is_available():
        logger.warning("GraphRAG LLM not available, using mock enhancement")
        return get_mock_enhancement(enhancement_prompt)

    try:
        # Use the LLM answerer directly
        response = graphrag.answerer.embedder.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert at creating inclusive, diverse image generation prompts."},
                {"role": "user", "content": enhancement_prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )

        enhanced_prompt = response.choices[0].message.content.strip()

        # Clean up the response
        if enhanced_prompt.startswith('"') and enhanced_prompt.endswith('"'):
            enhanced_prompt = enhanced_prompt[1:-1]

        return enhanced_prompt

    except Exception as e:
        logger.error(f"LLM enhancement failed: {e}")
        return get_mock_enhancement(enhancement_prompt)


def get_mock_enhancement(enhancement_prompt: str) -> str:
    """Provide mock enhancement when LLM is not available."""
    original_match = enhancement_prompt.find('ORIGINAL PROMPT: "') + len('ORIGINAL PROMPT: "')
    original_end = enhancement_prompt.find('"', original_match)
    original_prompt = enhancement_prompt[original_match:original_end]

    mock_additions = [
        "featuring people of diverse ages, ethnicities, and genders",
        "representing various cultural backgrounds and abilities",
        "with inclusive representation across all demographics",
        "showing diverse individuals from different walks of life",
        "depicting people of all backgrounds working together harmoniously"
    ]

    # Select addition based on prompt content
    addition_idx = hash(original_prompt) % len(mock_additions)
    return f"{original_prompt}, {mock_additions[addition_idx]}"


def run_sequential_enhancement(
    original_prompt: str,
    stereoset_rag: StereoSetRAG,
    graphrag: GraphRAG,
    args: argparse.Namespace
) -> Dict[str, any]:
    """Run the sequential enhancement process."""
    print(f"\n🎨 Original prompt: {original_prompt}")

    # Get negative examples from StereoSet
    negative_examples = []
    if stereoset_rag:
        print("\n🔍 Analyzing prompt for stereotypes...")
        negative_examples = stereoset_rag.get_negative_examples(original_prompt)
        if negative_examples:
            print(f"⚠️  Found {len(negative_examples)} stereotypical patterns to avoid")
        else:
            print("✅ No obvious stereotypical patterns detected")

    # Get GraphRAG triples
    graphrag_triples = {"bias": [], "cultural": []}
    if graphrag:
        print("\n📚 Retrieving knowledge from GraphRAG...")
        graphrag_triples = retrieve_graphrag_triples(
            graphrag, original_prompt, args.bias_top_k, args.cultural_top_k
        )

    # Sequential enhancement
    current_prompt = original_prompt
    previous_attempts = []
    iteration_results = []

    print(f"\n🔄 Starting sequential enhancement (target: {args.threshold}/100)")

    for iteration in range(1, args.max_iterations + 1):
        print(f"\n--- Iteration {iteration} ---")

        # Create enhancement prompt
        enhancement_prompt = create_enhancement_prompt(
            original_prompt,
            graphrag_triples["bias"],
            graphrag_triples["cultural"],
            negative_examples,
            previous_attempts,
            iteration
        )

        # Get enhancement
        enhanced_prompt = get_llm_enhancement(enhancement_prompt, graphrag)

        # Score diversity
        diversity_score = score_diversity(enhanced_prompt)

        print(f"Enhanced: {enhanced_prompt}")
        print(f"Diversity Score: {diversity_score['total_score']}/100")

        # Store results
        iteration_results.append({
            "iteration": iteration,
            "prompt": enhanced_prompt,
            "diversity_score": diversity_score,
            "met_threshold": diversity_score["total_score"] >= args.threshold
        })

        # Check if threshold met
        if diversity_score["total_score"] >= args.threshold:
            print(f"✅ Threshold achieved! ({diversity_score['total_score']}/100)")
            current_prompt = enhanced_prompt
            break
        print(f"📈 Progress: {diversity_score['total_score']}/100 (need {args.threshold})")
        previous_attempts.append(enhanced_prompt)
        current_prompt = enhanced_prompt

    else:
        print(f"\n⏰ Reached maximum iterations ({args.max_iterations})")

    return {
        "original_prompt": original_prompt,
        "final_prompt": current_prompt,
        "iterations": iteration_results,
        "negative_examples": negative_examples,
        "graphrag_triples": graphrag_triples,
        "final_score": iteration_results[-1]["diversity_score"] if iteration_results else None
    }


def display_results(results: Dict[str, any]) -> None:
    """Display enhancement results in a nice format."""
    print("\n" + "=" * 100)
    print("ENHANCED PROMPT RESULTS")
    print("=" * 100)

    print("\n🎨 ORIGINAL PROMPT:")
    print(f"   {results['original_prompt']}")

    print("\n✨ FINAL ENHANCED PROMPT:")
    print(f"   {results['final_prompt']}")

    if results["final_score"]:
        print(f"\n📊 FINAL DIVERSITY SCORE: {results['final_score']['total_score']}/100")
        breakdown = results["final_score"]["breakdown"]
        print("   Breakdown:")
        print(f"   • Age Diversity: {breakdown['age_diversity']}/15")
        print(f"   • Ethnic Diversity: {breakdown['ethnic_diversity']}/20")
        print(f"   • Gender Diversity: {breakdown['gender_diversity']}/15")
        print(f"   • Cultural Diversity: {breakdown['cultural_diversity']}/20")
        print(f"   • Ability Inclusion: {breakdown['ability_inclusion']}/10")
        print(f"   • Socioeconomic Diversity: {breakdown['socioeconomic_diversity']}/10")
        print(f"   • Specificity: {breakdown['specificity']}/10")

    print("\n🔄 ENHANCEMENT PROCESS:")
    for result in results["iterations"]:
        print(f"   Iteration {result['iteration']}: {result['diversity_score']['total_score']}/100 "
              f"{'✅' if result['met_threshold'] else '📈'}")

    if results["negative_examples"]:
        print(f"\n⚠️  STEREOTYPES AVOIDED: {len(results['negative_examples'])} patterns")
        print("   Examples:")
        for example in results["negative_examples"][:3]:
            print(f"   • {example}")

    print("\n📚 KNOWLEDGE SOURCES USED:")
    if results["graphrag_triples"]["bias"]:
        print(f"   • Bias patterns: {len(results['graphrag_triples']['bias'])} triples")
    if results["graphrag_triples"]["cultural"]:
        print(f"   • Cultural values: {len(results['graphrag_triples']['cultural'])} triples")
    if results["negative_examples"]:
        print(f"   • StereoSet patterns: {len(results['negative_examples'])} examples")


def main() -> None:
    """Main function to run enhanced sequential prompt enhancement."""
    setup_logging()
    args = parse_arguments()

    print("🎨 Enhanced Image Prompt Enhancement System")
    print("Combining GraphRAG with StereoSet for comprehensive bias mitigation")

    # Get prompt
    if args.prompt:
        original_prompt = args.prompt
    else:
        original_prompt = get_prompt_input()

    # Load systems
    stereoset_rag, graphrag = load_systems(
        args.use_stereoset,
        args.use_graphrag_bias,
        args.cache_file,
        args.stereoset_cache
    )

    if not stereoset_rag and not graphrag:
        print("❌ No enhancement systems available. Exiting.")
        return

    # Run enhancement
    try:
        results = run_sequential_enhancement(original_prompt, stereoset_rag, graphrag, args)
        display_results(results)

        # Save results
        output_file = "enhanced_prompt_results.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Results saved to {output_file}")

    except Exception as e:
        logger.error(f"Enhancement failed: {e}")
        return

    print("\n🎉 Enhancement completed successfully!")


if __name__ == "__main__":
    main()
