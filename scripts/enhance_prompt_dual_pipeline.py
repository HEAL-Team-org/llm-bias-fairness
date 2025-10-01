"""Enhanced Dual-Pipeline Image Prompt Enhancement.

This script combines bias mitigation and diversity enhancement in a sequential approach:
1. StereoSet RAG for stereotype detection and bias mitigation
2. CultureBank RAG for diversity enhancement with keyword search
3. GraphRAG for additional cultural awareness and bias patterns
4. Dual scoring system: separate thresholds for bias and diversity
5. Sequential enhancement until both thresholds are met

Usage:
    python enhance_prompt_dual_pipeline.py                           # Interactive mode
    python enhance_prompt_dual_pipeline.py -p "a doctor"            # Direct prompt
    python enhance_prompt_dual_pipeline.py --bias-threshold 75 --diversity-threshold 80  # Custom thresholds
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.knowledge import DiversityRAG, GraphRAG, StereoSetRAG

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
        description="Enhanced Dual-Pipeline Image Prompt Enhancement",
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
        help="Use StereoSet RAG for bias detection (default: True)"
    )

    parser.add_argument(
        "--use-diversity-rag",
        action="store_true",
        default=True,
        help="Use CultureBank RAG for diversity enhancement (default: True)"
    )

    parser.add_argument(
        "--use-graphrag",
        action="store_true",
        default=True,
        help="Use GraphRAG for additional cultural awareness (default: True)"
    )

    parser.add_argument(
        "--bias-threshold",
        type=int,
        default=75,
        help="Bias mitigation threshold to achieve (default: 75)"
    )

    parser.add_argument(
        "--diversity-threshold",
        type=int,
        default=80,
        help="Diversity threshold to achieve (default: 80)"
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Maximum enhancement iterations (default: 5)"
    )

    parser.add_argument(
        "--stereoset-top-k",
        type=int,
        default=10,
        help="Number of StereoSet records to retrieve (default: 10)"
    )

    parser.add_argument(
        "--diversity-top-k",
        type=int,
        default=8,
        help="Number of diversity examples to retrieve (default: 8)"
    )

    parser.add_argument(
        "--graphrag-cache",
        type=str,
        default="embeddings.pkl",
        help="GraphRAG embedding cache file (default: embeddings.pkl)"
    )

    parser.add_argument(
        "--stereoset-cache",
        type=str,
        default="stereoset_embeddings.pkl",
        help="StereoSet embedding cache file (default: stereoset_embeddings.pkl)"
    )

    parser.add_argument(
        "--diversity-cache",
        type=str,
        default="diversity_embeddings.pkl",
        help="Diversity embedding cache file (default: diversity_embeddings.pkl)"
    )

    return parser.parse_args()


def get_prompt_input() -> str:
    """Get image prompt from user input with a nice interface."""
    print("\n" + "🎨 " + "=" * 75)
    print("   DUAL-PIPELINE IMAGE PROMPT ENHANCEMENT SYSTEM")
    print("=" * 79)
    print("This system combines three powerful approaches for comprehensive enhancement:")
    print("  🛡️  StereoSet RAG - Stereotype detection and bias mitigation")
    print("  🌍 CultureBank RAG - Diversity enhancement with keyword search")
    print("  📚 GraphRAG - Cultural awareness and bias pattern analysis")
    print("\nEnter an image generation prompt to enhance for diversity and bias reduction.")
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


def load_enhancement_systems(args: argparse.Namespace) -> Tuple[
    Optional[StereoSetRAG],
    Optional[DiversityRAG],
    Optional[GraphRAG]
]:
    """Load and initialize all enhancement systems.
    
    Args:
        args: Command line arguments
        
    Returns:
        Tuple of (stereoset_rag, diversity_rag, graphrag) systems

    """
    stereoset_rag = None
    diversity_rag = None
    graphrag = None

    # Initialize StereoSet RAG for bias detection
    if args.use_stereoset:
        print("\n🛡️  Initializing StereoSet RAG system...")
        stereoset_rag = StereoSetRAG(cache_file=args.stereoset_cache, top_k=args.stereoset_top_k)

        if stereoset_rag.load_dataset():
            stats = stereoset_rag.get_dataset_stats()
            print(f"✅ StereoSet loaded: {stats['total_records']} records, "
                  f"{stats['total_stereotypical_sentences']} stereotypical sentences")
        else:
            print("❌ Failed to load StereoSet dataset")
            stereoset_rag = None

    # Initialize CultureBank RAG for diversity enhancement
    if args.use_diversity_rag:
        print("\n🌍 Initializing CultureBank Diversity RAG system...")
        diversity_rag = DiversityRAG(cache_file=args.diversity_cache, top_k=args.diversity_top_k)

        if diversity_rag.load_datasets():
            stats = diversity_rag.get_dataset_stats()
            print(f"✅ CultureBank loaded: {stats['total_records']} records, "
                  f"{stats['total_cultural_groups']} cultural groups")
        else:
            print("❌ Failed to load CultureBank dataset")
            diversity_rag = None

    # Initialize GraphRAG for additional cultural awareness
    if args.use_graphrag:
        print("\n📚 Initializing GraphRAG system...")
        graphrag = GraphRAG(cache_file=args.graphrag_cache)

        # Load bias and cultural data
        bias_loaded = load_bias_graph(graphrag)
        cultural_loaded = load_cultural_graphs(graphrag)

        if not bias_loaded and not cultural_loaded:
            print("❌ Failed to load any GraphRAG datasets")
            graphrag = None
        else:
            print("✅ GraphRAG system loaded successfully")

    return stereoset_rag, diversity_rag, graphrag


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


def score_bias_mitigation(prompt: str) -> Dict[str, any]:
    """Score the bias mitigation aspects of a prompt."""
    inclusive_terms = ["diverse", "inclusive", "various", "different", "multiple", "range of"]
    anti_bias_terms = ["regardless of", "irrespective of", "without regard to", "all backgrounds"]
    stereotype_avoidance = ["people", "individuals", "persons", "professionals", "experts"]

    prompt_lower = prompt.lower()

    # Score inclusive language (0-30)
    inclusive_score = min(30, sum(8 for term in inclusive_terms if term in prompt_lower))

    # Score anti-bias language (0-25)
    anti_bias_score = min(25, sum(8 for term in anti_bias_terms if term in prompt_lower))

    # Score stereotype avoidance (0-25)
    avoidance_score = min(25, sum(5 for term in stereotype_avoidance if term in prompt_lower))

    # Penalty for potentially problematic terms
    problematic_terms = ["typical", "normal", "usual", "standard", "traditional"]
    penalty = min(20, sum(4 for term in problematic_terms if term in prompt_lower))

    # Bonus for explicit diversity mentions
    diversity_bonus = min(20, len(prompt.split()) // 10)  # Bonus for detailed descriptions

    total_score = inclusive_score + anti_bias_score + avoidance_score + diversity_bonus - penalty
    total_score = max(0, min(100, total_score))  # Clamp to 0-100

    return {
        "total_score": total_score,
        "max_score": 100,
        "breakdown": {
            "inclusive_language": inclusive_score,
            "anti_bias_language": anti_bias_score,
            "stereotype_avoidance": avoidance_score,
            "detail_bonus": diversity_bonus,
            "problematic_penalty": penalty
        }
    }


def score_diversity(prompt: str) -> Dict[str, any]:
    """Score the diversity aspects of a prompt across multiple dimensions."""
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
    specificity_score = min(10, len(prompt.split()) // 8)

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


def create_dual_enhancement_prompt(
    original_prompt: str,
    bias_focus: bool,
    diversity_focus: bool,
    negative_examples: List[str],
    diversity_recommendations: List[str],
    graphrag_triples: Dict[str, List[Triple]],
    previous_attempts: List[str] = None,
    iteration: int = 1
) -> str:
    """Create enhancement prompt focusing on both bias and diversity aspects."""
    # Determine focus areas
    focus_description = []
    if bias_focus and diversity_focus:
        focus_description.append("BOTH bias mitigation AND diversity enhancement")
    elif bias_focus:
        focus_description.append("bias mitigation and stereotype avoidance")
    elif diversity_focus:
        focus_description.append("diversity enhancement and cultural inclusion")
    else:
        focus_description.append("general improvement")

    enhancement_prompt = f"""You are an expert AI assistant specializing in creating inclusive, bias-free, and diverse image generation prompts. Your task is to enhance the given prompt with focus on {focus_description[0]}.

ORIGINAL PROMPT: "{original_prompt}"

ITERATION: {iteration}"""

    if previous_attempts:
        enhancement_prompt += f"""

PREVIOUS ATTEMPTS:
{chr(10).join(f"{i+1}. {attempt}" for i, attempt in enumerate(previous_attempts))}

Please improve upon these previous attempts while avoiding repetition."""

    if bias_focus and negative_examples:
        enhancement_prompt += f"""

BIAS MITIGATION - STEREOTYPES TO AVOID (from StereoSet dataset):
The following examples contain stereotypical content that should be actively avoided:
{chr(10).join(f"- {example}" for example in negative_examples[:8])}

Make sure your enhanced prompt does NOT reinforce these types of stereotypes."""

    if diversity_focus and diversity_recommendations:
        enhancement_prompt += f"""

DIVERSITY ENHANCEMENT - CULTURAL RECOMMENDATIONS (from CultureBank dataset):
Incorporate these diversity elements and cultural perspectives:
{chr(10).join(f"- {rec}" for rec in diversity_recommendations[:8])}

Focus on authentic and meaningful cultural representation."""

    if graphrag_triples.get("bias"):
        enhancement_prompt += f"""

ADDITIONAL BIAS AWARENESS (from knowledge graph):
Consider these bias-related patterns to avoid:
{chr(10).join(f"- {subj} {pred} {obj}" for subj, pred, obj in graphrag_triples["bias"][:6])}"""

    if graphrag_triples.get("cultural"):
        enhancement_prompt += f"""

CULTURAL VALUES TO INCORPORATE (from knowledge graph):
Draw inspiration from these diverse cultural elements:
{chr(10).join(f"- {subj} {pred} {obj}" for subj, pred, obj in graphrag_triples["cultural"][:6])}"""

    enhancement_prompt += """

ENHANCEMENT REQUIREMENTS:
"""

    if bias_focus:
        enhancement_prompt += """
BIAS MITIGATION:
- Use inclusive language that avoids stereotypes
- Avoid assumptions based on appearance, gender, race, or profession
- Use neutral, respectful terminology
- Include explicit diversity markers to counter bias
"""

    if diversity_focus:
        enhancement_prompt += """
DIVERSITY ENHANCEMENT:
- Age Diversity: Include multiple age groups (young, middle-aged, elderly)
- Ethnic/Racial Diversity: Represent various ethnicities and races  
- Gender Diversity: Include diverse gender identities
- Cultural Diversity: Incorporate elements from different cultures
- Ability Inclusion: Consider people with different abilities
- Socioeconomic Diversity: Represent varied backgrounds respectfully
"""

    enhancement_prompt += """
GENERAL GUIDELINES:
- Be specific and descriptive
- Maintain the original creative intent
- Ensure natural language flow
- Avoid tokenistic or forced diversity
- Make diversity feel authentic and meaningful

OUTPUT INSTRUCTIONS:
- Provide ONLY the enhanced prompt text
- Do not include explanations or meta-commentary
- Focus on the specific improvement areas identified above

Enhanced Prompt:"""

    return enhancement_prompt


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
                {"role": "system", "content": "You are an expert at creating inclusive, diverse, and bias-free image generation prompts."},
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

    # Determine enhancement type from prompt content
    if "BIAS MITIGATION" in enhancement_prompt and "DIVERSITY ENHANCEMENT" in enhancement_prompt:
        addition = "featuring diverse people from various cultural backgrounds, ages, and abilities, while avoiding stereotypical representations"
    elif "BIAS MITIGATION" in enhancement_prompt:
        addition = "showing people without stereotypical assumptions, with inclusive and respectful representation"
    elif "DIVERSITY ENHANCEMENT" in enhancement_prompt:
        addition = "representing multiple cultural groups, age ranges, and diverse backgrounds"
    else:
        addition = "with improved inclusivity and diversity"

    return f"{original_prompt}, {addition}"


def run_dual_pipeline_enhancement(
    original_prompt: str,
    stereoset_rag: StereoSetRAG,
    diversity_rag: DiversityRAG,
    graphrag: GraphRAG,
    args: argparse.Namespace
) -> Dict[str, any]:
    """Run the dual-pipeline enhancement process."""
    print(f"\n🎨 Original prompt: {original_prompt}")

    # Get initial scores
    initial_bias_score = score_bias_mitigation(original_prompt)
    initial_diversity_score = score_diversity(original_prompt)

    print(f"📊 Initial scores - Bias: {initial_bias_score['total_score']}/100, "
          f"Diversity: {initial_diversity_score['total_score']}/100")

    # Get enhancement data
    negative_examples = []
    diversity_recommendations = []
    graphrag_triples = {"bias": [], "cultural": []}

    if stereoset_rag:
        print("\n🛡️  Analyzing prompt for stereotypes...")
        negative_examples = stereoset_rag.get_negative_examples(original_prompt)
        if negative_examples:
            print(f"⚠️  Found {len(negative_examples)} stereotypical patterns to avoid")
        else:
            print("✅ No obvious stereotypical patterns detected")

    if diversity_rag:
        print("\n🌍 Analyzing prompt for diversity opportunities...")
        diversity_recommendations = diversity_rag.get_diversity_recommendations(original_prompt)
        if diversity_recommendations:
            print(f"💡 Generated {len(diversity_recommendations)} diversity recommendations")
        else:
            print("ℹ️  Using general diversity guidelines")

    if graphrag:
        print("\n📚 Retrieving knowledge from GraphRAG...")
        if "bias" in graphrag.graphs:
            try:
                bias_result = graphrag.query("bias", original_prompt, top_k=10, include_answer=False)
                graphrag_triples["bias"] = bias_result["triples"]
                logger.info(f"Retrieved {len(graphrag_triples['bias'])} bias triples")
            except Exception as e:
                logger.warning(f"Failed to retrieve bias triples: {e}")

        # Retrieve from cultural graphs
        cultural_triples = []
        for graph_name in graphrag.graphs:
            if graph_name.startswith("cultural_"):
                try:
                    cultural_result = graphrag.query(graph_name, original_prompt, top_k=5, include_answer=False)
                    cultural_triples.extend(cultural_result["triples"])
                except Exception as e:
                    logger.warning(f"Failed to retrieve from {graph_name}: {e}")

        if cultural_triples:
            cultural_triples.sort(key=lambda x: float(x[0].split(":")[-1]) if ":" in str(x[0]) else 0, reverse=True)
            graphrag_triples["cultural"] = cultural_triples[:10]
            logger.info(f"Retrieved {len(graphrag_triples['cultural'])} cultural triples")

    # Sequential enhancement with dual scoring
    current_prompt = original_prompt
    previous_attempts = []
    iteration_results = []

    print("\n🔄 Starting dual-pipeline enhancement")
    print(f"   🎯 Bias threshold: {args.bias_threshold}/100")
    print(f"   🎯 Diversity threshold: {args.diversity_threshold}/100")

    for iteration in range(1, args.max_iterations + 1):
        print(f"\n--- Iteration {iteration} ---")

        # Score current prompt
        bias_score = score_bias_mitigation(current_prompt)
        diversity_score = score_diversity(current_prompt)

        bias_met = bias_score["total_score"] >= args.bias_threshold
        diversity_met = diversity_score["total_score"] >= args.diversity_threshold

        print(f"Current scores - Bias: {bias_score['total_score']}/100 {'✅' if bias_met else '❌'}, "
              f"Diversity: {diversity_score['total_score']}/100 {'✅' if diversity_met else '❌'}")

        # Check if both thresholds met
        if bias_met and diversity_met:
            print("🎉 Both thresholds achieved!")
            break

        # Determine focus areas for this iteration
        bias_focus = not bias_met
        diversity_focus = not diversity_met

        focus_msg = []
        if bias_focus:
            focus_msg.append("bias mitigation")
        if diversity_focus:
            focus_msg.append("diversity enhancement")
        print(f"🔍 Focusing on: {' and '.join(focus_msg)}")

        # Create enhancement prompt
        enhancement_prompt = create_dual_enhancement_prompt(
            original_prompt,
            bias_focus,
            diversity_focus,
            negative_examples,
            diversity_recommendations,
            graphrag_triples,
            previous_attempts,
            iteration
        )

        # Get enhancement
        enhanced_prompt = get_llm_enhancement(enhancement_prompt, graphrag)

        print(f"Enhanced: {enhanced_prompt}")

        # Store results
        iteration_results.append({
            "iteration": iteration,
            "prompt": enhanced_prompt,
            "bias_score": bias_score,
            "diversity_score": diversity_score,
            "bias_met": bias_met,
            "diversity_met": diversity_met,
            "focus_areas": {"bias": bias_focus, "diversity": diversity_focus}
        })

        previous_attempts.append(current_prompt)
        current_prompt = enhanced_prompt

    else:
        print(f"\n⏰ Reached maximum iterations ({args.max_iterations})")

    # Final scoring
    final_bias_score = score_bias_mitigation(current_prompt)
    final_diversity_score = score_diversity(current_prompt)

    return {
        "original_prompt": original_prompt,
        "final_prompt": current_prompt,
        "initial_scores": {
            "bias": initial_bias_score,
            "diversity": initial_diversity_score
        },
        "final_scores": {
            "bias": final_bias_score,
            "diversity": final_diversity_score
        },
        "iterations": iteration_results,
        "enhancement_data": {
            "negative_examples": negative_examples,
            "diversity_recommendations": diversity_recommendations,
            "graphrag_triples": graphrag_triples
        },
        "thresholds": {
            "bias": args.bias_threshold,
            "diversity": args.diversity_threshold
        }
    }


def display_dual_results(results: Dict[str, any]) -> None:
    """Display dual-pipeline enhancement results."""
    print("\n" + "=" * 100)
    print("DUAL-PIPELINE ENHANCEMENT RESULTS")
    print("=" * 100)

    print("\n🎨 ORIGINAL PROMPT:")
    print(f"   {results['original_prompt']}")

    print("\n✨ FINAL ENHANCED PROMPT:")
    print(f"   {results['final_prompt']}")

    # Score comparison
    print("\n📊 SCORE PROGRESSION:")
    initial_bias = results["initial_scores"]["bias"]["total_score"]
    initial_diversity = results["initial_scores"]["diversity"]["total_score"]
    final_bias = results["final_scores"]["bias"]["total_score"]
    final_diversity = results["final_scores"]["diversity"]["total_score"]

    print(f"   🛡️  Bias Mitigation: {initial_bias}/100 → {final_bias}/100 "
          f"({'✅' if final_bias >= results['thresholds']['bias'] else '❌'})")
    print(f"   🌍 Diversity Enhancement: {initial_diversity}/100 → {final_diversity}/100 "
          f"({'✅' if final_diversity >= results['thresholds']['diversity'] else '❌'})")

    # Detailed breakdowns
    print("\n📈 FINAL SCORE BREAKDOWNS:")
    print("   Bias Mitigation:")
    bias_breakdown = results["final_scores"]["bias"]["breakdown"]
    for key, value in bias_breakdown.items():
        print(f"     • {key.replace('_', ' ').title()}: {value}")

    print("   Diversity Enhancement:")
    diversity_breakdown = results["final_scores"]["diversity"]["breakdown"]
    for key, value in diversity_breakdown.items():
        print(f"     • {key.replace('_', ' ').title()}: {value}")

    # Enhancement process
    print("\n🔄 ENHANCEMENT PROCESS:")
    for result in results["iterations"]:
        bias_icon = "✅" if result["bias_met"] else "❌"
        diversity_icon = "✅" if result["diversity_met"] else "❌"
        print(f"   Iteration {result['iteration']}: Bias {result['bias_score']['total_score']}/100 {bias_icon}, "
              f"Diversity {result['diversity_score']['total_score']}/100 {diversity_icon}")

    # Enhancement sources
    print("\n📚 ENHANCEMENT SOURCES USED:")
    enhancement_data = results["enhancement_data"]
    if enhancement_data["negative_examples"]:
        print(f"   🛡️  StereoSet patterns: {len(enhancement_data['negative_examples'])} examples")
        print("     Examples:")
        for example in enhancement_data["negative_examples"][:3]:
            print(f"       • {example}")

    if enhancement_data["diversity_recommendations"]:
        print(f"   🌍 CultureBank recommendations: {len(enhancement_data['diversity_recommendations'])} suggestions")
        print("     Examples:")
        for rec in enhancement_data["diversity_recommendations"][:3]:
            print(f"       • {rec}")

    if enhancement_data["graphrag_triples"]["bias"]:
        print(f"   📊 GraphRAG bias patterns: {len(enhancement_data['graphrag_triples']['bias'])} triples")

    if enhancement_data["graphrag_triples"]["cultural"]:
        print(f"   📚 GraphRAG cultural values: {len(enhancement_data['graphrag_triples']['cultural'])} triples")


def main() -> None:
    """Main function to run dual-pipeline prompt enhancement."""
    setup_logging()
    args = parse_arguments()

    print("🎨 Dual-Pipeline Image Prompt Enhancement System")
    print("Combining bias mitigation and diversity enhancement with separate scoring")

    # Get prompt
    if args.prompt:
        original_prompt = args.prompt
    else:
        original_prompt = get_prompt_input()

    # Load systems
    stereoset_rag, diversity_rag, graphrag = load_enhancement_systems(args)

    if not stereoset_rag and not diversity_rag and not graphrag:
        print("❌ No enhancement systems available. Exiting.")
        return

    # Run enhancement
    try:
        results = run_dual_pipeline_enhancement(
            original_prompt, stereoset_rag, diversity_rag, graphrag, args
        )
        display_dual_results(results)

        # Save results
        output_file = "dual_pipeline_results.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Results saved to {output_file}")

    except Exception as e:
        logger.error(f"Enhancement failed: {e}")
        return

    print("\n🎉 Dual-pipeline enhancement completed successfully!")


if __name__ == "__main__":
    main()
