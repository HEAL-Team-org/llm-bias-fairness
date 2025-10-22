"""Image Prompt Enhancement using GraphRAG with Sequential Improvement and Diversity Scoring.

This script uses the GraphRAG system to sequentially enhance image generation prompts by:
1. Retrieving relevant bias information to avoid stereotypes
2. Retrieving cultural values to promote inclusivity  
3. Using an LLM to enhance the original prompt with diversity and bias mitigation
4. Scoring the diversity of the enhanced prompt (0-100)
5. Iteratively improving until a diversity threshold is met

Usage:
    python enhance_prompt_sequential.py                                    # Interactive mode
    python enhance_prompt_sequential.py -p "a doctor in a hospital"       # Direct prompt
    python enhance_prompt_sequential.py --prompt "students in classroom"  # Long form
    python enhance_prompt_sequential.py --threshold 80 --max-iterations 5 # Custom settings
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.data.parsers import DataParserFactory, Triple
from src.knowledge import GraphRAG


def setup_logging() -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
        handlers=[logging.StreamHandler()]
    )


def get_prompt_input() -> str:
    """Get image prompt from user input with a nice interface."""
    print("\n" + "🎨 " + "=" * 70)
    print("   IMAGE PROMPT ENHANCEMENT WITH SEQUENTIAL IMPROVEMENT")
    print("=" * 74)
    print("Enter an image generation prompt that you want to enhance for diversity")
    print("and bias mitigation. The system will iteratively improve the prompt")
    print("until it meets a diversity threshold.")
    print("\nExamples:")
    print("  • a doctor examining a patient")
    print("  • students studying in a classroom")
    print("  • a family having dinner")
    print("  • engineers working on a project")
    print("  • people celebrating a festival")
    print("-" * 74)

    while True:
        prompt = input("🎨 Your image prompt: ").strip()
        if prompt:
            return prompt
        print("⚠️  Please enter a non-empty prompt.")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Image Prompt Enhancement with Sequential Improvement and Diversity Scoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python enhance_prompt_sequential.py                                    # Interactive mode
  python enhance_prompt_sequential.py -p "a doctor in hospital"         # Direct prompt
  python enhance_prompt_sequential.py --prompt "students in classroom"  # Long form
  python enhance_prompt_sequential.py --threshold 80 --max-iterations 5 # Custom settings
        """
    )

    parser.add_argument(
        "-p", "--prompt",
        type=str,
        help="Image generation prompt to enhance (optional - will prompt if not provided)"
    )

    parser.add_argument(
        "--bias-top-k",
        type=int,
        default=25,
        help="Number of bias triples to retrieve (default: 25)"
    )

    parser.add_argument(
        "--cultural-top-k",
        type=int,
        default=25,
        help="Number of cultural value triples to retrieve (default: 25)"
    )

    parser.add_argument(
        "--cache-file",
        type=str,
        default="embeddings.pkl",
        help="Embedding cache file to use (default: embeddings.pkl)"
    )

    parser.add_argument(
        "--threshold",
        type=int,
        default=75,
        help="Diversity score threshold (0-100) to meet before stopping (default: 75)"
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum number of enhancement iterations (default: 3)"
    )

    return parser.parse_args()


def load_bias_graph(graphrag: GraphRAG) -> bool:
    """Load the bias dataset into GraphRAG."""
    bias_file = "data/biases/ADV_GRAPH_20240119 - ADV_GRAPH_20240119.csv"
    if not Path(bias_file).exists():
        print(f"❌ Bias file not found: {bias_file}")
        return False

    try:
        parser = DataParserFactory.create_parser(bias_file)
        graph = graphrag.add_graph("bias_graph", parser)
        print(f"✅ Loaded bias graph: {graph}")
        return True
    except Exception as e:
        print(f"❌ Error loading bias data: {e}")
        return False


def load_cultural_graphs(graphrag: GraphRAG) -> bool:
    """Load cultural values datasets into separate graphs."""
    cultural_dir = Path("data/cultural_values")
    if not cultural_dir.exists():
        print(f"❌ Cultural values directory not found: {cultural_dir}")
        return False

    txt_files = list(cultural_dir.glob("*.txt"))
    if not txt_files:
        print("❌ No text files found in cultural values directory")
        return False

    # Load all cultural datasets into a single combined graph
    print("\n--- Loading Cultural Values Data ---")
    combined_triples = []

    for file_path in txt_files:
        try:
            parser = DataParserFactory.create_parser(file_path)
            triples = parser.parse()
            combined_triples.extend(triples)
            country = file_path.stem.replace(" triples", "").replace("_", " ").title()
            print(f"  • {country}: {len(triples)} triples")
        except Exception as e:
            print(f"  ⚠️  Failed to load {file_path.name}: {e}")

    if combined_triples:
        # Create a combined cultural graph
        from src.data.parsers import BaseDataParser

        class CombinedParser(BaseDataParser):
            def __init__(self, triples: List[Triple]):
                self.triples = triples

            def parse(self) -> List[Triple]:
                return self.triples

        combined_parser = CombinedParser(combined_triples)
        graph = graphrag.add_graph("cultural_values", combined_parser)
        print(f"✅ Created combined cultural graph: {graph}")
        return True

    return False


def retrieve_relevant_triples(graphrag: GraphRAG, prompt: str, bias_top_k: int, cultural_top_k: int) -> Dict[str, List[Triple]]:
    """Retrieve relevant triples from both bias and cultural graphs using semantic similarity."""
    results = {}

    # Retrieve bias triples
    print(f"\n🔍 Retrieving top {bias_top_k} bias-related triples...")
    try:
        bias_result = graphrag.query(
            graph_name="bias_graph",
            question=prompt,
            top_k=bias_top_k,
            include_answer=False
        )
        results["bias"] = bias_result["triples"]
        print(f"✅ Retrieved {len(results['bias'])} bias triples")
    except Exception as e:
        print(f"❌ Failed to retrieve bias triples: {e}")
        results["bias"] = []

    # Retrieve cultural value triples
    print(f"🌍 Retrieving top {cultural_top_k} cultural value triples...")
    try:
        cultural_result = graphrag.query(
            graph_name="cultural_values",
            question=prompt,
            top_k=cultural_top_k,
            include_answer=False
        )
        results["cultural"] = cultural_result["triples"]
        print(f"✅ Retrieved {len(results['cultural'])} cultural value triples")
    except Exception as e:
        print(f"❌ Failed to retrieve cultural triples: {e}")
        results["cultural"] = []

    return results


def format_triples_for_llm(triples: List[Triple], category: str) -> str:
    """Format triples for LLM prompt."""
    if not triples:
        return f"No {category} data available."

    formatted = f"\n{category.upper()} INFORMATION:\n"
    formatted += "=" * 50 + "\n"

    for i, triple in enumerate(triples[:25], 1):  # Limit to top 25
        if hasattr(triple, "subject"):
            # Triple object with attributes
            formatted += f"{i}. {triple.subject} -> {triple.predicate} -> {triple.object}\n"
        else:
            # Tuple format (subject, predicate, object)
            subject, predicate, obj = triple
            formatted += f"{i}. {subject} -> {predicate} -> {obj}\n"

    return formatted


def format_triples_for_display(triples: List[Triple], category: str, max_display: int = 5) -> str:
    """Format triples for console display (limited number)."""
    if not triples:
        return f"   No {category} triples retrieved."

    formatted = f"   {category.title()} triples (top {min(max_display, len(triples))}):\n"

    for i, triple in enumerate(triples[:max_display], 1):
        if hasattr(triple, "subject"):
            formatted += f"     {i}. {triple.subject} -> {triple.predicate} -> {triple.object}\n"
        else:
            subject, predicate, obj = triple
            formatted += f"     {i}. {subject} -> {predicate} -> {obj}\n"

    if len(triples) > max_display:
        formatted += f"     ... and {len(triples) - max_display} more\n"

    return formatted.rstrip()


def create_enhancement_prompt(
    original_prompt: str,
    bias_triples: List[Triple],
    cultural_triples: List[Triple],
    iteration: int = 1,
    previous_prompt: Optional[str] = None,
    previous_score: Optional[int] = None
) -> str:
    """Create the prompt for LLM to enhance the original image prompt."""
    bias_info = format_triples_for_llm(bias_triples, "bias and stereotype")
    cultural_info = format_triples_for_llm(cultural_triples, "cultural values and diversity")

    # Add context for iterative improvement
    iteration_context = ""
    if iteration > 1 and previous_prompt and previous_score is not None:
        iteration_context = f"""
PREVIOUS ITERATION CONTEXT:
This is iteration {iteration} of the enhancement process.
Previous enhanced prompt: {previous_prompt}
Previous diversity score: {previous_score}/100

The previous prompt did not meet the minimum diversity threshold. Please focus on improving the areas that scored poorly in diversity assessment.
"""

    return f"""You are an AI assistant specialized in creating inclusive and diverse image generation prompts that avoid stereotypes and biases.

ORIGINAL IMAGE PROMPT:
{original_prompt}

{iteration_context}

{bias_info}

{cultural_info}

TASK:
Please enhance the {"original" if iteration == 1 else "previous"} image prompt to:

1. AVOID BIASES AND STEREOTYPES:
   - Use the bias information above to identify potential stereotypes
   - Modify the prompt to avoid reinforcing harmful biases
   - Ensure fair representation across different groups

2. PROMOTE CULTURAL DIVERSITY AND INCLUSION:
   - Incorporate relevant cultural values from the information above
   - Add elements that promote diversity in:
     * Age (various age groups)
     * Nationality and ethnicity (global representation)
     * Skin color and appearance (diverse physical features)
     * Gender and sex (inclusive gender representation)
     * Abilities and disabilities (accessibility and inclusion)
     * Socioeconomic backgrounds
     * Cultural practices and traditions

3. ENHANCE FOR VISUAL DIVERSITY:
   - Add specific descriptive elements that encourage diverse representation
   - Include cultural artifacts, clothing, or settings when appropriate
   - Suggest inclusive environments and contexts

REQUIREMENTS:
- Keep the core concept of the original prompt
- Make the enhanced prompt specific and actionable for image generation
- Provide concrete details rather than general statements
- Ensure the prompt flows naturally and is not overly complex
- Focus on positive representation rather than just avoiding negatives
{"- Improve upon the previous iteration's weaknesses" if iteration > 1 else ""}

Please provide:
1. ENHANCED PROMPT: A single, well-crafted image generation prompt
2. EXPLANATION: Brief explanation of key enhancements made
3. DIVERSITY ELEMENTS: List of specific diversity aspects incorporated

Enhanced Image Prompt:"""


def create_diversity_scoring_prompt(prompt: str) -> str:
    """Create the prompt for LLM to score diversity of the enhanced prompt."""
    scoring_prompt = f"""You are an AI diversity assessment expert. Your task is to evaluate the diversity and inclusivity of an image generation prompt.

IMAGE GENERATION PROMPT TO EVALUATE:
{prompt}

EVALUATION CRITERIA:
Score the prompt on a scale of 0-100 based on how well it promotes diversity and inclusion across these dimensions:

1. AGE DIVERSITY (15 points max):
   - Does it include or allow for multiple age groups?
   - Are different generations represented or implied?

2. ETHNIC/RACIAL DIVERSITY (20 points max):
   - Does it promote representation across different ethnicities?
   - Are various racial backgrounds encouraged or implied?

3. GENDER DIVERSITY (15 points max):
   - Does it include diverse gender representation?
   - Is the language gender-inclusive?

4. CULTURAL DIVERSITY (20 points max):
   - Does it incorporate diverse cultural elements?
   - Are different cultural practices, clothing, or traditions represented?

5. ABILITY/DISABILITY INCLUSION (10 points max):
   - Does it consider people with different abilities?
   - Is accessibility and inclusion addressed?

6. SOCIOECONOMIC DIVERSITY (10 points max):
   - Does it avoid assumptions about wealth or class?
   - Are different socioeconomic backgrounds considered?

7. SPECIFICITY AND ACTIONABILITY (10 points max):
   - Are the diversity elements specific and concrete?
   - Would an AI image generator understand how to create diverse imagery?

Please provide your response in this exact JSON format:

{{
  "overall_score": [0-100],
  "breakdown": {{
    "age_diversity": [0-15],
    "ethnic_racial_diversity": [0-20],
    "gender_diversity": [0-15],
    "cultural_diversity": [0-20],
    "ability_inclusion": [0-10],
    "socioeconomic_diversity": [0-10],
    "specificity": [0-10]
  }},
  "strengths": ["list", "of", "strengths"],
  "weaknesses": ["list", "of", "areas", "for", "improvement"],
  "suggestions": ["specific", "suggestions", "for", "improvement"]
}}"""

    return scoring_prompt


def score_diversity(prompt: str, graphrag: GraphRAG) -> Dict:
    """Score the diversity of a prompt using LLM."""
    print("\n📊 Scoring diversity of prompt...")

    if not graphrag.embedder.is_available():
        print("⚠️  OpenAI API not available, using mock diversity score...")
        return get_mock_diversity_score(prompt)

    try:
        scoring_prompt = create_diversity_scoring_prompt(prompt)

        response = graphrag.embedder.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in diversity and inclusion assessment. Provide detailed, accurate scoring in the exact JSON format requested."
                },
                {
                    "role": "user",
                    "content": scoring_prompt
                }
            ],
        )

        score_text = response.choices[0].message.content.strip()

        # Try to parse JSON response
        try:
            # Extract JSON from response if it's wrapped in markdown
            if "```json" in score_text:
                json_start = score_text.find("```json") + 7
                json_end = score_text.find("```", json_start)
                score_text = score_text[json_start:json_end].strip()
            elif "```" in score_text:
                json_start = score_text.find("```") + 3
                json_end = score_text.find("```", json_start)
                score_text = score_text[json_start:json_end].strip()

            score_data = json.loads(score_text)
            print(f"✅ Diversity score: {score_data['overall_score']}/100")
            return score_data

        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse diversity score JSON: {e}")
            print("⚠️  Falling back to mock diversity score...")
            return get_mock_diversity_score(prompt)

    except Exception as e:
        print(f"❌ Error getting diversity score: {e}")
        print("⚠️  Falling back to mock diversity score...")
        return get_mock_diversity_score(prompt)


def get_mock_diversity_score(prompt: str) -> Dict:
    """Create a mock diversity score when LLM is not available."""
    # Simple heuristic scoring based on keywords
    diversity_keywords = {
        "age": ["young", "old", "elderly", "senior", "child", "adult", "teenager", "generations"],
        "ethnic": ["diverse", "multicultural", "various", "different", "global", "international"],
        "gender": ["men", "women", "non-binary", "gender", "inclusive"],
        "cultural": ["cultural", "traditional", "heritage", "customs", "practices"],
        "ability": ["accessibility", "disabilities", "abilities", "inclusive"],
        "socioeconomic": ["backgrounds", "communities", "varied"]
    }

    prompt_lower = prompt.lower()
    scores = {}

    for category, keywords in diversity_keywords.items():
        found_keywords = sum(1 for keyword in keywords if keyword in prompt_lower)
        # Simple scoring: max points if 2+ keywords found, proportional otherwise
        max_scores = {"age": 15, "ethnic": 20, "gender": 15, "cultural": 20, "ability": 10, "socioeconomic": 10}
        scores[category] = min(max_scores[category], found_keywords * (max_scores[category] // 2))

    specificity_score = 8 if len(prompt.split()) > 15 else 5  # Longer prompts tend to be more specific

    overall_score = sum(scores.values()) + specificity_score

    return {
        "overall_score": overall_score,
        "breakdown": {
            "age_diversity": scores["age"],
            "ethnic_racial_diversity": scores["ethnic"],
            "gender_diversity": scores["gender"],
            "cultural_diversity": scores["cultural"],
            "ability_inclusion": scores["ability"],
            "socioeconomic_diversity": scores["socioeconomic"],
            "specificity": specificity_score
        },
        "strengths": ["Includes basic diversity elements"],
        "weaknesses": ["Could be more specific about diversity aspects"],
        "suggestions": ["Add more explicit diversity descriptors", "Include specific cultural elements"]
    }


def get_llm_enhancement(enhancement_prompt: str, graphrag: GraphRAG) -> str:
    """Get enhancement from LLM using OpenAI API."""
    print("\n🤖 Generating enhanced prompt with LLM...")

    if not graphrag.embedder.is_available():
        print("⚠️  OpenAI API not available, using mock response...")
        return get_mock_enhancement(enhancement_prompt)

    try:
        # Use the LLM to enhance the prompt
        response = graphrag.embedder.client.chat.completions.create(
            model="gpt-4o-mini",  # Use a good model for creative tasks
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in creating inclusive and diverse image generation prompts that avoid stereotypes and promote cultural sensitivity."
                },
                {
                    "role": "user",
                    "content": enhancement_prompt
                }
            ],
        )

        enhanced_result = response.choices[0].message.content.strip()
        print("✅ Enhanced prompt generated successfully!")
        return enhanced_result

    except Exception as e:
        print(f"❌ Error generating enhancement: {e}")
        print("⚠️  Falling back to mock response...")
        return get_mock_enhancement(enhancement_prompt)


def get_mock_enhancement(enhancement_prompt: str) -> str:
    """Create a mock enhancement when LLM is not available."""
    original_prompt = ""
    lines = enhancement_prompt.split("\n")
    for i, line in enumerate(lines):
        if "ORIGINAL IMAGE PROMPT:" in line and i + 1 < len(lines):
            original_prompt = lines[i + 1].strip()
            break

    if not original_prompt:
        original_prompt = "diverse people"

    mock_response = f"""
ENHANCED PROMPT:
{original_prompt}, featuring people of diverse ages including young adults, middle-aged individuals, and seniors, representing various ethnicities including Asian, African, Latino, Middle Eastern, and European backgrounds, with different skin tones and physical appearances, wearing culturally diverse clothing and accessories, in an inclusive environment that celebrates global diversity, with both men and women and non-binary individuals, including people with visible and invisible disabilities, showcasing different socioeconomic backgrounds through varied but respectful styling, with authentic cultural elements like traditional patterns, diverse architectural styles, and inclusive symbols that promote unity and respect across all communities

EXPLANATION:
Enhanced the original prompt to explicitly include age diversity (young to senior), ethnic and racial diversity (multiple specific backgrounds), gender inclusivity (men, women, non-binary), disability representation, and cultural authenticity through clothing, architecture, and symbols.

DIVERSITY ELEMENTS:
- Age: Multiple generations represented
- Ethnicity: Asian, African, Latino, Middle Eastern, European
- Gender: Men, women, non-binary individuals  
- Abilities: People with visible and invisible disabilities
- Culture: Traditional clothing, patterns, architectural diversity
- Socioeconomic: Varied but respectful representation
- Setting: Inclusive, globally-inspired environment
"""
    return mock_response


def extract_enhanced_prompt(llm_response: str) -> str:
    """Extract just the enhanced prompt from LLM response."""
    lines = llm_response.split("\n")
    enhanced_prompt = ""
    capturing = False

    for line in lines:
        if "ENHANCED PROMPT:" in line.upper():
            capturing = True
            continue
        if capturing and (
            "EXPLANATION:" in line.upper() or "DIVERSITY ELEMENTS:" in line.upper()
        ):
            break
        if capturing:
            enhanced_prompt += line + " "

    return enhanced_prompt.strip()


def sequential_enhance_prompt(
    graphrag: GraphRAG,
    original_prompt: str,
    bias_triples: List[Triple],
    cultural_triples: List[Triple],
    threshold: int,
    max_iterations: int
) -> Tuple[str, List[Dict]]:
    """Sequentially enhance prompt until diversity threshold is met."""
    print("\n🔄 Starting sequential enhancement process...")
    print(f"   Target diversity threshold: {threshold}/100")
    print(f"   Maximum iterations: {max_iterations}")

    current_prompt = original_prompt
    iteration_history = []

    for iteration in range(1, max_iterations + 1):
        print(f"\n{'='*20} ITERATION {iteration} {'='*20}")

        # Get previous context for iterations beyond the first
        previous_prompt = None
        previous_score = None
        if iteration > 1:
            previous_prompt = iteration_history[-1]["enhanced_prompt"]
            previous_score = iteration_history[-1]["diversity_score"]["overall_score"]

        # Create enhancement prompt
        enhancement_prompt = create_enhancement_prompt(
            original_prompt, bias_triples, cultural_triples,
            iteration, previous_prompt, previous_score
        )

        # Get LLM enhancement
        enhanced_result = get_llm_enhancement(enhancement_prompt, graphrag)
        enhanced_prompt = extract_enhanced_prompt(enhanced_result)

        # Score diversity
        diversity_score = score_diversity(enhanced_prompt, graphrag)

        # Store iteration data
        iteration_data = {
            "iteration": iteration,
            "enhanced_prompt": enhanced_prompt,
            "diversity_score": diversity_score,
            "full_response": enhanced_result,
            "threshold_met": diversity_score["overall_score"] >= threshold
        }
        iteration_history.append(iteration_data)

        # Display iteration results
        print(f"\n📊 ITERATION {iteration} RESULTS:")
        print(f"   Diversity Score: {diversity_score['overall_score']}/100")
        print(f"   Threshold Met: {'✅ Yes' if iteration_data['threshold_met'] else '❌ No'}")

        # Check if threshold is met
        if iteration_data["threshold_met"]:
            print(f"\n🎉 Diversity threshold reached in {iteration} iteration(s)!")
            break

        print(f"   🔄 Continuing to iteration {iteration + 1}...")

        # Update current prompt for next iteration
        current_prompt = enhanced_prompt

    final_prompt = iteration_history[-1]["enhanced_prompt"]
    final_score = iteration_history[-1]["diversity_score"]["overall_score"]

    if final_score >= threshold:
        print(f"\n✅ FINAL SUCCESS: Achieved diversity score of {final_score}/100 (>= {threshold})")
    else:
        print(f"\n⚠️  FINAL RESULT: Reached maximum iterations. Best score: {final_score}/100")

    return final_prompt, iteration_history


def display_final_results(original_prompt: str, final_prompt: str, iteration_history: List[Dict],
                         relevant_triples: Dict[str, List[Triple]]) -> None:
    """Display comprehensive final results."""
    print("\n" + "=" * 80)
    print("SEQUENTIAL PROMPT ENHANCEMENT RESULTS")
    print("=" * 80)

    print("\n🎨 ORIGINAL PROMPT:")
    print(f"   {original_prompt}")

    print("\n📊 RETRIEVED DATA:")
    if relevant_triples.get("bias"):
        print(f"   • Bias triples: {len(relevant_triples['bias'])}")
        print(format_triples_for_display(relevant_triples["bias"], "bias", 3))
    if relevant_triples.get("cultural"):
        print(f"   • Cultural triples: {len(relevant_triples['cultural'])}")
        print(format_triples_for_display(relevant_triples["cultural"], "cultural", 3))

    print("\n🔄 ENHANCEMENT ITERATIONS:")
    for i, iteration in enumerate(iteration_history):
        print(f"\n   Iteration {iteration['iteration']}:")
        print(f"     Score: {iteration['diversity_score']['overall_score']}/100")
        print(f"     Threshold Met: {'✅' if iteration['threshold_met'] else '❌'}")

        # Show breakdown for final iteration
        if i == len(iteration_history) - 1:
            breakdown = iteration["diversity_score"]["breakdown"]
            print("     Breakdown:")
            print(f"       • Age Diversity: {breakdown['age_diversity']}/15")
            print(f"       • Ethnic/Racial: {breakdown['ethnic_racial_diversity']}/20")
            print(f"       • Gender: {breakdown['gender_diversity']}/15")
            print(f"       • Cultural: {breakdown['cultural_diversity']}/20")
            print(f"       • Ability Inclusion: {breakdown['ability_inclusion']}/10")
            print(f"       • Socioeconomic: {breakdown['socioeconomic_diversity']}/10")
            print(f"       • Specificity: {breakdown['specificity']}/10")

    print("\n✨ FINAL ENHANCED PROMPT:")
    print(f"   {final_prompt}")

    # Show final diversity analysis
    final_diversity = iteration_history[-1]["diversity_score"]
    if final_diversity.get("strengths"):
        print("\n💪 STRENGTHS:")
        for strength in final_diversity["strengths"]:
            print(f"   • {strength}")

    if final_diversity.get("weaknesses"):
        print("\n🔧 AREAS FOR IMPROVEMENT:")
        for weakness in final_diversity["weaknesses"]:
            print(f"   • {weakness}")

    print("\n🎉 Sequential enhancement completed!")
    print("The enhanced prompt promotes diversity and inclusion while avoiding")
    print("stereotypes and biases based on your knowledge graphs.")


def main() -> None:
    """Run the sequential image prompt enhancement script."""
    args = parse_arguments()
    setup_logging()

    print("🎨 Sequential Image Prompt Enhancement System")
    print("Enhancing prompts for diversity with iterative improvement and scoring")

    # Get prompt from args or user input
    if args.prompt:
        original_prompt = args.prompt
        print(f"\n📝 Original prompt: {original_prompt}")
    else:
        original_prompt = get_prompt_input()

    # Initialize GraphRAG system
    print("\n⚙️  Initializing GraphRAG system...")
    graphrag = GraphRAG(cache_file=args.cache_file)

    # Load datasets
    print("\n📚 Loading knowledge graphs...")

    bias_loaded = load_bias_graph(graphrag)
    cultural_loaded = load_cultural_graphs(graphrag)

    if not bias_loaded and not cultural_loaded:
        print("❌ Failed to load any datasets. Exiting.")
        return

    # Retrieve relevant triples (once, used for all iterations)
    relevant_triples = retrieve_relevant_triples(
        graphrag, original_prompt, args.bias_top_k, args.cultural_top_k
    )

    # Sequential enhancement process
    final_prompt, iteration_history = sequential_enhance_prompt(
        graphrag, original_prompt,
        relevant_triples.get("bias", []),
        relevant_triples.get("cultural", []),
        args.threshold, args.max_iterations
    )

    # Display comprehensive results
    display_final_results(original_prompt, final_prompt, iteration_history, relevant_triples)


if __name__ == "__main__":
    main()
