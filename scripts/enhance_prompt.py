"""Image Prompt Enhancement using GraphRAG for Bias Mitigation and Cultural Awareness.

This script uses the GraphRAG system to enhance image generation prompts by:
1. Retrieving relevant bias information to avoid stereotypes
2. Retrieving cultural values to promote inclusivity  
3. Using an LLM to enhance the original prompt with diversity and bias mitigation

Usage:
    python enhance_prompt.py                                    # Interactive mode
    python enhance_prompt.py -p "a doctor in a hospital"       # Direct prompt
    python enhance_prompt.py --prompt "students in classroom"  # Long form
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List

from src.graphrag import GraphRAG
from src.parsers import DataParserFactory, Triple


def setup_logging() -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
        handlers=[logging.StreamHandler()]
    )


def get_prompt_input() -> str:
    """Get image prompt from user input with a nice interface."""
    print("\n" + "🎨 " + "=" * 60)
    print("   IMAGE PROMPT ENHANCEMENT FOR DIVERSITY & BIAS MITIGATION")
    print("=" * 64)
    print("Enter an image generation prompt that you want to enhance for diversity")
    print("and bias mitigation. The system will use cultural values and bias data")
    print("to suggest improvements.")
    print("\nExamples:")
    print("  • a doctor examining a patient")
    print("  • students studying in a classroom")
    print("  • a family having dinner")
    print("  • engineers working on a project")
    print("  • people celebrating a festival")
    print("-" * 64)
    
    while True:
        prompt = input("🎨 Your image prompt: ").strip()
        if prompt:
            return prompt
        print("⚠️  Please enter a non-empty prompt.")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Image Prompt Enhancement for Diversity and Bias Mitigation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python enhance_prompt.py                                    # Interactive mode
  python enhance_prompt.py -p "a doctor in hospital"         # Direct prompt
  python enhance_prompt.py --prompt "students in classroom"  # Long form
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
    print(f"\n--- Loading Cultural Values Data ---")
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
        from src.parsers import BaseDataParser
        
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


def create_enhancement_prompt(original_prompt: str, bias_triples: List[Triple], cultural_triples: List[Triple]) -> str:
    """Create the prompt for LLM to enhance the original image prompt."""
    
    bias_info = format_triples_for_llm(bias_triples, "bias and stereotype")
    cultural_info = format_triples_for_llm(cultural_triples, "cultural values and diversity")
    
    enhancement_prompt = f"""You are an AI assistant specialized in creating inclusive and diverse image generation prompts that avoid stereotypes and biases.

ORIGINAL IMAGE PROMPT:
{original_prompt}

{bias_info}

{cultural_info}

TASK:
Please enhance the original image prompt to:

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

Please provide:
1. ENHANCED PROMPT: A single, well-crafted image generation prompt
2. EXPLANATION: Brief explanation of key enhancements made
3. DIVERSITY ELEMENTS: List of specific diversity aspects incorporated

Enhanced Image Prompt:"""

    return enhancement_prompt


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
            temperature=0.3,  # Slight creativity but mostly focused
            max_tokens=1500
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


def main() -> None:
    """Run the image prompt enhancement script."""
    args = parse_arguments()
    setup_logging()
    
    print("🎨 Image Prompt Enhancement System")
    print("Enhancing prompts for diversity and bias mitigation using GraphRAG")
    
    # Get prompt from args or user input
    if args.prompt:
        original_prompt = args.prompt
        print(f"\n📝 Original prompt: {original_prompt}")
    else:
        original_prompt = get_prompt_input()
    
    # Initialize GraphRAG system
    print(f"\n⚙️  Initializing GraphRAG system...")
    graphrag = GraphRAG(cache_file=args.cache_file)
    
    # Load datasets
    print("\n📚 Loading knowledge graphs...")
    
    bias_loaded = load_bias_graph(graphrag)
    cultural_loaded = load_cultural_graphs(graphrag)
    
    if not bias_loaded and not cultural_loaded:
        print("❌ Failed to load any datasets. Exiting.")
        return
    
    # Retrieve relevant triples
    relevant_triples = retrieve_relevant_triples(
        graphrag, original_prompt, args.bias_top_k, args.cultural_top_k
    )
    
    # Create enhancement prompt for LLM
    print("\n📝 Creating enhancement prompt...")
    enhancement_prompt = create_enhancement_prompt(
        original_prompt,
        relevant_triples.get("bias", []),
        relevant_triples.get("cultural", [])
    )
    
    # Get LLM enhancement
    enhanced_result = get_llm_enhancement(enhancement_prompt, graphrag)
    
    # Display results
    print("\n" + "=" * 80)
    print("PROMPT ENHANCEMENT RESULTS")
    print("=" * 80)
    
    print(f"\n🎨 ORIGINAL PROMPT:")
    print(f"   {original_prompt}")
    
    print(f"\n📊 RETRIEVED DATA:")
    if relevant_triples.get("bias"):
        print(f"   • Bias triples: {len(relevant_triples['bias'])}")
        print(format_triples_for_display(relevant_triples['bias'], "bias", 3))
    if relevant_triples.get("cultural"):
        print(f"   • Cultural triples: {len(relevant_triples['cultural'])}")
        print(format_triples_for_display(relevant_triples['cultural'], "cultural", 3))
    
    print(f"\n✨ ENHANCED RESULT:")
    print(enhanced_result)
    
    print("\n🎉 Prompt enhancement completed!")
    print("\nThe enhanced prompt promotes diversity and inclusion while avoiding")
    print("stereotypes and biases based on your knowledge graphs.")


if __name__ == "__main__":
    main()
