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
import csv
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

from dataclasses import dataclass

from helper.logger import get_logger, log_timer
from src.graphrag import GraphRAG
from src.parsers import DataParserFactory, Triple
import src.generator  # noqa: F401 — ensures all subclasses are registered
from src.generator import BaseLLM

logger = get_logger(__name__)

_LLM_CLASSES = BaseLLM.registry  # populated automatically by __init_subclass__


# ── Robust JSON extractor ────────────────────────────────────────────────────


def extract_json_from_response(text: str) -> str:
    """Robustly extract a JSON object from an LLM response.

    Extraction priority:
    1. Last <json>...</json> pair  (our explicit prompt format — uses the
       *last* </json> tag and the last <json> tag that precedes it, so that
       any stray earlier tags from e.g. chain-of-thought reasoning are ignored)
    2. Last ```json ... ``` or ``` ... ``` markdown fence
    3. First '{' … last '}'  (catches any remaining prose wrapper)

    Also strips Qwen3 <think>...</think> reasoning blocks before all checks.
    """
    # 0. Strip <think>...</think> blocks — greedy so the whole block is removed
    text = re.sub(r"<think>.*</think>", "", text, flags=re.DOTALL)
    # 1. Preferred: last </json> tag, then last <json> tag that comes before it
    last_close = text.rfind("</json>")
    if last_close != -1:
        last_open = text.rfind("<json>", 0, last_close)
        if last_open != -1:
            inner = text[last_open + len("<json>") : last_close].strip()
            # Isolate the JSON object in case there is leading/trailing prose
            first_brace = inner.find("{")
            last_brace = inner.rfind("}")
            if first_brace != -1 and last_brace != -1:
                return inner[first_brace : last_brace + 1].strip()
            return inner
    # 2. Fallback: last ```json ... ``` or ``` ... ``` markdown fence
    fence_matches = list(
        re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    )
    if fence_matches:
        return fence_matches[-1].group(1).strip()
    # 3. Last resort: find first '{' and last '}' — handles any prose wrapper
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return text[first_brace : last_brace + 1].strip()
    # 4. Nothing found — return stripped text so the caller sees a clear error
    return text.strip()


def _quote_unquoted_json_keys(text: str) -> str:
    """Quote unquoted JSON object keys.

    Example: {overall_score: 10} -> {"overall_score": 10}
    """
    key_pattern = re.compile(r"([\{,])\s*([A-Za-z_][A-Za-z0-9_\-]*)\s*:")
    previous = None
    current = text
    for _ in range(5):
        if current == previous:
            break
        previous = current
        current = key_pattern.sub(r'\1"\2":', current)
    return current


def _replace_single_quoted_strings(text: str) -> str:
    """Convert single-quoted strings to double-quoted JSON strings."""
    single_str = re.compile(r"'((?:\\.|[^'\\])*)'")

    def _repl(match: re.Match) -> str:
        inner = match.group(1)
        inner = inner.replace('\\"', '"')
        inner = inner.replace('"', '\\"')
        return f'"{inner}"'

    return single_str.sub(_repl, text)


def repair_json_text(text: str) -> str:
    """Repair common malformed-JSON patterns in LLM responses."""
    repaired = text.strip()

    repaired = (
        repaired.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )

    repaired = re.sub(r"/\*.*?\*/", "", repaired, flags=re.DOTALL)
    repaired = re.sub(r"(^|\s)//.*?$", "", repaired, flags=re.MULTILINE)

    repaired = re.sub(r"\bTrue\b", "true", repaired)
    repaired = re.sub(r"\bFalse\b", "false", repaired)
    repaired = re.sub(r"\bNone\b", "null", repaired)

    # Remove dangling quote-only lines such as:
    #   "
    #   "ability_inclusion": 10
    repaired = re.sub(r'^\s*"\s*,?\s*$', "", repaired, flags=re.MULTILINE)

    # Fix malformed keys like: ability_inclusion": 10  ->  "ability_inclusion": 10
    repaired = re.sub(
        r'([\{,]\s*)([A-Za-z_][A-Za-z0-9_\-]*)"\s*:',
        r'\1"\2":',
        repaired,
    )

    # Fix split malformed key pattern:
    #   "
    #   ability_inclusion": 10
    # -> "ability_inclusion": 10
    repaired = re.sub(
        r'"\s*\n\s*([A-Za-z_][A-Za-z0-9_\-]*)"\s*:',
        r'"\1":',
        repaired,
    )

    repaired = _replace_single_quoted_strings(repaired)
    repaired = _quote_unquoted_json_keys(repaired)

    # Fix missing comma between adjacent string items in arrays:
    #   "item 1"
    #   "item 2"
    # -> "item 1",
    #    "item 2"
    repaired = re.sub(r'("\s*)\n(\s*")', r"\1,\n\2", repaired)

    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)

    return repaired.strip()


def loads_json_with_repair(text: str) -> Dict:
    """Load JSON and retry after repairing common LLM formatting issues."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.loads(repair_json_text(text))


# ── Diversity keyword lexicon ─────────────────────────────────────────────────

_LEXICON: Dict[str, Dict] = {
    "age": {
        "max_score": 15,
        "keywords": [
            "young",
            "old",
            "elderly",
            "senior",
            "child",
            "children",
            "adult",
            "teenager",
            "teen",
            "middle-aged",
            "infant",
            "toddler",
            "generation",
            "generations",
            "age",
            "aged",
            "youth",
            "elder",
        ],
    },
    "ethnic_racial": {
        "max_score": 20,
        "keywords": [
            "diverse",
            "multicultural",
            "multiracial",
            "ethnicity",
            "ethnic",
            "asian",
            "african",
            "latino",
            "hispanic",
            "middle eastern",
            "european",
            "indigenous",
            "aboriginal",
            "south asian",
            "east asian",
            "southeast asian",
            "black",
            "white",
            "brown",
            "mixed",
            "biracial",
            "multiethnic",
            "global",
            "international",
            "various backgrounds",
            "different backgrounds",
        ],
    },
    "gender": {
        "max_score": 15,
        "keywords": [
            "men",
            "women",
            "man",
            "woman",
            "male",
            "female",
            "non-binary",
            "nonbinary",
            "gender",
            "genderqueer",
            "transgender",
            "trans",
            "gender-inclusive",
            "gender-diverse",
            "they",
            "them",
            "inclusive",
            "all genders",
        ],
    },
    "cultural": {
        "max_score": 20,
        "keywords": [
            "cultural",
            "culture",
            "tradition",
            "traditional",
            "heritage",
            "customs",
            "practice",
            "practices",
            "attire",
            "clothing",
            "garment",
            "festival",
            "celebration",
            "ritual",
            "ceremony",
            "religious",
            "spiritual",
            "community",
            "communities",
            "background",
            "backgrounds",
            "diverse clothing",
        ],
    },
    "ability": {
        "max_score": 10,
        "keywords": [
            "disability",
            "disabilities",
            "wheelchair",
            "accessible",
            "accessibility",
            "hearing impaired",
            "visually impaired",
            "blind",
            "deaf",
            "prosthetic",
            "mobility",
            "neurodivergent",
            "diverse abilities",
            "differently abled",
            "inclusive",
        ],
    },
    "socioeconomic": {
        "max_score": 10,
        "keywords": [
            "background",
            "backgrounds",
            "community",
            "communities",
            "various",
            "different",
            "socioeconomic",
            "working class",
            "professional",
            "trade",
            "occupation",
            "diverse occupations",
            "mixed",
            "varied",
        ],
    },
    "specificity": {
        "max_score": 10,
        "keywords": [],  # scored by word count, not keywords
    },
}

# Bias / stereotype words — each occurrence penalises the keyword score by 5 pts
_BIAS_WORDS: List[str] = [
    "housewife",
    "mankind",
    "manpower",
    "stewardess",
    "fireman",
    "policeman",
    "mailman",
    "chairman",
    "steward",
    "exotic",
    "oriental",
    "colored",
    "primitive",
    "senile",
    "crone",
    "hag",
    "crippled",
    "retarded",
    "insane",
    "crazy",
    "psycho",
]

# Maps lexicon dimension key → LLM-style JSON breakdown key
_DIM_KEY_MAP: Dict[str, str] = {
    "age": "age_diversity",
    "ethnic_racial": "ethnic_racial_diversity",
    "gender": "gender_diversity",
    "cultural": "cultural_diversity",
    "ability": "ability_inclusion",
    "socioeconomic": "socioeconomic_diversity",
    "specificity": "specificity",
}


@dataclass
class _DimScore:
    name: str
    score: int
    max_score: int
    matched_keywords: List[str]


@dataclass
class _KeywordScoreResult:
    dimensions: List[_DimScore]
    bias_words_found: List[str]
    bias_penalty: int
    raw_score: int
    final_score: int
    word_count: int


def _score_keyword_dimension(prompt_lower: str, name: str, meta: Dict) -> _DimScore:
    """Score a single diversity dimension by keyword matching."""
    if name == "specificity":
        wc = len(prompt_lower.split())
        score = min(meta["max_score"], int(wc / 20 * meta["max_score"]))
        return _DimScore(name, score, meta["max_score"], [])
    matched = [kw for kw in meta["keywords"] if kw in prompt_lower]
    unique = list(dict.fromkeys(matched))
    ratio = min(1.0, len(unique) / 3)
    score = round(meta["max_score"] * ratio)
    return _DimScore(name, score, meta["max_score"], unique[:5])


def keyword_score(prompt: str) -> _KeywordScoreResult:
    """Keyword-heuristic diversity score. Always runs without any LLM.

    Scores the prompt across 7 dimensions (age, ethnic/racial, gender,
    cultural, ability, socioeconomic, specificity) and deducts 5 points
    per detected bias/stereotype word.  Returns a ``_KeywordScoreResult``
    whose ``final_score`` is clamped to 0-100.
    """
    prompt_lower = prompt.lower()
    dimensions = [
        _score_keyword_dimension(prompt_lower, name, meta)
        for name, meta in _LEXICON.items()
    ]
    bias_found = [w for w in _BIAS_WORDS if w in prompt_lower]
    bias_penalty = len(bias_found) * 5
    raw = sum(d.score for d in dimensions)
    final = max(0, min(100, raw - bias_penalty))
    return _KeywordScoreResult(
        dimensions=dimensions,
        bias_words_found=bias_found,
        bias_penalty=bias_penalty,
        raw_score=raw,
        final_score=final,
        word_count=len(prompt.split()),
    )


def _keyword_result_to_score_dict(kw: _KeywordScoreResult) -> Dict:
    """Convert a _KeywordScoreResult to the standard score-dict shape."""
    breakdown = {_DIM_KEY_MAP.get(d.name, d.name): d.score for d in kw.dimensions}
    breakdown_detail = {
        _DIM_KEY_MAP.get(d.name, d.name): {
            "score": d.score,
            "max": d.max_score,
            "matched": d.matched_keywords,
        }
        for d in kw.dimensions
    }
    return {
        "overall_score": kw.final_score,
        "score_method": "keyword_score",
        "breakdown": breakdown,
        "keyword_breakdown_detail": breakdown_detail,
        "bias_words_found": kw.bias_words_found,
        "bias_penalty": kw.bias_penalty,
        "word_count": kw.word_count,
        "strengths": [],
        "weaknesses": [],
        "suggestions": [],
    }


def _log_score_result(result: Dict, label: str = "") -> None:
    """Log a score-dict at INFO level with a consistent format."""
    tag = f"[{label}] " if label else ""
    method = result.get("score_method", "?")
    overall = result.get("overall_score", "?")
    kw_score = result.get(
        "keyword_score",
        result.get("overall_score", "?") if method == "keyword_score" else "N/A",
    )
    logger.info(
        "%sOverall=%s/100  method=%s  keyword=%s/100  words=%s",
        tag,
        overall,
        method,
        kw_score,
        result.get("word_count", "?"),
    )
    bd = result.get("breakdown", {})
    if bd:
        logger.info(
            "%s  age=%s/15  ethnic=%s/20  gender=%s/15  cultural=%s/20  "
            "ability=%s/10  socio=%s/10  specificity=%s/10",
            tag,
            bd.get("age_diversity", "?"),
            bd.get("ethnic_racial_diversity", "?"),
            bd.get("gender_diversity", "?"),
            bd.get("cultural_diversity", "?"),
            bd.get("ability_inclusion", "?"),
            bd.get("socioeconomic_diversity", "?"),
            bd.get("specificity", "?"),
        )
    if result.get("bias_words_found"):
        logger.warning(
            "%s  Bias words detected (-%d pts): %s",
            tag,
            result.get("bias_penalty", 0),
            ", ".join(result["bias_words_found"]),
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
        """,
    )

    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        help="Image generation prompt to enhance (optional - will prompt if not provided)",
    )

    parser.add_argument(
        "--bias-top-k",
        type=int,
        default=25,
        help="Number of bias triples to retrieve (default: 25)",
    )

    parser.add_argument(
        "--cultural-top-k",
        type=int,
        default=25,
        help="Number of cultural value triples to retrieve (default: 25)",
    )

    parser.add_argument(
        "--cache-file",
        type=str,
        default="embeddings.pkl",
        help="Embedding cache file to use (default: embeddings.pkl)",
    )

    parser.add_argument(
        "--threshold",
        type=int,
        default=75,
        help="Diversity score threshold (0-100) to meet before stopping (default: 75)",
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum number of enhancement iterations (default: 3)",
    )

    parser.add_argument(
        "--llm",
        type=str,
        default="openai",
        choices=list(_LLM_CLASSES.keys()),
        help="Which LLM backend to use (default: openai)",
    )
    parser.add_argument(
        "--chat-mode",
        action="store_true",
        default=False,
        help="Use chat endpoint instead of generate",
    )

    # for cls in _LLM_CLASSES.values():
    #     cls.add_args(parser)
    args = parser.parse_known_args()
    _LLM_CLASSES[args.llm].add_args(parser)

    return parser.parse_args()


def load_bias_graph(graphrag: GraphRAG) -> bool:
    """Load the bias dataset into GraphRAG."""
    bias_file = "data/biases/ADV_GRAPH_20240119 - ADV_GRAPH_20240119.csv"
    if not Path(bias_file).exists():
        logger.error("Bias file not found: %s", bias_file)
        return False

    try:
        parser = DataParserFactory.create_parser(bias_file)
        with log_timer(logger, "Load bias graph"):
            graph = graphrag.add_graph("bias_graph", parser)
        logger.info("Loaded bias graph: %s", graph)
        return True
    except Exception as e:
        logger.exception("Error loading bias data: %s", e)
        return False


def load_cultural_graphs(graphrag: GraphRAG) -> bool:
    """Load cultural values datasets into separate graphs."""
    cultural_dir = Path("data/cultural_values")
    if not cultural_dir.exists():
        logger.error("Cultural values directory not found: %s", cultural_dir)
        return False

    txt_files = list(cultural_dir.glob("*.txt"))
    if not txt_files:
        logger.error("No text files found in cultural values directory")
        return False

    # Load all cultural datasets into a single combined graph
    logger.info("Loading cultural values data...")
    combined_triples = []

    for file_path in txt_files:
        try:
            parser = DataParserFactory.create_parser(file_path)
            triples = parser.parse()
            combined_triples.extend(triples)
            country = file_path.stem.replace(" triples", "").replace("_", " ").title()
            logger.info("  %s: %d triples", country, len(triples))
        except Exception as e:
            logger.warning("Failed to load %s: %s", file_path.name, e)

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
        logger.info("Created combined cultural graph: %s", graph)
        return True

    return False


def retrieve_relevant_triples(
    graphrag: GraphRAG, prompt: str, bias_top_k: int, cultural_top_k: int
) -> Dict[str, List[Triple]]:
    """Retrieve relevant triples from both bias and cultural graphs using semantic similarity."""
    results = {}

    # Retrieve bias triples
    logger.info("Retrieving top %d bias-related triples...", bias_top_k)
    try:
        with log_timer(logger, "Query bias graph"):
            bias_result = graphrag.query(
                graph_name="bias_graph",
                question=prompt,
                top_k=bias_top_k,
                include_answer=False,
            )
        results["bias"] = bias_result["triples"]
        logger.info("Retrieved %d bias triples", len(results["bias"]))
    except Exception as e:
        logger.exception("Failed to retrieve bias triples: %s", e)
        results["bias"] = []

    # Retrieve cultural value triples
    logger.info("Retrieving top %d cultural value triples...", cultural_top_k)
    try:
        with log_timer(logger, "Query cultural graph"):
            cultural_result = graphrag.query(
                graph_name="cultural_values",
                question=prompt,
                top_k=cultural_top_k,
                include_answer=False,
            )
        results["cultural"] = cultural_result["triples"]
        logger.info("Retrieved %d cultural value triples", len(results["cultural"]))
    except Exception as e:
        logger.exception("Failed to retrieve cultural triples: %s", e)
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
            formatted += (
                f"{i}. {triple.subject} -> {triple.predicate} -> {triple.object}\n"
            )
        else:
            # Tuple format (subject, predicate, object)
            subject, predicate, obj = triple
            formatted += f"{i}. {subject} -> {predicate} -> {obj}\n"

    return formatted


def format_triples_for_display(
    triples: List[Triple], category: str, max_display: int = 5
) -> str:
    """Format triples for console display (limited number)."""
    if not triples:
        return f"   No {category} triples retrieved."

    formatted = (
        f"   {category.title()} triples (top {min(max_display, len(triples))}):\n"
    )

    for i, triple in enumerate(triples[:max_display], 1):
        if hasattr(triple, "subject"):
            formatted += (
                f"     {i}. {triple.subject} -> {triple.predicate} -> {triple.object}\n"
            )
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
    previous_score: Optional[int] = None,
) -> str:
    """Create the prompt for LLM to enhance the original image prompt."""

    bias_info = format_triples_for_llm(bias_triples, "bias and stereotype")
    cultural_info = format_triples_for_llm(
        cultural_triples, "cultural values and diversity"
    )

    # Add context for iterative improvement
    iteration_context = ""
    if iteration > 1 and previous_prompt and previous_score is not None:
        iteration_context = f"""
PREVIOUS ITERATION CONTEXT:
This is iteration {iteration} of the enhancement process.
Previous enhanced prompt: {previous_prompt}
Previous diversity score: {previous_score}/100

The previous prompt did not meet the minimum diversity threshold.

Can you think again? Use the newly retrieved bias and cultural knowledge above to reconsider your previous reasoning and identify what was missed or underrepresented.
"""

    return f"""You are an AI assistant specialized in creating inclusive and diverse image generation prompts that avoid stereotypes and biases.

ORIGINAL IMAGE PROMPT:
{original_prompt}

{iteration_context}

BIAS INFO:
{bias_info}

CULTURAL INFO:
{cultural_info}

TASK:
Before generating the enhanced prompt, think step by step using the bias and cultural knowledge provided above:
- For each bias triple, identify whether it reveals a stereotype present in the {"original" if iteration == 1 else "previous"} prompt and how to address it.
- For each cultural triple, identify what diversity element it suggests adding to the prompt.
- Use your reasoning above to decide on the final set of changes before writing the prompt.
- Write your thinking process in <think></think> tags.

Then, enhance the {"original" if iteration == 1 else "previous"} image prompt to:

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

Please provide your response in this exact JSON format.

You MUST wrap your final JSON object inside <json> and </json> tags.
You may reason briefly before the <json> tag, but do not add any text after the </json> tag.

CRITICAL FORMATTING RULES:
1. You MUST output valid, strict JSON wrapped exactly inside <json> and </json> tags.
2. Every single key and string value MUST be enclosed in double quotes ("). 
3. Do not use single quotes ('). 
4. Do not leave keys unquoted (e.g., write "ability_inclusion": 5, NOT ability_inclusion: 5).
5. You may reason briefly before the <json> tag, but you MUST NOT add any text whatsoever after the </json> tag.

<json>
{{
  "enhanced_prompt": "<single well-crafted image generation prompt>",
  "explanation": "<brief explanation of key enhancements made>",
  "diversity_elements": ["<aspect 1>", "<aspect 2>", "<aspect 3>"]
}}
</json>
"""


def create_enhancement_prompt_chat(
    original_prompt: str,
    bias_triples: List[Triple],
    cultural_triples: List[Triple],
    iteration: int = 1,
    previous_score: int | None = None,
) -> str:

    bias_info = format_triples_for_llm(bias_triples, "bias and stereotype")
    cultural_info = format_triples_for_llm(
        cultural_triples, "cultural values and diversity"
    )

    if iteration == 1:
        return f"""
ORIGINAL IMAGE PROMPT:
{original_prompt}

BIAS INFO:
{bias_info}

CULTURAL INFO:
{cultural_info}

Before generating the enhanced prompt, think step by step using the bias and cultural knowledge provided above:
- For each bias triple, identify whether it reveals a stereotype present in the original prompt and how to address it.
- For each cultural triple, identify what diversity element it suggests adding to the prompt.
- Use your reasoning above to decide on the final set of changes before writing the prompt.
- Write your thinking process in <think></think> tags.

Then, enhance the original image prompt to:

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

Please provide your response in this exact JSON format.

You MUST wrap your final JSON object inside <json> and </json> tags.
You may reason briefly before the <json> tag, but do not add any text after the </json> tag.

CRITICAL FORMATTING RULES:
1. You MUST output valid, strict JSON wrapped exactly inside <json> and </json> tags.
2. Every single key and string value MUST be enclosed in double quotes ("). 
3. Do not use single quotes ('). 
4. Do not leave keys unquoted (e.g., write "ability_inclusion": 5, NOT ability_inclusion: 5).
5. You may reason briefly before the <json> tag, but you MUST NOT add any text whatsoever after the </json> tag.

<json>
{{
  "enhanced_prompt": "<single well-crafted image generation prompt>",
  "explanation": "<brief explanation of key enhancements made>",
  "diversity_elements": ["<aspect 1>", "<aspect 2>", "<aspect 3>"]
}}
</json>
"""

    else:
        return f"""
Can you think again? Here is newly retrieved bias and cultural knowledge — use it to reconsider your previous reasoning and identify what was missed or underrepresented.
Write your thinking process in <think></think> tags.

PREVIOUS ITERATION CONTEXT:
This is iteration {iteration} of the enhancement process.
Previous diversity score: {previous_score}/100

BIAS INFO:
{bias_info}

CULTURAL INFO:
{cultural_info}

Please provide your response in this exact JSON format.

You MUST wrap your final JSON object inside <json> and </json> tags.
You may reason briefly before the <json> tag, but do not add any text after the </json> tag.

CRITICAL FORMATTING RULES:
1. You MUST output valid, strict JSON wrapped exactly inside <json> and </json> tags.
2. Every single key and string value MUST be enclosed in double quotes ("). 
3. Do not use single quotes ('). 
4. Do not leave keys unquoted (e.g., write "ability_inclusion": 5, NOT ability_inclusion: 5).
5. You may reason briefly before the <json> tag, but you MUST NOT add any text whatsoever after the </json> tag.

<json>
{{
  "enhanced_prompt": "<single well-crafted image generation prompt>",
  "explanation": "<brief explanation of key enhancements made>",
  "diversity_elements": ["<aspect 1>", "<aspect 2>", "<aspect 3>"]
}}
</json>
"""


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

Please provide your evaluation in this exact JSON format. 

CRITICAL FORMATTING RULES:
1. You MUST output valid, strict JSON wrapped exactly inside <json> and </json> tags.
2. Every single key MUST be enclosed in double quotes (").
3. Do not use single quotes ('). 
4. Do not leave keys unquoted (e.g., write "ability_inclusion": 5, NOT ability_inclusion: 5).
5. You may reason briefly before the <json> tag, but you MUST NOT add any text whatsoever after the </json> tag.

<json>
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
}}
</json>
"""

    return scoring_prompt


def llm_score(
    prompt: str,
    llm: BaseLLM = None,
    graphrag: GraphRAG = None,  # noqa: ARG001
    out_dir: Optional[Path] = None,
    use_llm: bool = True,
) -> Dict:
    """Score the diversity of a prompt.

    **Always** runs keyword scoring first.  Then attempts the LLM judge
    (unless *use_llm* is ``False`` or the server is unreachable).  The LLM
    score is authoritative when available; keyword score is the fallback.

    Every returned dict shares the same shape:
        overall_score          – int 0-100 (LLM when available, keyword otherwise)
        score_method           – "llm_score" | "keyword_score"
        breakdown              – 7-dimension dict (LLM values when available)
        keyword_score          – int (always present)
        keyword_breakdown_detail – per-dimension detail (always present)
        bias_words_found       – list[str]
        bias_penalty           – int
        word_count             – int
        strengths / weaknesses / suggestions – list[str]
    """
    # ── 1. Keyword scoring — always ───────────────────────────────────────────
    kw = keyword_score(prompt)
    kw_dict = _keyword_result_to_score_dict(kw)
    logger.info(
        "Keyword score: %d/100  (word_count=%d  bias_penalty=%d)",
        kw.final_score,
        kw.word_count,
        kw.bias_penalty,
    )
    if kw.bias_words_found:
        logger.warning(
            "Bias words detected in prompt (-%d pts): %s",
            kw.bias_penalty,
            ", ".join(kw.bias_words_found),
        )

    if not use_llm:
        logger.info("LLM judge skipped — keyword score is final")
        if llm is None or not llm.is_available():
            logger.warning(
                "LLM judge server not reachable — using keyword score as final score"
            )
        return kw_dict

    # ── 3. LLM judge ─────────────────────────────────────────────────────────
    score_text = "<not extracted yet>"
    try:
        scoring_prompt = create_diversity_scoring_prompt(prompt)
        with log_timer(logger, "LLM diversity judge (independent)"):
            raw_text = llm.generate(scoring_prompt)

        logger.debug("[llm_score] raw output (first 600 chars): %r", raw_text[:600])

        if out_dir is not None:
            raw_dir = out_dir / "raw_llm_outputs"
            raw_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%H%M%S_%f")
            raw_path = raw_dir / f"score_diversity_{ts}.txt"
            raw_path.write_text(raw_text, encoding="utf-8")
            logger.debug("[llm_score] raw output saved → %s", raw_path)

        score_text = extract_json_from_response(raw_text)
        logger.debug(
            "[llm_score] extracted JSON (first 400 chars): %r", score_text[:400]
        )

        score_data = loads_json_with_repair(score_text)
        if "overall_score" not in score_data:
            raise ValueError("LLM score JSON missing 'overall_score'")
        if "breakdown" in score_data and not isinstance(
            score_data.get("breakdown"), dict
        ):
            raise ValueError("LLM score JSON field 'breakdown' is not an object")

        llm_overall = int(score_data["overall_score"])
        logger.info(
            "LLM judge score: %d/100  (keyword was %d/100)",
            llm_overall,
            kw.final_score,
        )
        return {
            "score_method": "llm_score",
            "overall_score": llm_overall,
            "breakdown": score_data.get("breakdown", {}),
            "keyword_score": kw.final_score,
            "keyword_breakdown_detail": kw_dict["keyword_breakdown_detail"],
            "bias_words_found": kw.bias_words_found,
            "bias_penalty": kw.bias_penalty,
            "word_count": kw.word_count,
            "strengths": score_data.get("strengths", []),
            "weaknesses": score_data.get("weaknesses", []),
            "suggestions": score_data.get("suggestions", []),
        }

    except json.JSONDecodeError as e:
        logger.error("LLM judge returned invalid JSON: %s", e)
        logger.error("  Extracted text: %r", score_text[:300])
        logger.warning("Falling back to keyword score")
        return kw_dict
    except Exception as e:
        logger.exception("LLM judge error: %s", e)
        logger.warning("Falling back to keyword score")
        return kw_dict


def get_llm_enhancement(
    enhancement_prompt: str, llm: BaseLLM, graphrag: GraphRAG
) -> str:  # noqa: ARG001
    """Get enhancement from the LLM."""
    logger.info("Generating enhanced prompt with LLM...")

    if not llm.is_available():
        logger.warning("LLM not reachable, using mock response")
        return get_mock_enhancement(enhancement_prompt)

    try:
        with log_timer(logger, "Prompt enhancement via LLM"):
            enhanced_result = llm.generate(enhancement_prompt)
        logger.info("Enhanced prompt generated successfully")
        return enhanced_result
    except Exception as e:
        logger.exception("Error generating enhancement: %s", e)
        logger.warning("Falling back to mock response")
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

    import json as _json

    mock_data = {
        "enhanced_prompt": (
            f"{original_prompt}, featuring people of diverse ages including young adults, "
            "middle-aged individuals, and seniors, representing various ethnicities including "
            "Asian, African, Latino, Middle Eastern, and European backgrounds, with different "
            "skin tones and physical appearances, wearing culturally diverse clothing and "
            "accessories, in an inclusive environment that celebrates global diversity, with "
            "both men and women and non-binary individuals, including people with visible and "
            "invisible disabilities, showcasing different socioeconomic backgrounds through "
            "varied but respectful styling, with authentic cultural elements like traditional "
            "patterns, diverse architectural styles, and inclusive symbols that promote unity "
            "and respect across all communities"
        ),
        "explanation": (
            "Enhanced the original prompt to explicitly include age diversity (young to senior), "
            "ethnic and racial diversity (multiple specific backgrounds), gender inclusivity "
            "(men, women, non-binary), disability representation, and cultural authenticity "
            "through clothing, architecture, and symbols."
        ),
        "diversity_elements": [
            "Age: Multiple generations represented",
            "Ethnicity: Asian, African, Latino, Middle Eastern, European",
            "Gender: Men, women, non-binary individuals",
            "Abilities: People with visible and invisible disabilities",
            "Culture: Traditional clothing, patterns, architectural diversity",
            "Socioeconomic: Varied but respectful representation",
            "Setting: Inclusive, globally-inspired environment",
        ],
    }
    return f"<json>\n{_json.dumps(mock_data, indent=2)}\n</json>"


def extract_enhanced_prompt(llm_response: str) -> str:
    """Extract just the enhanced prompt from LLM response (JSON format)."""
    try:
        json_text = extract_json_from_response(llm_response)
        data = loads_json_with_repair(json_text)
        enhanced_prompt = data.get("enhanced_prompt", "").strip()
        if not enhanced_prompt:
            raise ValueError("'enhanced_prompt' missing or empty")
        return enhanced_prompt
    except (json.JSONDecodeError, KeyError, ValueError):
        # Fallback: legacy text parsing for non-JSON responses
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
    llm: BaseLLM,
    original_prompt: str,
    bias_triples: List[Triple],
    cultural_triples: List[Triple],
    threshold: int,
    max_iterations: int,
    out_dir: Optional[Path] = None,
) -> Tuple[str, List[Dict]]:
    """Sequentially enhance prompt until diversity threshold is met."""

    logger.info(
        "Starting sequential enhancement | threshold=%d | max_iterations=%d",
        threshold,
        max_iterations,
    )

    current_prompt = original_prompt
    iteration_history = []
    system_prompt = "You are an AI assistant specialized in creating inclusive and diverse image generation prompts that avoid stereotypes and biases."
    messages = [{"role": "system", "content": system_prompt}]

    for iteration in range(1, max_iterations + 1):
        logger.info("── ITERATION %d / %d ──", iteration, max_iterations)

        # Get previous context for iterations beyond the first
        previous_prompt = None
        previous_score = None

        if iteration > 1:
            previous_prompt = iteration_history[-1]["enhanced_prompt"]
            previous_score = iteration_history[-1]["diversity_score"]["overall_score"]

        # Create enhancement prompt
        enhancement_prompt = create_enhancement_prompt(
            original_prompt,
            bias_triples,
            cultural_triples,
            iteration,
            previous_prompt,
            previous_score,
        )

        enhancement_prompt_chat = create_enhancement_prompt_chat(
            original_prompt,
            bias_triples,
            cultural_triples,
            iteration,
            previous_score,
        )
        messages.append({"role": "user", "content": enhancement_prompt_chat})

        # Get LLM enhancement
        enhanced_result = (
            get_llm_enhancement(enhancement_prompt, llm, graphrag)
            if not llm.args.chat_mode
            else llm.is_available() and llm.chat(messages)
        )
        messages.append({"role": "assistant", "content": enhanced_result})

        # ── Save raw enhancement output to disk ──────────────────────────────────
        logger.debug(
            "[enhancement iter %d] raw LLM output (first 600 chars): %r",
            iteration,
            enhanced_result[:600],
        )
        if out_dir is not None:
            raw_dir = out_dir / "raw_llm_outputs"
            raw_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%H%M%S_%f")
            raw_path = raw_dir / f"enhancement_iter{iteration}_{ts}.txt"
            raw_path.write_text(enhanced_result, encoding="utf-8")
            logger.debug(
                "[enhancement iter %d] raw output saved to %s", iteration, raw_path
            )
            messages_dir = out_dir / "messages_llm_outputs"
            messages_dir.mkdir(parents=True, exist_ok=True)
            try:
                filename = (
                    f"{messages_dir}/{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                )

                with open(filename, "w") as f:
                    json.dump(
                        {"original_prompt": original_prompt, "messages": messages},
                        f,
                        indent=2,
                    )
            except:
                pass

        enhanced_prompt = extract_enhanced_prompt(enhanced_result)
        if not enhanced_prompt:
            fallback_prompt = current_prompt if current_prompt else original_prompt
            logger.warning(
                "Iteration %d produced empty enhanced_prompt; using fallback prompt for scoring",
                iteration,
            )
            enhanced_prompt = fallback_prompt

        # Score diversity — keyword always + LLM judge when available
        diversity_score = llm_score(enhanced_prompt, out_dir=out_dir)

        # Store iteration data
        iteration_data = {
            "iteration": iteration,
            "enhanced_prompt": enhanced_prompt,
            "diversity_score": diversity_score,
            "full_response": enhanced_result,
            "threshold_met": diversity_score["overall_score"] >= threshold,
        }
        iteration_history.append(iteration_data)

        # Log iteration results with both scores
        logger.info(
            "Iteration %d/%d | overall=%d/100 (%s) | keyword=%s/100 | threshold_met=%s",
            iteration,
            max_iterations,
            diversity_score["overall_score"],
            diversity_score.get("score_method", "?"),
            diversity_score.get("keyword_score", diversity_score["overall_score"]),
            iteration_data["threshold_met"],
        )
        _log_score_result(diversity_score, label=f"iter{iteration}")

        if iteration_data["threshold_met"]:
            logger.info(
                "Diversity threshold %d reached after %d iteration(s)",
                threshold,
                iteration,
            )
            break

        logger.info("Threshold not yet met — continuing to iteration %d", iteration + 1)

        # Update current prompt for next iteration
        current_prompt = enhanced_prompt

    final_prompt = iteration_history[-1]["enhanced_prompt"]
    final_ds = iteration_history[-1]["diversity_score"]
    final_score = final_ds["overall_score"]
    final_method = final_ds.get("score_method", "?")

    if final_score >= threshold:
        logger.info(
            "Enhancement complete — score %d/100 (%s) meets threshold %d",
            final_score,
            final_method,
            threshold,
        )
    else:
        logger.warning(
            "Max iterations reached — best score %d/100 (%s), threshold was %d",
            final_score,
            final_method,
            threshold,
        )

    return final_prompt, iteration_history


def display_final_results(
    original_prompt: str,
    final_prompt: str,
    iteration_history: List[Dict],
    relevant_triples: Dict[str, List[Triple]],
) -> None:
    """Display comprehensive final results."""

    logger.info("═" * 60)
    logger.info("SEQUENTIAL PROMPT ENHANCEMENT RESULTS")
    logger.info("═" * 60)
    logger.info("ORIGINAL PROMPT: %s", original_prompt)

    if relevant_triples.get("bias"):
        logger.info("Bias triples retrieved: %d", len(relevant_triples["bias"]))
    if relevant_triples.get("cultural"):
        logger.info("Cultural triples retrieved: %d", len(relevant_triples["cultural"]))

    for i, iteration in enumerate(iteration_history):
        ds = iteration["diversity_score"]
        logger.info(
            "Iteration %d | overall=%d/100 (%s) | keyword=%s/100 | threshold_met=%s",
            iteration["iteration"],
            ds["overall_score"],
            ds.get("score_method", "?"),
            ds.get("keyword_score", ds["overall_score"]),
            iteration["threshold_met"],
        )
        if i == len(iteration_history) - 1:
            logger.info("Final score breakdown:")
            _log_score_result(ds, label="final")

    logger.info("FINAL ENHANCED PROMPT: %s", final_prompt)

    final_diversity = iteration_history[-1]["diversity_score"]
    for strength in final_diversity.get("strengths", []):
        logger.info("  + strength: %s", strength)
    for weakness in final_diversity.get("weaknesses", []):
        logger.warning("  - weakness: %s", weakness)
    for suggestion in final_diversity.get("suggestions", []):
        logger.info("  ? suggestion: %s", suggestion)

    logger.info("Sequential enhancement completed")

    # ── console summary ───────────────────────────────────────────────────────
    final_ds = iteration_history[-1]["diversity_score"]
    print("\n" + "=" * 80)
    print("FINAL ENHANCED PROMPT")
    print("=" * 80)
    print(f"Original : {original_prompt}")
    print(f"Enhanced : {final_prompt}")
    print(
        f"Score    : {final_ds['overall_score']}/100  (method: {final_ds.get('score_method', '?')})"
    )
    if final_ds.get("score_method") == "llm_score":
        print(f"Keyword  : {final_ds.get('keyword_score', '?')}/100")
    print("=" * 80)


# ─────────────────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────────────────


def make_output_dir() -> Path:
    """Create and return a timestamped run directory under outputs/qwen32b-<datetime>."""
    run_dir = Path("outputs") / f"qwen32b-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", run_dir)
    return run_dir


def _best_iteration(iteration_history: List[Dict]) -> Dict:
    """Return the iteration dict with the highest overall diversity score."""
    return max(iteration_history, key=lambda it: it["diversity_score"]["overall_score"])


def write_output_csvs(
    out_dir: Path,
    rows_data: List[Dict],
    max_iters: int,
) -> None:
    """Write both output CSVs into *out_dir*.

    Parameters
    ----------
    out_dir:
        Destination folder (created if absent).
    rows_data:
        List of dicts, one per processed prompt:
        {
            "row_id":           int,
            "original_prompt":  str,
            "original_score":   int | None,   # diversity score of generic prompt
            "iteration_history": List[Dict],  # from sequential_enhance_prompt()
        }
    max_iters:
        Maximum number of enhancement iterations (used to build fixed-width columns).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. prompts_output.csv  (mirrors inputs/prompts.csv + finalized_prompt) ─
    prompts_csv = out_dir / "prompts_output.csv"
    prompts_fields = [
        "prompt",
        "__row_id__",
        "generic_prompt",
        "enhanced_prompts",
        "finalized_prompt",
    ]
    with prompts_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=prompts_fields)
        writer.writeheader()
        for row in rows_data:
            history = row["iteration_history"]
            all_enhanced = [it["enhanced_prompt"] for it in history]
            best = (
                _best_iteration(history)["enhanced_prompt"]
                if history
                else row["original_prompt"]
            )
            writer.writerow(
                {
                    "prompt": row["original_prompt"],
                    "__row_id__": row["row_id"],
                    "generic_prompt": row["original_prompt"],
                    "enhanced_prompts": json.dumps(all_enhanced, ensure_ascii=False),
                    "finalized_prompt": best,
                }
            )
    logger.info("Saved: %s", prompts_csv)

    # ── 2. detailed_scores.csv  (wide format: prompt+score per column) ─────────
    # Build column list dynamically up to max_iters.
    # Each score column is followed by a *_method column that records whether
    # the score came from the LLM judge ("llm_score") or keyword fallback
    # ("keyword_score"), so the CSV always reflects the true source.
    detail_fields = [
        "row_id",
        "generic_prompt",
        "generic_score",
        "generic_score_method",
    ]
    for i in range(1, max_iters + 1):
        detail_fields += [f"enhanced_prompt_{i}", f"score_{i}", f"score_{i}_method"]
    detail_fields += ["final_prompt", "final_score", "final_score_method"]

    detail_csv = out_dir / "detailed_scores.csv"
    with detail_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=detail_fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows_data:
            history = row["iteration_history"]
            record: Dict = {
                "row_id": row["row_id"],
                "generic_prompt": row["original_prompt"],
                "generic_score": row.get("original_score", ""),
                "generic_score_method": row.get("original_score_method", ""),
            }
            for it in history:
                n = it["iteration"]
                if n <= max_iters:
                    record[f"enhanced_prompt_{n}"] = it["enhanced_prompt"]
                    record[f"score_{n}"] = it["diversity_score"]["overall_score"]
                    record[f"score_{n}_method"] = it["diversity_score"].get(
                        "score_method", ""
                    )

            if history:
                best = _best_iteration(history)
                record["final_prompt"] = best["enhanced_prompt"]
                # Prefer the evaluation-judge score when run_batch.py provides it
                # (eval_final_score is the same number as llm_enhanced in summary.csv)
                if row.get("eval_final_score") is not None:
                    record["final_score"] = row["eval_final_score"]
                    record["final_score_method"] = row.get(
                        "eval_final_score_method", "llm_score"
                    )
                else:
                    record["final_score"] = best["diversity_score"]["overall_score"]
                    record["final_score_method"] = best["diversity_score"].get(
                        "score_method", ""
                    )
            else:
                record["final_prompt"] = row["original_prompt"]
                record["final_score"] = row.get(
                    "eval_final_score", row.get("original_score", "")
                )
                record["final_score_method"] = row.get(
                    "eval_final_score_method", row.get("original_score_method", "")
                )

            writer.writerow(record)
    logger.info("Saved: %s", detail_csv)


def save_run_json(out_dir: Path, data: Dict) -> None:
    """Save a full-detail JSON of the run to *out_dir/run_details.json*."""
    out_path = out_dir / "run_details.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info("Saved: %s", out_path)


def main() -> None:
    """Run the sequential image prompt enhancement script."""
    args = parse_arguments()

    llm = _LLM_CLASSES[args.llm](args)
    logger.info("Using LLM backend: %s", args.llm)

    logger.info("Sequential Image Prompt Enhancement System starting")

    # Get prompt from args or user input
    if args.prompt:
        original_prompt = args.prompt
        logger.info("Original prompt: %s", original_prompt)
    else:
        original_prompt = get_prompt_input()
        logger.info("Original prompt (from input): %s", original_prompt)

    # ── create timestamped output directory ───────────────────────────────────
    out_dir = make_output_dir()

    # Initialize GraphRAG system
    logger.info("Initializing GraphRAG system...")
    graphrag = GraphRAG(cache_file=args.cache_file)

    # Load datasets
    logger.info("Loading knowledge graphs...")

    bias_loaded = load_bias_graph(graphrag)
    cultural_loaded = load_cultural_graphs(graphrag)

    if not bias_loaded and not cultural_loaded:
        logger.error("Failed to load any datasets. Exiting.")
        sys.exit(1)

    # ── score the original prompt (keyword always + LLM judge when available) ────
    logger.info("Scoring original prompt diversity (keyword + LLM judge)...")
    original_score_data = llm_score(original_prompt, llm=llm, out_dir=out_dir)
    original_score = original_score_data["overall_score"]
    original_score_method = original_score_data["score_method"]
    logger.info(
        "Original prompt score: %d/100 (method: %s)",
        original_score,
        original_score_method,
    )
    _log_score_result(original_score_data, label="original")

    if original_score >= args.threshold:
        logger.info(
            "Original score %d/100 (%s) already meets threshold %d — no enhancement needed",
            original_score,
            original_score_method,
            args.threshold,
        )
        relevant_triples: Dict[str, List[Triple]] = {"bias": [], "cultural": []}
        final_prompt = original_prompt
        final_score_data = original_score_data
        iteration_history: List[Dict] = []
    else:
        # Retrieve relevant triples (once, used for all iterations)
        relevant_triples = retrieve_relevant_triples(
            graphrag, original_prompt, args.bias_top_k, args.cultural_top_k
        )

        # Sequential enhancement process (each iteration is scored in-loop)
        final_prompt, iteration_history = sequential_enhance_prompt(
            graphrag,
            llm,
            original_prompt,
            relevant_triples.get("bias", []),
            relevant_triples.get("cultural", []),
            args.threshold,
            args.max_iterations,
            out_dir=out_dir,
        )
        final_score_data = iteration_history[-1]["diversity_score"]

        # Display comprehensive results
        display_final_results(
            original_prompt, final_prompt, iteration_history, relevant_triples
        )

    # ── persist outputs ───────────────────────────────────────────────────────
    rows_data = [
        {
            "row_id": 0,
            "original_prompt": original_prompt,
            "original_score": original_score,
            "original_score_method": original_score_method,
            "iteration_history": iteration_history,
        }
    ]
    write_output_csvs(out_dir, rows_data, max_iters=args.max_iterations)
    save_run_json(
        out_dir,
        {
            "timestamp": datetime.now().isoformat(),
            "original_prompt": original_prompt,
            "original_score": original_score,
            "original_score_method": original_score_method,
            "threshold": args.threshold,
            "max_iterations": args.max_iterations,
            "final_prompt": final_prompt,
            "final_score": final_score_data.get("overall_score"),
            "final_score_method": final_score_data.get("score_method", ""),
            "iterations": [
                {
                    "iteration": it["iteration"],
                    "enhanced_prompt": it["enhanced_prompt"],
                    "overall_score": it["diversity_score"]["overall_score"],
                    "score_method": it["diversity_score"].get("score_method", ""),
                    "breakdown": it["diversity_score"].get("breakdown"),
                    "strengths": it["diversity_score"].get("strengths", []),
                    "weaknesses": it["diversity_score"].get("weaknesses", []),
                    "threshold_met": it["threshold_met"],
                }
                for it in iteration_history
            ],
        },
    )

    print(f"\nOutputs saved to: {out_dir}/")
    print("  prompts_output.csv   — enhanced prompts + finalized_prompt column")
    print("  detailed_scores.csv  — per-iteration scores in wide format")
    print("  run_details.json     — full structured run data")


if __name__ == "__main__":
    main()
