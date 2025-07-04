# Sequential Image Prompt Enhancement with Diversity Scoring

## Overview

The sequential enhancement system introduces an iterative approach to improving image generation prompts for diversity and bias mitigation. This system uses a diversity scoring agent that evaluates enhanced prompts and continues improving them until a configurable threshold is met.

## Features

### 🔄 Sequential Improvement Process
- **Iterative Enhancement**: Prompts are enhanced multiple times until diversity goals are met
- **Diversity Scoring**: Each iteration is scored on a 0-100 scale across multiple diversity dimensions
- **Threshold-Based Stopping**: Process stops when diversity score meets or exceeds the configured threshold
- **Maximum Iterations**: Configurable limit to prevent infinite loops

### 📊 Diversity Scoring Dimensions

The diversity scoring agent evaluates prompts across seven key dimensions:

1. **Age Diversity (15 points max)**: Multiple age groups represented
2. **Ethnic/Racial Diversity (20 points max)**: Various ethnic and racial backgrounds
3. **Gender Diversity (15 points max)**: Inclusive gender representation
4. **Cultural Diversity (20 points max)**: Different cultural elements and practices
5. **Ability/Disability Inclusion (10 points max)**: Accessibility considerations
6. **Socioeconomic Diversity (10 points max)**: Different socioeconomic backgrounds
7. **Specificity and Actionability (10 points max)**: Concrete, implementable diversity elements

### 🎯 Intelligent Enhancement Context

Each iteration after the first includes:
- Previous enhanced prompt
- Previous diversity score
- Specific areas that need improvement
- Focus on addressing weaknesses from prior iterations

## Usage

### Command Line Interface

```bash
# Basic usage
python enhance_prompt_sequential.py --prompt "a doctor in a hospital"

# With custom threshold and iterations
python enhance_prompt_sequential.py \
    --prompt "students in classroom" \
    --threshold 80 \
    --max-iterations 5

# Interactive mode
python enhance_prompt_sequential.py
```

### Parameters

- `--prompt, -p`: Image generation prompt to enhance
- `--threshold`: Diversity score threshold (0-100, default: 75)
- `--max-iterations`: Maximum enhancement iterations (default: 3)
- `--bias-top-k`: Number of bias triples to retrieve (default: 25)
- `--cultural-top-k`: Number of cultural triples to retrieve (default: 25)
- `--cache-file`: Embedding cache file (default: embeddings.pkl)

## Example Output

```
🎨 Sequential Image Prompt Enhancement System
Enhancing prompts for diversity with iterative improvement and scoring

📝 Original prompt: a doctor in a hospital

⚙️  Initializing GraphRAG system...
✅ Loaded bias graph: bias_graph: 13705 nodes, 51301 triples
✅ Created combined cultural graph: cultural_values: 5693 nodes, 6469 triples

🔄 Starting sequential enhancement process...
   Target diversity threshold: 75/100
   Maximum iterations: 3

==================== ITERATION 1 ====================
🤖 Generating enhanced prompt with LLM...
✅ Enhanced prompt generated successfully!
📊 Scoring diversity of prompt...
✅ Diversity score: 98/100

📊 ITERATION 1 RESULTS:
   Diversity Score: 98/100
   Threshold Met: ✅ Yes

🎉 Diversity threshold reached in 1 iteration(s)!

✅ FINAL SUCCESS: Achieved diversity score of 98/100 (>= 75)

================================================================================
SEQUENTIAL PROMPT ENHANCEMENT RESULTS
================================================================================

🎨 ORIGINAL PROMPT:
   a doctor in a hospital

🔄 ENHANCEMENT ITERATIONS:
   Iteration 1:
     Score: 98/100
     Threshold Met: ✅
     Breakdown:
       • Age Diversity: 15/15
       • Ethnic/Racial: 20/20
       • Gender: 15/15
       • Cultural: 20/20
       • Ability Inclusion: 10/10
       • Socioeconomic: 10/10
       • Specificity: 8/10

✨ FINAL ENHANCED PROMPT:
   a doctor in a hospital, featuring people of diverse ages including young adults, 
   middle-aged individuals, and seniors, representing various ethnicities including 
   Asian, African, Latino, Middle Eastern, and European backgrounds, with different 
   skin tones and physical appearances, wearing culturally diverse clothing and 
   accessories, in an inclusive environment that celebrates global diversity...

💪 STRENGTHS:
   • Includes basic diversity elements
   • Comprehensive age representation
   • Multi-ethnic and cultural inclusion

🔧 AREAS FOR IMPROVEMENT:
   • Could be more specific about diversity aspects
```

## Technical Architecture

### Sequential Enhancement Flow

1. **Initialization**: Load GraphRAG system with bias and cultural data
2. **Triple Retrieval**: Get relevant knowledge for the original prompt
3. **Iterative Enhancement Loop**:
   - Create context-aware enhancement prompt
   - Generate enhanced prompt using LLM
   - Score diversity of enhanced prompt
   - Check if threshold is met
   - If not met and iterations remain, continue with improvements
4. **Results Display**: Show comprehensive enhancement history and final results

### Diversity Scoring Agent

The scoring agent uses structured prompts to evaluate diversity across multiple dimensions:

```python
def score_diversity(prompt: str, graphrag: GraphRAG) -> Dict:
    """Score the diversity of a prompt using LLM."""
    scoring_prompt = create_diversity_scoring_prompt(prompt)
    
    response = graphrag.embedder.client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert in diversity assessment..."},
            {"role": "user", "content": scoring_prompt}
        ],
        temperature=0.1  # Low temperature for consistent scoring
    )
    
    return parse_json_response(response)
```

### Iterative Context Building

Each iteration builds upon previous results:

```python
def create_enhancement_prompt(
    original_prompt: str, 
    bias_triples: List[Triple], 
    cultural_triples: List[Triple], 
    iteration: int = 1, 
    previous_prompt: Optional[str] = None, 
    previous_score: Optional[int] = None
) -> str:
    # Include previous iteration context for iterations > 1
    if iteration > 1 and previous_prompt and previous_score is not None:
        iteration_context = f"""
        PREVIOUS ITERATION CONTEXT:
        This is iteration {iteration} of the enhancement process.
        Previous enhanced prompt: {previous_prompt}
        Previous diversity score: {previous_score}/100
        
        The previous prompt did not meet the minimum diversity threshold. 
        Please focus on improving the areas that scored poorly.
        """
```

## Configuration Options

### Threshold Settings
- **Conservative (60-70)**: Basic diversity requirements
- **Moderate (75-85)**: Good diversity coverage
- **Strict (90-100)**: Comprehensive diversity requirements

### Iteration Limits
- **1-2 iterations**: Quick enhancement
- **3-4 iterations**: Balanced improvement
- **5+ iterations**: Thorough optimization

## Benefits

1. **Quality Assurance**: Systematic evaluation ensures diversity goals are met
2. **Iterative Improvement**: Each iteration builds upon previous strengths and addresses weaknesses
3. **Configurable Standards**: Adjustable thresholds for different use cases
4. **Comprehensive Coverage**: Multi-dimensional diversity assessment
5. **Transparent Process**: Clear visibility into enhancement decisions and scoring

## Integration with Original System

The sequential enhancement system extends the original `enhance_prompt.py` functionality:

- **Original**: Single-pass enhancement with basic output
- **Sequential**: Multi-iteration enhancement with scoring and threshold-based optimization
- **Compatibility**: Both systems use the same GraphRAG backend and data sources
- **Flexibility**: Users can choose the appropriate system based on their needs

## Future Enhancements

1. **Custom Scoring Weights**: Allow users to weight different diversity dimensions
2. **Domain-Specific Scoring**: Specialized scoring for different image categories
3. **Learning from History**: Use previous enhancement patterns to improve future iterations
4. **Real-time Feedback**: Interactive adjustment of thresholds during enhancement
5. **Batch Processing**: Enhancement of multiple prompts with consistent standards
