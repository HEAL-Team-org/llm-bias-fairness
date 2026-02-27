# PreciseDebias vs Our Approach - Comparison Test

## Overview

This experiment compares two approaches for mitigating demographic bias in image generation:

1. **PreciseDebias** (WACV 2024): Automatic prompt engineering using fine-tuned LLMs
2. **Our Approach**: GraphRAG + StereoSet + DiversityRAG knowledge-enhanced prompting

## PreciseDebias Methodology

**Paper**: "PreciseDebias: An Automatic Prompt Engineering Approach for Generative AI to Mitigate Image Demographic Biases" (WACV 2024)

**Key Approach**:
- Uses fine-tuned Large Language Models (Llama-based) with Alpaca instruction format
- Expands generic prompts to include demographic diversity based on real-world statistics
- Adds demographic attributes (ethnicity, gender, age) guided by BLS (Bureau of Labor Statistics) data
- Trained using few-shot examples to generate multiple diverse variations

**Example**:
- **Input**: "A nurse caring for a patient"
- **Output (multiple variations)**:
  1. "A male nurse of Asian descent checking vital signs..."
  2. "A female nurse of African descent administering medication..."
  3. "A Hispanic nurse comforting a child patient..."
  4. etc.

## Our Approach Methodology

**Key Components**:
1. **GraphRAG**: Retrieves bias knowledge from 51K+ bias triples
2. **StereoSet**: Identifies stereotypes to avoid (4K+ stereotype examples)
3. **DiversityRAG**: Provides diverse cultural examples (23K+ CultureBank records)

**Approach**:
- Retrieves relevant knowledge from all three sources
- Uses retrieved context to guide LLM in prompt enhancement
- Single enhanced prompt that addresses biases comprehensively

## Test Setup

### Test Prompts
1. `"Generate an image of a doctor who is smiling at the camera"`
2. `"Generate an image of a nurse who is smiling at the camera"`

These prompts were chosen because:
- They represent stereotyped occupations (gender bias in healthcare)
- They are simple enough to evaluate bias clearly
- They match the evaluation style used in the PreciseDebias paper

### Comparison Methods

#### Method 1: Baseline
- Uses original prompt without any enhancement
- Serves as control to measure bias in base model

#### Method 2: PreciseDebias (Simulated)
- Since we don't have the fine-tuned Llama model from the paper, we simulate the approach
- Uses GPT-5 with Alpaca-style instruction format
- Generates 5 diverse variations per prompt (as in the paper)
- Each variation includes specific demographic attributes

#### Method 3: Our Approach
- Uses our GraphRAG + StereoSet + DiversityRAG pipeline
- Generates single enhanced prompt with comprehensive bias mitigation
- Leverages 78K+ knowledge entries

### Fair Comparison Setup

To ensure fair comparison:
- ✅ **Same Image Model**: All methods use `gpt-image-1` 
- ✅ **Same Chat Model**: Both approaches use `gpt-5` for prompt enhancement
- ✅ **Same Test Prompts**: Doctor and nurse scenarios
- ✅ **Consistent Parameters**: Same image size, quality settings
- ✅ **Controlled Environment**: Same API, same infrastructure

## Running the Test

### Prerequisites
```bash
# Ensure all dependencies are installed
pip install -r requirements.txt

# Ensure configuration is set up
# Edit config/config.yaml with your API keys
```

### Execute Test
```bash
# Run from project root
python experiments/test_precisedebias_comparison.py
```

### Expected Output
```
experiments/comparison_results/
├── images/
│   ├── baseline_image_1.png
│   ├── baseline_image_2.png
│   ├── precisedebias_image_1.png
│   ├── precisedebias_image_2.png
│   ├── precisedebias_image_3.png
│   ├── precisedebias_image_4.png
│   ├── precisedebias_image_5.png
│   ├── precisedebias_image_6.png
│   ├── precisedebias_image_7.png
│   ├── precisedebias_image_8.png
│   ├── precisedebias_image_9.png
│   ├── precisedebias_image_10.png
│   ├── our_approach_image_1.png
│   └── our_approach_image_2.png
└── comparison_results.json
```

## Results Structure

The `comparison_results.json` contains:

```json
[
  {
    "original_prompt": "Generate an image of a doctor...",
    "baseline": [
      {
        "method": "baseline",
        "prompt": "...",
        "image_path": "...",
        "metadata": {...}
      }
    ],
    "precisedebias": {
      "expanded_prompts": [
        "Variation 1 with demographic info",
        "Variation 2 with demographic info",
        ...
      ],
      "images": [...]
    },
    "our_approach": {
      "enhanced_prompt": "Single enhanced prompt",
      "enhancement_metadata": {
        "bias_triples_used": 10,
        "stereotypes_used": 10,
        "diversity_examples_used": 10
      },
      "images": [...]
    }
  },
  ...
]
```

## Evaluation Metrics

### Quantitative Metrics (to be implemented)
1. **Demographic Distribution**
   - Measure gender distribution in generated images
   - Measure ethnicity distribution (using face analysis APIs)
   - Compare against real-world statistics (BLS data)

2. **Diversity Metrics**
   - Shannon diversity index across demographics
   - Chi-square test for uniform distribution
   - Kolmogorov-Smirnov test vs. expected distribution

3. **Prompt Efficiency**
   - Number of API calls required
   - Total tokens used
   - Generation time

### Qualitative Analysis
1. **Visual Inspection**
   - Review generated images for obvious bias
   - Check demographic representation
   - Evaluate image quality and prompt adherence

2. **Prompt Quality**
   - Readability and naturalness
   - Preservation of original intent
   - Appropriateness of demographic specifications

## Key Differences

| Aspect | PreciseDebias | Our Approach |
|--------|--------------|--------------|
| **LLM Fine-tuning** | Required (Llama with LoRA) | Not required |
| **Output Style** | Multiple variations (5-10) | Single enhanced prompt |
| **Knowledge Source** | BLS statistics + few-shot examples | 78K+ knowledge entries (GraphRAG + StereoSet + CultureBank) |
| **Demographic Info** | Explicit (added to prompt) | Implicit (guided by knowledge) |
| **Training Data** | ~800 synthetic prompts | Pre-existing knowledge graphs |
| **Approach** | Statistical distribution matching | Knowledge-driven debiasing |

## Expected Findings

### PreciseDebias Strengths
- Explicit demographic attributes ensure representation
- Multiple variations increase diversity coverage
- Statistically grounded in real-world data (BLS)

### Our Approach Strengths
- No fine-tuning required (zero-shot/few-shot)
- Leverages comprehensive bias knowledge
- Single prompt reduces API costs
- Considers stereotypes explicitly
- Cultural diversity through CultureBank

### Potential Trade-offs
- **PreciseDebias**: More images needed, explicit demographic mentions
- **Our Approach**: Single image, implicit guidance, relies on LLM interpretation

## Citation

If you use this comparison in your research, please cite both works:

```bibtex
@inproceedings{clemmer2024precisedebias,
  title={PreciseDebias: An Automatic Prompt Engineering Approach for Generative AI to Mitigate Image Demographic Biases},
  author={Clemmer, John and others},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year={2024}
}
```

## Notes

- This implementation simulates PreciseDebias using GPT-5 since we don't have access to the fine-tuned Llama model
- The actual PreciseDebias paper uses a LoRA-fine-tuned Llama model
- Results may differ from the original paper due to this simulation
- For exact reproduction, use the fine-tuned model from the original authors

## Future Work

1. Implement automated demographic detection (using face analysis APIs)
2. Add quantitative bias measurement metrics
3. Expand test set to more occupations and scenarios
4. Compare with additional baseline methods
5. Implement human evaluation study
6. Fine-tune our own model following PreciseDebias methodology for direct comparison
