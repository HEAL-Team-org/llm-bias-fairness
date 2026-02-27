# PreciseDebias vs Our GraphRAG Approach - Preliminary Findings

## Executive Summary

We implemented a head-to-head comparison between the PreciseDebias paper methodology (WACV 2024) and our GraphRAG-based debiasing approach using the same models (GPT-5 for chat, DALL-E 3 for image generation) to ensure fair comparison.

## Test Design

### Test Prompts
1. **Doctor**: "Generate an image of a doctor who is smiling at the camera"
2. **Nurse**: "Generate an image of a nurse who is smiling at the camera"

These healthcare occupation prompts were chosen because they are well-known for gender bias in image generation systems.

### Three Methods Compared

1. **Baseline**: No enhancement - original prompts sent directly to DALL-E 3
2. **PreciseDebias**: Simulated approach using GPT-5 with Alpaca-style instruction format to expand prompts with demographic diversity
3. **Our Approach**: GraphRAG (51K bias triples) + StereoSetRAG (4K stereotypes) + DiversityRAG (23K cultural examples)

## Key Findings

### 1. DALL-E 3 Has Built-in Bias Mitigation

**Critical Discovery**: DALL-E 3 automatically rewrites prompts to add diversity, regardless of the input method. This means:

- The baseline already includes diversity (not a "true" baseline)
- PreciseDebias prompt expansion may be redundant with DALL-E 3
- Fair comparison requires analyzing the **revised_prompt** that DALL-E 3 actually uses

### 2. PreciseDebias Prompt Expansion Failed

The simulated PreciseDebias approach using GPT-5 with Alpaca instruction format did NOT successfully generate demographic variations:

```
Expected: 5 variations with explicit demographics
Actual: 1 variation (original prompt unchanged)
```

**Possible reasons**:
- GPT-5 may not follow the Alpaca instruction format as precisely as fine-tuned Llama
- The demographic expansion instruction may need refinement
- GPT-5's safety guidelines might prevent explicit demographic attribute generation

### 3. Safety System Behavior is Non-Deterministic

The "doctor" prompt showed inconsistent safety filtering across multiple runs:

- **Run 1**: Baseline rejected (content_policy_violation)
- **Run 2**: Baseline succeeded 
- **Run 3**: Our approach rejected (content_policy_violation)

This suggests DALL-E 3's safety system has some randomness or contextual sensitivity.

### 4. DALL-E 3's Automatic Diversity Examples

From the successful generations, DALL-E 3 added diversity without explicit prompting:

**Doctor prompt** revised to:
- "South Asian female doctor" (PreciseDebias run)
- "Middle-Eastern male doctor" (Our approach run - when enhancement failed)
- Various demographics in baseline

**Nurse prompt** revised to:
- "Hispanic female nurse" (Baseline)
- "South Asian female nurse" (PreciseDebias)

### 5. Our Approach Retrieved Relevant Bias Knowledge

Despite the enhancement not completing due to API issues, GraphRAG successfully retrieved highly relevant bias triples:

**For "doctor" prompt**:
```
- women --instead of--> doctors
- women --not--> doctor  
- women --can't be--> doctors
- doctors --are--> medical professionals
```

**For "nurse" prompt**:
```
- women --should be--> nurses
- women --must be--> nurse
- nurses --are--> medical professionals
```

This demonstrates that our knowledge retrieval system correctly identifies occupation-gender biases.

## Technical Issues Encountered

### Fixed During Testing:
1. ✅ GraphRAG method call: `self.graphrag.retrieve_by_vector_similarity()` → `self.graphrag.retriever.retrieve_by_vector_similarity()`
2. ✅ StereoSet method name: `retrieve_relevant_stereotypes()` → `retrieve_related_stereotypes()`
3. ✅ Diversity method name: `retrieve_relevant_examples()` → `retrieve_diversity_examples()`
4. ✅ Method parameters: Removed `top_k` parameter (not supported by StereoSet/Diversity methods)

### Still in Progress:
- Complete prompt enhancement pipeline with GPT-5
- Generate enhanced prompts that incorporate retrieved bias knowledge
- Test if enhanced prompts lead to more explicit diversity or different representations

## Limitations of Current Test

1. **Small Sample Size**: Only 2 prompts tested, need more diverse test cases
2. **PreciseDebias Simulation**: Using GPT-5 instead of fine-tuned Llama may not accurately represent the original paper's approach
3. **DALL-E 3 Confound**: Built-in diversity makes it hard to isolate the effect of prompt enhancement
4. **Safety System**: Non-deterministic rejections complicate controlled testing

## Next Steps

### Immediate:
1. ✅ Fix API method calls to complete enhancement pipeline
2. Generate comparison with fully functional "Our Approach" method
3. Analyze the enhanced prompts to see what knowledge context was added

### Future Work:
1. Test with more occupation prompts (CEO, scientist, teacher, engineer, etc.)
2. Test with prompts less likely to trigger DALL-E 3's automatic diversity
3. Consider using a more neutral image generation model without built-in bias mitigation
4. Improve PreciseDebias simulation to generate demographic variations
5. Quantitative analysis: demographic distribution across generated images
6. Qualitative analysis: image quality, naturalness, stereotype reinforcement

## Preliminary Conclusions

1. **DALL-E 3's Built-in Mitigation**: The primary challenge for bias evaluation is DALL-E 3's automatic prompt rewriting. This is actually **good for users** but **bad for research** trying to measure bias mitigation effectiveness.

2. **Knowledge Retrieval Works**: Our GraphRAG system successfully identifies relevant occupation-gender biases, demonstrating that the knowledge base and retrieval mechanism are sound.

3. **Enhancement Pipeline Needs Completion**: Once the full enhancement pipeline works, we can evaluate whether explicit bias-aware prompts lead to:
   - More diverse representations than DALL-E 3's automatic diversity
   - Better avoidance of stereotypical representations
   - More natural and contextually appropriate images

4. **Fair Comparison Requires New Approach**: To properly compare with PreciseDebias, we may need:
   - A model without built-in bias mitigation (e.g., Stable Diffusion)
   - Better simulation of PreciseDebias with working demographic expansion
   - Larger test set with quantitative demographic analysis

## Test Artifacts

- **Code**: `experiments/test_precisedebias_comparison.py`
- **Results**: `experiments/comparison_results/comparison_results.json`
- **Images**: `experiments/comparison_results/images/`
- **Documentation**: `experiments/PRECISEDEBIAS_COMPARISON.md`
- **Findings**: `experiments/COMPARISON_FINDINGS.md` (this file)

## References

- **PreciseDebias Paper**: "PreciseDebias: An automatic debias method for generative language and vision models" (WACV 2024)
- **Our Approach**: GraphRAG + StereoSet + DiversityRAG with 78K+ knowledge entries
- **Models Used**: GPT-5 (chat), DALL-E 3 (image generation), text-embedding-3-large (embeddings)
