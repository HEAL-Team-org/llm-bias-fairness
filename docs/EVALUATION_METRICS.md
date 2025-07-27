# Bias Evaluation Metrics Implementation

## Overview

This document describes the implementation of two key evaluation metrics for assessing bias mitigation and diversity improvement in image generation prompts:

1. **DI (Disparate Impact)**: Measures the ratio of occurrence rates between demographic subgroups
2. **CMMD (Conditional Maximum Mean Discrepancy)**: Uses CLIP embeddings to measure distribution distance between groups

## Theoretical Background

### Disparate Impact (DI)

**Definition**: Disparate Impact measures whether a system treats different demographic groups fairly by comparing the rates at which positive outcomes occur for each group.

**Formula**: 
```
DI = P(outcome | group1) / P(outcome | group2)
```

**Interpretation**:
- DI ≈ 1.0: Fair treatment of both groups
- DI < 0.8: Group2 favored over Group1 (bias detected)
- DI > 1.25: Group1 favored over Group2 (bias detected)

**Application in Prompt Enhancement**:
In our context, we measure how frequently different demographic groups are mentioned in prompts before and after enhancement. The goal is to achieve more balanced representation (DI closer to 1.0).

**Why DI is Important**:
- **Legal Compliance**: Based on the "80% rule" from employment law
- **Quantitative**: Provides clear numerical thresholds for bias detection
- **Interpretable**: Easy to understand and explain to stakeholders
- **Actionable**: Directly indicates which groups are under/over-represented

### Conditional Maximum Mean Discrepancy (CMMD)

**Definition**: CMMD extends Maximum Mean Discrepancy (MMD) to compare conditional distributions using CLIP embeddings, measuring how similar the semantic representations of different groups are.

**Background on MMD**:
MMD is a statistical distance measure between probability distributions. It uses kernel functions to map data into a reproducing kernel Hilbert space (RKHS) where linear comparisons become meaningful.

**Formula**:
```
MMD²(X, Y) = ||μ_X - μ_Y||²_H
```

Where μ_X and μ_Y are mean embeddings in the RKHS.

**CLIP Integration**:
CMMD uses CLIP (Contrastive Language-Image Pre-training) embeddings because:
- **Semantic Understanding**: CLIP captures semantic meaning, not just syntactic patterns
- **Cross-Modal**: Trained on image-text pairs, relevant for image generation
- **Rich Representations**: High-dimensional embeddings capture nuanced differences
- **Robust**: Less sensitive to surface-level text variations

**Kernel Types**:
1. **RBF Kernel**: `K(x,y) = exp(-γ||x-y||²)` - Captures non-linear relationships
2. **Linear Kernel**: `K(x,y) = x·y` - Simpler, computationally efficient

**Interpretation**:
- **Lower CMMD**: Groups have similar semantic distributions (less bias)
- **Higher CMMD**: Groups have different semantic distributions (potential bias)

## Implementation Details

### Dataset Requirements

For meaningful evaluation, we need datasets that contain:

1. **Demographic Annotations**: Clear labels for gender, ethnicity, age, etc.
2. **Sufficient Samples**: Minimum 5-10 samples per group for statistical significance
3. **Domain Relevance**: Appropriate for the target application (image generation)

### Recommended Datasets

Based on research, the following datasets are suitable for bias evaluation:

#### 1. FairFace Dataset
- **Source**: [FairFace GitHub](https://github.com/joojs/fairface)
- **Content**: 108,501 face images with balanced demographics
- **Demographics**: Race (7 categories), Gender (2 categories), Age (9 groups)
- **Use Case**: Ideal for evaluating facial representation bias
- **License**: CC BY 4.0

#### 2. COCO Dataset (Subset)
- **Source**: [MS COCO](https://cocodataset.org/)
- **Content**: 330K images with captions
- **Demographics**: Can be annotated for people in scenes
- **Use Case**: General scene understanding and person representation
- **Advantage**: Widely used benchmark

#### 3. Custom Evaluation Sets
For our specific use case, we create evaluation sets by:
- Generating prompts with known demographic specifications
- Using our existing bias and cultural knowledge graphs
- Creating synthetic test cases with controlled demographic content

### Code Architecture

#### Core Classes

1. **`PromptAnalyzer`**: Extracts demographic mentions from text prompts
   - Uses pattern matching for gender, ethnicity, age, profession
   - Extensible to new demographic categories
   - Handles synonyms and variations

2. **`DisparateImpactCalculator`**: Computes DI metrics
   - Configurable bias threshold (default: 0.8)
   - Supports multiple demographic categories
   - Provides detailed breakdown of group rates

3. **`CMMDCalculator`**: Computes CMMD using CLIP embeddings
   - Supports multiple CLIP models
   - Implements RBF and linear kernels
   - Handles CUDA/CPU computation
   - Graceful fallback when CLIP unavailable

4. **`BiasEvaluator`**: Orchestrates comprehensive evaluation
   - Combines DI and CMMD analysis
   - Generates summary statistics
   - Saves results for further analysis

### Usage Examples

#### Basic Evaluation
```python
from src.evaluation_metrics import BiasEvaluator

# Initialize evaluator
evaluator = BiasEvaluator(di_threshold=0.8, clip_model="ViT-B/32")

# Prepare data
original_prompts = ["a doctor", "a nurse", "an engineer"]
enhanced_prompts = ["a doctor of diverse background", ...]

# Run evaluation
results = evaluator.evaluate_enhancement_impact(
    original_prompts, enhanced_prompts
)

# Check improvements
print(f"Bias reduction achieved: {results['summary']['overall_bias_reduction']}")
```

#### Custom Group Comparisons
```python
# Define specific groups to evaluate
evaluation_groups = [
    {"category": "gender", "group1": "female", "group2": "male"},
    {"category": "ethnicity", "group1": "asian", "group2": "african"},
    {"category": "profession", "group1": "doctor", "group2": "nurse"}
]

results = evaluator.evaluate_enhancement_impact(
    original_prompts, enhanced_prompts, evaluation_groups
)
```

### Validation and Testing

#### Mock Implementation
When CLIP is unavailable, the system uses mock embeddings to ensure functionality:
- Generates random embeddings with appropriate dimensions
- Maintains API compatibility
- Allows testing without GPU requirements

#### Sample Data Generation
The `create_sample_evaluation_data()` function provides realistic test cases:
- Original prompts with potential bias patterns
- Enhanced prompts with explicit diversity elements
- Covers multiple professions and scenarios

## Metrics Interpretation

### Expected Results

#### Successful Bias Mitigation
- **DI Improvement**: Ratios moving closer to 1.0
- **CMMD Improvement**: Lower distribution distances
- **Coverage**: More demographic groups mentioned

#### Warning Signs
- **DI Worsening**: New biases introduced
- **CMMD Increase**: Groups becoming more separated
- **Over-correction**: Unrealistic demographic distributions

### Limitations

1. **Text-Based Analysis**: Only evaluates prompts, not generated images
2. **Pattern Matching**: May miss implicit demographic references
3. **Cultural Bias**: Evaluation patterns reflect training data biases
4. **Statistical Power**: Requires sufficient samples for meaningful results

### Future Enhancements

1. **Image-Based Evaluation**: Analyze actual generated images
2. **Intersectionality**: Evaluate combinations of demographic attributes
3. **Dynamic Patterns**: Learn demographic patterns from data
4. **Real-World Validation**: Compare with human judgment studies

## Mathematical Foundations

### MMD Computation Details

The empirical MMD between samples X = {x₁, ..., xₙ} and Y = {y₁, ..., yₘ} is:

```
MMD²(X, Y) = (1/n²)∑ᵢⱼ k(xᵢ, xⱼ) + (1/m²)∑ᵢⱼ k(yᵢ, yⱼ) - (2/nm)∑ᵢⱼ k(xᵢ, yⱼ)
```

### RBF Kernel Implementation

For the RBF kernel with parameter γ:
```
K(x, y) = exp(-γ||x - y||²)
```

The implementation uses efficient batch computation:
```python
def rbf_kernel(X, Y, gamma):
    X_norm = (X ** 2).sum(1).view(-1, 1)
    Y_norm = (Y ** 2).sum(1).view(1, -1)
    dist = X_norm + Y_norm - 2.0 * torch.mm(X, Y.T)
    return torch.exp(-gamma * dist)
```

### Statistical Significance

While not implemented in the current version, statistical significance testing could be added using:
- **Permutation Tests**: Randomly permute group labels
- **Bootstrap Confidence Intervals**: Estimate uncertainty in metrics
- **Multiple Comparisons Correction**: Adjust for multiple group comparisons

## Conclusion

The implemented evaluation system provides a comprehensive framework for measuring bias mitigation effectiveness. By combining established legal concepts (Disparate Impact) with modern semantic understanding (CLIP-based CMMD), it offers both interpretable and sophisticated bias measurement capabilities.

The modular design allows for easy extension to new demographic categories, evaluation metrics, and datasets, making it a valuable tool for responsible AI development in image generation and beyond.
