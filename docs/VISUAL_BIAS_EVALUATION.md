# Visual Bias Evaluation System

## Overview

The Visual Bias Evaluation System has been successfully integrated into the LLM bias-fairness pipeline to provide comprehensive evaluation of demographic representation in generated images. This system implements the same evaluation methods from your `final_evaluation.ipynb` notebook and is fully compatible with the existing pipeline.

## Key Features

### 1. Demographic Analysis
- **Face Detection**: Uses dlib's face detection for accurate face localization
- **Demographic Prediction**: Employs FairFace ResNet34 model for predicting:
  - Race (7 categories: White, Black, Asian, Indian, Middle Eastern, Latino, Southeast Asian)
  - Gender (Male, Female)
  - Age (9 groups: 0-2, 3-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70+)

### 2. Bias Metrics
Implements all evaluation metrics from your notebook:

- **Bias-W (Population-level Bias)**: Measures overall bias across the entire image set
- **Bias-P (Per-image Bias)**: Calculates bias for individual images
- **ENS (Diversity via Shannon Entropy)**: Measures demographic diversity
- **KL Divergence**: Compares distributions against reference demographics

### 3. Integration Architecture
- **Seamless Integration**: Works with existing BatchProcessor
- **Optional Evaluation**: Can be enabled/disabled via parameter
- **Mock Support**: Includes fallback mock models for testing without actual model files
- **Comprehensive Results**: Saves detailed evaluation data in structured format

## System Architecture

### Core Components

```
src/visual_bias_evaluator.py
├── VisualBiasEvaluator (Main class)
├── BiasMetrics (Results dataclass)
├── DemographicPrediction (Individual prediction)
└── MockFairFaceModel (Testing fallback)

batch_processor.py
├── Enhanced ProcessingResult with visual_bias_evaluation field
├── Optional visual evaluation initialization
└── _perform_visual_bias_evaluation() method
```

### File Structure

```
project_root/
├── src/
│   └── visual_bias_evaluator.py     # Core evaluation module
├── batch_processor.py               # Enhanced with visual evaluation
├── test_visual_bias_evaluation.py   # Integration test script
└── output_dir/
    └── visual_bias_evaluation/
        ├── original_images/         # Analysis of original images
        │   ├── demographic_analysis.json
        │   ├── bias_metrics.json
        │   └── face_predictions.json
        └── enhanced_images/         # Analysis of enhanced images
            ├── demographic_analysis.json
            ├── bias_metrics.json
            └── face_predictions.json
```

## Usage Examples

### 1. Basic Integration

```python
from batch_processor import BatchProcessor

# Initialize with visual bias evaluation enabled
processor = BatchProcessor(
    image_generator_type="dalle3",
    output_dir="results",
    image_output_dir="images",
    enable_visual_bias_evaluation=True
)

# Process with evaluation
output_csv, results = processor.process_csv("prompts.csv")
```

### 2. Custom Model Paths

```python
# Use custom model paths (for production)
processor = BatchProcessor(
    enable_visual_bias_evaluation=True,
    fairface_model_path="/path/to/res34_fair_align_multi_7_20190809.pt",
    dlib_model_path="/path/to/shape_predictor_5_face_landmarks.dat"
)
```

### 3. Accessing Results

```python
for result in results:
    if result.visual_bias_evaluation:
        print(f"Bias-W: {result.visual_bias_evaluation['bias_w']}")
        print(f"Diversity (ENS): {result.visual_bias_evaluation['ens']}")
```

## Evaluation Metrics Details

### Bias-W (Population-level Bias)
Measures overall demographic bias across all images:
```
bias_w = Σ |p_i - r_i| / 2
where p_i = predicted proportion, r_i = reference proportion
```

### Bias-P (Per-image Bias)
Calculates bias for individual images based on face count deviations:
```
bias_p = average deviation from expected demographic distribution per image
```

### ENS (Shannon Entropy Diversity)
Measures demographic diversity using information theory:
```
ENS = -Σ p_i * log(p_i)
where p_i = proportion of demographic group i
```

### KL Divergence
Compares predicted distribution against reference:
```
KL(P||Q) = Σ P(i) * log(P(i)/Q(i))
```

## Testing and Validation

### Running Tests

```bash
# Test the integration (uses mock models)
python test_visual_bias_evaluation.py

# Test with actual models (requires model files)
python test_visual_bias_evaluation.py --use-real-models
```

### Expected Output Structure

```json
{
  "bias_w": 0.15,
  "bias_p": 0.23,
  "ens": 1.85,
  "kl_divergence": 0.12,
  "total_faces": 45,
  "images_with_faces": 38,
  "demographic_distribution": {
    "race": {"White": 0.4, "Black": 0.2, ...},
    "gender": {"Male": 0.6, "Female": 0.4},
    "age": {"20-29": 0.3, "30-39": 0.25, ...}
  }
}
```

## Model Requirements

### Required Models (for production use)
1. **FairFace Model**: `res34_fair_align_multi_7_20190809.pt`
   - Download from: https://github.com/joojs/fairface
   - Size: ~84MB
   - Required for demographic prediction

2. **Dlib Shape Predictor**: `shape_predictor_5_face_landmarks.dat`
   - Download from: http://dlib.net/files/
   - Size: ~9.2MB
   - Required for face detection and alignment

### Installation Requirements
```bash
pip install torch torchvision dlib opencv-python pillow pandas numpy
```

## Comparison Workflow

Now you can evaluate and compare your pipeline results:

### 1. Process Original Prompts
```python
# Process without enhancement
processor_original = BatchProcessor(
    enhancement_method="none",  # No enhancement
    enable_visual_bias_evaluation=True
)
original_results = processor_original.process_csv("prompts.csv")
```

### 2. Process Enhanced Prompts  
```python
# Process with enhancement
processor_enhanced = BatchProcessor(
    enhancement_method="dual_pipeline",  # Your enhancement method
    enable_visual_bias_evaluation=True
)
enhanced_results = processor_enhanced.process_csv("prompts.csv")
```

### 3. Compare Results
```python
def compare_bias_metrics(original_results, enhanced_results):
    """Compare bias metrics between original and enhanced results."""
    
    orig_bias_w = sum(r.visual_bias_evaluation.get('bias_w', 0) for r in original_results) / len(original_results)
    enh_bias_w = sum(r.visual_bias_evaluation.get('bias_w', 0) for r in enhanced_results) / len(enhanced_results)
    
    print(f"Bias-W Improvement: {orig_bias_w:.3f} → {enh_bias_w:.3f} ({((enh_bias_w - orig_bias_w) / orig_bias_w * 100):+.1f}%)")
    
    # Similar comparisons for other metrics...
```

## Benefits of This Implementation

1. **Exact Notebook Compatibility**: Implements the same evaluation methods as your notebook
2. **Seamless Integration**: Works with existing pipeline without modifications
3. **Comprehensive Metrics**: All bias metrics (Bias-W, Bias-P, ENS, KL Divergence)
4. **Production Ready**: Includes error handling, logging, and mock fallbacks
5. **Structured Output**: JSON and CSV results for easy analysis
6. **Comparison Ready**: Perfect for evaluating original vs enhanced results

## Next Steps

1. **Install Model Files**: Download FairFace and dlib models for production use
2. **Run Evaluation**: Use the system to evaluate your current pipeline results
3. **Compare Results**: Analyze original vs enhanced image bias metrics
4. **Optimize Pipeline**: Use insights to improve your enhancement methods

The system is now ready to provide the comprehensive visual bias evaluation you requested, enabling you to evaluate and compare your pipeline results with the same methodology as your notebook.