"""Test script for bias evaluation metrics.

This script demonstrates how to use the DI and CMMD metrics to evaluate
the effectiveness of prompt enhancement for bias mitigation.
"""

import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.evaluation_metrics import BiasEvaluator, create_sample_evaluation_data


def setup_logging() -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
        handlers=[logging.StreamHandler()]
    )


def run_evaluation_demo() -> None:
    """Run a demonstration of the bias evaluation metrics."""
    logger = logging.getLogger(__name__)
    
    logger.info("🧪 Starting Bias Evaluation Metrics Demo")
    logger.info("=" * 60)
    
    # Create sample data
    logger.info("📊 Creating sample evaluation data...")
    original_prompts, enhanced_prompts = create_sample_evaluation_data()
    
    logger.info(f"Generated {len(original_prompts)} original prompts")
    logger.info(f"Generated {len(enhanced_prompts)} enhanced prompts")
    
    # Initialize evaluator
    logger.info("\n⚙️  Initializing bias evaluator...")
    evaluator = BiasEvaluator(di_threshold=0.8, clip_model="ViT-B/32")
    
    # Define evaluation groups
    evaluation_groups = [
        {"category": "gender", "group1": "female", "group2": "male"},
        {"category": "ethnicity", "group1": "asian", "group2": "european"}, 
        {"category": "age", "group1": "young", "group2": "senior"}
    ]
    
    logger.info(f"Will evaluate {len(evaluation_groups)} group comparisons")
    
    # Run comprehensive evaluation
    logger.info("\n🔍 Running bias evaluation...")
    results = evaluator.evaluate_enhancement_impact(
        original_prompts, 
        enhanced_prompts,
        evaluation_groups
    )
    
    # Display results
    logger.info("\n" + "=" * 80)
    logger.info("📈 BIAS EVALUATION RESULTS SUMMARY")
    logger.info("=" * 80)
    
    summary = results["summary"]
    logger.info(f"Total comparisons: {summary['total_comparisons']}")
    logger.info(f"DI improvements: {summary['di_improvements']}")
    logger.info(f"CMMD improvements: {summary['cmmd_improvements']}")
    logger.info(f"Overall bias reduction: {summary['overall_bias_reduction']}")
    
    # Detailed analysis
    logger.info("\n📊 DETAILED ANALYSIS:")
    
    for comparison in results["disparate_impact"]:
        di_result = results["disparate_impact"][comparison]
        if "error" not in di_result:
            logger.info(f"\n  {comparison} - Disparate Impact:")
            orig = di_result["original"]
            enh = di_result["enhanced"]
            logger.info(f"    Original: ratio={orig['ratio']:.3f}, biased={orig['is_biased']}")
            logger.info(f"    Enhanced: ratio={enh['ratio']:.3f}, biased={enh['is_biased']}")
            logger.info(f"    Improvement: {di_result['improvement']}")
    
    for comparison in results["cmmd"]:
        cmmd_result = results["cmmd"][comparison]
        if "error" not in cmmd_result:
            logger.info(f"\n  {comparison} - CMMD:")
            orig = cmmd_result["original"]
            enh = cmmd_result["enhanced"]
            logger.info(f"    Original: distance={orig['distance']:.4f}")
            logger.info(f"    Enhanced: distance={enh['distance']:.4f}")
            logger.info(f"    Improvement: {cmmd_result['improvement']}")
    
    # Save results
    output_path = "bias_evaluation_demo_results.json"
    evaluator.save_evaluation_results(results, output_path)
    logger.info(f"\n💾 Results saved to {output_path}")
    
    # Performance insights
    logger.info("\n🎯 KEY INSIGHTS:")
    
    if summary["overall_bias_reduction"]:
        logger.info("✅ The prompt enhancement system shows measurable bias reduction!")
    else:
        logger.info("⚠️  Limited bias reduction detected. Consider refining enhancement strategies.")
    
    if summary["di_improvements"] > 0:
        logger.info(f"✅ Disparate Impact improved in {summary['di_improvements']} comparisons")
    
    if summary["cmmd_improvements"] > 0:
        logger.info(f"✅ CMMD (distribution distance) improved in {summary['cmmd_improvements']} comparisons")
    
    logger.info("\n🔬 EVALUATION METHODOLOGY:")
    logger.info("• DI (Disparate Impact): Measures occurrence rate ratios between groups")
    logger.info("  - Ratio < 0.8 or > 1.25 indicates potential bias")
    logger.info("  - Closer to 1.0 indicates more fair representation")
    logger.info("• CMMD: Measures distribution distance in CLIP embedding space")
    logger.info("  - Lower distance indicates more similar group representations")
    logger.info("  - Uses semantic understanding via CLIP text embeddings")


def main() -> None:
    """Main function to run the evaluation demo."""
    setup_logging()
    
    try:
        run_evaluation_demo()
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error running evaluation demo: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
