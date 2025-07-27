#!/usr/bin/env python3
"""
Quick test script to demonstrate the bias evaluation metrics implementation.
This script runs a focused test to show that the system is working correctly.
"""

import logging
from src.evaluation_metrics import BiasEvaluator, create_sample_evaluation_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def run_quick_test():
    """Run a quick test of the evaluation system."""
    print("🧪 BIAS EVALUATION METRICS - QUICK TEST")
    print("=" * 50)
    
    # Create test data
    print("📊 Creating sample test data...")
    original_prompts, enhanced_prompts = create_sample_evaluation_data()
    
    print(f"✅ Created {len(original_prompts)} original prompts")
    print(f"✅ Created {len(enhanced_prompts)} enhanced prompts")
    
    # Initialize evaluator
    print("\n⚙️ Initializing bias evaluator...")
    evaluator = BiasEvaluator(di_threshold=0.8, clip_model="ViT-B/32")
    
    # Run evaluation
    print("\n🔍 Running bias evaluation...")
    results = evaluator.evaluate_enhancement_impact(
        original_prompts=original_prompts,
        enhanced_prompts=enhanced_prompts
    )
    
    # Display results
    print("\n📈 RESULTS SUMMARY:")
    summary = results['summary']
    print(f"  • Total comparisons: {summary['total_comparisons']}")
    print(f"  • DI improvements: {summary['di_improvements']}")
    print(f"  • CMMD improvements: {summary['cmmd_improvements']}")
    print(f"  • Overall bias reduction: {summary['overall_bias_reduction']}")
    
    # Show some example improvements
    print("\n🎯 DETAILED RESULTS:")
    for comparison, result in results['disparate_impact'].items():
        if 'error' not in result:
            print(f"\n  {comparison}:")
            orig = result['original']
            enh = result['enhanced']
            improved = result['improvement']
            
            print(f"    Original ratio: {orig['ratio']:.3f} (biased: {orig['is_biased']})")
            print(f"    Enhanced ratio: {enh['ratio']:.3f} (biased: {enh['is_biased']})")
            print(f"    Improvement: {'✅ YES' if improved else '❌ NO'}")
    
    print(f"\n✨ Test completed successfully!")
    
    # Save results
    evaluator.save_evaluation_results(results, "quick_test_results.json")
    print("📄 Results saved to quick_test_results.json")

if __name__ == "__main__":
    run_quick_test()
