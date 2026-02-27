#!/usr/bin/env python3
"""Test script for visual bias evaluation integration.

This script demonstrates how to use the complete pipeline with visual bias evaluation.
"""

import logging
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from scripts.batch_processor import BatchProcessor


def main():
    """Test the complete pipeline with visual bias evaluation."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    logger = logging.getLogger(__name__)

    print("🧪 Testing Visual Bias Evaluation Pipeline")
    print("=" * 60)

    # Test configuration
    input_csv = "test_prompts.csv"
    output_dir = "visual_bias_test_results"
    image_dir = "visual_bias_test_images"

    try:
        print("Initializing batch processor with visual bias evaluation...")

        # Initialize processor with visual evaluation enabled
        processor = BatchProcessor(
            image_generator_type="dalle3",  # Use real DALL-E 3 for actual images
            output_dir=output_dir,
            image_output_dir=image_dir,
            enable_visual_bias_evaluation=True,  # Enable visual bias evaluation
            # Note: FairFace and dlib models will use mock versions if not available
            # In production, provide actual model paths:
            # fairface_model_path="/path/to/res34_fair_align_multi_7_20190809.pt",
            # dlib_model_path="/path/to/shape_predictor_5_face_landmarks.dat"
        )

        print("Processing prompts with image generation and visual bias evaluation...")

        # Process 3 prompts for testing
        output_csv_path, results = processor.process_csv(
            input_csv_path=input_csv,
            prompt_column="prompt",
            max_rows=3,  # Test with 3 prompts
            save_intermediate=True
        )

        # Print results
        print("\n" + "=" * 60)
        print("VISUAL BIAS EVALUATION TEST RESULTS")
        print("=" * 60)

        for i, result in enumerate(results[:3]):
            print(f"\n📝 Prompt {i+1}: {result.original_prompt[:50]}...")

            if result.enhanced_prompt:
                print(f"✨ Enhanced: {result.enhanced_prompt[:100]}...")
                print(f"📊 Text Scores: Bias={result.bias_score}/100, Diversity={result.diversity_score}/100")

            if result.original_image_path:
                print("🖼️  Images:")
                print(f"   Original: {result.original_image_path}")
                print(f"   Enhanced: {result.enhanced_image_path}")

            if result.visual_bias_evaluation:
                print(f"🔍 Visual Evaluation: {result.visual_bias_evaluation}")

            if result.error:
                print(f"❌ Error: {result.error}")

        print("\n💾 Complete Results:")
        print(f"   CSV: {output_csv_path}")
        print(f"   JSON: {output_csv_path.replace('.csv', '.json')}")

        # Check if visual bias evaluation was performed
        evaluation_dir = Path(output_dir) / "visual_bias_evaluation"
        if evaluation_dir.exists():
            print("\n🎯 Visual Bias Evaluation Results:")
            print(f"   Evaluation directory: {evaluation_dir}")

            # List evaluation files
            for subdir in evaluation_dir.iterdir():
                if subdir.is_dir():
                    print(f"   📁 {subdir.name}/")
                    for file in subdir.iterdir():
                        if file.is_file():
                            print(f"      📄 {file.name}")
        else:
            print("⚠️  Visual bias evaluation directory not found")

        print("\n✅ Test completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"❌ Test failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
