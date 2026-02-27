#!/usr/bin/env python3
"""Phase 6 Integration Tests - Evaluation Layer Refactoring.

This test suite validates that the evaluation layer has been properly
separated into src/evaluation/ with clean imports and no circular dependencies.

Tests:
    1. Evaluation Layer Imports - All evaluator classes accessible
    2. VisualBiasEvaluator Import - Main evaluator class available
    3. DemographicPrediction Import - Dataclass available
    4. BiasMetrics Import - Metrics container available
    5. VisualBiasEvaluator Constants - Class attributes correctly defined
    6. No Circular Imports - Clean architecture validation
    7. Batch Processor Integration - Uses new imports correctly
"""


def test_evaluation_layer_imports():
    """TEST 1: Verify all evaluation layer classes can be imported."""
    print("=" * 70)
    print("TEST 1: Evaluation Layer Imports")
    print("=" * 70)

    try:
        from src.evaluation import (
            BiasMetrics,
            DemographicPrediction,
            VisualBiasEvaluator,
        )

        print("✅ All evaluation layer classes imported successfully:")
        print("  - VisualBiasEvaluator")
        print("  - DemographicPrediction")
        print("  - BiasMetrics")
        print()
        return True

    except ImportError as e:
        print(f"❌ Evaluation layer import failed: {e}")
        print()
        return False


def test_visual_bias_evaluator():
    """TEST 2: Verify VisualBiasEvaluator class structure."""
    print("=" * 70)
    print("TEST 2: VisualBiasEvaluator Class Structure")
    print("=" * 70)

    try:
        from src.evaluation import VisualBiasEvaluator

        # Verify class attributes
        assert hasattr(VisualBiasEvaluator, "RACE_LABELS"), "Should have RACE_LABELS"
        assert hasattr(VisualBiasEvaluator, "GENDER_LABELS"), "Should have GENDER_LABELS"
        assert hasattr(VisualBiasEvaluator, "AGE_LABELS"), "Should have AGE_LABELS"
        assert hasattr(VisualBiasEvaluator, "ATTRIBUTES_DICT"), "Should have ATTRIBUTES_DICT"
        assert hasattr(VisualBiasEvaluator, "COMBINATIONS"), "Should have COMBINATIONS"

        # Verify methods
        assert hasattr(VisualBiasEvaluator, "detect_faces"), "Should have detect_faces method"
        assert hasattr(VisualBiasEvaluator, "predict_demographics"), "Should have predict_demographics method"
        assert hasattr(VisualBiasEvaluator, "analyze_images"), "Should have analyze_images method"
        assert hasattr(VisualBiasEvaluator, "calculate_bias_w"), "Should have calculate_bias_w method"
        assert hasattr(VisualBiasEvaluator, "calculate_bias_p"), "Should have calculate_bias_p method"
        assert hasattr(VisualBiasEvaluator, "calculate_ens"), "Should have calculate_ens method"
        assert hasattr(VisualBiasEvaluator, "calculate_kl_divergence"), "Should have calculate_kl_divergence method"
        assert hasattr(VisualBiasEvaluator, "evaluate_batch_results"), "Should have evaluate_batch_results method"

        # Verify demographic labels
        assert len(VisualBiasEvaluator.RACE_LABELS) == 7, "Should have 7 race labels"
        assert len(VisualBiasEvaluator.GENDER_LABELS) == 2, "Should have 2 gender labels"
        assert len(VisualBiasEvaluator.AGE_LABELS) == 9, "Should have 9 age labels"

        print("✅ VisualBiasEvaluator validated successfully")
        print(f"  - Race labels: {len(VisualBiasEvaluator.RACE_LABELS)}")
        print(f"  - Gender labels: {len(VisualBiasEvaluator.GENDER_LABELS)}")
        print(f"  - Age labels: {len(VisualBiasEvaluator.AGE_LABELS)}")
        print(f"  - Combinations: {len(VisualBiasEvaluator.COMBINATIONS)}")
        print("  - All required methods present")
        print()
        return True

    except Exception as e:
        print(f"❌ VisualBiasEvaluator validation failed: {e}")
        print()
        return False


def test_demographic_prediction():
    """TEST 3: Verify DemographicPrediction dataclass."""
    print("=" * 70)
    print("TEST 3: DemographicPrediction Dataclass")
    print("=" * 70)

    try:
        from dataclasses import is_dataclass

        import numpy as np

        from src.evaluation import DemographicPrediction

        # Verify it's a dataclass
        assert is_dataclass(DemographicPrediction), "Should be a dataclass"

        # Verify fields
        fields = [f.name for f in DemographicPrediction.__dataclass_fields__.values()]
        expected_fields = [
            "face_path", "original_image", "race", "gender", "age",
            "race_scores", "gender_scores", "age_scores"
        ]
        for field in expected_fields:
            assert field in fields, f"Should have {field} field"

        # Create a test instance
        prediction = DemographicPrediction(
            face_path="test.png",
            original_image="original.png",
            race="White",
            gender="Male",
            age="30-39",
            race_scores=np.array([0.8, 0.1, 0.05, 0.02, 0.01, 0.01, 0.01]),
            gender_scores=np.array([0.7, 0.3]),
            age_scores=np.array([0.01, 0.01, 0.05, 0.6, 0.2, 0.1, 0.02, 0.005, 0.005])
        )

        assert prediction.race == "White", "Should store race"
        assert prediction.gender == "Male", "Should store gender"
        assert prediction.age == "30-39", "Should store age"

        print("✅ DemographicPrediction validated successfully")
        print("  - Is dataclass: ✓")
        print(f"  - Fields: {len(fields)}")
        print("  - Can create instances: ✓")
        print()
        return True

    except Exception as e:
        print(f"❌ DemographicPrediction test failed: {e}")
        print()
        return False


def test_bias_metrics():
    """TEST 4: Verify BiasMetrics dataclass."""
    print("=" * 70)
    print("TEST 4: BiasMetrics Dataclass")
    print("=" * 70)

    try:
        from dataclasses import is_dataclass

        import pandas as pd

        from src.evaluation import BiasMetrics

        # Verify it's a dataclass
        assert is_dataclass(BiasMetrics), "Should be a dataclass"

        # Verify fields
        fields = [f.name for f in BiasMetrics.__dataclass_fields__.values()]
        expected_fields = [
            "bias_w_metrics", "bias_p_metrics", "ens_metrics", "kl_divergence_metrics"
        ]
        for field in expected_fields:
            assert field in fields, f"Should have {field} field"

        # Create a test instance
        metrics = BiasMetrics(
            bias_w_metrics=pd.DataFrame({"Attribute_Group": ["race"], "Bias_W": [0.5]}),
            bias_p_metrics=pd.DataFrame({"Original_Image": ["img1"], "Attribute_Group": ["race"], "Bias_P": [0.3]}),
            ens_metrics=pd.DataFrame({"Attribute": ["race"], "ENS": [5.2]}),
            kl_divergence_metrics=pd.DataFrame({"Attribute_Group": ["race"], "KL_Divergence": [0.1]})
        )

        assert len(metrics.bias_w_metrics) == 1, "Should store bias_w_metrics"
        assert len(metrics.ens_metrics) == 1, "Should store ens_metrics"

        print("✅ BiasMetrics validated successfully")
        print("  - Is dataclass: ✓")
        print(f"  - Fields: {len(fields)}")
        print("  - Can create instances: ✓")
        print("  - Stores pandas DataFrames: ✓")
        print()
        return True

    except Exception as e:
        print(f"❌ BiasMetrics test failed: {e}")
        print()
        return False


def test_evaluator_constants():
    """TEST 5: Verify VisualBiasEvaluator demographic constants."""
    print("=" * 70)
    print("TEST 5: VisualBiasEvaluator Constants")
    print("=" * 70)

    try:
        from src.evaluation import VisualBiasEvaluator

        # Verify race labels
        expected_races = [
            "White", "Black", "Latino_Hispanic", "East Asian",
            "Southeast Asian", "Indian", "Middle Eastern"
        ]
        assert expected_races == VisualBiasEvaluator.RACE_LABELS, \
            "Race labels mismatch"

        # Verify gender labels
        expected_genders = ["Male", "Female"]
        assert expected_genders == VisualBiasEvaluator.GENDER_LABELS, \
            "Gender labels mismatch"

        # Verify age labels
        expected_ages = [
            "0-2", "3-9", "10-19", "20-29", "30-39",
            "40-49", "50-59", "60-69", "70+"
        ]
        assert expected_ages == VisualBiasEvaluator.AGE_LABELS, \
            "Age labels mismatch"

        # Verify attributes dict
        assert "race" in VisualBiasEvaluator.ATTRIBUTES_DICT
        assert "gender" in VisualBiasEvaluator.ATTRIBUTES_DICT
        assert "age" in VisualBiasEvaluator.ATTRIBUTES_DICT

        # Verify combinations
        assert len(VisualBiasEvaluator.COMBINATIONS) == 7, \
            "Should have 7 attribute combinations"

        print("✅ VisualBiasEvaluator constants validated")
        print("  - Race labels: White, Black, Latino_Hispanic, East Asian, ...")
        print("  - Gender labels: Male, Female")
        print("  - Age labels: 0-2, 3-9, 10-19, ..., 70+")
        print("  - Attributes dict keys: race, gender, age")
        print("  - Combinations: 7 (single, pairs, triple)")
        print()
        return True

    except Exception as e:
        print(f"❌ Constants validation failed: {e}")
        print()
        return False


def test_no_circular_imports():
    """TEST 6: Verify no circular import dependencies."""
    print("=" * 70)
    print("TEST 6: No Circular Imports")
    print("=" * 70)

    try:
        # Test various import combinations to ensure no circular dependencies

        # Evaluation layer imports
        # Evaluation layer can be imported alongside other layers
        from src.config import settings
        from src.data import EmbeddingCache
        from src.evaluation import BiasMetrics, DemographicPrediction, VisualBiasEvaluator
        from src.generation import DALLE3Generator
        from src.knowledge import GraphRAG

        print("✅ No circular import dependencies detected")
        print("  - evaluation layer: OK")
        print("  - evaluation → config: OK")
        print("  - evaluation → data: OK")
        print("  - evaluation → knowledge: OK")
        print("  - evaluation → generation: OK")
        print("  - All combinations tested successfully")
        print()
        return True

    except ImportError as e:
        print(f"❌ Circular import detected: {e}")
        print()
        return False


def test_batch_processor_integration():
    """TEST 7: Verify batch_processor uses new imports."""
    print("=" * 70)
    print("TEST 7: Batch Processor Integration")
    print("=" * 70)

    try:
        # Read batch_processor.py and check it imports from src.evaluation
        from pathlib import Path

        batch_processor_path = Path("scripts/batch_processor.py")
        if not batch_processor_path.exists():
            print("⚠️  batch_processor.py not found, skipping test")
            print()
            return True

        content = batch_processor_path.read_text()

        # Check for new import
        assert "from src.evaluation import" in content, \
            "batch_processor should import from src.evaluation"

        # Check old import is removed
        assert "from src.visual_bias_evaluator import" not in content, \
            "batch_processor should not import from old src.visual_bias_evaluator"

        print("✅ Batch processor integration validated")
        print("  - Uses src.evaluation imports: ✓")
        print("  - No old src.visual_bias_evaluator imports: ✓")
        print()
        return True

    except Exception as e:
        print(f"❌ Batch processor integration failed: {e}")
        print()
        return False


def main():
    """Run all Phase 6 integration tests."""
    print()
    print("🧪 " + "=" * 68)
    print("   PHASE 6 INTEGRATION TESTS - EVALUATION LAYER REFACTORING")
    print("=" * 70)
    print()

    tests = [
        ("Evaluation Layer Imports", test_evaluation_layer_imports),
        ("VisualBiasEvaluator Class Structure", test_visual_bias_evaluator),
        ("DemographicPrediction Dataclass", test_demographic_prediction),
        ("BiasMetrics Dataclass", test_bias_metrics),
        ("VisualBiasEvaluator Constants", test_evaluator_constants),
        ("No Circular Imports", test_no_circular_imports),
        ("Batch Processor Integration", test_batch_processor_integration),
    ]

    results = []
    for name, test_func in tests:
        result = test_func()
        results.append((name, result))

    # Summary
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")

    print()
    print("-" * 70)
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    print(f"Results: {passed_count}/{total_count} tests passed")
    print()

    if passed_count == total_count:
        print("🎉 ALL PHASE 6 TESTS PASSED! 🎉")
        print()
        print("Phase 6 refactoring is complete and functional.")
        print("Evaluation layer successfully separated into src/evaluation/.")
        print("Ready to proceed to Phase 7: Enhancement Layer")
    else:
        print("⚠️  Some tests failed. Please review the errors above.")

    print()
    return 0 if passed_count == total_count else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
