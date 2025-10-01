#!/usr/bin/env python3
"""Phase 7 Integration Tests - Enhancement Layer Refactoring.

This test suite validates that the enhancement layer has been properly
created in src/enhancement/ with clean imports and no circular dependencies.

Tests:
    1. Enhancement Layer Imports - All enhancer classes accessible
    2. EnhancementConfig Dataclass - Configuration structure
    3. EnhancementResult Dataclass - Result structure  
    4. EnhancementSystem Instantiation - Can create system
    5. EnhancementSystem Basic Enhancement - Can enhance prompts
    6. Enhancement Scoring - Bias and diversity scoring
    7. No Circular Imports - Clean architecture validation
"""


def test_enhancement_layer_imports():
    """TEST 1: Verify all enhancement layer classes can be imported."""
    print("=" * 70)
    print("TEST 1: Enhancement Layer Imports")
    print("=" * 70)

    try:
        from src.enhancement import (
            EnhancementConfig,
            EnhancementResult,
            EnhancementSystem,
        )

        print("✅ All enhancement layer classes imported successfully:")
        print("  - EnhancementSystem")
        print("  - EnhancementConfig")
        print("  - EnhancementResult")
        print()
        return True

    except ImportError as e:
        print(f"❌ Enhancement layer import failed: {e}")
        print()
        return False


def test_enhancement_config():
    """TEST 2: Verify EnhancementConfig dataclass."""
    print("=" * 70)
    print("TEST 2: EnhancementConfig Dataclass")
    print("=" * 70)

    try:
        from dataclasses import is_dataclass

        from src.enhancement import EnhancementConfig

        # Verify it's a dataclass
        assert is_dataclass(EnhancementConfig), "Should be a dataclass"

        # Create with defaults
        config = EnhancementConfig()

        # Verify default values
        assert config.use_stereoset == True, "Should enable StereoSet by default"
        assert config.use_diversity_rag == True, "Should enable DiversityRAG by default"
        assert config.use_graphrag == True, "Should enable GraphRAG by default"
        assert config.bias_threshold == 75, "Should have bias threshold of 75"
        assert config.diversity_threshold == 80, "Should have diversity threshold of 80"
        assert config.max_iterations == 5, "Should have max 5 iterations"

        # Create with custom values
        custom_config = EnhancementConfig(
            bias_threshold=90,
            diversity_threshold=85,
            max_iterations=10
        )
        assert custom_config.bias_threshold == 90
        assert custom_config.diversity_threshold == 85
        assert custom_config.max_iterations == 10

        print("✅ EnhancementConfig validated successfully")
        print("  - Is dataclass: ✓")
        print("  - Default values: ✓")
        print("  - Custom values: ✓")
        print(f"  - Bias threshold: {config.bias_threshold}")
        print(f"  - Diversity threshold: {config.diversity_threshold}")
        print()
        return True

    except Exception as e:
        print(f"❌ EnhancementConfig test failed: {e}")
        print()
        return False


def test_enhancement_result():
    """TEST 3: Verify EnhancementResult dataclass."""
    print("=" * 70)
    print("TEST 3: EnhancementResult Dataclass")
    print("=" * 70)

    try:
        from dataclasses import is_dataclass

        from src.enhancement import EnhancementResult

        # Verify it's a dataclass
        assert is_dataclass(EnhancementResult), "Should be a dataclass"

        # Create a test result
        result = EnhancementResult(
            original_prompt="a doctor",
            final_prompt="a diverse doctor from various backgrounds",
            initial_bias_score=50.0,
            final_bias_score=75.0,
            initial_diversity_score=45.0,
            final_diversity_score=82.0,
            iterations=3,
            negative_examples=["stereotype1", "stereotype2"],
            diversity_recommendations=["rec1", "rec2", "rec3"],
            bias_threshold_met=True,
            diversity_threshold_met=True
        )

        # Verify fields
        assert result.original_prompt == "a doctor"
        assert result.final_prompt == "a diverse doctor from various backgrounds"
        assert result.iterations == 3
        assert len(result.negative_examples) == 2
        assert len(result.diversity_recommendations) == 3

        # Verify computed properties
        assert result.bias_improvement == 25.0, "Should calculate bias improvement"
        assert result.diversity_improvement == 37.0, "Should calculate diversity improvement"
        assert result.both_thresholds_met == True, "Should check both thresholds"

        print("✅ EnhancementResult validated successfully")
        print("  - Is dataclass: ✓")
        print("  - Can create instances: ✓")
        print(f"  - Bias improvement: {result.bias_improvement}")
        print(f"  - Diversity improvement: {result.diversity_improvement}")
        print(f"  - Both thresholds met: {result.both_thresholds_met}")
        print()
        return True

    except Exception as e:
        print(f"❌ EnhancementResult test failed: {e}")
        print()
        return False


def test_enhancement_system_instantiation():
    """TEST 4: Verify EnhancementSystem can be instantiated."""
    print("=" * 70)
    print("TEST 4: EnhancementSystem Instantiation")
    print("=" * 70)

    try:
        from src.enhancement import EnhancementConfig, EnhancementSystem

        # Create with default config
        system = EnhancementSystem()
        assert system.config is not None, "Should have config"
        assert isinstance(system.config, EnhancementConfig), "Should use EnhancementConfig"

        # Create with custom config
        custom_config = EnhancementConfig(bias_threshold=90)
        custom_system = EnhancementSystem(custom_config)
        assert custom_system.config.bias_threshold == 90

        # Verify methods exist
        assert hasattr(system, "enhance"), "Should have enhance method"
        assert hasattr(system, "batch_enhance"), "Should have batch_enhance method"
        assert hasattr(system, "_score_bias"), "Should have _score_bias method"
        assert hasattr(system, "_score_diversity"), "Should have _score_diversity method"

        print("✅ EnhancementSystem instantiated successfully")
        print(f"  - Config bias threshold: {system.config.bias_threshold}")
        print(f"  - Config diversity threshold: {system.config.diversity_threshold}")
        print("  - All methods present: ✓")
        print()
        return True

    except Exception as e:
        print(f"❌ EnhancementSystem instantiation failed: {e}")
        print()
        return False


def test_enhancement_basic():
    """TEST 5: Verify EnhancementSystem can enhance prompts."""
    print("=" * 70)
    print("TEST 5: EnhancementSystem Basic Enhancement")
    print("=" * 70)

    try:
        from src.enhancement import EnhancementResult, EnhancementSystem

        # Create system
        system = EnhancementSystem()

        # Enhance a prompt
        original = "a doctor"
        result = system.enhance(original)

        # Verify result type
        assert isinstance(result, EnhancementResult), "Should return EnhancementResult"

        # Verify basic fields
        assert result.original_prompt == original, "Should preserve original prompt"
        assert result.final_prompt != "", "Should generate enhanced prompt"
        assert result.iterations > 0, "Should track iterations"

        # Verify scores exist
        assert result.initial_bias_score >= 0, "Should have initial bias score"
        assert result.final_bias_score >= 0, "Should have final bias score"
        assert result.initial_diversity_score >= 0, "Should have initial diversity score"
        assert result.final_diversity_score >= 0, "Should have final diversity score"

        print("✅ Basic enhancement works successfully")
        print(f"  - Original: {result.original_prompt}")
        print(f"  - Enhanced: {result.final_prompt}")
        print(f"  - Bias: {result.initial_bias_score:.1f} → {result.final_bias_score:.1f}")
        print(f"  - Diversity: {result.initial_diversity_score:.1f} → {result.final_diversity_score:.1f}")
        print()
        return True

    except Exception as e:
        print(f"❌ Basic enhancement test failed: {e}")
        print()
        return False


def test_enhancement_scoring():
    """TEST 6: Verify enhancement scoring methods."""
    print("=" * 70)
    print("TEST 6: Enhancement Scoring")
    print("=" * 70)

    try:
        from src.enhancement import EnhancementSystem

        system = EnhancementSystem()

        # Test bias scoring
        basic_prompt = "a person"
        inclusive_prompt = "a diverse person from various backgrounds"

        basic_bias = system._score_bias(basic_prompt)
        inclusive_bias = system._score_bias(inclusive_prompt)

        assert 0 <= basic_bias <= 100, "Bias score should be 0-100"
        assert 0 <= inclusive_bias <= 100, "Bias score should be 0-100"
        assert inclusive_bias >= basic_bias, "Inclusive prompt should score higher on bias"

        # Test diversity scoring
        basic_diversity = system._score_diversity(basic_prompt)
        inclusive_diversity = system._score_diversity(inclusive_prompt)

        assert 0 <= basic_diversity <= 100, "Diversity score should be 0-100"
        assert 0 <= inclusive_diversity <= 100, "Diversity score should be 0-100"
        assert inclusive_diversity >= basic_diversity, "Inclusive prompt should score higher on diversity"

        print("✅ Enhancement scoring validated successfully")
        print(f"  - Bias scoring: {basic_bias:.1f} → {inclusive_bias:.1f}")
        print(f"  - Diversity scoring: {basic_diversity:.1f} → {inclusive_diversity:.1f}")
        print("  - Score ranges valid: ✓")
        print("  - Inclusive > basic: ✓")
        print()
        return True

    except Exception as e:
        print(f"❌ Enhancement scoring test failed: {e}")
        print()
        return False


def test_no_circular_imports():
    """TEST 7: Verify no circular import dependencies."""
    print("=" * 70)
    print("TEST 7: No Circular Imports")
    print("=" * 70)

    try:
        # Test various import combinations to ensure no circular dependencies

        # Enhancement layer imports
        # Enhancement layer can be imported alongside other layers
        from src.config import settings
        from src.data import EmbeddingCache
        from src.enhancement import EnhancementConfig, EnhancementResult, EnhancementSystem
        from src.evaluation import VisualBiasEvaluator
        from src.generation import DALLE3Generator
        from src.knowledge import GraphRAG

        print("✅ No circular import dependencies detected")
        print("  - enhancement layer: OK")
        print("  - enhancement → config: OK")
        print("  - enhancement → data: OK")
        print("  - enhancement → knowledge: OK")
        print("  - enhancement → generation: OK")
        print("  - enhancement → evaluation: OK")
        print("  - All combinations tested successfully")
        print()
        return True

    except ImportError as e:
        print(f"❌ Circular import detected: {e}")
        print()
        return False


def main():
    """Run all Phase 7 integration tests."""
    print()
    print("🧪 " + "=" * 68)
    print("   PHASE 7 INTEGRATION TESTS - ENHANCEMENT LAYER REFACTORING")
    print("=" * 70)
    print()

    tests = [
        ("Enhancement Layer Imports", test_enhancement_layer_imports),
        ("EnhancementConfig Dataclass", test_enhancement_config),
        ("EnhancementResult Dataclass", test_enhancement_result),
        ("EnhancementSystem Instantiation", test_enhancement_system_instantiation),
        ("EnhancementSystem Basic Enhancement", test_enhancement_basic),
        ("Enhancement Scoring", test_enhancement_scoring),
        ("No Circular Imports", test_no_circular_imports),
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
        print("🎉 ALL PHASE 7 TESTS PASSED! 🎉")
        print()
        print("Phase 7 refactoring is complete and functional.")
        print("Enhancement layer successfully created in src/enhancement/.")
        print("Ready to proceed to Phase 8: CLI Layer")
    else:
        print("⚠️  Some tests failed. Please review the errors above.")

    print()
    return 0 if passed_count == total_count else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
