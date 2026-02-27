#!/usr/bin/env python3
"""Phase 5 Integration Tests - Generation Layer Refactoring.

This test suite validates that the generation layer has been properly
separated into src/generation/ with clean imports and no circular dependencies.

Tests:
    1. Generation Layer Imports - All generator classes accessible
    2. BaseImageGenerator Import - Abstract base class available
    3. DALLE3Generator Import - DALL-E 3 implementation available
    4. MockImageGenerator Import - Testing implementation available
    5. Factory Function Import - create_image_generator() available
    6. ImageGenerationError Import - Custom exception available
    7. No Circular Imports - Clean architecture validation
"""


def test_generation_layer_imports():
    """TEST 1: Verify all generation layer classes can be imported."""
    print("=" * 70)
    print("TEST 1: Generation Layer Imports")
    print("=" * 70)

    try:
        from src.generation import (
            BaseImageGenerator,
            DALLE3Generator,
            ImageGenerationError,
            MockImageGenerator,
            create_image_generator,
        )

        print("✅ All generation layer classes imported successfully:")
        print("  - BaseImageGenerator")
        print("  - DALLE3Generator")
        print("  - MockImageGenerator")
        print("  - ImageGenerationError")
        print("  - create_image_generator")
        print()
        return True

    except ImportError as e:
        print(f"❌ Generation layer import failed: {e}")
        print()
        return False


def test_base_image_generator():
    """TEST 2: Verify BaseImageGenerator is an abstract base class."""
    print("=" * 70)
    print("TEST 2: BaseImageGenerator Abstract Class")
    print("=" * 70)

    try:
        from abc import ABC

        from src.generation import BaseImageGenerator

        # Verify it's an ABC
        assert issubclass(BaseImageGenerator, ABC), "BaseImageGenerator should inherit from ABC"

        # Verify it has the required abstract method
        assert hasattr(BaseImageGenerator, "generate_image"), "Should have generate_image method"

        # Verify it has helper methods
        assert hasattr(BaseImageGenerator, "_generate_filename"), "Should have _generate_filename method"
        assert hasattr(BaseImageGenerator, "_save_image"), "Should have _save_image method"

        print("✅ BaseImageGenerator validated successfully")
        print("  - Inherits from ABC")
        print("  - Has abstract generate_image() method")
        print("  - Has _generate_filename() helper")
        print("  - Has _save_image() helper")
        print()
        return True

    except Exception as e:
        print(f"❌ BaseImageGenerator validation failed: {e}")
        print()
        return False


def test_dalle3_generator():
    """TEST 3: Verify DALLE3Generator can be instantiated."""
    print("=" * 70)
    print("TEST 3: DALLE3Generator Instantiation")
    print("=" * 70)

    try:
        from src.generation import BaseImageGenerator, DALLE3Generator

        # Verify it's a subclass of BaseImageGenerator
        assert issubclass(DALLE3Generator, BaseImageGenerator), \
            "DALLE3Generator should inherit from BaseImageGenerator"

        # Try to instantiate (will warn about missing API key, but that's OK)
        generator = DALLE3Generator(api_key="test_key")

        # Verify attributes
        assert hasattr(generator, "client"), "Should have OpenAI client"
        assert hasattr(generator, "model"), "Should have model attribute"
        assert hasattr(generator, "size"), "Should have size attribute"
        assert hasattr(generator, "quality"), "Should have quality attribute"
        assert hasattr(generator, "output_dir"), "Should have output_dir attribute"

        print("✅ DALLE3Generator instantiated successfully")
        print(f"  - Model: {generator.model}")
        print(f"  - Size: {generator.size}")
        print(f"  - Quality: {generator.quality}")
        print(f"  - Output dir: {generator.output_dir}")
        print()
        return True

    except Exception as e:
        print(f"❌ DALLE3Generator instantiation failed: {e}")
        print()
        return False


def test_mock_generator():
    """TEST 4: Verify MockImageGenerator works without API."""
    print("=" * 70)
    print("TEST 4: MockImageGenerator Testing")
    print("=" * 70)

    try:
        import tempfile
        from pathlib import Path

        from src.generation import BaseImageGenerator, MockImageGenerator

        # Verify it's a subclass of BaseImageGenerator
        assert issubclass(MockImageGenerator, BaseImageGenerator), \
            "MockImageGenerator should inherit from BaseImageGenerator"

        # Create a temporary directory for test images
        with tempfile.TemporaryDirectory() as tmpdir:
            # Instantiate the mock generator
            generator = MockImageGenerator(output_dir=tmpdir)

            # Generate a test image
            filepath, metadata = generator.generate_image("test prompt")

            # Verify the image was created
            assert Path(filepath).exists(), f"Image file should exist: {filepath}"
            assert metadata["model"] == "mock_generator", "Should use mock_generator model"
            assert metadata["original_prompt"] == "test prompt", "Should store original prompt"

            print("✅ MockImageGenerator works successfully")
            print(f"  - Output directory: {generator.output_dir}")
            print(f"  - Generated image: {Path(filepath).name}")
            print(f"  - Image size: {metadata['size']}")
            print(f"  - Generation count: {generator.generation_count}")

        print()
        return True

    except Exception as e:
        print(f"❌ MockImageGenerator test failed: {e}")
        print()
        return False


def test_factory_function():
    """TEST 5: Verify create_image_generator factory function."""
    print("=" * 70)
    print("TEST 5: Factory Function (create_image_generator)")
    print("=" * 70)

    try:
        import tempfile

        from src.generation import (
            DALLE3Generator,
            MockImageGenerator,
            create_image_generator,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test creating a mock generator
            mock_gen = create_image_generator("mock", output_dir=tmpdir)
            assert isinstance(mock_gen, MockImageGenerator), "Should create MockImageGenerator"

            # Test creating a DALL-E 3 generator
            dalle_gen = create_image_generator("dalle3", api_key="test", output_dir=tmpdir)
            assert isinstance(dalle_gen, DALLE3Generator), "Should create DALLE3Generator"

            # Test invalid generator type
            try:
                create_image_generator("invalid_type")
                assert False, "Should raise ValueError for invalid type"
            except ValueError as e:
                assert "Unsupported generator type" in str(e), "Should have proper error message"

        print("✅ Factory function works correctly")
        print("  - Creates MockImageGenerator for 'mock'")
        print("  - Creates DALLE3Generator for 'dalle3'")
        print("  - Raises ValueError for invalid types")
        print()
        return True

    except Exception as e:
        print(f"❌ Factory function test failed: {e}")
        print()
        return False


def test_image_generation_error():
    """TEST 6: Verify ImageGenerationError exception class."""
    print("=" * 70)
    print("TEST 6: ImageGenerationError Exception")
    print("=" * 70)

    try:
        from src.generation import ImageGenerationError

        # Verify it's an exception class
        assert issubclass(ImageGenerationError, Exception), \
            "ImageGenerationError should inherit from Exception"

        # Test raising and catching the exception
        try:
            raise ImageGenerationError("Test error message")
        except ImageGenerationError as e:
            assert str(e) == "Test error message", "Should preserve error message"

        print("✅ ImageGenerationError validated successfully")
        print("  - Inherits from Exception")
        print("  - Can be raised and caught")
        print("  - Preserves error messages")
        print()
        return True

    except Exception as e:
        print(f"❌ ImageGenerationError test failed: {e}")
        print()
        return False


def test_no_circular_imports():
    """TEST 7: Verify no circular import dependencies."""
    print("=" * 70)
    print("TEST 7: No Circular Imports")
    print("=" * 70)

    try:
        # Test various import combinations to ensure no circular dependencies

        # Generation layer imports
        # Generation layer can be imported alongside other layers
        from src.config import settings
        from src.data import EmbeddingCache
        from src.generation import (
            BaseImageGenerator,
            DALLE3Generator,
            MockImageGenerator,
            create_image_generator,
        )
        from src.knowledge import GraphRAG

        print("✅ No circular import dependencies detected")
        print("  - generation layer: OK")
        print("  - generation → config: OK")
        print("  - generation → data: OK")
        print("  - generation → knowledge: OK")
        print("  - All combinations tested successfully")
        print()
        return True

    except ImportError as e:
        print(f"❌ Circular import detected: {e}")
        print()
        return False


def test_batch_processor_integration():
    """TEST 8: Verify batch_processor uses new imports."""
    print("=" * 70)
    print("TEST 8: Batch Processor Integration")
    print("=" * 70)

    try:
        # Read batch_processor.py and check it imports from src.generation
        from pathlib import Path

        batch_processor_path = Path("scripts/batch_processor.py")
        if not batch_processor_path.exists():
            print("⚠️  batch_processor.py not found, skipping test")
            print()
            return True

        content = batch_processor_path.read_text()

        # Check for new import
        assert "from src.generation import" in content, \
            "batch_processor should import from src.generation"

        # Check old import is removed
        assert "from src.image_generator import" not in content, \
            "batch_processor should not import from old src.image_generator"

        print("✅ Batch processor integration validated")
        print("  - Uses src.generation imports: ✓")
        print("  - No old src.image_generator imports: ✓")
        print()
        return True

    except Exception as e:
        print(f"❌ Batch processor integration failed: {e}")
        print()
        return False


def main():
    """Run all Phase 5 integration tests."""
    print()
    print("🧪 " + "=" * 68)
    print("   PHASE 5 INTEGRATION TESTS - GENERATION LAYER REFACTORING")
    print("=" * 70)
    print()

    tests = [
        ("Generation Layer Imports", test_generation_layer_imports),
        ("BaseImageGenerator Abstract Class", test_base_image_generator),
        ("DALLE3Generator Instantiation", test_dalle3_generator),
        ("MockImageGenerator Testing", test_mock_generator),
        ("Factory Function", test_factory_function),
        ("ImageGenerationError Exception", test_image_generation_error),
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
        print("🎉 ALL PHASE 5 TESTS PASSED! 🎉")
        print()
        print("Phase 5 refactoring is complete and functional.")
        print("Generation layer successfully separated into src/generation/.")
        print("Ready to proceed to Phase 6: Evaluation Layer")
    else:
        print("⚠️  Some tests failed. Please review the errors above.")

    print()
    return 0 if passed_count == total_count else 1


if __name__ == "__main__":
    exit(main())
