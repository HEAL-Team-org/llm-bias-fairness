"""Test prompt template loader.

Quick test to verify prompt templates load and format correctly.
"""

from src.utils.prompt_loader import (
    get_nested_prompt,
    get_prompt,
    get_prompt_manager,
    initialize_prompts,
    load_prompts,
)


def test_load_enhancement_prompts():
    """Test loading enhancement prompt templates."""
    manager = get_prompt_manager()
    load_prompts("enhancement_prompts.yaml")

    # Check that templates loaded
    assert manager.has_template("base_enhancement_prompt")
    assert manager.has_template("bias_mitigation_section")
    assert manager.has_template("diversity_enhancement_section")
    assert manager.has_template("focus_descriptions")

    print("✓ Enhancement prompts loaded successfully")
    print(f"  Loaded templates: {len(manager.list_templates())}")


def test_load_query_prompts():
    """Test loading query prompt templates."""
    manager = get_prompt_manager()
    load_prompts("query_prompts.yaml")

    # Check that templates loaded
    assert manager.has_template("graphrag_answer_prompt")
    assert manager.has_template("bias_detection_query")
    assert manager.has_template("triple_format")

    print("✓ Query prompts loaded successfully")


def test_format_simple_template():
    """Test formatting a simple template."""
    prompt = get_prompt(
        "base_enhancement_prompt",
        focus_description="bias mitigation",
        original_prompt="a doctor",
        iteration=1
    )

    assert "bias mitigation" in prompt
    assert "a doctor" in prompt
    assert "ITERATION: 1" in prompt

    print("✓ Simple template formatting works")
    print(f"  Sample output: {prompt[:100]}...")


def test_format_nested_template():
    """Test formatting a nested template."""
    focus = get_nested_prompt("focus_descriptions", "bias_only")

    assert "bias mitigation" in focus
    assert "stereotype" in focus

    print("✓ Nested template formatting works")
    print(f"  Focus description: {focus}")


def test_triple_formatting():
    """Test triple formatting template."""
    triple = get_prompt(
        "triple_format",
        subject="doctor",
        predicate="treats",
        object="patient"
    )

    assert "doctor" in triple
    assert "treats" in triple
    assert "patient" in triple

    print("✓ Triple formatting works")
    print(f"  Formatted triple: {triple}")


def test_initialize_all():
    """Test initializing all default prompts."""
    initialize_prompts()

    manager = get_prompt_manager()
    templates = manager.list_templates()

    assert len(templates) > 10, f"Expected > 10 templates, got {len(templates)}"

    print("✓ Default prompt initialization works")
    print(f"  Total templates loaded: {len(templates)}")


def test_missing_variable():
    """Test handling of missing variables in templates."""
    prompt = get_prompt(
        "base_enhancement_prompt",
        focus_description="test"
        # Missing: original_prompt, iteration
    )

    # Should still return the template (with placeholders)
    assert prompt
    assert "focus_description" in prompt or "test" in prompt

    print("✓ Missing variable handling works")


def main():
    """Run all tests."""
    print("Testing Prompt Template Loader\n" + "=" * 50)

    try:
        test_load_enhancement_prompts()
        test_load_query_prompts()
        test_format_simple_template()
        test_format_nested_template()
        test_triple_formatting()
        test_initialize_all()
        test_missing_variable()

        print("\n" + "=" * 50)
        print("✓ All tests passed!")
        return True
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
