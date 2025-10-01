"""Test configuration system.

Quick test to verify the configuration loading works correctly.
"""

import os

from src.config.settings import (
    Config,
    get_cache_file,
    get_chat_model,
    get_config,
    get_data_path,
    get_embedding_model,
    get_enhancement_param,
    get_image_gen_param,
    get_image_model,
    get_openai_api_key,
)


def test_config_singleton():
    """Test that Config is a singleton."""
    config1 = Config()
    config2 = Config()
    assert config1 is config2, "Config should be a singleton"
    print("✓ Config singleton pattern works")


def test_default_config_loading():
    """Test that default configuration loads correctly."""
    config = get_config()

    # Test OpenAI configuration
    assert config.get("openai.embedding_model") == "text-embedding-3-large"
    assert config.get("openai.chat_model") == "gpt-4o-mini"
    assert config.get("openai.image_model") == "dall-e-3"
    print("✓ OpenAI configuration loaded correctly")

    # Test cache configuration
    assert config.get("cache.embeddings") == "embeddings.pkl"
    assert config.get("cache.stereoset") == "stereoset_embeddings.pkl"
    print("✓ Cache configuration loaded correctly")

    # Test enhancement configuration
    assert config.get("enhancement.bias_threshold") == 75
    assert config.get("enhancement.diversity_threshold") == 80
    assert config.get("enhancement.max_iterations") == 3
    print("✓ Enhancement configuration loaded correctly")

    # Test image generation configuration
    assert config.get("image_generation.dalle.size") == "1024x1024"
    assert config.get("image_generation.dalle.quality") == "standard"
    print("✓ Image generation configuration loaded correctly")


def test_convenience_functions():
    """Test convenience functions for accessing config."""
    assert get_embedding_model() == "text-embedding-3-large"
    assert get_chat_model() == "gpt-4o-mini"
    assert get_image_model() == "dall-e-3"
    print("✓ Convenience functions work correctly")

    assert get_cache_file("embeddings") == "embeddings.pkl"
    assert get_cache_file("stereoset") == "stereoset_embeddings.pkl"
    print("✓ Cache file functions work correctly")

    assert get_data_path("bias_csv") == "data/biases/ADV_GRAPH_20240119 - ADV_GRAPH_20240119.csv"
    print("✓ Data path functions work correctly")

    assert get_enhancement_param("bias_threshold") == 75
    assert get_enhancement_param("max_iterations") == 3
    print("✓ Enhancement param functions work correctly")

    assert get_image_gen_param("dalle.size") == "1024x1024"
    print("✓ Image generation param functions work correctly")


def test_dot_notation():
    """Test dot notation for accessing nested config."""
    config = get_config()

    # Test nested access
    max_retries = config.get("openai.max_retries")
    print(f"  Max retries: {max_retries}")
    assert max_retries == 3, f"Expected 3, got {max_retries}"

    diversity_alpha = config.get("enhancement.diversity_alpha")
    print(f"  Diversity alpha: {diversity_alpha}")
    assert diversity_alpha == 0.6, f"Expected 0.6, got {diversity_alpha}"
    print("✓ Dot notation works for nested config")

    # Test default values
    assert config.get("nonexistent.key", "default") == "default"
    print("✓ Default values work correctly")


def test_env_override():
    """Test that environment variables override config."""
    # Set test environment variable
    os.environ["OPENAI_API_KEY"] = "test-key-123"

    # Reload config to pick up environment variable
    config = get_config()
    config.reload()

    # Test that environment variable is used
    api_key = get_openai_api_key()
    assert api_key == "test-key-123", f"Expected 'test-key-123', got '{api_key}'"
    print("✓ Environment variable override works")

    # Clean up
    del os.environ["OPENAI_API_KEY"]


def test_visual_evaluation_config():
    """Test visual evaluation configuration."""
    config = get_config()

    # Test demographic labels
    race_labels = config.get("visual_evaluation.race_labels")
    print(f"  Race labels: {race_labels}")
    assert race_labels is not None, "Race labels should not be None"
    assert len(race_labels) == 7, f"Expected 7 race labels, got {len(race_labels)}"
    assert "White" in race_labels
    assert "Black" in race_labels
    print("✓ Visual evaluation race labels loaded correctly")

    gender_labels = config.get("visual_evaluation.gender_labels")
    assert len(gender_labels) == 2, f"Expected 2 gender labels, got {len(gender_labels)}"
    assert "Male" in gender_labels
    assert "Female" in gender_labels
    print("✓ Visual evaluation gender labels loaded correctly")

    age_labels = config.get("visual_evaluation.age_labels")
    assert len(age_labels) == 9, f"Expected 9 age labels, got {len(age_labels)}"
    assert "0-2" in age_labels
    assert "60-69" in age_labels
    print("✓ Visual evaluation age labels loaded correctly")


def main():
    """Run all tests."""
    print("Testing Configuration System\n" + "=" * 50)

    try:
        test_config_singleton()
        test_default_config_loading()
        test_convenience_functions()
        test_dot_notation()
        test_env_override()
        test_visual_evaluation_config()

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
