"""Test data layer (embeddings and parsers).

Quick test to verify the data layer modules work correctly.
"""

import os
import tempfile

from src.data import (
    CachedEmbedder,
    EmbeddingCache,
    OpenAIEmbedder,
    Triple,
)


def test_embedding_cache():
    """Test EmbeddingCache functionality."""
    import numpy as np

    # Use temporary file for testing
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
        temp_cache = f.name

    try:
        # Create cache and add embedding
        cache = EmbeddingCache(temp_cache)
        test_text = "hello world"
        test_embedding = np.array([0.1, 0.2, 0.3])

        cache.set(test_text, test_embedding)
        cache.save()

        assert test_text in cache
        assert len(cache) == 1

        # Load cache in new instance
        cache2 = EmbeddingCache(temp_cache)
        retrieved = cache2.get(test_text)

        assert retrieved is not None
        assert np.allclose(retrieved, test_embedding)

        print("✓ EmbeddingCache works correctly")

    finally:
        # Cleanup
        if os.path.exists(temp_cache):
            os.remove(temp_cache)


def test_openai_embedder():
    """Test OpenAIEmbedder initialization."""
    # Test without API key (should handle gracefully)
    embedder = OpenAIEmbedder(api_key=None, model="text-embedding-3-large")

    # Check that it reports availability correctly
    has_key = os.environ.get("OPENAI_API_KEY") is not None
    assert embedder.is_available() == has_key

    print(f"✓ OpenAIEmbedder initialization works (API available: {has_key})")


def test_cached_embedder():
    """Test CachedEmbedder initialization."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
        temp_cache = f.name

    try:
        embedder = CachedEmbedder(cache_file=temp_cache)

        assert embedder is not None
        assert len(embedder) == 0  # Empty cache initially

        print("✓ CachedEmbedder initialization works")

    finally:
        if os.path.exists(temp_cache):
            os.remove(temp_cache)


def test_triple_type():
    """Test Triple type definition."""
    # Triple should be a tuple of 3 strings
    triple: Triple = ("subject", "predicate", "object")

    assert len(triple) == 3
    assert all(isinstance(s, str) for s in triple)

    print("✓ Triple type definition works")


def test_imports():
    """Test that all expected exports are available."""
    from src.data import (
        BaseDataParser,
        BiasCSVParser,
        CachedEmbedder,
        CulturalTriplesParser,
        DataParserFactory,
        EmbeddingCache,
        OpenAIEmbedder,
        StereoSetParser,
        StereoSetRecord,
        Triple,
    )

    # Check that classes are importable
    assert EmbeddingCache is not None
    assert OpenAIEmbedder is not None
    assert CachedEmbedder is not None
    assert BaseDataParser is not None
    assert BiasCSVParser is not None
    assert CulturalTriplesParser is not None
    assert DataParserFactory is not None
    assert StereoSetParser is not None
    assert StereoSetRecord is not None
    assert Triple is not None

    print("✓ All data layer exports are importable")


def main():
    """Run all tests."""
    print("Testing Data Layer (Embeddings & Parsers)\n" + "=" * 50)

    try:
        test_triple_type()
        test_imports()
        test_embedding_cache()
        test_openai_embedder()
        test_cached_embedder()

        print("\n" + "=" * 50)
        print("✓ All tests passed!")
        return True
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
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
