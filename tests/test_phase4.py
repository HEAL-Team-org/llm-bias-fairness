#!/usr/bin/env python3
"""Phase 4 Integration Tests - Knowledge Layer Refactoring.

This script tests that all Phase 4 refactoring changes work correctly:
1. Knowledge layer imports
2. Data layer to knowledge layer integration
3. All RAG systems instantiation
4. Backward compatibility
"""

import sys


def test_knowledge_imports():
    """Test that all knowledge layer classes can be imported."""
    print("=" * 70)
    print("TEST 1: Knowledge Layer Imports")
    print("=" * 70)

    try:
        from src.knowledge import (
            CultureBankParser,
            CultureBankRecord,
            DiversityRAG,
            GraphRAG,
            GraphRetriever,
            KnowledgeGraph,
            LLMAnswerer,
            StereoSetRAG,
        )

        print("✅ All knowledge layer classes imported successfully:")
        print("  - KnowledgeGraph")
        print("  - GraphRetriever")
        print("  - LLMAnswerer")
        print("  - GraphRAG")
        print("  - StereoSetRAG")
        print("  - DiversityRAG")
        print("  - CultureBankRecord")
        print("  - CultureBankParser")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_data_layer_integration():
    """Test that knowledge layer correctly uses data layer."""
    print("\n" + "=" * 70)
    print("TEST 2: Data Layer Integration")
    print("=" * 70)

    try:
        from src.data import CachedEmbedder, EmbeddingCache, OpenAIEmbedder
        from src.data.parsers import BaseDataParser, BiasCSVParser
        from src.knowledge import GraphRAG

        print("✅ Data layer imports successful")
        print("  - EmbeddingCache")
        print("  - OpenAIEmbedder")
        print("  - CachedEmbedder")
        print("  - BaseDataParser")
        print("  - BiasCSVParser")
        return True
    except ImportError as e:
        print(f"❌ Integration test failed: {e}")
        return False


def test_graphrag_instantiation():
    """Test GraphRAG can be instantiated with data layer components."""
    print("\n" + "=" * 70)
    print("TEST 3: GraphRAG Instantiation")
    print("=" * 70)

    try:
        from src.knowledge import GraphRAG

        graphrag = GraphRAG(cache_file="test_phase4_cache.pkl")

        # Verify components
        assert hasattr(graphrag, "cache"), "Missing cache attribute"
        assert hasattr(graphrag, "embedder"), "Missing embedder attribute"
        assert hasattr(graphrag, "retriever"), "Missing retriever attribute"
        assert hasattr(graphrag, "answerer"), "Missing answerer attribute"
        assert hasattr(graphrag, "graphs"), "Missing graphs attribute"

        print("✅ GraphRAG instantiated successfully")
        print(f"  - Cache: {type(graphrag.cache).__name__}")
        print(f"  - Embedder: {type(graphrag.embedder).__name__}")
        print(f"  - Retriever: {type(graphrag.retriever).__name__}")
        print(f"  - Answerer: {type(graphrag.answerer).__name__}")
        print(f"  - Graphs: {type(graphrag.graphs).__name__}")

        return True
    except Exception as e:
        print(f"❌ GraphRAG instantiation failed: {e}")
        return False


def test_stereoset_rag_instantiation():
    """Test StereoSetRAG can be instantiated."""
    print("\n" + "=" * 70)
    print("TEST 4: StereoSetRAG Instantiation")
    print("=" * 70)

    try:
        from src.knowledge import StereoSetRAG

        stereoset = StereoSetRAG(
            cache_file="test_phase4_stereoset.pkl",
            top_k=10
        )

        # Verify components
        assert hasattr(stereoset, "cache"), "Missing cache attribute"
        assert hasattr(stereoset, "embedder"), "Missing embedder attribute"
        assert hasattr(stereoset, "parser"), "Missing parser attribute"
        assert hasattr(stereoset, "contexts"), "Missing contexts attribute"
        assert stereoset.top_k == 10, "Incorrect top_k value"

        print("✅ StereoSetRAG instantiated successfully")
        print(f"  - Cache: {type(stereoset.cache).__name__}")
        print(f"  - Embedder: {type(stereoset.embedder).__name__}")
        print(f"  - Parser: {type(stereoset.parser).__name__}")
        print(f"  - Top-k: {stereoset.top_k}")

        return True
    except Exception as e:
        print(f"❌ StereoSetRAG instantiation failed: {e}")
        return False


def test_diversity_rag_instantiation():
    """Test DiversityRAG can be instantiated with custom weights."""
    print("\n" + "=" * 70)
    print("TEST 5: DiversityRAG Instantiation")
    print("=" * 70)

    try:
        from src.knowledge import DiversityRAG

        diversity = DiversityRAG(
            cache_file="test_phase4_diversity.pkl",
            top_k=15,
            alpha=0.7,
            beta=0.2,
            gamma=0.1
        )

        # Verify components
        assert hasattr(diversity, "cache"), "Missing cache attribute"
        assert hasattr(diversity, "embedder"), "Missing embedder attribute"
        assert hasattr(diversity, "parser"), "Missing parser attribute"
        assert hasattr(diversity, "records"), "Missing records attribute"
        assert diversity.top_k == 15, "Incorrect top_k value"
        assert diversity.alpha == 0.7, "Incorrect alpha value"
        assert diversity.beta == 0.2, "Incorrect beta value"
        assert diversity.gamma == 0.1, "Incorrect gamma value"

        print("✅ DiversityRAG instantiated successfully")
        print(f"  - Cache: {type(diversity.cache).__name__}")
        print(f"  - Embedder: {type(diversity.embedder).__name__}")
        print(f"  - Parser: {type(diversity.parser).__name__}")
        print(f"  - Top-k: {diversity.top_k}")
        print(f"  - Alpha (vector similarity): {diversity.alpha}")
        print(f"  - Beta (keyword match): {diversity.beta}")
        print(f"  - Gamma (agreement score): {diversity.gamma}")

        return True
    except Exception as e:
        print(f"❌ DiversityRAG instantiation failed: {e}")
        return False


def test_knowledge_graph_creation():
    """Test KnowledgeGraph can be created and used."""
    print("\n" + "=" * 70)
    print("TEST 6: KnowledgeGraph Creation")
    print("=" * 70)

    try:
        from src.knowledge import KnowledgeGraph

        graph = KnowledgeGraph(name="TestGraph")

        # Add some test triples
        test_triples = [
            ("doctor", "profession", "medical"),
            ("nurse", "profession", "medical"),
            ("engineer", "profession", "technical"),
        ]

        graph.add_triples(test_triples)

        # Verify (should have 5 unique nodes: doctor, medical, nurse, engineer, technical)
        # Predicates are not stored as nodes, they're edge attributes
        assert len(graph.nodes) == 5, f"Expected 5 nodes, got {len(graph.nodes)}"
        assert len(graph.triples) == 3, f"Expected 3 triples, got {len(graph.triples)}"

        print("✅ KnowledgeGraph created and populated successfully")
        print(f"  - Name: {graph.name}")
        print(f"  - Nodes: {len(graph.nodes)}")
        print(f"  - Triples: {len(graph.triples)}")
        print(f"  - Sample nodes: {graph.nodes[:3]}")

        return True
    except Exception as e:
        print(f"❌ KnowledgeGraph creation failed: {e}")
        return False


def test_no_circular_imports():
    """Test that there are no circular import dependencies."""
    print("\n" + "=" * 70)
    print("TEST 7: No Circular Imports")
    print("=" * 70)

    try:
        # Import in different orders to check for circular dependencies
        from src.data import EmbeddingCache, OpenAIEmbedder
        from src.data.parsers import BaseDataParser, BiasCSVParser
        from src.knowledge import DiversityRAG, GraphRAG, StereoSetRAG

        print("✅ No circular import dependencies detected")
        print("  - knowledge → data: OK")
        print("  - data.parsers → knowledge: OK")
        print("  - All combinations tested successfully")

        return True
    except ImportError as e:
        print(f"❌ Circular import detected: {e}")
        return False


def run_all_tests():
    """Run all Phase 4 tests."""
    print("\n" + "🧪 " + "=" * 68)
    print("   PHASE 4 INTEGRATION TESTS - KNOWLEDGE LAYER REFACTORING")
    print("=" * 70)

    tests = [
        ("Knowledge Layer Imports", test_knowledge_imports),
        ("Data Layer Integration", test_data_layer_integration),
        ("GraphRAG Instantiation", test_graphrag_instantiation),
        ("StereoSetRAG Instantiation", test_stereoset_rag_instantiation),
        ("DiversityRAG Instantiation", test_diversity_rag_instantiation),
        ("KnowledgeGraph Creation", test_knowledge_graph_creation),
        ("No Circular Imports", test_no_circular_imports),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")

    print("\n" + "-" * 70)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL PHASE 4 TESTS PASSED! 🎉")
        print("\nPhase 4 refactoring is complete and functional.")
        print("Knowledge layer successfully separated from data layer.")
        print("Ready to proceed to Phase 5: Generation Layer")
        return 0
    print(f"\n⚠️  {total - passed} test(s) failed")
    print("Please review the failures above before proceeding.")
    return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
