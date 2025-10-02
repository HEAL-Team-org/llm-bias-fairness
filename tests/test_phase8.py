#!/usr/bin/env python3
"""Phase 8 Integration Tests - CLI Layer Refactoring.

This test suite validates that the CLI layer has been properly
created in src/cli/ with clean imports and no circular dependencies.

Tests:
    1. CLI Layer Imports - All CLI classes accessible
    2. BaseCommand Structure - Base command class working
    3. EnhanceCommand Exists - Enhance command available
    4. BatchCommand Exists - Batch command available
    5. QueryCommand Exists - Query command available
    6. TestCommand Exists - Test command available
    7. CLI Parser Creation - Parser can be created
    8. CLI Runner Creation - Runner can be instantiated
    9. No Circular Imports - Clean architecture validation
"""


def test_cli_layer_imports():
    """TEST 1: Verify all CLI layer classes can be imported."""
    print("=" * 70)
    print("TEST 1: CLI Layer Imports")
    print("=" * 70)

    try:
        from src.cli import (
            BatchCommand,
            CLIRunner,
            EnhanceCommand,
            QueryCommand,
            TestCommand,
            create_cli_parser,
        )

        print("✅ All CLI layer classes imported successfully:")
        print("  - CLIRunner")
        print("  - create_cli_parser")
        print("  - EnhanceCommand")
        print("  - BatchCommand")
        print("  - QueryCommand")
        print("  - TestCommand")
        print()
        return True

    except ImportError as e:
        print(f"❌ CLI layer import failed: {e}")
        print()
        return False


def test_base_command():
    """TEST 2: Verify BaseCommand class structure."""
    print("=" * 70)
    print("TEST 2: BaseCommand Structure")
    print("=" * 70)

    try:
        from abc import ABC

        from src.cli.commands import BaseCommand

        # Verify it's an abstract base class
        assert issubclass(BaseCommand, ABC), "Should be an ABC"

        # Verify required methods exist
        assert hasattr(BaseCommand, "add_arguments"), "Should have add_arguments method"
        assert hasattr(BaseCommand, "execute"), "Should have execute method"
        assert hasattr(BaseCommand, "setup"), "Should have setup method"
        assert hasattr(BaseCommand, "cleanup"), "Should have cleanup method"

        print("✅ BaseCommand structure validated successfully")
        print("  - Is ABC: ✓")
        print("  - Has add_arguments: ✓")
        print("  - Has execute: ✓")
        print("  - Has setup: ✓")
        print("  - Has cleanup: ✓")
        print()
        return True

    except Exception as e:
        print(f"❌ BaseCommand test failed: {e}")
        print()
        return False


def test_enhance_command():
    """TEST 3: Verify EnhanceCommand exists and is a BaseCommand."""
    print("=" * 70)
    print("TEST 3: EnhanceCommand Exists")
    print("=" * 70)

    try:
        from src.cli import EnhanceCommand
        from src.cli.commands import BaseCommand

        # Verify it's a subclass of BaseCommand
        assert issubclass(EnhanceCommand, BaseCommand), "Should inherit from BaseCommand"

        # Create instance
        command = EnhanceCommand()
        assert command is not None, "Should be instantiable"

        # Verify methods exist
        assert hasattr(command, "add_arguments"), "Should have add_arguments"
        assert hasattr(command, "execute"), "Should have execute"

        print("✅ EnhanceCommand validated successfully")
        print("  - Inherits from BaseCommand: ✓")
        print("  - Can be instantiated: ✓")
        print("  - Has required methods: ✓")
        print()
        return True

    except Exception as e:
        print(f"❌ EnhanceCommand test failed: {e}")
        print()
        return False


def test_batch_command():
    """TEST 4: Verify BatchCommand exists and is a BaseCommand."""
    print("=" * 70)
    print("TEST 4: BatchCommand Exists")
    print("=" * 70)

    try:
        from src.cli import BatchCommand
        from src.cli.commands import BaseCommand

        assert issubclass(BatchCommand, BaseCommand), "Should inherit from BaseCommand"

        command = BatchCommand()
        assert command is not None, "Should be instantiable"
        assert hasattr(command, "add_arguments"), "Should have add_arguments"
        assert hasattr(command, "execute"), "Should have execute"

        print("✅ BatchCommand validated successfully")
        print("  - Inherits from BaseCommand: ✓")
        print("  - Can be instantiated: ✓")
        print("  - Has required methods: ✓")
        print()
        return True

    except Exception as e:
        print(f"❌ BatchCommand test failed: {e}")
        print()
        return False


def test_query_command():
    """TEST 5: Verify QueryCommand exists and is a BaseCommand."""
    print("=" * 70)
    print("TEST 5: QueryCommand Exists")
    print("=" * 70)

    try:
        from src.cli import QueryCommand
        from src.cli.commands import BaseCommand

        assert issubclass(QueryCommand, BaseCommand), "Should inherit from BaseCommand"

        command = QueryCommand()
        assert command is not None, "Should be instantiable"
        assert hasattr(command, "add_arguments"), "Should have add_arguments"
        assert hasattr(command, "execute"), "Should have execute"

        print("✅ QueryCommand validated successfully")
        print("  - Inherits from BaseCommand: ✓")
        print("  - Can be instantiated: ✓")
        print("  - Has required methods: ✓")
        print()
        return True

    except Exception as e:
        print(f"❌ QueryCommand test failed: {e}")
        print()
        return False


def test_test_command():
    """TEST 6: Verify TestCommand exists and is a BaseCommand."""
    print("=" * 70)
    print("TEST 6: TestCommand Exists")
    print("=" * 70)

    try:
        from src.cli import TestCommand
        from src.cli.commands import BaseCommand

        assert issubclass(TestCommand, BaseCommand), "Should inherit from BaseCommand"

        command = TestCommand()
        assert command is not None, "Should be instantiable"
        assert hasattr(command, "add_arguments"), "Should have add_arguments"
        assert hasattr(command, "execute"), "Should have execute"

        print("✅ TestCommand validated successfully")
        print("  - Inherits from BaseCommand: ✓")
        print("  - Can be instantiated: ✓")
        print("  - Has required methods: ✓")
        print()
        return True

    except Exception as e:
        print(f"❌ TestCommand test failed: {e}")
        print()
        return False


def test_cli_parser():
    """TEST 7: Verify CLI parser can be created."""
    print("=" * 70)
    print("TEST 7: CLI Parser Creation")
    print("=" * 70)

    try:
        import argparse

        from src.cli import create_cli_parser

        # Create parser
        parser = create_cli_parser()
        assert parser is not None, "Should create parser"
        assert isinstance(parser, argparse.ArgumentParser), "Should be ArgumentParser"

        # Verify subcommands exist
        # Parse help to see subcommands (without errors)
        help_text = parser.format_help()
        assert "enhance" in help_text, "Should have enhance command"
        assert "batch" in help_text, "Should have batch command"
        assert "query" in help_text, "Should have query command"
        assert "test" in help_text, "Should have test command"

        print("✅ CLI parser created successfully")
        print("  - Parser created: ✓")
        print("  - Has enhance command: ✓")
        print("  - Has batch command: ✓")
        print("  - Has query command: ✓")
        print("  - Has test command: ✓")
        print()
        return True

    except Exception as e:
        print(f"❌ CLI parser test failed: {e}")
        print()
        return False


def test_cli_runner():
    """TEST 8: Verify CLI runner can be created."""
    print("=" * 70)
    print("TEST 8: CLI Runner Creation")
    print("=" * 70)

    try:
        from src.cli import CLIRunner

        # Create runner
        runner = CLIRunner()
        assert runner is not None, "Should create runner"

        # Verify methods exist
        assert hasattr(runner, "run"), "Should have run method"
        assert hasattr(runner, "main"), "Should have main method"
        assert hasattr(runner, "setup_logging"), "Should have setup_logging method"

        # Verify commands are registered
        assert "enhance" in runner.commands, "Should have enhance command"
        assert "batch" in runner.commands, "Should have batch command"
        assert "query" in runner.commands, "Should have query command"
        assert "test" in runner.commands, "Should have test command"

        print("✅ CLI runner created successfully")
        print("  - Runner created: ✓")
        print("  - Has run method: ✓")
        print("  - Has main method: ✓")
        print("  - Commands registered: ✓")
        print()
        return True

    except Exception as e:
        print(f"❌ CLI runner test failed: {e}")
        print()
        return False


def test_no_circular_imports():
    """TEST 9: Verify no circular import dependencies."""
    print("=" * 70)
    print("TEST 9: No Circular Imports")
    print("=" * 70)

    try:
        # Test various import combinations to ensure no circular dependencies

        # CLI layer imports
        from src.cli import CLIRunner, create_cli_parser
        from src.cli.commands import BaseCommand
        from src.cli.commands.batch import BatchCommand
        from src.cli.commands.enhance import EnhanceCommand
        from src.cli.commands.query import QueryCommand
        from src.cli.commands.test import TestCommand

        # CLI layer can be imported alongside other layers
        from src.config import settings
        from src.data import EmbeddingCache
        from src.enhancement import EnhancementSystem
        from src.evaluation import VisualBiasEvaluator
        from src.generation import DALLE3Generator
        from src.knowledge import GraphRAG

        print("✅ No circular import dependencies detected")
        print("  - cli layer: OK")
        print("  - cli → config: OK")
        print("  - cli → data: OK")
        print("  - cli → knowledge: OK")
        print("  - cli → generation: OK")
        print("  - cli → evaluation: OK")
        print("  - cli → enhancement: OK")
        print("  - All combinations tested successfully")
        print()
        return True

    except ImportError as e:
        print(f"❌ Circular import detected: {e}")
        print()
        return False


def main():
    """Run all Phase 8 integration tests."""
    print()
    print("🧪 " + "=" * 68)
    print("   PHASE 8 INTEGRATION TESTS - CLI LAYER REFACTORING")
    print("=" * 70)
    print()

    tests = [
        ("CLI Layer Imports", test_cli_layer_imports),
        ("BaseCommand Structure", test_base_command),
        ("EnhanceCommand Exists", test_enhance_command),
        ("BatchCommand Exists", test_batch_command),
        ("QueryCommand Exists", test_query_command),
        ("TestCommand Exists", test_test_command),
        ("CLI Parser Creation", test_cli_parser),
        ("CLI Runner Creation", test_cli_runner),
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
        print("🎉 ALL PHASE 8 TESTS PASSED! 🎉")
        print()
        print("Phase 8 refactoring is complete and functional.")
        print("CLI layer successfully created in src/cli/.")
        print("Ready to proceed to Phase 9: Documentation Update")
    else:
        print("⚠️  Some tests failed. Please review the errors above.")

    print()
    return 0 if passed_count == total_count else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
