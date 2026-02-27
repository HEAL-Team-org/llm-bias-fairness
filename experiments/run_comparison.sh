#!/bin/bash
# Quick runner script for PreciseDebias comparison test

echo "=========================================="
echo "PreciseDebias vs Our Approach Comparison"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "config/config.yaml" ]; then
    echo "Error: Must run from project root directory"
    echo "Current directory: $(pwd)"
    exit 1
fi

echo "✓ Running from project root"
echo ""

# Check if config exists
if [ ! -f "config/config.yaml" ]; then
    echo "Error: config/config.yaml not found"
    echo "Please set up your configuration first"
    exit 1
fi

echo "✓ Configuration file found"
echo ""

# Check if required data files exist
if [ ! -f "data/biases/ADV_GRAPH_20240119 - ADV_GRAPH_20240119.csv" ]; then
    echo "Error: Bias data file not found"
    exit 1
fi

echo "✓ Required data files found"
echo ""

# Create experiments directory if it doesn't exist
mkdir -p experiments/comparison_results/images

echo "Starting comparison test..."
echo ""
echo "This will:"
echo "  1. Test baseline (no enhancement)"
echo "  2. Test PreciseDebias approach (5 variations per prompt)"
echo "  3. Test our GraphRAG approach (1 enhanced prompt)"
echo "  4. Generate images with gpt-image-1"
echo "  5. Save results to experiments/comparison_results/"
echo ""

# Run the test
python3 experiments/test_precisedebias_comparison.py

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Test completed successfully!"
    echo "=========================================="
    echo ""
    echo "Results saved to:"
    echo "  - experiments/comparison_results/comparison_results.json"
    echo "  - experiments/comparison_results/images/"
    echo ""
    echo "To view results:"
    echo "  cat experiments/comparison_results/comparison_results.json | jq"
    echo "  ls -lh experiments/comparison_results/images/"
else
    echo ""
    echo "=========================================="
    echo "Test failed!"
    echo "=========================================="
    echo ""
    echo "Check the error messages above"
    exit 1
fi
