#!/bin/bash
# Example usage script for precised image generation

echo "==================================================================="
echo "Precised Image Generation - Usage Examples"
echo "==================================================================="
echo ""

# Check environment
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set"
    echo "Please set it with: export OPENAI_API_KEY='your-key-here'"
    exit 1
fi

echo "✓ OPENAI_API_KEY is set"
echo "  Base URL: ${OPENAI_BASE_URL:-https://api.openai.com/v1}"
echo ""

# Show menu
echo "Select an option:"
echo "  1) Quick test (2 rows, sequential)"
echo "  2) Small batch (10 rows, parallel)"
echo "  3) Medium batch (50 rows, parallel)"
echo "  4) Default batch (100 rows, parallel)"
echo "  5) Large batch (500 rows, parallel)"
echo "  6) Full dataset (1000 rows, parallel)"
echo "  7) Custom configuration"
echo ""

read -p "Enter choice [1-7]: " choice

case $choice in
    1)
        echo ""
        echo "Running quick test (2 rows)..."
        python test_precised_generation.py
        ;;
    2)
        echo ""
        echo "Running small batch (10 rows, 4 workers)..."
        python generate_precised_images.py --num-rows 10 --workers 4
        ;;
    3)
        echo ""
        echo "Running medium batch (50 rows, 4 workers)..."
        python generate_precised_images.py --num-rows 50 --workers 4
        ;;
    4)
        echo ""
        echo "Running default batch (100 rows, 4 workers)..."
        python generate_precised_images.py --num-rows 100 --workers 4
        ;;
    5)
        echo ""
        echo "Running large batch (500 rows, 8 workers)..."
        echo "Estimated cost: ~$100"
        read -p "Continue? [y/N]: " confirm
        if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
            python generate_precised_images.py --num-rows 500 --workers 8
        else
            echo "Cancelled"
        fi
        ;;
    6)
        echo ""
        echo "Running FULL dataset (1000 rows, 8 workers)..."
        echo "Estimated cost: ~$200"
        echo "Estimated time: ~2-3 hours"
        read -p "Are you sure? [y/N]: " confirm
        if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
            python generate_precised_images.py --num-rows 1000 --workers 8
        else
            echo "Cancelled"
        fi
        ;;
    7)
        echo ""
        read -p "Number of rows: " rows
        read -p "Number of workers (1-10): " workers
        read -p "Output directory: " output_dir
        
        echo ""
        echo "Configuration:"
        echo "  Rows: $rows"
        echo "  Workers: $workers"
        echo "  Output: $output_dir"
        read -p "Continue? [y/N]: " confirm
        
        if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
            python generate_precised_images.py \
                --num-rows "$rows" \
                --workers "$workers" \
                --output-dir "$output_dir"
        else
            echo "Cancelled"
        fi
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "==================================================================="
echo "Generation complete!"
echo "==================================================================="
