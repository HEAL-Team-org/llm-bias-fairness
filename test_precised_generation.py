#!/usr/bin/env python3
"""Quick test script to verify precised image generation with 2 rows."""

import subprocess
import sys

def main():
    print("Testing precised image generation with 2 rows...")
    print("="*80)
    
    # Run generation script with 2 rows
    cmd = [
        sys.executable,
        "generate_precised_images.py",
        "--csv", "prompts_precised.csv",
        "--num-rows", "2",
        "--output-dir", "test_precised_images",
        "--sequential"  # Use sequential for testing
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("\n" + "="*80)
        print("✓ Test completed successfully!")
        print("Check 'test_precised_images/' directory for generated images")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Test failed with error: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        return 130
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
