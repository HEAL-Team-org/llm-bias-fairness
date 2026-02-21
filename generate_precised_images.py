#!/usr/bin/env python3
"""
Generate images for precised prompts from prompts_precised.csv
This script reads precised prompts and generates images using DALL-E 3.
Supports both standard OpenAI and Azure OpenAI providers.
"""

import argparse
import csv
import json
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_openai_client(provider='openai', azure_endpoint=None, azure_api_version='2024-12-01-preview'):
    """
    Setup OpenAI client with appropriate configuration.
    
    Args:
        provider: 'openai' for standard OpenAI or 'azure' for Azure OpenAI
        azure_endpoint: Azure OpenAI endpoint (required if provider is 'azure')
        azure_api_version: Azure API version (default: 2024-12-01-preview)
    
    Returns:
        OpenAI or AzureOpenAI client instance
    """
    try:
        from openai import OpenAI, AzureOpenAI
        
        provider = provider.lower()
        
        if provider == 'azure':
            # Azure OpenAI configuration
            api_key = os.getenv('AZURE_OPENAI_API_KEY')
            endpoint = azure_endpoint or os.getenv('AZURE_OPENAI_ENDPOINT')
            
            if not api_key:
                raise ValueError("AZURE_OPENAI_API_KEY environment variable not set")
            if not endpoint:
                raise ValueError("AZURE_OPENAI_ENDPOINT environment variable or --azure-endpoint argument required")
            
            client = AzureOpenAI(
                api_key=api_key,
                azure_endpoint=endpoint,
                api_version=azure_api_version
            )
            logger.info(f"Azure OpenAI client initialized")
            logger.info(f"  Endpoint: {endpoint}")
            logger.info(f"  API Version: {azure_api_version}")
            return client
        else:
            # Standard OpenAI configuration
            api_key = os.getenv('OPENAI_API_KEY')
            base_url = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
            
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            
            client = OpenAI(api_key=api_key, base_url=base_url)
            logger.info(f"Standard OpenAI client initialized")
            logger.info(f"  Base URL: {base_url}")
            return client
            
    except ImportError:
        logger.error("openai package not installed. Run: pip install openai")
        raise
    except Exception as e:
        logger.error(f"Failed to setup OpenAI client: {e}")
        raise


def read_precised_prompts(csv_path: str, num_rows: int = 100, seed: int = None) -> List[Dict]:
    """
    Read precised prompts from CSV file with optional random selection.
    
    Args:
        csv_path: Path to prompts_precised.csv
        num_rows: Number of rows to process (default: 100)
        seed: Random seed for reproducible selection (default: None for sequential)
    
    Returns:
        List of dictionaries with row_id, original_prompt, and precised_prompts
    """
    all_rows = []
    
    try:
        # First, read all rows from CSV
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                # Parse the precised_prompts field (JSON array of strings)
                try:
                    precised_prompts_list = json.loads(row['precised_prompts'])
                    
                    all_rows.append({
                        'csv_idx': idx,
                        'row_id': row.get('__row_id__', idx),
                        'original_prompt': row['prompt'],
                        'generic_prompt': row.get('generic_prompt', ''),
                        'precised_prompts': precised_prompts_list
                    })
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse precised_prompts for row {idx}: {e}")
                    continue
        
        logger.info(f"Read {len(all_rows)} total rows from {csv_path}")
        
        # Select rows (randomly or sequentially)
        if seed is not None:
            # Random selection with seed for reproducibility
            random.seed(seed)
            if num_rows > len(all_rows):
                logger.warning(f"Requested {num_rows} rows but only {len(all_rows)} available. Using all.")
                selected_rows = all_rows
            else:
                selected_rows = random.sample(all_rows, num_rows)
                # Sort by row_id to maintain consistent ordering for output
                selected_rows.sort(key=lambda x: int(x['row_id']))
            logger.info(f"Randomly selected {len(selected_rows)} rows with seed={seed}")
        else:
            # Sequential selection
            selected_rows = all_rows[:num_rows]
            logger.info(f"Selected first {len(selected_rows)} rows sequentially")
        
        # Add row_idx for tracking in output
        prompts_data = []
        for idx, row in enumerate(selected_rows):
            row['row_idx'] = idx
            prompts_data.append(row)
        
        # Log selected row IDs for traceability
        row_ids = [row['row_id'] for row in prompts_data]
        logger.info(f"Selected row IDs: {row_ids[:10]}{'...' if len(row_ids) > 10 else ''}")
        
        return prompts_data
    
    except FileNotFoundError:
        logger.error(f"CSV file not found: {csv_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading CSV: {e}")
        raise


def generate_image(client, prompt: str, output_path: str, model: str = "dall-e-3", size: str = "1024x1024") -> Dict:
    """
    Generate a single image using DALL-E 3.
    
    Args:
        client: OpenAI or AzureOpenAI client instance
        prompt: Text prompt for image generation
        output_path: Path to save the generated image
        model: Model name or Azure deployment name (default: dall-e-3)
        size: Image size (default: 1024x1024)
    
    Returns:
        Dictionary with status and metadata
    """
    try:
        logger.info(f"Generating image for: {prompt[:80]}...")
        
        response = client.images.generate(
            model=model,
            prompt=prompt,
            size=size,
            quality="standard",
            n=1
        )
        
        image_url = response.data[0].url
        revised_prompt = getattr(response.data[0], 'revised_prompt', None)
        
        # Download and save image
        import urllib.request
        urllib.request.urlretrieve(image_url, output_path)
        
        logger.info(f"✓ Image saved to: {output_path}")
        
        return {
            'status': 'success',
            'output_path': output_path,
            'revised_prompt': revised_prompt,
            'original_prompt': prompt,
            'model': model
        }
    
    except Exception as e:
        logger.error(f"✗ Failed to generate image: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'output_path': output_path,
            'original_prompt': prompt,
            'model': model
        }


def generate_images_for_row(client, row_data: Dict, output_dir: Path, model: str = "dall-e-3") -> List[Dict]:
    """
    Generate all images for a single row's precised prompts.
    
    Args:
        client: OpenAI or AzureOpenAI client instance
        row_data: Dictionary with row information
        output_dir: Output directory for images
        model: Model name or Azure deployment name
    
    Returns:
        List of result dictionaries
    """
    results = []
    row_idx = row_data['row_idx']
    row_id = row_data['row_id']
    precised_prompts = row_data['precised_prompts']
    
    # Create subdirectory using row_id for traceability
    row_dir = output_dir / f"row_{row_id}"
    row_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing Row {row_idx} (CSV ID: {row_id}) - {len(precised_prompts)} variants")
    logger.info(f"{'='*80}")
    
    for variant_idx, prompt in enumerate(precised_prompts):
        output_filename = f"row_{row_id}_variant_{variant_idx:02d}.png"
        output_path = row_dir / output_filename
        
        result = generate_image(client, prompt, str(output_path), model=model)
        result['row_idx'] = row_idx
        result['row_id'] = row_id
        result['variant_idx'] = variant_idx
        
        results.append(result)
        
        # Rate limiting: wait between requests
        time.sleep(1)
    
    return results


def process_parallel(client, prompts_data: List[Dict], output_dir: Path, model: str = "dall-e-3", num_workers: int = 4) -> List[Dict]:
    """
    Process multiple rows in parallel using thread pool.
    
    Args:
        client: OpenAI or AzureOpenAI client instance
        prompts_data: List of row data dictionaries
        output_dir: Output directory for images
        model: Model name or Azure deployment name
        num_workers: Number of parallel workers
    
    Returns:
        List of all results
    """
    all_results = []
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_row = {
            executor.submit(generate_images_for_row, client, row_data, output_dir, model): row_data
            for row_data in prompts_data
        }
        
        # Process completed tasks
        for future in as_completed(future_to_row):
            row_data = future_to_row[future]
            try:
                results = future.result()
                all_results.extend(results)
                logger.info(f"✓ Completed row {row_data['row_idx']} (CSV ID: {row_data['row_id']}): {len(results)} images")
            except Exception as e:
                logger.error(f"✗ Row {row_data['row_idx']} (CSV ID: {row_data['row_id']}) failed: {e}")
    
    return all_results


def save_results_metadata(metadata: Dict, output_path: str):
    """Save generation results metadata to JSON file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Results metadata saved to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save results metadata: {e}")


def print_summary(results: List[Dict]):
    """Print generation summary statistics."""
    total = len(results)
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = total - successful
    
    logger.info(f"\n{'='*80}")
    logger.info("GENERATION SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Total images attempted: {total}")
    logger.info(f"Successful: {successful} ({successful/total*100:.1f}%)")
    logger.info(f"Failed: {failed} ({failed/total*100:.1f}%)")
    
    if failed > 0:
        logger.info("\nFailed prompts:")
        for r in results:
            if r['status'] == 'error':
                logger.info(f"  - Row {r['row_idx']}, Variant {r['variant_idx']}: {r['error']}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate images for precised prompts from CSV. Supports OpenAI and Azure OpenAI.'
    )
    parser.add_argument(
        '--csv',
        type=str,
        default='prompts_precised.csv',
        help='Path to prompts_precised.csv (default: prompts_precised.csv)'
    )
    parser.add_argument(
        '--num-rows',
        type=int,
        default=100,
        help='Number of rows to process (default: 100)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='precised_generated_images',
        help='Output directory for images (default: precised_generated_images)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of parallel workers (default: 4)'
    )
    parser.add_argument(
        '--sequential',
        action='store_true',
        help='Process sequentially instead of parallel'
    )
    parser.add_argument(
        '--provider',
        type=str,
        choices=['openai', 'azure'],
        default='openai',
        help='API provider: openai or azure (default: openai)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='dall-e-3',
        help='Model name (OpenAI) or deployment name (Azure) (default: dall-e-3)'
    )
    parser.add_argument(
        '--azure-endpoint',
        type=str,
        help='Azure OpenAI endpoint (required for --provider azure, or set AZURE_OPENAI_ENDPOINT)'
    )
    parser.add_argument(
        '--azure-api-version',
        type=str,
        default='2024-12-01-preview',
        help='Azure API version (default: 2024-12-01-preview)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for row selection (if not set, select sequentially)'
    )
    
    args = parser.parse_args()
    
    # Setup
    logger.info("Starting precised image generation...")
    logger.info(f"Configuration:")
    logger.info(f"  Provider: {args.provider.upper()}")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  CSV file: {args.csv}")
    logger.info(f"  Number of rows: {args.num_rows}")
    logger.info(f"  Random seed: {args.seed if args.seed is not None else 'None (sequential)'}")
    logger.info(f"  Output directory: {args.output_dir}")
    logger.info(f"  Workers: {args.workers if not args.sequential else 1} {'(sequential)' if args.sequential else ''}")
    
    if args.provider == 'azure':
        logger.info(f"  Azure endpoint: {args.azure_endpoint or os.getenv('AZURE_OPENAI_ENDPOINT')}")
        logger.info(f"  Azure API version: {args.azure_api_version}")
    
    # Initialize client
    client = setup_openai_client(
        provider=args.provider,
        azure_endpoint=args.azure_endpoint,
        azure_api_version=args.azure_api_version
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read prompts (with optional random selection)
    prompts_data = read_precised_prompts(args.csv, args.num_rows, seed=args.seed)
    
    if not prompts_data:
        logger.error("No prompts loaded. Exiting.")
        return
    
    # Generate images
    start_time = time.time()
    
    if args.sequential:
        all_results = []
        for row_data in prompts_data:
            results = generate_images_for_row(client, row_data, output_dir, model=args.model)
            all_results.extend(results)
    else:
        all_results = process_parallel(client, prompts_data, output_dir, model=args.model, num_workers=args.workers)
    
    elapsed_time = time.time() - start_time
    
    # Save metadata
    metadata = {
        'configuration': {
            'provider': args.provider,
            'model': args.model,
            'num_rows': args.num_rows,
            'seed': args.seed,
            'csv_file': args.csv,
            'output_dir': args.output_dir,
            'workers': args.workers if not args.sequential else 1,
            'sequential': args.sequential
        },
        'results': all_results,
        'summary': {
            'total_images': len(all_results),
            'successful': sum(1 for r in all_results if r['status'] == 'success'),
            'failed': sum(1 for r in all_results if r['status'] == 'error'),
            'total_time_seconds': elapsed_time,
            'average_time_per_image': elapsed_time / len(all_results) if all_results else 0
        }
    }
    
    results_file = output_dir / "generation_results.json"
    save_results_metadata(metadata, str(results_file))
    
    # Print summary
    print_summary(all_results)
    logger.info(f"\nTotal time: {elapsed_time:.2f} seconds")
    logger.info(f"Average time per image: {elapsed_time/len(all_results):.2f} seconds")


if __name__ == '__main__':
    main()
