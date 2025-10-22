#!/usr/bin/env python3
"""
Parallel Pipeline Test Script with Multiple Workers

This script processes prompts in parallel using multiple workers:
1. Load prompts from CSV
2. Distribute work across 32 workers
3. Each worker: enhance prompts + generate images (original & enhanced)
4. Collect and save all results
"""

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config.settings import get_config, get_openai_base_url, get_openai_provider
from src.generation.image_generator import create_image_generator
from src.knowledge.diversity import DiversityRAG
from src.knowledge.graphrag import GraphRAG
from src.knowledge.stereoset import StereoSetRAG

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_prompts(csv_file):
    """Load prompts from CSV file."""
    logger.info(f"Loading prompts from: {csv_file}")
    
    prompts = []
    with open(csv_file) as f:
        lines = f.readlines()
    
    # Skip header
    for line in lines[1:]:
        prompt = line.strip()
        if prompt:
            prompts.append(prompt)
    
    logger.info(f"Loaded {len(prompts)} prompts")
    return prompts


def initialize_worker(output_dir="results"):
    """Initialize RAG systems and image generator for worker process."""
    try:
        # Reload config in worker process
        from src.config.settings import reload_config
        reload_config()
        
        config = get_config()
        api_key = config.get("openai.api_key")
        
        # Initialize GraphRAG
        graphrag = GraphRAG(cache_file="embeddings.pkl")
        from src.data.parsers import BiasCSVParser
        bias_parser = BiasCSVParser(
            "data/biases/ADV_GRAPH_20240119 - ADV_GRAPH_20240119.csv"
        )
        graphrag.add_graph("bias_graph", bias_parser)
        
        # Initialize StereoSet RAG
        stereoset_rag = StereoSetRAG(cache_file="stereoset_embeddings.pkl")
        stereoset_rag.load_dataset()
        
        # Initialize Diversity RAG
        diversity_rag = DiversityRAG(cache_file="diversity_embeddings.pkl")
        diversity_rag.load_datasets()
        
        # Initialize image generator with configurable output directory
        # Get image model from config (e.g., "gpt-image-1", "dall-e-3")
        from src.config.settings import get_image_model
        image_model = get_image_model()
        
        images_dir = f"{output_dir}/generated_images"
        generator = create_image_generator(
            "dalle3",
            output_dir=images_dir,
            api_key=api_key,
            model=image_model  # Pass the configured model name
        )
        
        return graphrag, stereoset_rag, diversity_rag, generator
        
    except Exception as e:
        logger.error(f"Worker initialization failed: {e}")
        raise


def enhance_prompt(prompt, graphrag, stereoset_rag, diversity_rag):
    """Enhance a single prompt using dual pipeline."""
    try:
        # Query GraphRAG for bias-aware context
        graph_result = graphrag.query(
            graph_name="bias_graph",
            question=prompt,
            top_k=8,
            include_answer=False
        )
        
        # Query StereoSet RAG
        stereoset_results = stereoset_rag.retrieve_related_stereotypes(prompt)
        
        # Query Diversity RAG
        diversity_results = diversity_rag.retrieve_diversity_examples(prompt)
        
        # Combine contexts
        bias_context = "\n".join([
            f"- {t[0]} {t[1]} {t[2]}" 
            for t in graph_result.get("triples", [])[:5]
        ])
        
        stereo_context = "\n".join([
            f"- {record.context} (bias: {record.bias_type})" 
            for record, score in stereoset_results[:3]
        ])
        
        diversity_context = "\n".join([
            f"- {record.doc_text[:100]}... (group: {record.cultural_group})" 
            for record, score in diversity_results[:3]
        ])
        
        # Generate enhancement prompt
        enhancement_request = f"""Original prompt: "{prompt}"

Bias awareness context:
{bias_context}

Stereotype awareness:
{stereo_context}

Diversity guidance:
{diversity_context}

Please rewrite the prompt to:
1. Ensure diverse and inclusive representation
2. Avoid reinforcing stereotypes
3. Promote fairness and cultural sensitivity
4. Maintain the original intent and context

Enhanced prompt:"""
        
        # Get enhancement using GraphRAG's answerer
        enhanced = graphrag.answerer.answer_question(
            enhancement_request,
            graph_result.get("triples", [])[:5]
        )
        
        return {
            "original_prompt": prompt,
            "enhanced_prompt": enhanced,
            "bias_triples_count": len(graph_result.get("triples", [])),
            "stereoset_matches": len(stereoset_results),
            "diversity_matches": len(diversity_results)
        }
        
    except Exception as e:
        logger.error(f"Enhancement failed: {e}")
        return {
            "original_prompt": prompt,
            "enhanced_prompt": prompt,
            "error": str(e)
        }


def generate_image(prompt, generator, index, suffix=""):
    """Generate image from prompt."""
    try:
        filename = f"sample_{index:03d}{suffix}"
        filepath, metadata = generator.generate_image(
            prompt,
            filename=filename
        )
        
        return {
            "filepath": filepath,
            "metadata": metadata,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Image generation failed: {e}")
        return {
            "filepath": None,
            "error": str(e),
            "success": False
        }


def process_single_prompt(args):
    """Process a single prompt (called by worker process)."""
    index, prompt, output_dir = args
    
    try:
        logger.info(f"[{index}] Starting: {prompt[:60]}...")
        
        # Initialize worker (lazy initialization per worker)
        if not hasattr(process_single_prompt, 'initialized'):
            process_single_prompt.graphrag, \
            process_single_prompt.stereoset_rag, \
            process_single_prompt.diversity_rag, \
            process_single_prompt.generator = initialize_worker(output_dir)
            process_single_prompt.initialized = True
            logger.info(f"[{index}] Worker initialized")
        
        graphrag = process_single_prompt.graphrag
        stereoset_rag = process_single_prompt.stereoset_rag
        diversity_rag = process_single_prompt.diversity_rag
        generator = process_single_prompt.generator
        
        # Generate image from ORIGINAL prompt
        logger.info(f"[{index}] Generating original image...")
        original_image_result = generate_image(
            prompt,
            generator,
            index,
            suffix="_original"
        )
        
        # Enhance prompt
        logger.info(f"[{index}] Enhancing prompt...")
        enhancement_result = enhance_prompt(
            prompt, graphrag, stereoset_rag, diversity_rag
        )
        
        # Generate image from ENHANCED prompt
        logger.info(f"[{index}] Generating enhanced image...")
        enhanced_image_result = generate_image(
            enhancement_result["enhanced_prompt"],
            generator,
            index,
            suffix="_enhanced"
        )
        
        result = {
            "index": index,
            "original_prompt": prompt,
            "enhancement": enhancement_result,
            "images": {
                "original": original_image_result,
                "enhanced": enhanced_image_result
            }
        }
        
        # Save intermediate result
        intermediate_file = Path(output_dir) / f"intermediate_result_{index:03d}.json"
        with open(intermediate_file, "w") as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"[{index}] ✓ Complete!")
        return result
        
    except Exception as e:
        logger.error(f"[{index}] Failed: {e}")
        return {
            "index": index,
            "original_prompt": prompt,
            "error": str(e)
        }


def main():
    """Run parallel pipeline test."""
    parser = argparse.ArgumentParser(
        description="Run parallel bias-aware image generation pipeline"
    )
    parser.add_argument(
        "csv_file",
        nargs="?",
        default="sample_prompts.csv",
        help="Path to CSV file containing prompts (default: sample_prompts.csv)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=32,
        help="Number of parallel workers (default: 32)"
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Output directory for results (default: results)"
    )
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("PARALLEL PIPELINE TEST")
    logger.info("=" * 70)
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Input CSV: {args.csv_file}")
    logger.info(f"Output directory: {args.output_dir}\n")
    
    # Create output directories
    Path(args.output_dir).mkdir(exist_ok=True)
    (Path(args.output_dir) / "generated_images").mkdir(exist_ok=True)
    
    # Load prompts
    prompts = load_prompts(args.csv_file)
    
    if not prompts:
        logger.error("No prompts loaded!")
        return
    
    # Prepare work items (index, prompt, output_dir tuples)
    work_items = [(i, prompt, args.output_dir) for i, prompt in enumerate(prompts, 1)]
    
    logger.info(f"\n{'=' * 70}")
    logger.info(f"PROCESSING {len(prompts)} PROMPTS WITH {args.workers} WORKERS")
    logger.info(f"{'=' * 70}\n")
    
    start_time = time.time()
    results = []
    completed = 0
    
    # Process prompts in parallel
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Submit all work
        future_to_index = {
            executor.submit(process_single_prompt, item): item[0]
            for item in work_items
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result = future.result()
                results.append(result)
                completed += 1
                
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                remaining = len(prompts) - completed
                eta = remaining / rate if rate > 0 else 0
                
                logger.info(
                    f"Progress: {completed}/{len(prompts)} "
                    f"({100*completed/len(prompts):.1f}%) | "
                    f"Rate: {rate:.2f} prompts/sec | "
                    f"ETA: {eta/60:.1f} min"
                )
                
            except Exception as e:
                logger.error(f"Task {index} generated exception: {e}")
                results.append({
                    "index": index,
                    "error": str(e)
                })
    
    # Sort results by index
    results.sort(key=lambda x: x["index"])
    
    # Save final results
    logger.info(f"\n{'=' * 70}")
    logger.info("SAVING FINAL RESULTS")
    logger.info(f"{'=' * 70}")
    
    output_file = Path(args.output_dir) / "complete_pipeline_results.json"
    with open(output_file, "w") as f:
        json.dump({
            "total_prompts": len(prompts),
            "results": results,
            "configuration": {
                "workers": args.workers,
                "provider": get_openai_provider(),
                "base_url": get_openai_base_url()
            },
            "timing": {
                "total_seconds": time.time() - start_time,
                "prompts_per_second": len(prompts) / (time.time() - start_time)
            }
        }, f, indent=2)
    
    logger.info(f"✓ Final results saved: {output_file}")
    
    # Summary
    logger.info(f"\n{'=' * 70}")
    logger.info("SUMMARY")
    logger.info(f"{'=' * 70}")
    
    successful_enhancements = sum(
        1 for r in results 
        if "enhancement" in r and "error" not in r["enhancement"]
    )
    successful_original_images = sum(
        1 for r in results 
        if "images" in r and r["images"]["original"]["success"]
    )
    successful_enhanced_images = sum(
        1 for r in results 
        if "images" in r and r["images"]["enhanced"]["success"]
    )
    
    elapsed = time.time() - start_time
    
    logger.info(f"Total prompts: {len(prompts)}")
    logger.info(f"Successful enhancements: {successful_enhancements}/{len(prompts)}")
    logger.info(f"Successful images (original): {successful_original_images}/{len(prompts)}")
    logger.info(f"Successful images (enhanced): {successful_enhanced_images}/{len(prompts)}")
    logger.info(f"Total time: {elapsed/60:.2f} minutes")
    logger.info(f"Average rate: {len(prompts)/elapsed:.2f} prompts/second")
    logger.info(f"Results directory: {args.output_dir}/")
    logger.info(f"Images directory: {args.output_dir}/generated_images/")
    
    logger.info("\n✅ Parallel pipeline test complete!\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\nTest interrupted by user")
    except Exception as e:
        logger.error(f"\n\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
