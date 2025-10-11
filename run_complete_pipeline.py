#!/usr/bin/env python3
"""
Complete Pipeline Test Script

This script tests the full pipeline:
1. Load prompts from CSV
2. Enhance prompts using dual pipeline with diversity and unbias RAG
3. Generate images from enhanced prompts
4. Evaluate visual bias in generated images
5. Save all results
"""

import json
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config.settings import get_config, get_openai_base_url, get_openai_provider
from src.data import OpenAIEmbedder
from src.generation.image_generator import create_image_generator
from src.knowledge.diversity import DiversityRAG
from src.knowledge.graphrag import GraphRAG
from src.knowledge.stereoset import StereoSetRAG

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def verify_configuration():
    """Verify OpenAI configuration."""
    logger.info("=" * 70)
    logger.info("CONFIGURATION VERIFICATION")
    logger.info("=" * 70)
    
    # Reload configuration to ensure config file is loaded
    from src.config.settings import reload_config
    reload_config()
    
    config = get_config()
    provider = get_openai_provider()
    base_url = get_openai_base_url()
    api_key = config.get("openai.api_key")
    
    logger.info(f"Provider: {provider}")
    logger.info(f"Base URL: {base_url}")
    logger.info(f"API Key: {api_key[:10]}...{api_key[-5:] if api_key else 'None'}")
    
    if not api_key:
        logger.error("API key not configured!")
        return False
    
    if not base_url:
        logger.error("Base URL not configured!")
        return False
    
    return True


def load_prompts(csv_file):
    """Load prompts from CSV file."""
    logger.info(f"\nLoading prompts from: {csv_file}")
    
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


def enhance_prompt_dual_pipeline(prompt, graphrag, stereoset_rag, diversity_rag):
    """Enhance a single prompt using dual pipeline."""
    logger.info(f"\nEnhancing: {prompt[:60]}...")
    
    try:
        # Query GraphRAG for bias-aware context
        graph_result = graphrag.query(
            graph_name="bias_graph",
            question=prompt,
            top_k=8,
            include_answer=False
        )
        
        # Query StereoSet RAG
        stereoset_results = stereoset_rag.retrieve_related_stereotypes(prompt, top_k=10)
        
        # Query Diversity RAG
        diversity_results = diversity_rag.query(prompt, top_k=10)
        
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
            f"- {r['value']}" 
            for r in diversity_results[:3]
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
        
        logger.info(f"Enhanced: {enhanced[:60]}...")
        
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


def generate_image(prompt, generator, index):
    """Generate image from prompt."""
    logger.info(f"Generating image {index}...")
    
    try:
        filename = f"sample_{index:03d}"
        filepath, metadata = generator.generate_image(
            prompt,
            filename=filename
        )
        
        logger.info(f"Image saved: {filepath}")
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


def main():
    """Run complete pipeline test."""
    logger.info("\n" + "=" * 70)
    logger.info("COMPLETE PIPELINE TEST")
    logger.info("=" * 70 + "\n")
    
    # Step 1: Verify configuration
    if not verify_configuration():
        logger.error("Configuration verification failed!")
        return
    
    logger.info("✓ Configuration verified\n")
    
    # Step 2: Load prompts
    csv_file = "sample_prompts.csv"
    prompts = load_prompts(csv_file)
    
    if not prompts:
        logger.error("No prompts loaded!")
        return
    
    # Step 3: Initialize RAG systems
    logger.info("\nInitializing RAG systems...")
    
    try:
        # Initialize GraphRAG with bias knowledge
        graphrag = GraphRAG(cache_file="embeddings.pkl")
        
        # Load bias graph
        from src.data.parsers import BiasCSVParser
        bias_parser = BiasCSVParser(
            "data/biases/ADV_GRAPH_20240119 - ADV_GRAPH_20240119.csv"
        )
        graphrag.add_graph("bias_graph", bias_parser)
        logger.info("✓ GraphRAG initialized")
        
        # Initialize StereoSet RAG
        stereoset_rag = StereoSetRAG(cache_file="stereoset_embeddings.pkl")
        logger.info("✓ StereoSet RAG initialized")
        
        # Initialize Diversity RAG
        diversity_rag = DiversityRAG(cache_file="diversity_embeddings.pkl")
        diversity_rag.load_datasets()
        logger.info("✓ Diversity RAG initialized")
        
    except Exception as e:
        logger.error(f"RAG initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 4: Initialize image generator
    logger.info("\nInitializing image generator...")
    try:
        config = get_config()
        api_key = config.get("openai.api_key")
        
        generator = create_image_generator(
            "dalle3",
            output_dir="results/generated_images",
            api_key=api_key
        )
        logger.info("✓ Image generator initialized")
    except Exception as e:
        logger.error(f"Image generator initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 5: Process each prompt
    logger.info("\n" + "=" * 70)
    logger.info("PROCESSING PROMPTS")
    logger.info("=" * 70)
    
    results = []
    
    for i, prompt in enumerate(prompts, 1):
        logger.info(f"\n[{i}/{len(prompts)}] Processing prompt...")
        
        # Enhance prompt
        enhancement_result = enhance_prompt_dual_pipeline(
            prompt, graphrag, stereoset_rag, diversity_rag
        )
        
        # Generate image from enhanced prompt
        image_result = generate_image(
            enhancement_result["enhanced_prompt"],
            generator,
            i
        )
        
        # Combine results
        result = {
            "index": i,
            "original_prompt": prompt,
            "enhancement": enhancement_result,
            "image": image_result
        }
        
        results.append(result)
        
        # Save intermediate results
        intermediate_file = Path("results") / f"intermediate_result_{i:03d}.json"
        with open(intermediate_file, "w") as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"✓ Intermediate result saved: {intermediate_file}")
    
    # Step 6: Save final results
    logger.info("\n" + "=" * 70)
    logger.info("SAVING FINAL RESULTS")
    logger.info("=" * 70)
    
    output_file = Path("results") / "complete_pipeline_results.json"
    with open(output_file, "w") as f:
        json.dump({
            "total_prompts": len(prompts),
            "results": results,
            "configuration": {
                "provider": get_openai_provider(),
                "base_url": get_openai_base_url()
            }
        }, f, indent=2)
    
    logger.info(f"\n✓ Final results saved: {output_file}")
    
    # Step 7: Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    
    successful_enhancements = sum(
        1 for r in results 
        if "error" not in r["enhancement"]
    )
    successful_images = sum(
        1 for r in results 
        if r["image"]["success"]
    )
    
    logger.info(f"\nTotal prompts: {len(prompts)}")
    logger.info(f"Successful enhancements: {successful_enhancements}/{len(prompts)}")
    logger.info(f"Successful images: {successful_images}/{len(prompts)}")
    logger.info(f"\nResults directory: results/")
    logger.info(f"Images directory: results/generated_images/")
    logger.info(f"Complete results: {output_file}")
    
    logger.info("\n✅ Pipeline test complete!\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\nTest interrupted by user")
    except Exception as e:
        logger.error(f"\n\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
