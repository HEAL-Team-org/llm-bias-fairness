#!/usr/bin/env python3
"""
Test script to compare PreciseDebias approach with our GraphRAG-based approach.

This script:
1. Uses PreciseDebias-style prompt expansion (demographic distribution-based)
2. Compares with our GraphRAG + StereoSet + DiversityRAG approach
3. Generates images using the same model (gpt-image-1) for fair comparison
4. Saves results with metadata for analysis
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import get_config, get_image_model
from src.data import OpenAIEmbedder
from src.generation.image_generator import create_image_generator
from src.knowledge.diversity import DiversityRAG
from src.knowledge.graphrag import GraphRAG
from src.knowledge.stereoset import StereoSetRAG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PreciseDebiasSimulator:
    """
    Simulates PreciseDebias approach using direct demographic insertion.
    
    PreciseDebias methodology (from actual implementation):
    - Explicitly adds demographic attributes to prompts
    - Uses fixed demographics: Black/White/Hispanic/Asian × male/female
    - Based on BLS workforce statistics distributions
    - Originally uses fine-tuned Llama, but demographics are directly inserted
    """
    
    def __init__(self, config: Dict):
        """Initialize PreciseDebias simulator."""
        self.config = config
    
    def expand_prompt(self, prompt: str, num_variations: int = 5) -> List[str]:
        """
        Expand prompt using PreciseDebias-style demographic distribution.
        
        PreciseDebias explicitly adds demographic attributes (ethnicity + gender)
        to prompts. This is based on the actual implementation that uses:
        "Black male", "Black female", "White male", "White female",
        "Hispanic male", "Hispanic female", "Asian male", "Asian female"
        
        Args:
            prompt: Original generic prompt
            num_variations: Number of diverse variations to generate
            
        Returns:
            List of expanded prompts with demographic information
        """
        # Demographics used in PreciseDebias training (from their code)
        demographics = [
            "Black male",
            "Black female", 
            "White male",
            "White female", 
            "Hispanic male", 
            "Hispanic female", 
            "Asian male",
            "Asian female"
        ]
        
        try:
            logger.info(f"Expanding prompt with PreciseDebias approach: '{prompt}'")
            
            # PreciseDebias method: Simply insert demographic attributes into the prompt
            # They identify the subject (e.g., "doctor", "nurse") and add demographics
            # For simplicity, we'll add it at the beginning of the entity mention
            
            variations = []
            
            # Select random demographics (or use all if num_variations >= 8)
            if num_variations >= len(demographics):
                selected_demos = demographics
            else:
                import random
                selected_demos = random.sample(demographics, num_variations)
            
            # Generate variations by prepending demographic to the prompt
            # This matches PreciseDebias's approach of "[ETHNICITY] doctor" -> "Black male doctor"
            for demo in selected_demos:
                # Insert demographic attribute naturally into the prompt
                # e.g., "a doctor" -> "a Black male doctor"
                # e.g., "a nurse" -> "an Asian female nurse"
                expanded = prompt
                
                # Replace common patterns
                for article_pattern in [" a doctor", " a nurse", " a CEO", " a scientist", 
                                       " a teacher", " a engineer", " a lawyer", " a manager"]:
                    if article_pattern in expanded.lower():
                        # Find actual case-sensitive match
                        for word in ["doctor", "nurse", "CEO", "scientist", "teacher", 
                                    "engineer", "lawyer", "manager"]:
                            patterns = [f" a {word}", f" an {word}", f" A {word}", f" An {word}"]
                            for pattern in patterns:
                                if pattern in expanded:
                                    # Insert demographic before the profession
                                    article = pattern.split()[0]  # " a" or " an"
                                    expanded = expanded.replace(
                                        pattern, 
                                        f"{article} {demo} {word}"
                                    )
                                    break
                
                variations.append(expanded)
            
            if not variations:
                logger.warning("No variations generated, using original prompt")
                variations = [prompt]
            
            logger.info(f"Generated {len(variations)} PreciseDebias variations")
            return variations
            
        except Exception as e:
            logger.error(f"PreciseDebias expansion failed: {e}")
            return [prompt]  # Fallback to original


class OurApproachEnhancer:
    """
    Our GraphRAG + StereoSet + DiversityRAG approach for prompt enhancement.
    """
    
    def __init__(self, config: Dict):
        """Initialize our approach components."""
        self.config = config
        
        # Initialize GraphRAG
        logger.info("Initializing GraphRAG...")
        self.graphrag = GraphRAG(cache_file="embeddings.pkl")
        from src.data.parsers import BiasCSVParser
        bias_parser = BiasCSVParser(
            "data/biases/ADV_GRAPH_20240119 - ADV_GRAPH_20240119.csv"
        )
        self.graphrag.add_graph("bias_graph", bias_parser)
        
        # Initialize StereoSet RAG
        logger.info("Initializing StereoSet RAG...")
        self.stereoset_rag = StereoSetRAG(cache_file="stereoset_embeddings.pkl")
        self.stereoset_rag.load_dataset()
        
        # Initialize Diversity RAG
        logger.info("Initializing Diversity RAG...")
        self.diversity_rag = DiversityRAG(cache_file="diversity_embeddings.pkl")
        self.diversity_rag.load_datasets()
        
        logger.info("All RAG systems initialized")
    
    def enhance_prompt(self, prompt: str) -> Tuple[str, Dict]:
        """
        Enhance prompt using our GraphRAG + StereoSet + DiversityRAG approach.
        
        Args:
            prompt: Original prompt
            
        Returns:
            Tuple of (enhanced_prompt, metadata)
        """
        try:
            logger.info(f"Enhancing prompt with our approach: '{prompt}'")
            
            # Step 1: Retrieve bias knowledge
            bias_graph = self.graphrag.graphs["bias_graph"]
            bias_triples = self.graphrag.retriever.retrieve_by_vector_similarity(
                bias_graph, prompt, top_k=10
            )
            logger.info(f"Retrieved {len(bias_triples)} bias triples")
            
            # Step 2: Retrieve stereotypes
            stereotypes = self.stereoset_rag.retrieve_related_stereotypes(
                prompt
            )
            logger.info(f"Retrieved {len(stereotypes)} stereotypes")
            
            # Step 3: Retrieve diversity examples
            diversity_examples = self.diversity_rag.retrieve_diversity_examples(
                prompt
            )
            logger.info(f"Retrieved {len(diversity_examples)} diversity examples")
            
            # Step 4: Generate enhanced prompt using GraphRAG
            context_parts = []
            
            if bias_triples:
                bias_text = "\n".join([
                    f"- {subject} {relation} {obj}"
                    for subject, relation, obj in bias_triples[:5]
                ])
                context_parts.append(f"Bias Knowledge:\n{bias_text}")
            
            if stereotypes:
                stereo_text = "\n".join([
                    f"- {record.context}" for record, score in stereotypes[:5]
                ])
                context_parts.append(f"Stereotypes to Avoid:\n{stereo_text}")
            
            if diversity_examples:
                div_text = "\n".join([
                    f"- {record.value}" for record, score in diversity_examples[:5]
                ])
                context_parts.append(f"Diversity Examples:\n{div_text}")
            
            context = "\n\n".join(context_parts)
            
            # Generate enhanced prompt using GPT-5
            enhancement_query = f"""Given the following knowledge about biases, stereotypes, and diversity:

{context}

Enhance this image generation prompt to be more fair, unbiased, and diverse:
"{prompt}"

Provide ONLY the enhanced prompt, nothing else. Make it natural and maintain the original intent."""
            
            # Use OpenAI client directly to generate enhanced prompt
            from openai import OpenAI
            client = OpenAI(api_key=self.config.get("openai.api_key"))
            
            response = client.chat.completions.create(
                model=self.config.get("chat_model", "gpt-5"),
                messages=[{"role": "user", "content": enhancement_query}],
                max_completion_tokens=200
            )
            
            enhanced_prompt = response.choices[0].message.content.strip()
            
            metadata = {
                "method": "GraphRAG + StereoSet + DiversityRAG",
                "bias_triples_used": len(bias_triples),
                "stereotypes_used": len(stereotypes),
                "diversity_examples_used": len(diversity_examples),
            }
            
            logger.info(f"Enhanced prompt: '{enhanced_prompt}'")
            return enhanced_prompt, metadata
            
        except Exception as e:
            logger.error(f"Our approach enhancement failed: {e}")
            return prompt, {"error": str(e)}


def generate_images_for_prompts(
    prompts: List[str],
    method_name: str,
    generator,
    output_dir: Path,
    prompt_idx: int = 0
) -> List[Dict]:
    """
    Generate images for a list of prompts.
    
    Args:
        prompts: List of prompts to generate images for
        method_name: Name of the method (for file naming)
        generator: Image generator instance
        output_dir: Output directory
        prompt_idx: Index of the test prompt (to avoid filename collisions)
        
    Returns:
        List of result dictionaries with image paths and metadata
    """
    results = []
    
    for idx, prompt in enumerate(prompts, 1):
        try:
            logger.info(f"Generating image {idx}/{len(prompts)} for {method_name}")
            
            # Generate filename with prompt index to avoid collisions
            safe_method = method_name.replace(" ", "_").lower()
            filename = f"{safe_method}_prompt{prompt_idx}_image_{idx}.png"
            
            # Generate image
            filepath, metadata = generator.generate_image(
                prompt=prompt,
                filename=filename
            )
            
            result = {
                "method": method_name,
                "prompt": prompt,
                "image_path": str(filepath),
                "metadata": metadata,
                "success": True
            }
            
            logger.info(f"Successfully generated: {filepath}")
            results.append(result)
            
        except Exception as e:
            logger.error(f"Failed to generate image for {method_name}: {e}")
            results.append({
                "method": method_name,
                "prompt": prompt,
                "error": str(e),
                "success": False
            })
    
    return results


def main():
    """Main comparison workflow."""
    logger.info("=" * 80)
    logger.info("PRECISEDEBIAS vs OUR APPROACH - COMPARISON TEST")
    logger.info("=" * 80)
    
    # Test prompts
    test_prompts = [
        "Generate an image of a doctor who is smiling at the camera",
        "Generate an image of a nurse who is smiling at the camera"
    ]
    
    # Setup output directory
    output_dir = Path("experiments/comparison_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    # Load configuration
    logger.info("Loading configuration...")
    config = get_config()
    api_key = config.get("openai.api_key")
    image_model = get_image_model()
    
    logger.info(f"Using image model: {image_model}")
    logger.info(f"Using chat model: {config.get('openai.chat_model')}")
    
    # Initialize image generator
    logger.info("Initializing image generator...")
    generator = create_image_generator(
        "dalle3",
        output_dir=str(images_dir),
        api_key=api_key,
        model=image_model
    )
    
    # Initialize both approaches
    logger.info("\n" + "=" * 80)
    logger.info("INITIALIZING METHODS")
    logger.info("=" * 80)
    
    precisedebias = PreciseDebiasSimulator(config)
    our_approach = OurApproachEnhancer(config)
    
    # Run comparison for each test prompt
    all_results = []
    
    for prompt_idx, original_prompt in enumerate(test_prompts, 1):
        logger.info("\n" + "=" * 80)
        logger.info(f"TEST PROMPT {prompt_idx}: {original_prompt}")
        logger.info("=" * 80)
        
        prompt_results = {
            "original_prompt": original_prompt,
            "baseline": None,
            "precisedebias": None,
            "our_approach": None
        }
        
        # Method 1: Baseline (no enhancement)
        logger.info("\n" + "-" * 80)
        logger.info("METHOD 1: BASELINE (No Enhancement)")
        logger.info("-" * 80)
        
        baseline_results = generate_images_for_prompts(
            [original_prompt],
            "baseline",
            generator,
            images_dir,
            prompt_idx=prompt_idx
        )
        prompt_results["baseline"] = baseline_results
        
        # Method 2: PreciseDebias approach
        logger.info("\n" + "-" * 80)
        logger.info("METHOD 2: PRECISEDEBIAS APPROACH")
        logger.info("-" * 80)
        
        precisedebias_prompts = precisedebias.expand_prompt(
            original_prompt, num_variations=5
        )
        logger.info(f"PreciseDebias generated {len(precisedebias_prompts)} variations:")
        for idx, p in enumerate(precisedebias_prompts, 1):
            logger.info(f"  {idx}. {p}")
        
        precisedebias_results = generate_images_for_prompts(
            precisedebias_prompts,
            "precisedebias",
            generator,
            images_dir,
            prompt_idx=prompt_idx
        )
        prompt_results["precisedebias"] = {
            "expanded_prompts": precisedebias_prompts,
            "images": precisedebias_results
        }
        
        # Method 3: Our approach
        logger.info("\n" + "-" * 80)
        logger.info("METHOD 3: OUR APPROACH (GraphRAG + StereoSet + DiversityRAG)")
        logger.info("-" * 80)
        
        enhanced_prompt, enhancement_metadata = our_approach.enhance_prompt(
            original_prompt
        )
        logger.info(f"Our approach enhanced prompt: {enhanced_prompt}")
        
        our_results = generate_images_for_prompts(
            [enhanced_prompt],
            "our_approach",
            generator,
            images_dir,
            prompt_idx=prompt_idx
        )
        prompt_results["our_approach"] = {
            "enhanced_prompt": enhanced_prompt,
            "enhancement_metadata": enhancement_metadata,
            "images": our_results
        }
        
        all_results.append(prompt_results)
    
    # Save comprehensive results
    logger.info("\n" + "=" * 80)
    logger.info("SAVING RESULTS")
    logger.info("=" * 80)
    
    results_file = output_dir / "comparison_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"Results saved to: {results_file}")
    
    # Generate summary
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON SUMMARY")
    logger.info("=" * 80)
    
    for idx, result in enumerate(all_results, 1):
        logger.info(f"\nPrompt {idx}: {result['original_prompt']}")
        logger.info(f"  Baseline images: {len([r for r in result['baseline'] if r['success']])}")
        logger.info(f"  PreciseDebias images: {len([r for r in result['precisedebias']['images'] if r['success']])}")
        logger.info(f"  Our approach images: {len([r for r in result['our_approach']['images'] if r['success']])}")
    
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON TEST COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"\nResults directory: {output_dir}")
    logger.info(f"Images directory: {images_dir}")
    logger.info(f"Results JSON: {results_file}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
