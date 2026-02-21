"""Batch runner for dual-pipeline enhancement with per-step result saving.

Runs dual-pipeline enhancement for multiple prompts and saves detailed results
for each iteration step, including generated images.
"""

import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from src.generation.image_generator import ImageGenerationError, create_image_generator

from enhance_prompt_dual_pipeline import (
    load_enhancement_systems,
    parse_arguments,
    run_dual_pipeline_enhancement,
    setup_logging,
)

logger = logging.getLogger(__name__)


def setup_image_generator(
    backend: str = "dalle",
    model: Optional[str] = None,
    output_dir: Optional[Path] = None,
):
    """Setup image generator with specified backend.
    
    Args:
        backend: Image generation backend ("dalle", "stable-diffusion")
        model: Model name/path (uses defaults if None)
        output_dir: Output directory for generated images
    
    Returns:
        Image generator instance or None if setup fails
    """
    try:
        if backend == "dalle":
            generator_type = "dalle3"
            default_model = "dall-e-3"
        elif backend in ["stable-diffusion", "stable_diffusion"]:
            generator_type = "stable-diffusion"
            default_model = "runwayml/stable-diffusion-v1-5"  # Smaller model (~4GB vs ~13GB for SDXL)
        else:
            logger.error(f"Unknown backend: {backend}")
            return None
        
        # Use provided model or default
        final_model = model or default_model
        
        logger.info(f"Setting up {backend} image generator...")
        logger.info(f"Model: {final_model}")
        
        generator = create_image_generator(
            generator_type=generator_type,
            model=final_model if generator_type != "dalle3" else None,
            output_dir=output_dir or "generated_images"
        )
        
        logger.info(f"✓ {backend} image generator initialized")
        return generator
    
    except ImageGenerationError as e:
        logger.error(f"Failed to setup {backend} generator: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error setting up {backend} generator: {e}")
        return None


def generate_single_image(
    generator,
    prompt: str,
    output_path: str,
    num_inference_steps: int = 50,
    seed: Optional[int] = None,
    **kwargs
) -> Dict:
    """Generate a single image using the configured generator."""
    try:
        logger.info(f"Generating image: {prompt[:60]}...")
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Generate image using the generator
        saved_path, metadata = generator.generate_image(
            prompt=prompt,
            filename=Path(output_path).name,
            num_inference_steps=num_inference_steps,
            seed=seed,
            **kwargs
        )
        
        logger.info(f"✓ Image saved to: {saved_path}")
        
        return {
            'status': 'success',
            'output_path': saved_path,
            'original_prompt': prompt,
            **metadata
        }
    
    except ImageGenerationError as e:
        logger.error(f"✗ Image generation failed: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'output_path': output_path,
            'original_prompt': prompt,
        }
    except Exception as e:
        logger.error(f"✗ Unexpected error: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'output_path': output_path,
            'original_prompt': prompt,
        }


def generate_images_for_iteration(
    generator,
    prompt: str,
    prompt_id: str,
    iteration: int,
    output_dir: Path,
    num_images: int = 5,
    num_workers: int = 3,
    num_inference_steps: int = 50,
    **kwargs
) -> List[Dict]:
    """Generate multiple images for a single iteration in parallel."""
    if not generator:
        logger.warning("No image generator available, skipping image generation")
        return []
    
    # Create images subdirectory
    images_dir = output_dir / "images" / prompt_id / f"iteration_{iteration}"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🖼️  Generating {num_images} images for iteration {iteration}...")
    
    results = []
    
    # Prepare tasks for parallel generation
    tasks = []
    for i in range(num_images):
        output_path = images_dir / f"image_{i+1:02d}.png"
        # Use different seeds for variety
        seed = (iteration * 1000 + i) if kwargs.get('use_seed', True) else None
        tasks.append((generator, prompt, str(output_path), num_inference_steps, seed, i+1))
    
    # Generate images in parallel (limit workers for GPU memory)
    max_workers = 1 if hasattr(generator, 'device') and generator.device == 'cuda' else num_workers
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                generate_single_image,
                task[0], task[1], task[2], task[3], task[4], **kwargs
            ): task[5]
            for task in tasks
        }
        
        for future in as_completed(futures):
            img_num = futures[future]
            try:
                result = future.result()
                results.append(result)
                
                if result['status'] == 'success':
                    print(f"   ✓ Image {img_num}/{num_images} generated")
                else:
                    print(f"   ✗ Image {img_num}/{num_images} failed: {result.get('error', 'Unknown error')}")
            
            except Exception as e:
                logger.error(f"Image {img_num} generation failed: {e}")
                results.append({
                    'status': 'error',
                    'error': str(e),
                    'image_num': img_num
                })
    
    successful = sum(1 for r in results if r.get('status') == 'success')
    print(f"   📊 Generated {successful}/{num_images} images successfully")
    
    return results


def save_iteration_results(
    prompt_id: str,
    results: Dict,
    output_dir: Path,
    generator: Optional[object] = None,
    generate_images: bool = True,
    num_images: int = 5,
    num_inference_steps: int = 50
) -> None:
    """Save results for each iteration to separate files and generate images."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save overall results
    overall_file = output_dir / f"{prompt_id}_complete_results.json"
    with open(overall_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"📄 Saved complete results to {overall_file}")
    
    # Save each iteration separately and generate images
    for iter_data in results.get("iterations", []):
        iteration_num = iter_data["iteration"]
        enhanced_prompt = iter_data["prompt"]
        
        # Generate images for this iteration
        image_results = []
        if generate_images and generator:
            try:
                image_results = generate_images_for_iteration(
                    generator,
                    enhanced_prompt,
                    prompt_id,
                    iteration_num,
                    output_dir,
                    num_images=num_images,
                    num_inference_steps=num_inference_steps
                )
            except Exception as e:
                logger.error(f"Failed to generate images for iteration {iteration_num}: {e}")
        
        iter_file = output_dir / f"{prompt_id}_iteration_{iteration_num}.json"
        
        iter_output = {
            "prompt_id": prompt_id,
            "original_prompt": results["original_prompt"],
            "iteration": iteration_num,
            "enhanced_prompt": enhanced_prompt,
            "scores": {
                "bias": {
                    "total": iter_data["bias_score"]["total_score"],
                    "details": iter_data["bias_score"]
                },
                "diversity": {
                    "total": iter_data["diversity_score"]["total_score"],
                    "details": iter_data["diversity_score"]
                }
            },
            "thresholds_met": {
                "bias": iter_data["bias_met"],
                "diversity": iter_data["diversity_met"]
            },
            "focus_areas": iter_data["focus_areas"],
            "generated_images": image_results,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(iter_file, "w") as f:
            json.dump(iter_output, f, indent=2, default=str)
        print(f"   └─ Iteration {iteration_num} saved to {iter_file}")
    
    # Save summary
    summary_file = output_dir / f"{prompt_id}_summary.json"
    summary = {
        "prompt_id": prompt_id,
        "original_prompt": results["original_prompt"],
        "final_prompt": results["final_prompt"],
        "total_iterations": len(results.get("iterations", [])),
        "initial_scores": {
            "bias": results["initial_scores"]["bias"]["total_score"],
            "diversity": results["initial_scores"]["diversity"]["total_score"]
        },
        "final_scores": {
            "bias": results["final_scores"]["bias"]["total_score"],
            "diversity": results["final_scores"]["diversity"]["total_score"]
        },
        "thresholds": results["thresholds"],
        "improvement": {
            "bias": results["final_scores"]["bias"]["total_score"] - 
                    results["initial_scores"]["bias"]["total_score"],
            "diversity": results["final_scores"]["diversity"]["total_score"] - 
                         results["initial_scores"]["diversity"]["total_score"]
        },
        "timestamp": datetime.now().isoformat()
    }
    
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"📊 Summary saved to {summary_file}")


def run_batch_enhancement(
    prompts: List[Dict[str, str]],
    output_dir: Path,
    max_iterations: int = 5,
    bias_threshold: int = 75,
    diversity_threshold: int = 80,
    generate_images: bool = True,
    num_images_per_iteration: int = 5,
    image_backend: str = "dalle",
    image_model: Optional[str] = None,
    num_inference_steps: int = 50
) -> None:
    """Run enhancement for multiple prompts with image generation."""
    setup_logging()
    
    print("=" * 80)
    print("🎨 Dual-Pipeline Batch Enhancement")
    print("=" * 80)
    print(f"Number of prompts: {len(prompts)}")
    print(f"Max iterations per prompt: {max_iterations}")
    print(f"Bias threshold: {bias_threshold}/100")
    print(f"Diversity threshold: {diversity_threshold}/100")
    print(f"Generate images: {generate_images}")
    if generate_images:
        print(f"Image backend: {image_backend}")
        if image_model:
            print(f"Image model: {image_model}")
        print(f"Images per iteration: {num_images_per_iteration}")
        print(f"Inference steps: {num_inference_steps}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)
    
    # Create mock args
    class Args:
        pass
    
    args = Args()
    args.use_stereoset = True
    args.use_diversity_rag = True
    args.use_graphrag = True
    args.bias_threshold = bias_threshold
    args.diversity_threshold = diversity_threshold
    args.max_iterations = max_iterations
    args.stereoset_top_k = 10
    args.diversity_top_k = 8
    args.graphrag_cache = "embeddings.pkl"
    args.stereoset_cache = "stereoset_embeddings.pkl"
    args.diversity_cache = "diversity_embeddings.pkl"
    
    # Load enhancement systems once
    print("\n🔧 Loading enhancement systems...")
    stereoset_rag, diversity_rag, graphrag = load_enhancement_systems(args)
    
    if not stereoset_rag and not diversity_rag and not graphrag:
        print("❌ No enhancement systems available. Exiting.")
        return
    
    # Setup image generator
    image_generator = None
    if generate_images:
        print(f"\n🖼️  Setting up {image_backend} image generator...")
        # Ensure output directory exists before initializing generator
        images_output_dir = output_dir / "images"
        images_output_dir.mkdir(parents=True, exist_ok=True)
        
        image_generator = setup_image_generator(
            backend=image_backend,
            model=image_model,
            output_dir=images_output_dir
        )
        if not image_generator:
            print("⚠️  Image generation disabled due to setup failure")
            generate_images = False
    
    # Process each prompt
    all_results = []
    for idx, prompt_data in enumerate(prompts, 1):
        prompt_id = prompt_data["id"]
        prompt_text = prompt_data["text"]
        
        print("\n" + "=" * 80)
        print(f"Processing {idx}/{len(prompts)}: {prompt_id}")
        print("=" * 80)
        print(f"Prompt: {prompt_text}")
        
        try:
            results = run_dual_pipeline_enhancement(
                prompt_text,
                stereoset_rag,
                diversity_rag,
                graphrag,
                args
            )
            
            # Save results for this prompt with image generation
            save_iteration_results(
                prompt_id,
                results,
                output_dir,
                generator=image_generator,
                generate_images=generate_images,
                num_images=num_images_per_iteration,
                num_inference_steps=num_inference_steps
            )
            
            all_results.append({
                "prompt_id": prompt_id,
                "success": True,
                "results": results
            })
            
            print(f"\n✅ Completed {prompt_id}")
            
        except Exception as e:
            logger.error(f"Failed to process {prompt_id}: {e}")
            all_results.append({
                "prompt_id": prompt_id,
                "success": False,
                "error": str(e)
            })
            print(f"\n❌ Failed {prompt_id}: {e}")
    
    # Save batch summary
    batch_summary_file = output_dir / "batch_summary.json"
    batch_summary = {
        "total_prompts": len(prompts),
        "successful": sum(1 for r in all_results if r["success"]),
        "failed": sum(1 for r in all_results if not r["success"]),
        "prompts": [
            {
                "id": r["prompt_id"],
                "success": r["success"],
                "final_scores": {
                    "bias": r["results"]["final_scores"]["bias"]["total_score"],
                    "diversity": r["results"]["final_scores"]["diversity"]["total_score"]
                } if r["success"] else None,
                "iterations": len(r["results"].get("iterations", [])) if r["success"] else 0
            }
            for r in all_results
        ],
        "timestamp": datetime.now().isoformat()
    }
    
    with open(batch_summary_file, "w") as f:
        json.dump(batch_summary, f, indent=2, default=str)
    
    print("\n" + "=" * 80)
    print("🎉 Batch Enhancement Complete!")
    print("=" * 80)
    print(f"Total prompts: {len(prompts)}")
    print(f"Successful: {batch_summary['successful']}")
    print(f"Failed: {batch_summary['failed']}")
    print(f"Results saved to: {output_dir}")
    print(f"Batch summary: {batch_summary_file}")
    print("=" * 80)


def main():
    """Main function."""
    # Define the two prompts
    prompts = [
        {
            "id": "prompt_001_doctor",
            "text": "Generate an image of a doctor who is smiling at the camera"
        },
        {
            "id": "prompt_002_nurse",
            "text": "Generate an image of a nurse who is smiling at the camera"
        }
    ]
    
    # Output directory
    output_dir = Path(__file__).parent.parent / "results" / "dual_pipeline_enhancement" / datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Run batch enhancement with image generation
    run_batch_enhancement(
        prompts=prompts,
        output_dir=output_dir,
        max_iterations=5,
        bias_threshold=50,
        diversity_threshold=50,
        generate_images=True,
        num_images_per_iteration=5,
        image_backend="stable-diffusion",  # Options: "dalle" or "stable-diffusion"
        image_model="runwayml/stable-diffusion-v1-5",  # SD 1.5: faster, smaller (~4GB)
        num_inference_steps=30  # For Stable Diffusion (ignored for DALL-E)
    )


if __name__ == "__main__":
    main()
