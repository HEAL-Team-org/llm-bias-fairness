"""Batch Processing Pipeline for Prompt Enhancement and Image Generation.

This module provides a complete pipeline to:
1. Read prompts from a CSV file
2. Enhance each prompt using the dual-pipeline system
3. Generate images for both original and enhanced prompts
4. Save results with trackable filenames
"""

import argparse
import json
import logging
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

# Import our modules
from enhance_prompt_dual_pipeline import (
    load_enhancement_systems,
    parse_arguments,
    run_dual_pipeline_enhancement,
)

from src.generation import ImageGenerationError, create_image_generator

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Result of processing a single prompt."""

    row_index: int
    original_prompt: str
    enhanced_prompt: Optional[str] = None
    bias_score: Optional[float] = None
    diversity_score: Optional[float] = None
    original_image_path: Optional[str] = None
    enhanced_image_path: Optional[str] = None
    original_image_metadata: Optional[Dict] = None
    enhanced_image_metadata: Optional[Dict] = None
    processing_time: Optional[float] = None
    error: Optional[str] = None
    enhancement_metadata: Optional[Dict] = None
    # Visual bias evaluation results
    visual_bias_evaluation: Optional[Dict] = None


class BatchProcessor:
    """Batch processor for prompt enhancement and image generation."""

    def __init__(
        self,
        image_generator_type: str = "dalle3",
        output_dir: Union[str, Path] = "batch_results",
        image_output_dir: Union[str, Path] = "generated_images",
        enable_visual_bias_evaluation: bool = False,
        fairface_model_path: Optional[str] = None,
        dlib_model_path: Optional[str] = None,
        **image_generator_kwargs
    ):
        """Initialize the batch processor.
        
        Args:
            image_generator_type: Type of image generator ("dalle3", "mock")
            output_dir: Directory for batch processing results
            image_output_dir: Directory for generated images
            enable_visual_bias_evaluation: Whether to enable visual bias evaluation
            fairface_model_path: Path to FairFace model file
            dlib_model_path: Path to dlib shape predictor model
            **image_generator_kwargs: Additional arguments for image generator

        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.enable_visual_bias_evaluation = enable_visual_bias_evaluation

        # Initialize visual bias evaluator if enabled
        self.visual_bias_evaluator = None
        if self.enable_visual_bias_evaluation:
            try:
                from src.evaluation import VisualBiasEvaluator
                self.visual_bias_evaluator = VisualBiasEvaluator(
                    fairface_model_path=fairface_model_path,
                    dlib_model_path=dlib_model_path
                )
                logger.info("Visual bias evaluator initialized")
            except ImportError as e:
                logger.warning(f"Visual bias evaluation disabled - missing dependencies: {e}")
                self.enable_visual_bias_evaluation = False

        # Initialize image generator
        try:
            self.image_generator = create_image_generator(
                generator_type=image_generator_type,
                output_dir=image_output_dir,
                **image_generator_kwargs
            )
            logger.info(f"Initialized {image_generator_type} image generator")
        except Exception as e:
            logger.error(f"Failed to initialize image generator: {e}")
            raise

        # Initialize enhancement systems
        try:
            args = parse_arguments()
            self.stereoset_rag, self.diversity_rag, self.graphrag = load_enhancement_systems(args)
            logger.info("Initialized enhancement systems")
        except Exception as e:
            logger.error(f"Failed to initialize enhancement systems: {e}")
            raise

        self.results: List[ProcessingResult] = []

    def process_csv(
        self,
        input_csv_path: Union[str, Path],
        prompt_column: str = "prompt",
        output_csv_path: Optional[Union[str, Path]] = None,
        start_row: int = 0,
        max_rows: Optional[int] = None,
        save_intermediate: bool = True
    ) -> Tuple[str, List[ProcessingResult]]:
        """Process prompts from a CSV file.
        
        Args:
            input_csv_path: Path to input CSV file
            prompt_column: Name of column containing prompts
            output_csv_path: Path for output CSV (auto-generated if None)
            start_row: Row to start processing from
            max_rows: Maximum number of rows to process
            save_intermediate: Whether to save results after each prompt
            
        Returns:
            Tuple of (output_csv_path, processing_results)

        """
        input_path = Path(input_csv_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input CSV not found: {input_path}")

        # Load CSV
        try:
            df = pd.read_csv(input_path)
            logger.info(f"Loaded CSV with {len(df)} rows")
        except Exception as e:
            raise ValueError(f"Failed to load CSV: {e}")

        if prompt_column not in df.columns:
            raise ValueError(f"Column '{prompt_column}' not found. Available: {list(df.columns)}")

        # Determine output path
        if output_csv_path is None:
            timestamp = int(time.time())
            output_csv_path = self.output_dir / f"enhanced_prompts_{timestamp}.csv"
        else:
            output_csv_path = Path(output_csv_path)

        # Select rows to process
        end_row = min(len(df), start_row + max_rows) if max_rows else len(df)
        rows_to_process = df.iloc[start_row:end_row]

        logger.info(f"Processing rows {start_row} to {end_row-1} ({len(rows_to_process)} total)")

        # Process each prompt
        self.results = []
        for idx, row in rows_to_process.iterrows():
            try:
                result = self._process_single_prompt(
                    row_index=idx,
                    prompt=row[prompt_column]
                )
                self.results.append(result)

                logger.info(f"Processed row {idx}: {result.original_prompt[:50]}...")

                # Save intermediate results
                if save_intermediate:
                    self._save_results_to_csv(df, output_csv_path, prompt_column)

            except Exception as e:
                logger.error(f"Failed to process row {idx}: {e}")
                result = ProcessingResult(
                    row_index=idx,
                    original_prompt=row[prompt_column],
                    error=str(e)
                )
                self.results.append(result)

        # Save final results
        final_output_path = self._save_results_to_csv(df, output_csv_path, prompt_column)

        # Save detailed results as JSON
        json_output_path = output_csv_path.with_suffix(".json")
        self._save_results_to_json(json_output_path)

        # Perform visual bias evaluation if enabled
        if self.enable_visual_bias_evaluation and self.visual_bias_evaluator:
            logger.info("Performing visual bias evaluation...")
            self._perform_visual_bias_evaluation(str(output_csv_path.parent))

        logger.info(f"Batch processing complete. Results saved to: {final_output_path}")
        return str(final_output_path), self.results

    def _process_single_prompt(self, row_index: int, prompt: str) -> ProcessingResult:
        """Process a single prompt through the enhancement and image generation pipeline.
        
        Args:
            row_index: Index of the row being processed
            prompt: Original prompt text
            
        Returns:
            ProcessingResult with all processing outcomes

        """
        start_time = time.time()
        result = ProcessingResult(row_index=row_index, original_prompt=prompt)

        try:
            # Step 1: Enhance the prompt
            logger.info(f"Enhancing prompt: '{prompt[:100]}...'")

            # Create args object for the enhancement with default values
            # PRIORITIZE StereoSet and Diversity RAG over GraphRAG
            args = argparse.Namespace(
                bias_threshold=75,
                diversity_threshold=80,
                max_iterations=5,
                stereoset_top_k=10,
                diversity_top_k=10,
                graphrag_top_k=10,
                use_stereoset=True,
                use_diversity_rag=True,
                use_graphrag=False  # DISABLED: Focus on StereoSet + Diversity RAG
            )

            enhancement_result = run_dual_pipeline_enhancement(
                original_prompt=prompt,
                stereoset_rag=self.stereoset_rag,
                diversity_rag=self.diversity_rag,
                graphrag=self.graphrag,
                args=args
            )

            result.enhanced_prompt = enhancement_result.get("final_prompt", prompt)
            result.bias_score = enhancement_result.get("final_scores", {}).get("bias", {}).get("total_score")
            result.diversity_score = enhancement_result.get("final_scores", {}).get("diversity", {}).get("total_score")
            result.enhancement_metadata = enhancement_result

            logger.info(f"Enhanced prompt (bias: {result.bias_score}, diversity: {result.diversity_score})")

            # Step 2: Generate image for original prompt
            try:
                original_filename = f"row_{row_index:04d}_original"
                original_path, original_metadata = self.image_generator.generate_image(
                    prompt=prompt,
                    filename=f"{original_filename}.png"
                )
                result.original_image_path = original_path
                result.original_image_metadata = original_metadata
                logger.info(f"Generated original image: {original_path}")
            except ImageGenerationError as e:
                logger.error(f"Failed to generate original image: {e}")
                result.error = f"Original image generation failed: {e}"

            # Step 3: Generate image for enhanced prompt
            try:
                enhanced_filename = f"row_{row_index:04d}_enhanced"
                enhanced_path, enhanced_metadata = self.image_generator.generate_image(
                    prompt=result.enhanced_prompt,
                    filename=f"{enhanced_filename}.png"
                )
                result.enhanced_image_path = enhanced_path
                result.enhanced_image_metadata = enhanced_metadata
                logger.info(f"Generated enhanced image: {enhanced_path}")
            except ImageGenerationError as e:
                logger.error(f"Failed to generate enhanced image: {e}")
                if result.error:
                    result.error += f"; Enhanced image generation failed: {e}"
                else:
                    result.error = f"Enhanced image generation failed: {e}"

            result.processing_time = time.time() - start_time
            logger.info(f"Completed processing row {row_index} in {result.processing_time:.2f}s")

        except Exception as e:
            result.error = f"Enhancement failed: {e}"
            result.processing_time = time.time() - start_time
            logger.error(f"Failed to process prompt: {e}")
            logger.debug(traceback.format_exc())

        return result

    def _save_results_to_csv(
        self,
        original_df: pd.DataFrame,
        output_path: Path,
        prompt_column: str
    ) -> str:
        """Save results back to CSV format.
        
        Args:
            original_df: Original dataframe
            output_path: Path to save CSV
            prompt_column: Name of original prompt column
            
        Returns:
            Path to saved CSV file

        """
        # Create results DataFrame
        results_data = []

        for result in self.results:
            row_data = {
                "row_index": result.row_index,
                "original_prompt": result.original_prompt,
                "enhanced_prompt": result.enhanced_prompt,
                "bias_score": result.bias_score,
                "diversity_score": result.diversity_score,
                "original_image_path": result.original_image_path,
                "enhanced_image_path": result.enhanced_image_path,
                "processing_time": result.processing_time,
                "error": result.error
            }
            results_data.append(row_data)

        results_df = pd.DataFrame(results_data)

        # Merge with original data if available
        if not results_df.empty and not original_df.empty:
            # Reset index to ensure proper merging
            results_df.set_index("row_index", inplace=True)

            # Merge with original dataframe
            merged_df = original_df.join(
                results_df[["enhanced_prompt", "bias_score", "diversity_score",
                           "original_image_path", "enhanced_image_path",
                           "processing_time", "error"]],
                how="left"
            )
        else:
            merged_df = results_df

        # Save to CSV
        merged_df.to_csv(output_path, index=True)
        logger.info(f"Results saved to CSV: {output_path}")

        return str(output_path)

    def _save_results_to_json(self, output_path: Path) -> str:
        """Save detailed results to JSON format.
        
        Args:
            output_path: Path to save JSON file
            
        Returns:
            Path to saved JSON file

        """
        json_data = []

        for result in self.results:
            data = {
                "row_index": result.row_index,
                "original_prompt": result.original_prompt,
                "enhanced_prompt": result.enhanced_prompt,
                "bias_score": result.bias_score,
                "diversity_score": result.diversity_score,
                "original_image_path": result.original_image_path,
                "enhanced_image_path": result.enhanced_image_path,
                "processing_time": result.processing_time,
                "error": result.error,
                "enhancement_metadata": result.enhancement_metadata,
                "original_image_metadata": result.original_image_metadata,
                "enhanced_image_metadata": result.enhanced_image_metadata
            }
            json_data.append(data)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Detailed results saved to JSON: {output_path}")
        return str(output_path)

    def _perform_visual_bias_evaluation(self, output_dir: str):
        """Perform visual bias evaluation on generated images.
        
        Args:
            output_dir: Directory containing the processing results

        """
        if not self.visual_bias_evaluator:
            logger.warning("Visual bias evaluator not available")
            return

        try:
            # Collect generated images
            original_images = []
            enhanced_images = []

            for result in self.results:
                if result.original_image_path and os.path.exists(result.original_image_path):
                    original_images.append(result.original_image_path)
                if result.enhanced_image_path and os.path.exists(result.enhanced_image_path):
                    enhanced_images.append(result.enhanced_image_path)

            if not original_images and not enhanced_images:
                logger.warning("No images found for visual bias evaluation")
                return

            # Perform evaluation
            evaluation_dir = os.path.join(output_dir, "visual_bias_evaluation")
            os.makedirs(evaluation_dir, exist_ok=True)

            if original_images and enhanced_images:
                # Evaluate both original and enhanced
                original_metrics, enhanced_metrics = self.visual_bias_evaluator.evaluate_batch_results(
                    original_images=original_images,
                    enhanced_images=enhanced_images,
                    output_dir=evaluation_dir
                )

                logger.info(f"Visual bias evaluation complete. Results saved to: {evaluation_dir}")

                # Store evaluation results in processing results
                for result in self.results:
                    result.visual_bias_evaluation = {
                        "evaluation_completed": True,
                        "evaluation_dir": evaluation_dir,
                        "original_images_count": len(original_images),
                        "enhanced_images_count": len(enhanced_images)
                    }

            else:
                # Evaluate available images
                all_images = original_images + enhanced_images
                image_type = "original" if original_images else "enhanced"

                logger.info(f"Evaluating {len(all_images)} {image_type} images...")

                df = self.visual_bias_evaluator.analyze_images(
                    all_images,
                    os.path.join(evaluation_dir, image_type)
                )

                if not df.empty:
                    # Calculate metrics
                    bias_w = self.visual_bias_evaluator.calculate_bias_w(df)
                    bias_p = self.visual_bias_evaluator.calculate_bias_p(df)
                    ens = self.visual_bias_evaluator.calculate_ens(df)
                    kl = self.visual_bias_evaluator.calculate_kl_divergence(df)

                    # Save metrics
                    metrics_dir = os.path.join(evaluation_dir, f"{image_type}_metrics")
                    os.makedirs(metrics_dir, exist_ok=True)

                    bias_w.to_csv(os.path.join(metrics_dir, "bias_w_metrics.csv"), index=False)
                    bias_p.to_csv(os.path.join(metrics_dir, "bias_p_metrics.csv"), index=False)
                    ens.to_csv(os.path.join(metrics_dir, "ens_metrics.csv"), index=False)
                    kl.to_csv(os.path.join(metrics_dir, "kl_divergence_metrics.csv"), index=False)

                    logger.info(f"Visual bias evaluation complete. Results saved to: {metrics_dir}")

        except Exception as e:
            logger.error(f"Visual bias evaluation failed: {e}")
            # Store error in results
            for result in self.results:
                result.visual_bias_evaluation = {
                    "evaluation_completed": False,
                    "error": str(e)
                }


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Batch process prompts with enhancement and image generation")
    parser.add_argument("input_csv", help="Path to input CSV file")
    parser.add_argument("--prompt-column", default="prompt", help="Name of column containing prompts")
    parser.add_argument("--output-csv", help="Path for output CSV file")
    parser.add_argument("--start-row", type=int, default=0, help="Row to start processing from")
    parser.add_argument("--max-rows", type=int, help="Maximum number of rows to process")
    parser.add_argument("--image-generator", default="dalle3", choices=["dalle3", "mock"],
                       help="Type of image generator to use")
    parser.add_argument("--output-dir", default="batch_results", help="Directory for batch results")
    parser.add_argument("--image-dir", default="generated_images", help="Directory for generated images")
    parser.add_argument("--no-save-intermediate", action="store_true",
                       help="Don't save intermediate results")

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    try:
        # Initialize processor
        processor = BatchProcessor(
            image_generator_type=args.image_generator,
            output_dir=args.output_dir,
            image_output_dir=args.image_dir
        )

        # Process CSV
        output_path, results = processor.process_csv(
            input_csv_path=args.input_csv,
            prompt_column=args.prompt_column,
            output_csv_path=args.output_csv,
            start_row=args.start_row,
            max_rows=args.max_rows,
            save_intermediate=not args.no_save_intermediate
        )

        # Print summary
        successful = len([r for r in results if r.error is None])
        total = len(results)

        print("\nBatch processing completed!")
        print(f"Processed: {successful}/{total} prompts successfully")
        print(f"Results saved to: {output_path}")

        if successful < total:
            print(f"Failed prompts: {total - successful}")
            for result in results:
                if result.error:
                    print(f"  Row {result.row_index}: {result.error}")

    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
