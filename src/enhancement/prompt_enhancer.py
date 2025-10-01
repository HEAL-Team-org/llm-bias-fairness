"""Prompt Enhancement System.

This module provides a unified interface for prompt enhancement,
combining bias mitigation and diversity enhancement strategies.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.knowledge import DiversityRAG, GraphRAG, StereoSetRAG

logger = logging.getLogger(__name__)


@dataclass
class EnhancementConfig:
    """Configuration for enhancement system."""

    # System flags
    use_stereoset: bool = True
    use_diversity_rag: bool = True
    use_graphrag: bool = True

    # Thresholds
    bias_threshold: int = 75
    diversity_threshold: int = 80

    # Iteration settings
    max_iterations: int = 5

    # Top-k settings
    stereoset_top_k: int = 10
    diversity_top_k: int = 10
    graphrag_top_k: int = 10

    # Model settings
    openai_model: str = "gpt-4"
    temperature: float = 0.7


@dataclass
class EnhancementResult:
    """Result of prompt enhancement."""

    original_prompt: str
    final_prompt: str

    # Scores
    initial_bias_score: float
    final_bias_score: float
    initial_diversity_score: float
    final_diversity_score: float

    # Enhancement data
    iterations: int
    negative_examples: List[str] = field(default_factory=list)
    diversity_recommendations: List[str] = field(default_factory=list)

    # Threshold achievement
    bias_threshold_met: bool = False
    diversity_threshold_met: bool = False

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def bias_improvement(self) -> float:
        """Calculate bias score improvement."""
        return self.final_bias_score - self.initial_bias_score

    @property
    def diversity_improvement(self) -> float:
        """Calculate diversity score improvement."""
        return self.final_diversity_score - self.initial_diversity_score

    @property
    def both_thresholds_met(self) -> bool:
        """Check if both thresholds were met."""
        return self.bias_threshold_met and self.diversity_threshold_met


class EnhancementSystem:
    """Unified prompt enhancement system.
    
    This class orchestrates multiple enhancement strategies:
    - Stereotype detection and bias mitigation (StereoSet)
    - Cultural diversity enhancement (CultureBank)
    - Knowledge-based enhancement (GraphRAG)
    
    Example:
        >>> config = EnhancementConfig(bias_threshold=75, diversity_threshold=80)
        >>> system = EnhancementSystem(config)
        >>> result = system.enhance("a doctor")
        >>> print(result.final_prompt)
        >>> print(f"Bias: {result.final_bias_score}, Diversity: {result.final_diversity_score}")

    """

    def __init__(
        self,
        config: Optional[EnhancementConfig] = None,
        stereoset_rag: Optional[StereoSetRAG] = None,
        diversity_rag: Optional[DiversityRAG] = None,
        graphrag: Optional[GraphRAG] = None
    ):
        """Initialize the enhancement system.
        
        Args:
            config: Enhancement configuration
            stereoset_rag: Pre-initialized StereoSetRAG instance
            diversity_rag: Pre-initialized DiversityRAG instance
            graphrag: Pre-initialized GraphRAG instance

        """
        self.config = config or EnhancementConfig()

        # Store RAG systems
        self.stereoset_rag = stereoset_rag if self.config.use_stereoset else None
        self.diversity_rag = diversity_rag if self.config.use_diversity_rag else None
        self.graphrag = graphrag if self.config.use_graphrag else None

        logger.info("EnhancementSystem initialized")
        if self.stereoset_rag:
            logger.info("  - StereoSet RAG enabled")
        if self.diversity_rag:
            logger.info("  - Diversity RAG enabled")
        if self.graphrag:
            logger.info("  - GraphRAG enabled")

    def enhance(self, prompt: str, **kwargs) -> EnhancementResult:
        """Enhance a prompt for bias mitigation and diversity.
        
        Args:
            prompt: Original prompt to enhance
            **kwargs: Additional enhancement parameters
            
        Returns:
            EnhancementResult with enhanced prompt and metrics

        """
        logger.info(f"Enhancing prompt: {prompt}")

        # Get initial scores
        initial_bias = self._score_bias(prompt)
        initial_diversity = self._score_diversity(prompt)

        # Gather enhancement data
        negative_examples = []
        diversity_recommendations = []

        if self.stereoset_rag:
            logger.info("Analyzing for stereotypes...")
            negative_examples = self._get_negative_examples(prompt)

        if self.diversity_rag:
            logger.info("Analyzing for diversity opportunities...")
            diversity_recommendations = self._get_diversity_recommendations(prompt)

        # For now, return a simple enhancement result
        # In a full implementation, this would iterate with LLM enhancement
        enhanced_prompt = self._simple_enhance(
            prompt,
            negative_examples,
            diversity_recommendations
        )

        # Get final scores
        final_bias = self._score_bias(enhanced_prompt)
        final_diversity = self._score_diversity(enhanced_prompt)

        # Create result
        result = EnhancementResult(
            original_prompt=prompt,
            final_prompt=enhanced_prompt,
            initial_bias_score=initial_bias,
            final_bias_score=final_bias,
            initial_diversity_score=initial_diversity,
            final_diversity_score=final_diversity,
            iterations=1,
            negative_examples=negative_examples,
            diversity_recommendations=diversity_recommendations,
            bias_threshold_met=final_bias >= self.config.bias_threshold,
            diversity_threshold_met=final_diversity >= self.config.diversity_threshold
        )

        logger.info(f"Enhancement complete: Bias {initial_bias:.1f}→{final_bias:.1f}, "
                   f"Diversity {initial_diversity:.1f}→{final_diversity:.1f}")

        return result

    def _score_bias(self, prompt: str) -> float:
        """Score prompt for bias mitigation (0-100).
        
        Higher scores indicate better bias mitigation.
        """
        # Simplified scoring - in full implementation would use more sophisticated metrics
        score = 50.0  # Base score

        # Check for inclusive language markers
        inclusive_terms = ["diverse", "various", "multiple", "different", "inclusive"]
        for term in inclusive_terms:
            if term in prompt.lower():
                score += 5.0

        # Check for stereotypical language (simple heuristic)
        if self.stereoset_rag and hasattr(self.stereoset_rag, "get_negative_examples"):
            negative_examples = self._get_negative_examples(prompt)
            if len(negative_examples) > 0:
                score -= len(negative_examples) * 5.0

        return max(0.0, min(100.0, score))

    def _score_diversity(self, prompt: str) -> float:
        """Score prompt for diversity (0-100).
        
        Higher scores indicate better diversity representation.
        """
        # Simplified scoring - in full implementation would use more sophisticated metrics
        score = 50.0  # Base score

        # Check for diversity indicators
        diversity_terms = [
            "diverse", "multicultural", "various ethnicities", "different backgrounds",
            "inclusive", "representation", "varied", "multiple cultures"
        ]
        for term in diversity_terms:
            if term in prompt.lower():
                score += 8.0

        # Check for cultural specificity
        if self.diversity_rag and hasattr(self.diversity_rag, "get_diversity_recommendations"):
            recommendations = self._get_diversity_recommendations(prompt)
            if len(recommendations) > 0:
                score += min(20.0, len(recommendations) * 3.0)

        return max(0.0, min(100.0, score))

    def _get_negative_examples(self, prompt: str) -> List[str]:
        """Get stereotype examples to avoid."""
        if not self.stereoset_rag:
            return []

        try:
            examples = self.stereoset_rag.get_negative_examples(prompt, top_k=self.config.stereoset_top_k)
            return examples if examples else []
        except Exception as e:
            logger.warning(f"Failed to get negative examples: {e}")
            return []

    def _get_diversity_recommendations(self, prompt: str) -> List[str]:
        """Get diversity enhancement recommendations."""
        if not self.diversity_rag:
            return []

        try:
            recommendations = self.diversity_rag.get_diversity_recommendations(
                prompt,
                top_k=self.config.diversity_top_k
            )
            return recommendations if recommendations else []
        except Exception as e:
            logger.warning(f"Failed to get diversity recommendations: {e}")
            return []

    def _simple_enhance(
        self,
        prompt: str,
        negative_examples: List[str],
        diversity_recommendations: List[str]
    ) -> str:
        """Apply simple prompt enhancement.
        
        This is a simplified enhancement for testing. In production,
        this would use LLM-based enhancement with the full dual-pipeline approach.
        """
        enhanced = prompt

        # Add diversity language if not present
        if diversity_recommendations and "diverse" not in enhanced.lower():
            # Extract key diversity elements
            if len(diversity_recommendations) > 0:
                enhanced = f"{enhanced} with diverse representation"

        # Add inclusive language if not present
        if "various" not in enhanced.lower() and "different" not in enhanced.lower():
            enhanced = enhanced.replace(" a ", " a person from various backgrounds as a ")
            if enhanced == prompt:  # If no replacement made
                enhanced = f"{enhanced} showing diversity"

        return enhanced

    def batch_enhance(self, prompts: List[str], **kwargs) -> List[EnhancementResult]:
        """Enhance multiple prompts.
        
        Args:
            prompts: List of prompts to enhance
            **kwargs: Additional enhancement parameters
            
        Returns:
            List of EnhancementResult objects

        """
        results = []
        for i, prompt in enumerate(prompts, 1):
            logger.info(f"Enhancing prompt {i}/{len(prompts)}")
            result = self.enhance(prompt, **kwargs)
            results.append(result)
        return results

    @classmethod
    def from_config_file(cls, config_path: str) -> "EnhancementSystem":
        """Create enhancement system from configuration file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Initialized EnhancementSystem

        """
        # This would load from a YAML config file
        # For now, just use defaults
        logger.info(f"Loading configuration from {config_path}")
        config = EnhancementConfig()
        return cls(config)
