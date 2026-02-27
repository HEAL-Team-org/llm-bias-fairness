"""Evaluation Layer - Bias and diversity evaluation metrics.

This module provides interfaces and implementations for evaluating bias
and diversity in generated images and text, including visual bias analysis
and demographic fairness metrics.

Exports:
    - VisualBiasEvaluator: Visual bias and diversity evaluator
    - DemographicPrediction: Demographic prediction dataclass
    - BiasMetrics: Bias metrics container dataclass
"""

from src.evaluation.visual_bias_evaluator import (
    BiasMetrics,
    DemographicPrediction,
    VisualBiasEvaluator,
)

__all__ = [
    "BiasMetrics",
    "DemographicPrediction",
    "VisualBiasEvaluator",
]
