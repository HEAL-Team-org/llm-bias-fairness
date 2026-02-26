"""Evaluation metrics for bias and fairness in image generation prompts.

This module implements two key metrics for evaluating bias mitigation:
1. DI (Disparate Impact): Measures ratio of occurrence rates between subgroups
2. CMMD (Conditional Maximum Mean Discrepancy): Uses CLIP embeddings to measure distribution distance

The metrics are designed to evaluate how well the prompt enhancement system
reduces bias and improves diversity in AI-generated content.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json
import numpy as np
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from collections import defaultdict, Counter

# Optional imports for CLIP functionality
try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    logging.warning("CLIP not available. CMMD metric will use mock implementation.")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logging.warning("PIL not available. Image processing will be limited.")

logger = logging.getLogger(__name__)


@dataclass
class DisparateImpactResult:
    """Results from Disparate Impact analysis."""
    ratio: float
    group1_rate: float
    group2_rate: float
    group1_name: str
    group2_name: str
    is_biased: bool
    threshold: float = 0.8
    
    def __str__(self) -> str:
        bias_status = "BIASED" if self.is_biased else "FAIR"
        return (f"DI({self.group1_name}/{self.group2_name}): {self.ratio:.3f} "
                f"({self.group1_rate:.3f}/{self.group2_rate:.3f}) - {bias_status}")


@dataclass
class CMMDResult:
    """Results from CMMD analysis."""
    distance: float
    group1_name: str
    group2_name: str
    n_samples_group1: int
    n_samples_group2: int
    kernel_type: str = "rbf"
    
    def __str__(self) -> str:
        return (f"CMMD({self.group1_name}, {self.group2_name}): {self.distance:.4f} "
                f"(n1={self.n_samples_group1}, n2={self.n_samples_group2})")


class PromptAnalyzer:
    """Analyzes enhanced prompts for demographic and diversity patterns."""
    
    def __init__(self):
        """Initialize the prompt analyzer with demographic detection patterns."""
        self.gender_patterns = {
            'male': ['man', 'men', 'male', 'gentleman', 'guy', 'boy', 'father', 'husband', 'brother', 'son'],
            'female': ['woman', 'women', 'female', 'lady', 'girl', 'mother', 'wife', 'sister', 'daughter'],
            'non_binary': ['non-binary', 'nonbinary', 'enby', 'genderqueer', 'genderfluid', 'they/them']
        }
        
        self.ethnicity_patterns = {
            'asian': ['asian', 'chinese', 'japanese', 'korean', 'indian', 'thai', 'vietnamese', 'filipino'],
            'african': ['african', 'black', 'nigerian', 'kenyan', 'ethiopian', 'ghanaian'],
            'latino': ['latino', 'latina', 'hispanic', 'mexican', 'colombian', 'argentinian', 'puerto rican'],
            'middle_eastern': ['middle eastern', 'arab', 'persian', 'iranian', 'lebanese', 'syrian'],
            'european': ['european', 'white', 'caucasian', 'british', 'german', 'french', 'italian', 'scandinavian'],
            'native': ['native american', 'indigenous', 'aboriginal', 'first nations']
        }
        
        self.age_patterns = {
            'young': ['young', 'teenager', 'teen', 'youth', 'adolescent', 'child', 'kid'],
            'middle_aged': ['middle-aged', 'adult', 'mature'],
            'senior': ['senior', 'elderly', 'older', 'aged', 'grandparent', 'grandmother', 'grandfather']
        }
        
        self.profession_patterns = {
            'doctor': ['doctor', 'physician', 'surgeon', 'medical'],
            'engineer': ['engineer', 'engineering', 'technical'],
            'teacher': ['teacher', 'educator', 'professor', 'instructor'],
            'nurse': ['nurse', 'nursing'],
            'lawyer': ['lawyer', 'attorney', 'legal'],
            'ceo': ['ceo', 'executive', 'manager', 'director'],
            'scientist': ['scientist', 'researcher', 'research']
        }
    
    def extract_demographics(self, prompt: str) -> Dict[str, List[str]]:
        """Extract demographic mentions from a prompt."""
        prompt_lower = prompt.lower()
        demographics = {
            'gender': [],
            'ethnicity': [],
            'age': [],
            'profession': []
        }
        
        # Extract gender mentions
        for gender, patterns in self.gender_patterns.items():
            if any(pattern in prompt_lower for pattern in patterns):
                demographics['gender'].append(gender)
        
        # Extract ethnicity mentions
        for ethnicity, patterns in self.ethnicity_patterns.items():
            if any(pattern in prompt_lower for pattern in patterns):
                demographics['ethnicity'].append(ethnicity)
        
        # Extract age mentions
        for age, patterns in self.age_patterns.items():
            if any(pattern in prompt_lower for pattern in patterns):
                demographics['age'].append(age)
        
        # Extract profession mentions
        for profession, patterns in self.profession_patterns.items():
            if any(pattern in prompt_lower for pattern in patterns):
                demographics['profession'].append(profession)
        
        return demographics


class DisparateImpactCalculator:
    """Calculates Disparate Impact (DI) metric for bias evaluation.
    
    DI measures the ratio of occurrence rates between different subgroups.
    A ratio below 0.8 typically indicates bias favoring one group over another.
    
    Formula: DI = P(outcome | group1) / P(outcome | group2)
    """
    
    def __init__(self, threshold: float = 0.8):
        """Initialize with bias threshold.
        
        Args:
            threshold: Threshold below which bias is considered present (default: 0.8)
        """
        self.threshold = threshold
        self.analyzer = PromptAnalyzer()
    
    def calculate_di_for_prompts(
        self, 
        original_prompts: List[str], 
        enhanced_prompts: List[str],
        demographic_category: str = 'gender',
        group1: str = 'female',
        group2: str = 'male'
    ) -> Tuple[DisparateImpactResult, DisparateImpactResult]:
        """Calculate DI for original vs enhanced prompts.
        
        Args:
            original_prompts: List of original prompts
            enhanced_prompts: List of enhanced prompts  
            demographic_category: Category to analyze ('gender', 'ethnicity', 'age')
            group1: First group to compare
            group2: Second group to compare
            
        Returns:
            Tuple of (original_di_result, enhanced_di_result)
        """
        def calculate_group_rates(prompts: List[str]) -> Tuple[float, float]:
            group1_count = 0
            group2_count = 0
            total_prompts = len(prompts)
            
            for prompt in prompts:
                demographics = self.analyzer.extract_demographics(prompt)
                category_mentions = demographics.get(demographic_category, [])
                
                if group1 in category_mentions:
                    group1_count += 1
                if group2 in category_mentions:
                    group2_count += 1
            
            group1_rate = group1_count / total_prompts if total_prompts > 0 else 0
            group2_rate = group2_count / total_prompts if total_prompts > 0 else 0
            
            return group1_rate, group2_rate
        
        # Calculate rates for original prompts
        orig_group1_rate, orig_group2_rate = calculate_group_rates(original_prompts)
        orig_ratio = orig_group1_rate / orig_group2_rate if orig_group2_rate > 0 else float('inf')
        orig_is_biased = orig_ratio < self.threshold or orig_ratio > (1 / self.threshold)
        
        # Calculate rates for enhanced prompts
        enh_group1_rate, enh_group2_rate = calculate_group_rates(enhanced_prompts)
        enh_ratio = enh_group1_rate / enh_group2_rate if enh_group2_rate > 0 else float('inf')
        enh_is_biased = enh_ratio < self.threshold or enh_ratio > (1 / self.threshold)
        
        original_result = DisparateImpactResult(
            ratio=orig_ratio,
            group1_rate=orig_group1_rate,
            group2_rate=orig_group2_rate,
            group1_name=group1,
            group2_name=group2,
            is_biased=orig_is_biased,
            threshold=self.threshold
        )
        
        enhanced_result = DisparateImpactResult(
            ratio=enh_ratio,
            group1_rate=enh_group1_rate,
            group2_rate=enh_group2_rate,
            group1_name=group1,
            group2_name=group2,
            is_biased=enh_is_biased,
            threshold=self.threshold
        )
        
        return original_result, enhanced_result


class CMMDCalculator:
    """Calculates Conditional Maximum Mean Discrepancy using CLIP embeddings.
    
    CMMD measures the distance between distributions of generated content
    in CLIP embedding space. Lower CMMD indicates more similar distributions,
    which suggests less bias between groups.
    """
    
    def __init__(self, model_name: str = "ViT-B/32", device: Optional[str] = None):
        """Initialize CMMD calculator with CLIP model.
        
        Args:
            model_name: CLIP model to use
            device: Device to run computations on
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.preprocess = None
        self.analyzer = PromptAnalyzer()
        
        if CLIP_AVAILABLE:
            try:
                self.model, self.preprocess = clip.load(model_name, device=self.device)
                logger.info(f"Loaded CLIP model {model_name} on {self.device}")
            except Exception as e:
                logger.warning(f"Failed to load CLIP model: {e}")
                self.model = None
        else:
            logger.warning("CLIP not available, using mock implementation")
    
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """Encode texts using CLIP text encoder.
        
        Args:
            texts: List of text prompts to encode
            
        Returns:
            Tensor of text embeddings
        """
        if self.model is None:
            # Mock implementation when CLIP is not available
            logger.warning("Using mock CLIP embeddings")
            return torch.randn(len(texts), 512)  # Mock 512-dim embeddings
        
        try:
            text_tokens = clip.tokenize(texts, truncate=True).to(self.device)
            with torch.no_grad():
                text_features = self.model.encode_text(text_tokens)
                text_features = F.normalize(text_features, p=2, dim=1)
            return text_features.cpu()
        except Exception as e:
            logger.error(f"Error encoding texts: {e}")
            return torch.randn(len(texts), 512)  # Fallback to mock embeddings
    
    def compute_mmd(
        self, 
        X: torch.Tensor, 
        Y: torch.Tensor, 
        kernel: str = "rbf", 
        gamma: float = 1.0
    ) -> float:
        """Compute Maximum Mean Discrepancy between two sets of embeddings.
        
        Args:
            X: First set of embeddings [n1, d]
            Y: Second set of embeddings [n2, d]
            kernel: Kernel type ('rbf' or 'linear')
            gamma: RBF kernel parameter
            
        Returns:
            MMD distance
        """
        def rbf_kernel(X: torch.Tensor, Y: torch.Tensor, gamma: float) -> torch.Tensor:
            """RBF kernel computation."""
            # Compute pairwise distances
            X_norm = (X ** 2).sum(1).view(-1, 1)
            Y_norm = (Y ** 2).sum(1).view(1, -1)
            dist = X_norm + Y_norm - 2.0 * torch.mm(X, Y.transpose(0, 1))
            return torch.exp(-gamma * dist)
        
        def linear_kernel(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
            """Linear kernel computation."""
            return torch.mm(X, Y.transpose(0, 1))
        
        n1, n2 = X.shape[0], Y.shape[0]
        
        if kernel == "rbf":
            K_XX = rbf_kernel(X, X, gamma)
            K_YY = rbf_kernel(Y, Y, gamma)
            K_XY = rbf_kernel(X, Y, gamma)
        else:  # linear kernel
            K_XX = linear_kernel(X, X)
            K_YY = linear_kernel(Y, Y)
            K_XY = linear_kernel(X, Y)
        
        # MMD computation
        term1 = K_XX.sum() / (n1 * n1)
        term2 = K_YY.sum() / (n2 * n2)
        term3 = K_XY.sum() / (n1 * n2)
        
        mmd = term1 + term2 - 2 * term3
        return float(mmd)
    
    def calculate_cmmd_for_groups(
        self,
        prompts: List[str],
        demographic_category: str = 'gender',
        group1: str = 'female',
        group2: str = 'male',
        kernel: str = "rbf"
    ) -> Optional[CMMDResult]:
        """Calculate CMMD between two demographic groups in prompts.
        
        Args:
            prompts: List of prompts to analyze
            demographic_category: Category to analyze
            group1: First group name
            group2: Second group name
            kernel: Kernel type for MMD computation
            
        Returns:
            CMMD result or None if insufficient data
        """
        # Separate prompts by groups
        group1_prompts = []
        group2_prompts = []
        
        for prompt in prompts:
            demographics = self.analyzer.extract_demographics(prompt)
            category_mentions = demographics.get(demographic_category, [])
            
            if group1 in category_mentions:
                group1_prompts.append(prompt)
            elif group2 in category_mentions:
                group2_prompts.append(prompt)
        
        # Need minimum samples for meaningful comparison
        if len(group1_prompts) < 5 or len(group2_prompts) < 5:
            logger.warning(f"Insufficient samples for CMMD: {group1}={len(group1_prompts)}, {group2}={len(group2_prompts)}")
            return None
        
        # Encode prompts
        group1_embeddings = self.encode_texts(group1_prompts)
        group2_embeddings = self.encode_texts(group2_prompts)
        
        # Compute CMMD
        cmmd_distance = self.compute_mmd(group1_embeddings, group2_embeddings, kernel=kernel)
        
        return CMMDResult(
            distance=cmmd_distance,
            group1_name=group1,
            group2_name=group2,
            n_samples_group1=len(group1_prompts),
            n_samples_group2=len(group2_prompts),
            kernel_type=kernel
        )


class BiasEvaluator:
    """Comprehensive bias evaluation system combining DI and CMMD metrics."""
    
    def __init__(self, di_threshold: float = 0.8, clip_model: str = "ViT-B/32"):
        """Initialize bias evaluator.
        
        Args:
            di_threshold: Threshold for disparate impact detection
            clip_model: CLIP model name for CMMD calculation
        """
        self.di_calculator = DisparateImpactCalculator(threshold=di_threshold)
        self.cmmd_calculator = CMMDCalculator(model_name=clip_model)
        
    def evaluate_enhancement_impact(
        self,
        original_prompts: List[str],
        enhanced_prompts: List[str],
        evaluation_groups: Optional[List[Dict]] = None
    ) -> Dict:
        """Evaluate the impact of prompt enhancement on bias metrics.
        
        Args:
            original_prompts: Original prompts before enhancement
            enhanced_prompts: Enhanced prompts after bias mitigation
            evaluation_groups: List of group specifications for evaluation
            
        Returns:
            Comprehensive evaluation results
        """
        if evaluation_groups is None:
            evaluation_groups = [
                {'category': 'gender', 'group1': 'female', 'group2': 'male'},
                {'category': 'ethnicity', 'group1': 'asian', 'group2': 'european'},
                {'category': 'age', 'group1': 'young', 'group2': 'senior'}
            ]
        
        results = {
            'disparate_impact': {},
            'cmmd': {},
            'summary': {}
        }
        
        # Calculate DI for each group comparison
        for group_spec in evaluation_groups:
            category = group_spec['category']
            group1 = group_spec['group1']
            group2 = group_spec['group2']
            
            comparison_key = f"{category}_{group1}_vs_{group2}"
            
            # Disparate Impact Analysis
            try:
                orig_di, enh_di = self.di_calculator.calculate_di_for_prompts(
                    original_prompts, enhanced_prompts,
                    demographic_category=category,
                    group1=group1, group2=group2
                )
                
                results['disparate_impact'][comparison_key] = {
                    'original': {
                        'ratio': orig_di.ratio,
                        'group1_rate': orig_di.group1_rate,
                        'group2_rate': orig_di.group2_rate,
                        'is_biased': orig_di.is_biased
                    },
                    'enhanced': {
                        'ratio': enh_di.ratio,
                        'group1_rate': enh_di.group1_rate,
                        'group2_rate': enh_di.group2_rate,
                        'is_biased': enh_di.is_biased
                    },
                    'improvement': orig_di.is_biased and not enh_di.is_biased
                }
                
                logger.info(f"DI Analysis - {comparison_key}:")
                logger.info(f"  Original: {orig_di}")
                logger.info(f"  Enhanced: {enh_di}")
                
            except Exception as e:
                logger.error(f"Error calculating DI for {comparison_key}: {e}")
                results['disparate_impact'][comparison_key] = {'error': str(e)}
            
            # CMMD Analysis
            try:
                orig_cmmd = self.cmmd_calculator.calculate_cmmd_for_groups(
                    original_prompts, category, group1, group2
                )
                enh_cmmd = self.cmmd_calculator.calculate_cmmd_for_groups(
                    enhanced_prompts, category, group1, group2
                )
                
                if orig_cmmd and enh_cmmd:
                    results['cmmd'][comparison_key] = {
                        'original': {
                            'distance': orig_cmmd.distance,
                            'n_samples_group1': orig_cmmd.n_samples_group1,
                            'n_samples_group2': orig_cmmd.n_samples_group2
                        },
                        'enhanced': {
                            'distance': enh_cmmd.distance,
                            'n_samples_group1': enh_cmmd.n_samples_group1,
                            'n_samples_group2': enh_cmmd.n_samples_group2
                        },
                        'improvement': enh_cmmd.distance < orig_cmmd.distance
                    }
                    
                    logger.info(f"CMMD Analysis - {comparison_key}:")
                    logger.info(f"  Original: {orig_cmmd}")
                    logger.info(f"  Enhanced: {enh_cmmd}")
                else:
                    results['cmmd'][comparison_key] = {'error': 'Insufficient samples'}
                    
            except Exception as e:
                logger.error(f"Error calculating CMMD for {comparison_key}: {e}")
                results['cmmd'][comparison_key] = {'error': str(e)}
        
        # Generate summary
        results['summary'] = self._generate_summary(results)
        
        return results
    
    def _generate_summary(self, results: Dict) -> Dict:
        """Generate summary statistics from evaluation results."""
        summary = {
            'total_comparisons': len(results['disparate_impact']),
            'di_improvements': 0,
            'cmmd_improvements': 0,
            'overall_bias_reduction': False
        }
        
        di_improvements = 0
        cmmd_improvements = 0
        
        for key in results['disparate_impact']:
            di_result = results['disparate_impact'][key]
            if not isinstance(di_result, dict) or 'error' in di_result:
                continue
                
            if di_result.get('improvement', False):
                di_improvements += 1
        
        for key in results['cmmd']:
            cmmd_result = results['cmmd'][key]
            if not isinstance(cmmd_result, dict) or 'error' in cmmd_result:
                continue
                
            if cmmd_result.get('improvement', False):
                cmmd_improvements += 1
        
        summary['di_improvements'] = di_improvements
        summary['cmmd_improvements'] = cmmd_improvements
        summary['overall_bias_reduction'] = (di_improvements + cmmd_improvements) > 0
        
        return summary
    
    def save_evaluation_results(self, results: Dict, output_path: str) -> None:
        """Save evaluation results to JSON file."""
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Evaluation results saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")


# Example usage and testing functions
def create_sample_evaluation_data() -> Tuple[List[str], List[str]]:
    """Create sample data for testing the evaluation metrics."""
    original_prompts = [
        "a doctor examining a patient",
        "an engineer working on a project", 
        "a nurse caring for patients",
        "a teacher in a classroom",
        "a scientist in a laboratory",
        "a lawyer in court",
        "a CEO in a meeting",
        "a chef cooking in a kitchen"
    ]
    
    enhanced_prompts = [
        "a doctor examining a patient, featuring people of diverse ages including young adults and seniors, representing various ethnicities including Asian, African, Latino, and European backgrounds, with both men and women medical professionals",
        "an engineer working on a project, showing a diverse team of engineers including women, men, and non-binary individuals from different cultural backgrounds, with various ages and abilities represented",
        "a nurse caring for patients, depicting nurses of different genders, ethnicities, and ages, including male nurses, female nurses, and non-binary healthcare workers from diverse cultural backgrounds",
        "a teacher in a classroom, featuring educators of various ethnicities, genders, and ages, teaching students from diverse backgrounds in an inclusive educational environment",
        "a scientist in a laboratory, showing researchers of different genders, ethnicities, and ages working together, including women in STEM, people with disabilities, and scientists from various cultural backgrounds",
        "a lawyer in court, depicting legal professionals of diverse backgrounds, genders, and ethnicities, representing the global diversity of the legal profession",
        "a CEO in a meeting, featuring business executives of various genders, ethnicities, and ages, including women leaders, people of color, and leaders from different cultural backgrounds",
        "a chef cooking in a kitchen, showing culinary professionals of diverse backgrounds, genders, and cultural traditions, representing global cuisine and inclusive kitchens"
    ]
    
    return original_prompts, enhanced_prompts


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    original_prompts, enhanced_prompts = create_sample_evaluation_data()
    
    # Initialize evaluator
    evaluator = BiasEvaluator()
    
    # Run evaluation
    results = evaluator.evaluate_enhancement_impact(original_prompts, enhanced_prompts)
    
    # Print summary
    print("\n" + "=" * 80)
    print("BIAS EVALUATION RESULTS")
    print("=" * 80)
    
    summary = results['summary']
    print(f"Total comparisons: {summary['total_comparisons']}")
    print(f"DI improvements: {summary['di_improvements']}")
    print(f"CMMD improvements: {summary['cmmd_improvements']}")
    print(f"Overall bias reduction: {summary['overall_bias_reduction']}")
    
    # Save results
    evaluator.save_evaluation_results(results, "bias_evaluation_results.json")
