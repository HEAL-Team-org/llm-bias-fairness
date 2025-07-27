#!/usr/bin/env python3
"""
Integration example showing how to use evaluation metrics with the GraphRAG system.
This script demonstrates end-to-end bias evaluation for prompt enhancement.
"""

import json
import logging
from typing import List, Dict, Any
from pathlib import Path

# Local imports
from src.evaluation_metrics import BiasEvaluator
from src.graphrag import GraphRAG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IntegratedBiasEvaluator:
    """
    Integrated evaluation system combining GraphRAG enhancement with bias metrics.
    """
    
    def __init__(self, config_path: str = None):
        """Initialize the integrated evaluator."""
        self.config = self._load_config(config_path)
        self.graph_rag = GraphRAG()
        self.bias_evaluator = BiasEvaluator(
            di_threshold=self.config.get('di_threshold', 0.8),
            clip_model=self.config.get('clip_model', 'ViT-B/32')
        )
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        
        return {
            'di_threshold': 0.8,
            'clip_model': 'ViT-B/32',
            'evaluation_groups': [
                {'category': 'gender', 'group1': 'female', 'group2': 'male'},
                {'category': 'ethnicity', 'group1': 'white', 'group2': 'black'},
                {'category': 'age', 'group1': 'young', 'group2': 'elderly'},
                {'category': 'profession', 'group1': 'doctor', 'group2': 'nurse'}
            ],
            'enhancement_methods': ['semantic', 'cultural', 'professional']
        }
    
    def enhance_prompts_with_graphrag(self, prompts: List[str]) -> List[str]:
        """Enhance prompts using GraphRAG system."""
        enhanced_prompts = []
        
        for prompt in prompts:
            try:
                # For now, use a simple mock enhancement that adds diversity language
                # In a real implementation, this would use the full GraphRAG pipeline
                enhanced = self._mock_enhance_prompt(prompt)
                enhanced_prompts.append(enhanced)
                logger.debug(f"Enhanced '{prompt}' -> '{enhanced}'")
            except Exception as e:
                logger.warning(f"Failed to enhance prompt '{prompt}': {e}")
                enhanced_prompts.append(prompt)  # Fallback to original
                
        return enhanced_prompts
    
    def _mock_enhance_prompt(self, prompt: str) -> str:
        """Mock enhancement that adds diversity elements for testing."""
        diversity_additions = [
            "featuring people of diverse ages, ethnicities, and genders",
            "representing various cultural backgrounds and abilities", 
            "with inclusive representation across all demographics",
            "showing diverse individuals from different walks of life",
            "depicting people of all backgrounds working together"
        ]
        
        # Simple heuristic: add diversity language if not already present
        if any(word in prompt.lower() for word in ['diverse', 'inclusive', 'various', 'all']):
            return prompt  # Already enhanced
        
        # Add diversity element based on prompt hash for consistency
        diversity_idx = hash(prompt) % len(diversity_additions)
        return f"{prompt}, {diversity_additions[diversity_idx]}"
    
    def evaluate_enhancement_pipeline(
        self, 
        input_prompts: List[str],
        save_results: bool = True,
        output_path: str = 'evaluation_results.json'
    ) -> Dict[str, Any]:
        """
        Run complete evaluation pipeline:
        1. Enhance prompts with GraphRAG
        2. Evaluate bias metrics
        3. Generate comprehensive report
        """
        logger.info(f"Starting evaluation pipeline with {len(input_prompts)} prompts")
        
        # Step 1: Enhance prompts
        logger.info("Enhancing prompts with GraphRAG...")
        enhanced_prompts = self.enhance_prompts_with_graphrag(input_prompts)
        
        # Step 2: Evaluate bias metrics
        logger.info("Evaluating bias metrics...")
        evaluation_results = self.bias_evaluator.evaluate_enhancement_impact(
            original_prompts=input_prompts,
            enhanced_prompts=enhanced_prompts,
            evaluation_groups=self.config['evaluation_groups']
        )
        
        # Step 3: Generate comprehensive report
        report = self._generate_evaluation_report(
            input_prompts, enhanced_prompts, evaluation_results
        )
        
        # Step 4: Save results if requested
        if save_results:
            self._save_results(report, output_path)
            logger.info(f"Results saved to {output_path}")
        
        return report
    
    def _generate_evaluation_report(
        self,
        original_prompts: List[str],
        enhanced_prompts: List[str],
        evaluation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive evaluation report."""
        
        report = {
            'metadata': {
                'total_prompts': len(original_prompts),
                'enhancement_success_rate': self._calculate_success_rate(
                    original_prompts, enhanced_prompts
                ),
                'evaluation_timestamp': evaluation_results.get('metadata', {}).get('timestamp'),
                'config': self.config
            },
            'prompt_analysis': {
                'original_prompts': original_prompts,
                'enhanced_prompts': enhanced_prompts,
                'enhancement_examples': self._get_enhancement_examples(
                    original_prompts, enhanced_prompts
                )
            },
            'bias_metrics': evaluation_results,
            'summary': self._generate_summary(evaluation_results),
            'recommendations': self._generate_recommendations(evaluation_results)
        }
        
        return report
    
    def _calculate_success_rate(
        self, 
        original: List[str], 
        enhanced: List[str]
    ) -> float:
        """Calculate the rate of successful enhancements."""
        if not original:
            return 0.0
        
        successful = sum(1 for orig, enh in zip(original, enhanced) if orig != enh)
        return successful / len(original)
    
    def _get_enhancement_examples(
        self,
        original: List[str],
        enhanced: List[str],
        max_examples: int = 5
    ) -> List[Dict[str, str]]:
        """Get examples of prompt enhancements."""
        examples = []
        
        for orig, enh in zip(original, enhanced):
            if orig != enh and len(examples) < max_examples:
                examples.append({
                    'original': orig,
                    'enhanced': enh,
                    'improvement': self._describe_improvement(orig, enh)
                })
        
        return examples
    
    def _describe_improvement(self, original: str, enhanced: str) -> str:
        """Describe the type of improvement made."""
        improvements = []
        
        if 'diverse' in enhanced.lower() and 'diverse' not in original.lower():
            improvements.append('added diversity language')
        
        if 'inclusive' in enhanced.lower() and 'inclusive' not in original.lower():
            improvements.append('added inclusive language')
        
        if len(enhanced) > len(original):
            improvements.append('expanded description')
        
        if 'cultural' in enhanced.lower():
            improvements.append('added cultural context')
        
        return ', '.join(improvements) if improvements else 'general enhancement'
    
    def _generate_summary(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary of evaluation results."""
        summary = evaluation_results.get('summary', {})
        
        return {
            'overall_assessment': self._assess_overall_performance(summary),
            'key_findings': self._extract_key_findings(evaluation_results),
            'bias_reduction_achieved': summary.get('overall_bias_reduction', False),
            'most_improved_groups': self._find_most_improved_groups(evaluation_results),
            'areas_needing_attention': self._find_problematic_areas(evaluation_results)
        }
    
    def _assess_overall_performance(self, summary: Dict[str, Any]) -> str:
        """Assess overall performance level."""
        bias_reduction = summary.get('overall_bias_reduction', False)
        di_improvements = summary.get('di_improvements', 0)
        cmmd_improvements = summary.get('cmmd_improvements', 0)
        
        if bias_reduction and di_improvements > 0 and cmmd_improvements > 0:
            return 'Excellent - Significant bias reduction achieved'
        elif bias_reduction:
            return 'Good - Some bias reduction achieved'
        elif di_improvements > 0 or cmmd_improvements > 0:
            return 'Fair - Mixed results with some improvements'
        else:
            return 'Needs Improvement - No clear bias reduction'
    
    def _extract_key_findings(self, results: Dict[str, Any]) -> List[str]:
        """Extract key findings from evaluation results."""
        findings = []
        
        di_results = results.get('di_results', {})
        cmmd_results = results.get('cmmd_results', {})
        
        # DI findings
        for category, result in di_results.items():
            if result.get('improved', False):
                findings.append(f"Disparate Impact improved for {category}")
            elif result.get('bias_detected', False):
                findings.append(f"Bias still detected in {category}")
        
        # CMMD findings
        for category, result in cmmd_results.items():
            if result.get('improved', False):
                findings.append(f"Semantic similarity improved for {category}")
        
        return findings
    
    def _find_most_improved_groups(self, results: Dict[str, Any]) -> List[str]:
        """Find groups with most improvement."""
        improved_groups = []
        
        di_results = results.get('di_results', {})
        for category, result in di_results.items():
            if result.get('improved', False):
                improvement = result.get('improvement_magnitude', 0)
                improved_groups.append((category, improvement))
        
        # Sort by improvement magnitude
        improved_groups.sort(key=lambda x: x[1], reverse=True)
        return [group[0] for group in improved_groups[:3]]
    
    def _find_problematic_areas(self, results: Dict[str, Any]) -> List[str]:
        """Find areas that still need attention."""
        problematic = []
        
        di_results = results.get('di_results', {})
        for category, result in di_results.items():
            if result.get('bias_detected', False) and not result.get('improved', False):
                problematic.append(f"{category} - persistent bias")
        
        return problematic
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        summary = results.get('summary', {})
        
        if not summary.get('overall_bias_reduction', False):
            recommendations.append(
                "Consider expanding the bias knowledge graph with more diverse patterns"
            )
        
        problematic_areas = self._find_problematic_areas(results)
        if problematic_areas:
            recommendations.append(
                f"Focus enhancement efforts on: {', '.join(problematic_areas)}"
            )
        
        di_results = results.get('di_results', {})
        low_di_categories = [
            cat for cat, result in di_results.items() 
            if result.get('ratio', 1.0) < 0.8
        ]
        
        if low_di_categories:
            recommendations.append(
                f"Increase representation for underrepresented groups in: {', '.join(low_di_categories)}"
            )
        
        if not recommendations:
            recommendations.append("Continue current enhancement approach - good results achieved")
        
        return recommendations
    
    def _save_results(self, report: Dict[str, Any], output_path: str):
        """Save evaluation results to file."""
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)


def main():
    """Demo function showing integrated evaluation."""
    # Initialize evaluator
    evaluator = IntegratedBiasEvaluator()
    
    # Sample prompts for evaluation
    test_prompts = [
        "a doctor in a hospital",
        "a nurse caring for patients",
        "an engineer designing software",
        "a teacher in a classroom",
        "a CEO leading a meeting",
        "a scientist in a laboratory",
        "a chef cooking in a kitchen",
        "an artist creating a painting"
    ]
    
    logger.info("Starting integrated bias evaluation demo")
    
    # Run evaluation pipeline
    try:
        results = evaluator.evaluate_enhancement_pipeline(
            input_prompts=test_prompts,
            save_results=True,
            output_path='integrated_evaluation_results.json'
        )
        
        # Print summary
        print("\n" + "="*60)
        print("BIAS EVALUATION SUMMARY")
        print("="*60)
        print(f"Assessment: {results['summary']['overall_assessment']}")
        print(f"Prompts Enhanced: {results['metadata']['total_prompts']}")
        print(f"Success Rate: {results['metadata']['enhancement_success_rate']:.1%}")
        
        print("\nKey Findings:")
        for finding in results['summary']['key_findings']:
            print(f"  • {finding}")
        
        print("\nRecommendations:")
        for rec in results['recommendations']:
            print(f"  • {rec}")
        
        print("\nExample Enhancements:")
        for example in results['prompt_analysis']['enhancement_examples']:
            print(f"  Original: '{example['original']}'")
            print(f"  Enhanced: '{example['enhanced']}'")
            print(f"  Improvement: {example['improvement']}")
            print()
        
        logger.info("Evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
