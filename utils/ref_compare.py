"""
Reference comparison utilities for cricket cover drive analysis.
Compares actual metrics to ideal technique benchmarks and calculates deviations.
"""

import numpy as np
import yaml
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from .metrics import FrameMetrics


@dataclass
class DeviationAnalysis:
    """Analysis of deviation from ideal technique."""
    metric_name: str
    actual_value: float
    ideal_value: float
    deviation: float
    deviation_percentage: float
    severity: str  # slight, moderate, significant, major
    score_impact: float
    feedback: str


@dataclass
class ReferenceComparison:
    """Complete reference comparison results."""
    skill_level: str
    overall_score: float
    metric_deviations: List[DeviationAnalysis]
    phase_analysis: Dict[str, Any]
    smoothness_analysis: Dict[str, Any]
    recommendations: List[str]


class ReferenceComparator:
    """Compares actual technique to ideal reference values."""
    
    def __init__(self, ideal_config_path: str = "configs/ideal.yml"):
        self.ideal_config_path = Path(ideal_config_path)
        self.ideal_config = self._load_ideal_config()
        
        # Deviation severity thresholds
        self.severity_thresholds = {
            'slight': 0.10,      # Within 10% of optimal
            'moderate': 0.25,    # Within 25% of optimal
            'significant': 0.50, # Within 50% of optimal
            'major': float('inf') # Beyond 50% of optimal
        }
        
    def _load_ideal_config(self) -> Dict[str, Any]:
        """Load ideal technique configuration."""
        try:
            with open(self.ideal_config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Could not load ideal config: {e}")
            return self._get_default_ideal_config()
    
    def _get_default_ideal_config(self) -> Dict[str, Any]:
        """Get default ideal configuration if file not found."""
        return {
            'skill_levels': {
                'intermediate': {
                    'front_elbow_angle': {'min': 100, 'max': 180, 'optimal': 140, 'weight': 0.25},
                    'spine_lean_angle': {'min': 0, 'max': 25, 'optimal': 12, 'weight': 0.20},
                    'head_over_front_knee': {'min': 0, 'max': 0.25, 'optimal': 0.12, 'weight': 0.25},
                    'front_foot_direction': {'min': 0, 'max': 30, 'optimal': 15, 'weight': 0.15}
                }
            },
            'deviation_penalties': {
                'slight_deviation': 0.1,
                'moderate_deviation': 0.3,
                'significant_deviation': 0.6,
                'major_deviation': 1.0
            }
        }
    
    def compare_to_reference(self, metrics_history: List[FrameMetrics], 
                           smoothness_analysis: Optional[Dict[str, Any]] = None,
                           phase_analysis: Optional[Dict[str, Any]] = None,
                           skill_level: str = "intermediate") -> ReferenceComparison:
        """Compare actual metrics to ideal reference values."""
        
        if not metrics_history:
            return self._create_empty_comparison()
        
        # Calculate median values for each metric
        median_metrics = self._calculate_median_metrics(metrics_history)
        
        # Get ideal values for the specified skill level
        ideal_values = self._get_ideal_values(skill_level)
        
        # Analyze deviations for each metric
        deviations = []
        total_score = 0.0
        total_weight = 0.0
        
        for metric_name, actual_value in median_metrics.items():
            if metric_name in ideal_values:
                deviation_analysis = self._analyze_metric_deviation(
                    metric_name, actual_value, ideal_values[metric_name]
                )
                deviations.append(deviation_analysis)
                
                # Calculate weighted score impact
                weight = ideal_values[metric_name].get('weight', 0.1)
                score_impact = (1.0 - deviation_analysis.score_impact) * weight
                total_score += score_impact
                total_weight += weight
        
        # Normalize overall score
        overall_score = (total_score / total_weight) * 10 if total_weight > 0 else 0.0
        
        # Analyze phase-specific performance
        phase_analysis = self._analyze_phase_performance(metrics_history, phase_analysis)
        
        # Analyze smoothness against reference
        smoothness_analysis = self._analyze_smoothness_reference(smoothness_analysis, skill_level)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(deviations, smoothness_analysis, phase_analysis)
        
        return ReferenceComparison(
            skill_level=skill_level,
            overall_score=overall_score,
            metric_deviations=deviations,
            phase_analysis=phase_analysis,
            smoothness_analysis=smoothness_analysis,
            recommendations=recommendations
        )
    
    def _calculate_median_metrics(self, metrics_history: List[FrameMetrics]) -> Dict[str, float]:
        """Calculate median values for each metric across the video."""
        valid_metrics = [m for m in metrics_history if m is not None]
        
        if not valid_metrics:
            return {}
        
        median_metrics = {}
        
        # Calculate median for each metric
        for metric_name in ['front_elbow_angle', 'spine_lean_angle', 'head_over_front_knee', 
                          'front_foot_direction', 'back_elbow_angle', 'shoulder_rotation']:
            values = [getattr(m, metric_name, None) for m in valid_metrics]
            values = [v for v in values if v is not None]
            
            if values:
                median_metrics[metric_name] = np.median(values)
        
        return median_metrics
    
    def _get_ideal_values(self, skill_level: str) -> Dict[str, Any]:
        """Get ideal values for the specified skill level."""
        if skill_level not in self.ideal_config.get('skill_levels', {}):
            skill_level = 'intermediate'  # Default fallback
        
        return self.ideal_config['skill_levels'][skill_level]
    
    def _analyze_metric_deviation(self, metric_name: str, actual_value: float, 
                                ideal_config: Dict[str, Any]) -> DeviationAnalysis:
        """Analyze deviation of a single metric from ideal."""
        
        optimal_value = ideal_config.get('optimal', 0.0)
        min_value = ideal_config.get('min', optimal_value * 0.8)
        max_value = ideal_config.get('max', optimal_value * 1.2)
        
        # Calculate deviation
        if optimal_value == 0:
            deviation = abs(actual_value)
            deviation_percentage = 1.0 if actual_value != 0 else 0.0
        else:
            deviation = abs(actual_value - optimal_value)
            deviation_percentage = deviation / abs(optimal_value)
        
        # Determine severity
        severity = self._determine_severity(deviation_percentage)
        
        # Calculate score impact
        score_impact = self._calculate_score_impact(deviation_percentage, severity)
        
        # Generate feedback
        feedback = self._generate_metric_feedback(metric_name, actual_value, optimal_value, severity)
        
        return DeviationAnalysis(
            metric_name=metric_name,
            actual_value=actual_value,
            ideal_value=optimal_value,
            deviation=deviation,
            deviation_percentage=deviation_percentage,
            severity=severity,
            score_impact=score_impact,
            feedback=feedback
        )
    
    def _determine_severity(self, deviation_percentage: float) -> str:
        """Determine severity of deviation."""
        for severity, threshold in self.severity_thresholds.items():
            if deviation_percentage <= threshold:
                return severity
        return 'major'
    
    def _calculate_score_impact(self, deviation_percentage: float, severity: str) -> float:
        """Calculate score impact based on deviation severity."""
        penalties = self.ideal_config.get('deviation_penalties', {})
        
        if severity == 'slight':
            return penalties.get('slight_deviation', 0.1)
        elif severity == 'moderate':
            return penalties.get('moderate_deviation', 0.3)
        elif severity == 'significant':
            return penalties.get('significant_deviation', 0.6)
        else:  # major
            return penalties.get('major_deviation', 1.0)
    
    def _generate_metric_feedback(self, metric_name: str, actual: float, ideal: float, 
                                severity: str) -> str:
        """Generate feedback for a metric deviation."""
        
        metric_display_names = {
            'front_elbow_angle': 'Front Elbow Angle',
            'spine_lean_angle': 'Spine Lean',
            'head_over_front_knee': 'Head Position',
            'front_foot_direction': 'Foot Direction',
            'back_elbow_angle': 'Back Elbow Angle',
            'shoulder_rotation': 'Shoulder Rotation'
        }
        
        display_name = metric_display_names.get(metric_name, metric_name.replace('_', ' ').title())
        
        if severity == 'slight':
            return f"{display_name} is close to ideal ({actual:.1f} vs {ideal:.1f})"
        elif severity == 'moderate':
            if actual > ideal:
                return f"{display_name} is too high ({actual:.1f} vs ideal {ideal:.1f})"
            else:
                return f"{display_name} is too low ({actual:.1f} vs ideal {ideal:.1f})"
        elif severity == 'significant':
            return f"{display_name} needs significant improvement ({actual:.1f} vs ideal {ideal:.1f})"
        else:  # major
            return f"{display_name} requires major correction ({actual:.1f} vs ideal {ideal:.1f})"
    
    def _analyze_phase_performance(self, metrics_history: List[FrameMetrics], 
                                 phase_analysis: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance during different phases."""
        if not phase_analysis:
            return {'available': False}
        
        phase_performance = {
            'available': True,
            'phase_scores': {},
            'phase_deviations': {}
        }
        
        # Analyze each phase if available
        if 'phases' in phase_analysis:
            for phase_name, phase_data in phase_analysis.get('phases', {}).items():
                if 'metrics' in phase_data:
                    phase_metrics = phase_data['metrics']
                    phase_score = self._calculate_phase_score(phase_metrics, phase_name)
                    phase_performance['phase_scores'][phase_name] = phase_score
        
        return phase_performance
    
    def _calculate_phase_score(self, phase_metrics: Dict[str, Any], phase_name: str) -> float:
        """Calculate score for a specific phase."""
        if not phase_metrics:
            return 0.0
        
        # Get ideal values for this phase
        ideal_phases = self.ideal_config.get('phases', {})
        if phase_name not in ideal_phases:
            return 7.0  # Default score
        
        ideal_values = ideal_phases[phase_name]
        total_score = 0.0
        metric_count = 0
        
        for metric_name, actual_value in phase_metrics.items():
            if metric_name in ideal_values and actual_value is not None:
                ideal_value = ideal_values[metric_name]
                deviation = abs(actual_value - ideal_value)
                
                # Calculate score based on deviation
                if ideal_value == 0:
                    score = max(0, 10 - deviation * 10)
                else:
                    deviation_percentage = deviation / abs(ideal_value)
                    score = max(0, 10 - deviation_percentage * 10)
                
                total_score += score
                metric_count += 1
        
        return total_score / metric_count if metric_count > 0 else 7.0
    
    def _analyze_smoothness_reference(self, smoothness_analysis: Optional[Dict[str, Any]], 
                                    skill_level: str) -> Dict[str, Any]:
        """Analyze smoothness against reference targets."""
        if not smoothness_analysis:
            return {'available': False}
        
        smoothness_targets = self.ideal_config.get('smoothness_targets', {})
        if skill_level not in smoothness_targets:
            skill_level = 'intermediate'
        
        targets = smoothness_targets[skill_level]
        overall_smoothness = smoothness_analysis.get('overall_smoothness', 0.0)
        
        # Calculate smoothness score
        min_smoothness = targets.get('min_smoothness_score', 0.65)
        smoothness_score = max(0, (overall_smoothness - min_smoothness) / (1.0 - min_smoothness) * 10)
        
        return {
            'available': True,
            'overall_smoothness': overall_smoothness,
            'target_smoothness': min_smoothness,
            'smoothness_score': smoothness_score,
            'meets_target': overall_smoothness >= min_smoothness
        }
    
    def _generate_recommendations(self, deviations: List[DeviationAnalysis], 
                                smoothness_analysis: Dict[str, Any],
                                phase_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        # Recommendations based on metric deviations
        major_deviations = [d for d in deviations if d.severity in ['significant', 'major']]
        moderate_deviations = [d for d in deviations if d.severity == 'moderate']
        
        if major_deviations:
            recommendations.append(f"Focus on correcting {len(major_deviations)} major technique issues")
        
        if moderate_deviations:
            recommendations.append(f"Improve {len(moderate_deviations)} moderate technique areas")
        
        # Specific recommendations for major deviations
        for deviation in major_deviations[:3]:  # Top 3 issues
            recommendations.append(deviation.feedback)
        
        # Smoothness recommendations
        if smoothness_analysis.get('available', False):
            if not smoothness_analysis.get('meets_target', True):
                recommendations.append("Work on movement smoothness and consistency")
        
        # Phase-specific recommendations
        if phase_analysis.get('available', False):
            phase_scores = phase_analysis.get('phase_scores', {})
            weak_phases = [phase for phase, score in phase_scores.items() if score < 6.0]
            if weak_phases:
                recommendations.append(f"Focus on improving: {', '.join(weak_phases)}")
        
        if not recommendations:
            recommendations.append("Technique is within acceptable range")
        
        return recommendations
    
    def _create_empty_comparison(self) -> ReferenceComparison:
        """Create empty comparison when no data is available."""
        return ReferenceComparison(
            skill_level="intermediate",
            overall_score=0.0,
            metric_deviations=[],
            phase_analysis={'available': False},
            smoothness_analysis={'available': False},
            recommendations=["No data available for analysis"]
        )
    
    def get_skill_level_assessment(self, overall_score: float) -> str:
        """Assess skill level based on overall score."""
        if overall_score >= 8.5:
            return "Professional"
        elif overall_score >= 7.0:
            return "Advanced"
        elif overall_score >= 5.0:
            return "Intermediate"
        else:
            return "Beginner"
    
    def export_comparison_report(self, comparison: ReferenceComparison, 
                               output_path: str) -> str:
        """Export comparison results to a detailed report."""
        report_lines = [
            "# Cricket Cover Drive - Reference Comparison Report",
            f"Skill Level: {comparison.skill_level}",
            f"Overall Score: {comparison.overall_score:.1f}/10",
            f"Assessment: {self.get_skill_level_assessment(comparison.overall_score)}",
            "",
            "## Metric Deviations",
            ""
        ]
        
        for deviation in comparison.metric_deviations:
            report_lines.extend([
                f"### {deviation.metric_name.replace('_', ' ').title()}",
                f"- Actual: {deviation.actual_value:.1f}",
                f"- Ideal: {deviation.ideal_value:.1f}",
                f"- Deviation: {deviation.deviation:.1f} ({deviation.deviation_percentage:.1%})",
                f"- Severity: {deviation.severity.title()}",
                f"- Feedback: {deviation.feedback}",
                ""
            ])
        
        report_lines.extend([
            "## Recommendations",
            ""
        ])
        
        for i, rec in enumerate(comparison.recommendations, 1):
            report_lines.append(f"{i}. {rec}")
        
        report_content = "\n".join(report_lines)
        
        try:
            with open(output_path, 'w') as f:
                f.write(report_content)
            return output_path
        except Exception as e:
            print(f"Error writing report: {e}")
            return ""
