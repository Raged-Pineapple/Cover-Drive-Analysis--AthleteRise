"""
Skill grade prediction for cricket cover drive analysis.
Maps final category scores and smoothness into skill levels.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from .metrics import FrameMetrics


class SkillLevel(Enum):
    """Enumeration of skill levels."""
    BEGINNER = "Beginner"
    INTERMEDIATE = "Intermediate"
    ADVANCED = "Advanced"
    PROFESSIONAL = "Professional"


@dataclass
class SkillAssessment:
    """Complete skill assessment with detailed breakdown."""
    skill_level: SkillLevel
    overall_score: float
    category_scores: Dict[str, float]
    smoothness_score: float
    confidence: float
    strengths: List[str]
    weaknesses: List[str]
    improvement_areas: List[str]
    next_level_threshold: float


class SkillGrader:
    """Predicts skill level based on analysis results."""
    
    def __init__(self):
        # Skill level thresholds
        self.thresholds = {
            SkillLevel.BEGINNER: {
                'min_score': 0.0,
                'max_score': 4.9,
                'min_smoothness': 0.0,
                'max_smoothness': 0.49
            },
            SkillLevel.INTERMEDIATE: {
                'min_score': 5.0,
                'max_score': 6.9,
                'min_smoothness': 0.50,
                'max_smoothness': 0.69
            },
            SkillLevel.ADVANCED: {
                'min_score': 7.0,
                'max_score': 8.4,
                'min_smoothness': 0.70,
                'max_smoothness': 0.84
            },
            SkillLevel.PROFESSIONAL: {
                'min_score': 8.5,
                'max_score': 10.0,
                'min_smoothness': 0.85,
                'max_smoothness': 1.0
            }
        }
        
        # Category weights for skill assessment
        self.category_weights = {
            'Footwork': 0.20,
            'Head Position': 0.25,
            'Swing Control': 0.25,
            'Balance': 0.15,
            'Follow-through': 0.15
        }
        
        # Skill level characteristics
        self.level_characteristics = {
            SkillLevel.BEGINNER: {
                'description': 'Learning basic technique with significant room for improvement',
                'typical_issues': ['Inconsistent footwork', 'Poor head position', 'Unstable balance'],
                'focus_areas': ['Basic stance', 'Head position', 'Balance fundamentals']
            },
            SkillLevel.INTERMEDIATE: {
                'description': 'Solid foundation with some technical areas needing refinement',
                'typical_issues': ['Minor footwork inconsistencies', 'Occasional head movement', 'Timing issues'],
                'focus_areas': ['Consistency', 'Timing', 'Advanced techniques']
            },
            SkillLevel.ADVANCED: {
                'description': 'High-level technique with minor refinements possible',
                'typical_issues': ['Fine-tuning required', 'Minor timing adjustments', 'Advanced positioning'],
                'focus_areas': ['Precision', 'Advanced tactics', 'Mental game']
            },
            SkillLevel.PROFESSIONAL: {
                'description': 'Elite-level technique suitable for professional competition',
                'typical_issues': ['Minimal technical issues', 'Game situation adaptation', 'Mental preparation'],
                'focus_areas': ['Game strategy', 'Mental toughness', 'Consistency under pressure']
            }
        }
    
    def assess_skill_level(self, category_scores: Dict[str, float], 
                          smoothness_analysis: Optional[Dict[str, Any]] = None,
                          reference_comparison: Optional[Dict[str, Any]] = None) -> SkillAssessment:
        """Assess skill level based on category scores and additional metrics."""
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(category_scores)
        
        # Get smoothness score
        smoothness_score = self._get_smoothness_score(smoothness_analysis)
        
        # Determine skill level
        skill_level = self._determine_skill_level(overall_score, smoothness_score)
        
        # Calculate confidence in assessment
        confidence = self._calculate_assessment_confidence(category_scores, smoothness_score)
        
        # Identify strengths and weaknesses
        strengths, weaknesses = self._identify_strengths_weaknesses(category_scores)
        
        # Generate improvement areas
        improvement_areas = self._generate_improvement_areas(skill_level, category_scores, weaknesses)
        
        # Calculate next level threshold
        next_level_threshold = self._calculate_next_level_threshold(skill_level)
        
        return SkillAssessment(
            skill_level=skill_level,
            overall_score=overall_score,
            category_scores=category_scores,
            smoothness_score=smoothness_score,
            confidence=confidence,
            strengths=strengths,
            weaknesses=weaknesses,
            improvement_areas=improvement_areas,
            next_level_threshold=next_level_threshold
        )
    
    def _calculate_overall_score(self, category_scores: Dict[str, float]) -> float:
        """Calculate weighted overall score from category scores."""
        
        if not category_scores:
            return 0.0
        
        total_score = 0.0
        total_weight = 0.0
        
        for category, score in category_scores.items():
            weight = self.category_weights.get(category, 0.1)
            total_score += score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _get_smoothness_score(self, smoothness_analysis: Optional[Dict[str, Any]]) -> float:
        """Extract smoothness score from analysis."""
        
        if not smoothness_analysis:
            return 0.5  # Default neutral score
        
        # Get overall smoothness if available
        overall_smoothness = smoothness_analysis.get('overall_smoothness', 0.5)
        
        # Normalize to 0-1 range
        return max(0.0, min(1.0, overall_smoothness))
    
    def _determine_skill_level(self, overall_score: float, smoothness_score: float) -> SkillLevel:
        """Determine skill level based on overall score and smoothness."""
        
        # Combine score and smoothness (70% score, 30% smoothness)
        combined_score = overall_score * 0.7 + smoothness_score * 10 * 0.3
        
        # Find appropriate skill level
        for level in reversed(list(SkillLevel)):  # Check from highest to lowest
            thresholds = self.thresholds[level]
            if combined_score >= thresholds['min_score']:
                return level
        
        return SkillLevel.BEGINNER
    
    def _calculate_assessment_confidence(self, category_scores: Dict[str, float], 
                                       smoothness_score: float) -> float:
        """Calculate confidence in the skill assessment."""
        
        if not category_scores:
            return 0.0
        
        # Confidence based on score consistency
        scores = list(category_scores.values())
        score_variance = np.var(scores) if len(scores) > 1 else 0.0
        
        # Lower variance = higher confidence (more consistent performance)
        consistency_confidence = max(0.0, 1.0 - score_variance / 10.0)
        
        # Confidence based on smoothness availability
        smoothness_confidence = 1.0 if smoothness_score != 0.5 else 0.7
        
        # Overall confidence
        overall_confidence = (consistency_confidence * 0.7 + smoothness_confidence * 0.3)
        
        return min(1.0, overall_confidence)
    
    def _identify_strengths_weaknesses(self, category_scores: Dict[str, float]) -> tuple[List[str], List[str]]:
        """Identify strengths and weaknesses based on category scores."""
        
        strengths = []
        weaknesses = []
        
        if not category_scores:
            return strengths, weaknesses
        
        # Calculate average score
        avg_score = np.mean(list(category_scores.values()))
        
        # Identify strengths (above average)
        for category, score in category_scores.items():
            if score >= avg_score + 1.0:  # Significantly above average
                strengths.append(f"Strong {category.lower()} (score: {score:.1f})")
            elif score <= avg_score - 1.0:  # Significantly below average
                weaknesses.append(f"Needs improvement in {category.lower()} (score: {score:.1f})")
        
        # If no clear strengths/weaknesses, provide general feedback
        if not strengths and not weaknesses:
            if avg_score >= 7.0:
                strengths.append("Overall solid technique across all areas")
            elif avg_score <= 4.0:
                weaknesses.append("Fundamental technique needs work")
            else:
                strengths.append("Balanced performance with room for improvement")
        
        return strengths, weaknesses
    
    def _generate_improvement_areas(self, skill_level: SkillLevel, 
                                  category_scores: Dict[str, float],
                                  weaknesses: List[str]) -> List[str]:
        """Generate specific improvement areas based on skill level and scores."""
        
        improvement_areas = []
        
        # Add level-specific focus areas
        level_info = self.level_characteristics[skill_level]
        improvement_areas.extend(level_info['focus_areas'])
        
        # Add specific areas based on lowest scores
        if category_scores:
            sorted_categories = sorted(category_scores.items(), key=lambda x: x[1])
            lowest_category = sorted_categories[0]
            
            if lowest_category[1] < 6.0:
                improvement_areas.append(f"Focus on {lowest_category[0].lower()} improvement")
        
        # Add areas from weaknesses
        for weakness in weaknesses[:2]:  # Top 2 weaknesses
            improvement_areas.append(weakness)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_areas = []
        for area in improvement_areas:
            if area not in seen:
                seen.add(area)
                unique_areas.append(area)
        
        return unique_areas[:5]  # Limit to top 5 areas
    
    def _calculate_next_level_threshold(self, current_level: SkillLevel) -> float:
        """Calculate the score needed to reach the next skill level."""
        
        if current_level == SkillLevel.PROFESSIONAL:
            return 10.0  # Already at highest level
        
        # Get next level
        levels = list(SkillLevel)
        current_index = levels.index(current_level)
        next_level = levels[current_index + 1]
        
        # Return minimum score for next level
        return self.thresholds[next_level]['min_score']
    
    def get_skill_level_description(self, skill_level: SkillLevel) -> Dict[str, Any]:
        """Get detailed description for a skill level."""
        
        return self.level_characteristics.get(skill_level, {
            'description': 'Unknown skill level',
            'typical_issues': [],
            'focus_areas': []
        })
    
    def generate_skill_report(self, assessment: SkillAssessment) -> Dict[str, Any]:
        """Generate comprehensive skill assessment report."""
        
        level_info = self.get_skill_level_description(assessment.skill_level)
        
        report = {
            'skill_level': assessment.skill_level.value,
            'overall_score': assessment.overall_score,
            'confidence': assessment.confidence,
            'description': level_info['description'],
            'category_breakdown': assessment.category_scores,
            'smoothness_score': assessment.smoothness_score,
            'strengths': assessment.strengths,
            'weaknesses': assessment.weaknesses,
            'improvement_areas': assessment.improvement_areas,
            'next_level_threshold': assessment.next_level_threshold,
            'typical_issues': level_info['typical_issues'],
            'focus_areas': level_info['focus_areas']
        }
        
        return report
    
    def export_skill_assessment(self, assessment: SkillAssessment, 
                              output_path: str) -> str:
        """Export skill assessment to a detailed report file."""
        
        report = self.generate_skill_report(assessment)
        
        # Create markdown report
        lines = [
            "# Cricket Cover Drive - Skill Assessment Report",
            "",
            f"## Skill Level: {report['skill_level']}",
            f"**Overall Score:** {report['overall_score']:.1f}/10",
            f"**Assessment Confidence:** {report['confidence']:.1%}",
            "",
            f"### Description",
            report['description'],
            "",
            "## Category Breakdown",
            ""
        ]
        
        # Add category scores
        for category, score in report['category_breakdown'].items():
            lines.append(f"- **{category}:** {score:.1f}/10")
        
        lines.extend([
            "",
            f"## Smoothness Score: {report['smoothness_score']:.2f}",
            "",
            "## Strengths",
            ""
        ])
        
        for strength in report['strengths']:
            lines.append(f"- {strength}")
        
        lines.extend([
            "",
            "## Areas for Improvement",
            ""
        ])
        
        for area in report['improvement_areas']:
            lines.append(f"- {area}")
        
        lines.extend([
            "",
            "## Next Level",
            f"To reach the next skill level, aim for a score of **{report['next_level_threshold']:.1f}/10**",
            "",
            "## Typical Issues at This Level",
            ""
        ])
        
        for issue in report['typical_issues']:
            lines.append(f"- {issue}")
        
        lines.extend([
            "",
            "## Recommended Focus Areas",
            ""
        ])
        
        for area in report['focus_areas']:
            lines.append(f"- {area}")
        
        report_content = "\n".join(lines)
        
        try:
            with open(output_path, 'w') as f:
                f.write(report_content)
            return output_path
        except Exception as e:
            print(f"Error writing skill assessment: {e}")
            return ""
