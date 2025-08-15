"""
Scoring utilities for cricket cover drive analysis.
Calculates final scores and generates evaluation JSON with feedback.
"""

import numpy as np
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from collections import defaultdict

from .metrics import FrameMetrics


def convert_numpy_types(obj):
    """Convert NumPy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj


@dataclass
class ScoringThresholds:
    """Thresholds for scoring different aspects of the cover drive."""
    
    # Angle thresholds (degrees)
    elbow_angle_good: tuple = (120, 160)
    elbow_angle_warning: tuple = (100, 180)
    spine_lean_good: tuple = (0, 15)
    spine_lean_warning: tuple = (0, 25)
    foot_direction_good: tuple = (0, 20)
    foot_direction_warning: tuple = (0, 35)
    
    # Distance thresholds (normalized)
    head_knee_distance_good: tuple = (0, 0.15)
    head_knee_distance_warning: tuple = (0, 0.3)
    
    # Velocity thresholds (degrees/frame)
    elbow_velocity_good: tuple = (-10, 10)
    spine_velocity_good: tuple = (-5, 5)
    
    # Category weights
    footwork_weights: Dict[str, float] = None
    head_position_weights: Dict[str, float] = None
    swing_control_weights: Dict[str, float] = None
    balance_weights: Dict[str, float] = None
    follow_through_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.footwork_weights is None:
            self.footwork_weights = {
                'front_foot_direction': 0.6,
                'knee_separation': 0.4
            }
        
        if self.head_position_weights is None:
            self.head_position_weights = {
                'head_over_front_knee': 0.8,
                'spine_lean_angle': 0.2
            }
        
        if self.swing_control_weights is None:
            self.swing_control_weights = {
                'front_elbow_angle': 0.5,
                'back_elbow_angle': 0.3,
                'front_elbow_velocity': 0.2
            }
        
        if self.balance_weights is None:
            self.balance_weights = {
                'spine_lean_angle': 0.6,
                'spine_lean_velocity': 0.4
            }
        
        if self.follow_through_weights is None:
            self.follow_through_weights = {
                'front_elbow_angle': 0.4,
                'shoulder_rotation': 0.3,
                'hip_rotation': 0.3
            }


class ScoreCalculator:
    """Calculate final scores and generate evaluation JSON."""
    
    def __init__(self, thresholds: Optional[ScoringThresholds] = None):
        self.thresholds = thresholds or ScoringThresholds()
        
    def calculate_final_scores(self, 
                             metrics_history: List[FrameMetrics],
                             stats: Dict[str, Any],
                             handedness: str) -> Dict[str, Any]:
        """Calculate final scores and generate evaluation JSON."""
        
        if not metrics_history:
            return self._create_error_result("No metrics available")
        
        # Filter out None metrics
        valid_metrics = [m for m in metrics_history if m is not None]
        
        if len(valid_metrics) < 1:
            return self._create_error_result("No valid metrics available")
        
        # For very short videos, we'll still provide basic analysis
        if len(valid_metrics) < 5:
            print(f"Warning: Only {len(valid_metrics)} valid frames available. Analysis may be limited.")
        
        # Calculate category scores
        footwork_score = self._calculate_footwork_score(valid_metrics)
        head_position_score = self._calculate_head_position_score(valid_metrics)
        swing_control_score = self._calculate_swing_control_score(valid_metrics)
        balance_score = self._calculate_balance_score(valid_metrics)
        follow_through_score = self._calculate_follow_through_score(valid_metrics)
        
        # Generate feedback
        footwork_feedback = self._generate_footwork_feedback(valid_metrics, footwork_score)
        head_position_feedback = self._generate_head_position_feedback(valid_metrics, head_position_score)
        swing_control_feedback = self._generate_swing_control_feedback(valid_metrics, swing_control_score)
        balance_feedback = self._generate_balance_feedback(valid_metrics, balance_score)
        follow_through_feedback = self._generate_follow_through_feedback(valid_metrics, follow_through_score)
        
        # Calculate global statistics
        avg_fps = self._calculate_avg_fps(stats)
        frames_analyzed = len(valid_metrics)
        frames_with_missing_keypoints = stats.get("frames_with_missing_keypoints", 0)
        
        # Create evaluation JSON with graceful handling of None scores
        evaluation = {
            "summary": {
                "Footwork": {
                    "score": footwork_score if footwork_score is not None else 0,
                    "feedback": footwork_feedback if footwork_score is not None else "Insufficient data for footwork analysis"
                },
                "Head Position": {
                    "score": head_position_score if head_position_score is not None else 0,
                    "feedback": head_position_feedback if head_position_score is not None else "Insufficient data for head position analysis"
                },
                "Swing Control": {
                    "score": swing_control_score if swing_control_score is not None else 0,
                    "feedback": swing_control_feedback if swing_control_score is not None else "Insufficient data for swing control analysis"
                },
                "Balance": {
                    "score": balance_score if balance_score is not None else 0,
                    "feedback": balance_feedback if balance_score is not None else "Insufficient data for balance analysis"
                },
                "Follow-through": {
                    "score": follow_through_score if follow_through_score is not None else 0,
                    "feedback": follow_through_feedback if follow_through_score is not None else "Insufficient data for follow-through analysis"
                }
            },
            "global": {
                "avg_fps": avg_fps,
                "frames_analyzed": frames_analyzed,
                "frames_with_missing_keypoints": frames_with_missing_keypoints,
                "handedness": handedness
            }
        }
        
        # Convert any NumPy types to native Python types for JSON serialization
        return convert_numpy_types(evaluation)
    
    def _calculate_footwork_score(self, metrics: List[FrameMetrics]) -> int:
        """Calculate footwork score based on foot direction and knee separation."""
        
        scores = []
        
        # Foot direction component
        foot_direction_scores = []
        for metric in metrics:
            if metric.front_foot_direction is not None:
                score = self._score_metric(
                    metric.front_foot_direction,
                    self.thresholds.foot_direction_good,
                    self.thresholds.foot_direction_warning
                )
                foot_direction_scores.append(score)
        
        if foot_direction_scores:
            avg_foot_score = np.mean(foot_direction_scores)
            scores.append(avg_foot_score * self.thresholds.footwork_weights['front_foot_direction'])
        
        # Knee separation component
        knee_separation_scores = []
        for metric in metrics:
            if metric.knee_separation is not None:
                # Good knee separation is typically 0.3-0.6 normalized
                if 0.3 <= metric.knee_separation <= 0.6:
                    score = 10.0
                elif 0.2 <= metric.knee_separation <= 0.7:
                    score = 7.0
                else:
                    score = 4.0
                knee_separation_scores.append(score)
        
        if knee_separation_scores:
            avg_knee_score = np.mean(knee_separation_scores)
            scores.append(avg_knee_score * self.thresholds.footwork_weights['knee_separation'])
        
        if scores:
            return int(round(sum(scores)))
        else:
            # If no valid metrics available, return None to indicate insufficient data
            return None
    
    def _calculate_head_position_score(self, metrics: List[FrameMetrics]) -> int:
        """Calculate head position score based on head-knee distance and spine lean."""
        
        scores = []
        
        # Head over front knee component
        head_knee_scores = []
        for metric in metrics:
            if metric.head_over_front_knee is not None:
                score = self._score_metric(
                    metric.head_over_front_knee,
                    self.thresholds.head_knee_distance_good,
                    self.thresholds.head_knee_distance_warning
                )
                head_knee_scores.append(score)
        
        if head_knee_scores:
            avg_head_knee_score = np.mean(head_knee_scores)
            scores.append(avg_head_knee_score * self.thresholds.head_position_weights['head_over_front_knee'])
        
        # Spine lean component
        spine_lean_scores = []
        for metric in metrics:
            if metric.spine_lean_angle is not None:
                score = self._score_metric(
                    metric.spine_lean_angle,
                    self.thresholds.spine_lean_good,
                    self.thresholds.spine_lean_warning
                )
                spine_lean_scores.append(score)
        
        if spine_lean_scores:
            avg_spine_score = np.mean(spine_lean_scores)
            scores.append(avg_spine_score * self.thresholds.head_position_weights['spine_lean_angle'])
        
        if scores:
            return int(round(sum(scores)))
        else:
            # If no valid metrics available, return None to indicate insufficient data
            return None
    
    def _calculate_swing_control_score(self, metrics: List[FrameMetrics]) -> int:
        """Calculate swing control score based on elbow angles and velocity."""
        
        scores = []
        
        # Front elbow angle component
        front_elbow_scores = []
        for metric in metrics:
            if metric.front_elbow_angle is not None:
                score = self._score_metric(
                    metric.front_elbow_angle,
                    self.thresholds.elbow_angle_good,
                    self.thresholds.elbow_angle_warning
                )
                front_elbow_scores.append(score)
        
        if front_elbow_scores:
            avg_front_elbow_score = np.mean(front_elbow_scores)
            scores.append(avg_front_elbow_score * self.thresholds.swing_control_weights['front_elbow_angle'])
        
        # Back elbow angle component
        back_elbow_scores = []
        for metric in metrics:
            if metric.back_elbow_angle is not None:
                score = self._score_metric(
                    metric.back_elbow_angle,
                    self.thresholds.elbow_angle_good,
                    self.thresholds.elbow_angle_warning
                )
                back_elbow_scores.append(score)
        
        if back_elbow_scores:
            avg_back_elbow_score = np.mean(back_elbow_scores)
            scores.append(avg_back_elbow_score * self.thresholds.swing_control_weights['back_elbow_angle'])
        
        # Elbow velocity component
        elbow_velocity_scores = []
        for metric in metrics:
            if metric.front_elbow_velocity is not None:
                score = self._score_metric(
                    abs(metric.front_elbow_velocity),
                    self.thresholds.elbow_velocity_good,
                    (0, 20)  # Warning threshold for velocity
                )
                elbow_velocity_scores.append(score)
        
        if elbow_velocity_scores:
            avg_velocity_score = np.mean(elbow_velocity_scores)
            scores.append(avg_velocity_score * self.thresholds.swing_control_weights['front_elbow_velocity'])
        
        if scores:
            return int(round(sum(scores)))
        else:
            # If no valid metrics available, return None to indicate insufficient data
            return None
    
    def _calculate_balance_score(self, metrics: List[FrameMetrics]) -> int:
        """Calculate balance score based on spine lean and its stability."""
        
        scores = []
        
        # Spine lean component
        spine_lean_scores = []
        for metric in metrics:
            if metric.spine_lean_angle is not None:
                score = self._score_metric(
                    metric.spine_lean_angle,
                    self.thresholds.spine_lean_good,
                    self.thresholds.spine_lean_warning
                )
                spine_lean_scores.append(score)
        
        if spine_lean_scores:
            avg_spine_score = float(np.mean(spine_lean_scores))  # Convert to native Python float
            scores.append(avg_spine_score * self.thresholds.balance_weights['spine_lean_angle'])
        
        # Spine velocity component (stability)
        spine_velocity_scores = []
        for metric in metrics:
            if metric.spine_lean_velocity is not None:
                score = self._score_metric(
                    abs(metric.spine_lean_velocity),
                    self.thresholds.spine_velocity_good,
                    (0, 10)  # Warning threshold for velocity
                )
                spine_velocity_scores.append(score)
        
        if spine_velocity_scores:
            avg_velocity_score = np.mean(spine_velocity_scores)
            scores.append(avg_velocity_score * self.thresholds.balance_weights['spine_lean_velocity'])
        
        if scores:
            return int(round(sum(scores)))
        else:
            # If no valid metrics available, return None to indicate insufficient data
            return None
    
    def _calculate_follow_through_score(self, metrics: List[FrameMetrics]) -> int:
        """Calculate follow-through score based on elbow angle, shoulder and hip rotation."""
        
        scores = []
        
        # Front elbow angle component (should maintain good angle through follow-through)
        front_elbow_scores = []
        for metric in metrics:
            if metric.front_elbow_angle is not None:
                score = self._score_metric(
                    metric.front_elbow_angle,
                    self.thresholds.elbow_angle_good,
                    self.thresholds.elbow_angle_warning
                )
                front_elbow_scores.append(score)
        
        if front_elbow_scores:
            avg_front_elbow_score = float(np.mean(front_elbow_scores))  # Convert to native Python float
            scores.append(avg_front_elbow_score * self.thresholds.follow_through_weights['front_elbow_angle'])
        
        # Shoulder rotation component
        shoulder_rotation_scores = []
        for metric in metrics:
            if metric.shoulder_rotation is not None:
                # Good shoulder rotation is typically 15-45 degrees
                if 15 <= abs(metric.shoulder_rotation) <= 45:
                    score = 10.0
                elif 10 <= abs(metric.shoulder_rotation) <= 60:
                    score = 7.0
                else:
                    score = 4.0
                shoulder_rotation_scores.append(score)
        
        if shoulder_rotation_scores:
            avg_shoulder_score = float(np.mean(shoulder_rotation_scores))  # Convert to native Python float
            scores.append(avg_shoulder_score * self.thresholds.follow_through_weights['shoulder_rotation'])
        
        # Hip rotation component
        hip_rotation_scores = []
        for metric in metrics:
            if metric.hip_rotation is not None:
                # Good hip rotation is typically 10-35 degrees
                if 10 <= abs(metric.hip_rotation) <= 35:
                    score = 10.0
                elif 5 <= abs(metric.hip_rotation) <= 45:
                    score = 7.0
                else:
                    score = 4.0
                hip_rotation_scores.append(score)
        
        if hip_rotation_scores:
            avg_hip_score = np.mean(hip_rotation_scores)
            scores.append(avg_hip_score * self.thresholds.follow_through_weights['hip_rotation'])
        
        if scores:
            return int(round(sum(scores)))
        else:
            # If no valid metrics available, return None to indicate insufficient data
            return None
    
    def _score_metric(self, value: float, good_range: tuple, warning_range: tuple) -> float:
        """Score a metric value based on good and warning ranges."""
        
        abs_value = abs(value)
        
        # Check if in good range
        if good_range[0] <= abs_value <= good_range[1]:
            return 10.0
        # Check if in warning range
        elif warning_range[0] <= abs_value <= warning_range[1]:
            return 7.0
        # Outside acceptable ranges
        else:
            return 4.0
    
    def _generate_footwork_score(self, metrics: List[FrameMetrics], score: int) -> str:
        """Generate feedback for footwork based on score and metrics."""
        
        if score >= 8:
            return "Excellent foot positioning and alignment throughout the shot."
        elif score >= 6:
            return "Good footwork with minor alignment issues."
        else:
            return "Foot positioning needs improvement; focus on alignment and balance."
    
    def _generate_footwork_feedback(self, metrics: List[FrameMetrics], score: int) -> str:
        """Generate feedback for footwork based on score and metrics."""
        
        if score >= 8:
            return "Foot aligned; slight toe-out late in follow-through."
        elif score >= 6:
            return "Footwork generally good; minor alignment issues detected."
        else:
            return "Foot positioning needs work; improve alignment and stance."
    
    def _generate_head_position_feedback(self, metrics: List[FrameMetrics], score: int) -> str:
        """Generate feedback for head position based on score and metrics."""
        
        if score >= 8:
            return "Head remains over front knee during impact."
        elif score >= 6:
            return "Head position mostly stable; slight movement detected."
        else:
            return "Head position inconsistent; maintain focus over front knee."
    
    def _generate_swing_control_feedback(self, metrics: List[FrameMetrics], score: int) -> str:
        """Generate feedback for swing control based on score and metrics."""
        
        if score >= 8:
            return "Excellent swing control with consistent elbow angles."
        elif score >= 6:
            return "Elbow drops near impact; tighten arc."
        else:
            return "Swing control needs work; focus on elbow position and timing."
    
    def _generate_balance_feedback(self, metrics: List[FrameMetrics], score: int) -> str:
        """Generate feedback for balance based on score and metrics."""
        
        if score >= 8:
            return "Good stability and low lateral sway."
        elif score >= 6:
            return "Balance generally good; minor instability detected."
        else:
            return "Balance needs improvement; reduce lateral movement."
    
    def _generate_follow_through_feedback(self, metrics: List[FrameMetrics], score: int) -> str:
        """Generate feedback for follow-through based on score and metrics."""
        
        if score >= 8:
            return "Follow-through is consistent and complete."
        elif score >= 6:
            return "Follow-through is consistent; hold longer."
        else:
            return "Follow-through needs work; complete the motion fully."
    
    def _calculate_avg_fps(self, stats: Dict[str, Any]) -> float:
        """Calculate average FPS from processing statistics."""
        
        processing_times = stats.get("processing_times", [])
        if not processing_times:
            return 0.0
        
        avg_time = float(np.mean(processing_times))  # Convert to native Python float
        return 1.0 / avg_time if avg_time > 0 else 0.0
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create error result when analysis fails."""
        
        return {
            "error": error_message,
            "summary": {
                "Footwork": {"score": 0, "feedback": "Analysis failed"},
                "Head Position": {"score": 0, "feedback": "Analysis failed"},
                "Swing Control": {"score": 0, "feedback": "Analysis failed"},
                "Balance": {"score": 0, "feedback": "Analysis failed"},
                "Follow-through": {"score": 0, "feedback": "Analysis failed"}
            },
            "global": {
                "avg_fps": 0.0,
                "frames_analyzed": 0,
                "frames_with_missing_keypoints": 0,
                "handedness": "unknown"
            }
        }


class DetailedAnalyzer:
    """Provides detailed analysis of specific aspects of the cover drive."""
    
    def __init__(self):
        self.score_calculator = ScoreCalculator()
    
    def analyze_phase_breakdown(self, metrics: List[FrameMetrics]) -> Dict[str, Any]:
        """Analyze the cover drive in different phases."""
        
        if len(metrics) < 10:
            return {"error": "Insufficient data for phase analysis"}
        
        # Divide into phases (simplified: setup, backswing, downswing, impact, follow-through)
        total_frames = len(metrics)
        phase_size = total_frames // 5
        
        phases = {
            "setup": metrics[:phase_size],
            "backswing": metrics[phase_size:2*phase_size],
            "downswing": metrics[2*phase_size:3*phase_size],
            "impact": metrics[3*phase_size:4*phase_size],
            "follow_through": metrics[4*phase_size:]
        }
        
        phase_analysis = {}
        for phase_name, phase_metrics in phases.items():
            if phase_metrics:
                phase_analysis[phase_name] = {
                    "elbow_angle_avg": self._calculate_metric_average(phase_metrics, 'front_elbow_angle'),
                    "spine_lean_avg": self._calculate_metric_average(phase_metrics, 'spine_lean_angle'),
                    "head_position_avg": self._calculate_metric_average(phase_metrics, 'head_over_front_knee'),
                    "foot_direction_avg": self._calculate_metric_average(phase_metrics, 'front_foot_direction')
                }
        
        return phase_analysis
    
    def _calculate_metric_average(self, metrics: List[FrameMetrics], metric_name: str) -> Optional[float]:
        """Calculate average of a metric across a list of frames."""
        
        values = []
        for metric in metrics:
            value = getattr(metric, metric_name)
            if value is not None:
                values.append(value)
        
        if values:
            return float(np.mean(values))  # Convert to native Python float
        return None
    
    def identify_improvement_areas(self, metrics: List[FrameMetrics]) -> List[str]:
        """Identify specific areas for improvement."""
        
        improvements = []
        
        # Analyze elbow angle consistency
        elbow_angles = [m.front_elbow_angle for m in metrics if m.front_elbow_angle is not None]
        if elbow_angles:
            elbow_std = np.std(elbow_angles)
            if elbow_std > 15:
                improvements.append("Reduce elbow angle variability")
        
        # Analyze spine lean stability
        spine_leans = [m.spine_lean_angle for m in metrics if m.spine_lean_angle is not None]
        if spine_leans:
            spine_std = np.std(spine_leans)
            if spine_std > 8:
                improvements.append("Improve spine stability")
        
        # Analyze head position consistency
        head_positions = [m.head_over_front_knee for m in metrics if m.head_over_front_knee is not None]
        if head_positions:
            head_std = np.std(head_positions)
            if head_std > 0.1:
                improvements.append("Maintain consistent head position")
        
        return improvements
