"""
Plotting utilities for cricket cover drive analysis.
Generates charts for temporal smoothness and consistency analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Any, Optional, Tuple
import cv2

from .metrics import FrameMetrics


class SmoothnessAnalyzer:
    """Analyzes temporal smoothness and consistency of metrics."""
    
    def __init__(self):
        # Smoothness analysis parameters
        self.thresholds = {
            'max_angle_jump': 20.0,  # Maximum allowed angle change between frames
            'max_distance_jump': 0.05,  # Maximum allowed distance change
            'smoothness_window': 5,  # Window for calculating local smoothness
            'outlier_threshold': 2.0  # Standard deviations for outlier detection
        }
        
    def analyze_smoothness(self, metrics_history: List[FrameMetrics]) -> Dict[str, Any]:
        """Analyze temporal smoothness of all metrics."""
        if len(metrics_history) < 10:
            return {'error': 'Insufficient data for smoothness analysis'}
        
        # Extract metric time series
        elbow_angles = [m.front_elbow_angle or 0.0 for m in metrics_history]
        spine_leans = [m.spine_lean_angle or 0.0 for m in metrics_history]
        head_knee_dists = [m.head_over_front_knee or 0.0 for m in metrics_history]
        foot_directions = [m.front_foot_direction or 0.0 for m in metrics_history]
        
        # Calculate smoothness metrics
        smoothness_scores = {
            'elbow_angle': self._calculate_metric_smoothness(elbow_angles, 'angle'),
            'spine_lean': self._calculate_metric_smoothness(spine_leans, 'angle'),
            'head_knee_distance': self._calculate_metric_smoothness(head_knee_dists, 'distance'),
            'foot_direction': self._calculate_metric_smoothness(foot_directions, 'angle')
        }
        
        # Calculate overall smoothness
        overall_smoothness = np.mean([
            smoothness_scores['elbow_angle']['overall_score'],
            smoothness_scores['spine_lean']['overall_score'],
            smoothness_scores['head_knee_distance']['overall_score'],
            smoothness_scores['foot_direction']['overall_score']
        ])
        
        # Identify problematic regions
        problematic_regions = self._identify_problematic_regions(metrics_history, smoothness_scores)
        
        return {
            'overall_smoothness': overall_smoothness,
            'metric_smoothness': smoothness_scores,
            'problematic_regions': problematic_regions,
            'recommendations': self._generate_smoothness_recommendations(smoothness_scores)
        }
    
    def _calculate_metric_smoothness(self, values: List[float], metric_type: str) -> Dict[str, Any]:
        """Calculate smoothness metrics for a single metric."""
        if len(values) < 3:
            return {'overall_score': 0.0, 'issues': ['Insufficient data']}
        
        # Calculate frame-to-frame changes
        changes = [abs(values[i] - values[i-1]) for i in range(1, len(values))]
        
        # Calculate statistics
        mean_change = np.mean(changes)
        std_change = np.std(changes)
        max_change = max(changes)
        
        # Identify outliers
        outlier_threshold = mean_change + self.thresholds['outlier_threshold'] * std_change
        outliers = [i for i, change in enumerate(changes) if change > outlier_threshold]
        
        # Calculate local smoothness (using rolling window)
        local_smoothness = []
        window = self.thresholds['smoothness_window']
        
        for i in range(len(values)):
            start_idx = max(0, i - window // 2)
            end_idx = min(len(values), i + window // 2 + 1)
            window_values = values[start_idx:end_idx]
            
            if len(window_values) > 1:
                window_changes = [abs(window_values[j] - window_values[j-1]) 
                                for j in range(1, len(window_values))]
                local_smoothness.append(1.0 / (1.0 + np.mean(window_changes)))
            else:
                local_smoothness.append(1.0)
        
        # Calculate overall smoothness score
        if metric_type == 'angle':
            max_allowed_change = self.thresholds['max_angle_jump']
        else:  # distance
            max_allowed_change = self.thresholds['max_distance_jump']
        
        # Score based on how well changes stay within acceptable range
        acceptable_changes = sum(1 for change in changes if change <= max_allowed_change)
        change_score = acceptable_changes / len(changes) if changes else 0.0
        
        # Score based on consistency (lower variance is better)
        consistency_score = max(0.0, 1.0 - (std_change / max_allowed_change))
        
        # Score based on local smoothness
        local_score = np.mean(local_smoothness)
        
        # Overall score (weighted average)
        overall_score = (change_score * 0.4 + consistency_score * 0.3 + local_score * 0.3)
        
        return {
            'overall_score': overall_score,
            'mean_change': mean_change,
            'std_change': std_change,
            'max_change': max_change,
            'outlier_count': len(outliers),
            'outlier_frames': outliers,
            'local_smoothness': local_smoothness,
            'change_score': change_score,
            'consistency_score': consistency_score,
            'local_score': local_score
        }
    
    def _identify_problematic_regions(self, metrics_history: List[FrameMetrics],
                                    smoothness_scores: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify regions with poor smoothness."""
        problematic_regions = []
        
        # Find regions with multiple outliers
        all_outlier_frames = set()
        for metric_name, scores in smoothness_scores.items():
            if 'outlier_frames' in scores:
                all_outlier_frames.update(scores['outlier_frames'])
        
        # Group consecutive outlier frames into regions
        if all_outlier_frames:
            outlier_frames = sorted(list(all_outlier_frames))
            regions = []
            current_region = [outlier_frames[0]]
            
            for frame in outlier_frames[1:]:
                if frame == current_region[-1] + 1:
                    current_region.append(frame)
                else:
                    if len(current_region) >= 2:  # Only consider regions with 2+ frames
                        regions.append(current_region)
                    current_region = [frame]
            
            if len(current_region) >= 2:
                regions.append(current_region)
            
            # Analyze each region
            for region in regions:
                region_analysis = {
                    'start_frame': region[0],
                    'end_frame': region[-1],
                    'duration': len(region),
                    'severity': self._calculate_region_severity(region, smoothness_scores),
                    'affected_metrics': self._identify_affected_metrics(region, smoothness_scores)
                }
                problematic_regions.append(region_analysis)
        
        return problematic_regions
    
    def _calculate_region_severity(self, region: List[int], 
                                 smoothness_scores: Dict[str, Any]) -> float:
        """Calculate severity of a problematic region."""
        severity = 0.0
        
        for metric_name, scores in smoothness_scores.items():
            if 'outlier_frames' in scores:
                region_outliers = [f for f in region if f in scores['outlier_frames']]
                if region_outliers:
                    # Severity based on number of outliers and their magnitude
                    severity += len(region_outliers) / len(region)
        
        return min(1.0, severity)
    
    def _identify_affected_metrics(self, region: List[int], 
                                 smoothness_scores: Dict[str, Any]) -> List[str]:
        """Identify which metrics are affected in a region."""
        affected = []
        
        for metric_name, scores in smoothness_scores.items():
            if 'outlier_frames' in scores:
                region_outliers = [f for f in region if f in scores['outlier_frames']]
                if region_outliers:
                    affected.append(metric_name)
        
        return affected
    
    def _generate_smoothness_recommendations(self, smoothness_scores: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on smoothness analysis."""
        recommendations = []
        
        for metric_name, scores in smoothness_scores.items():
            if scores['overall_score'] < 0.7:
                recommendations.append(f"Improve {metric_name} smoothness (score: {scores['overall_score']:.2f})")
            
            if scores['outlier_count'] > len(scores.get('local_smoothness', [])) * 0.1:
                recommendations.append(f"Reduce {metric_name} outliers ({scores['outlier_count']} detected)")
        
        if not recommendations:
            recommendations.append("Overall smoothness is good")
        
        return recommendations


class ChartGenerator:
    """Generates charts and visualizations for analysis."""
    
    def __init__(self):
        # Chart styling
        plt.style.use('default')
        self.colors = {
            'elbow': '#1f77b4',
            'spine': '#ff7f0e',
            'head_knee': '#2ca02c',
            'foot': '#d62728',
            'good': '#2ca02c',
            'warning': '#ff7f0e',
            'poor': '#d62728'
        }
    
    def create_smoothness_chart(self, metrics_history: List[FrameMetrics], 
                              smoothness_analysis: Dict[str, Any],
                              output_path: str = "output/metrics_smoothness.png") -> str:
        """Create comprehensive smoothness analysis chart."""
        
        if len(metrics_history) < 10:
            return ""
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Cricket Cover Drive - Temporal Smoothness Analysis', fontsize=16, fontweight='bold')
        
        # Extract metric data
        elbow_angles = [m.front_elbow_angle or 0.0 for m in metrics_history]
        spine_leans = [m.spine_lean_angle or 0.0 for m in metrics_history]
        head_knee_dists = [m.head_over_front_knee or 0.0 for m in metrics_history]
        foot_directions = [m.front_foot_direction or 0.0 for m in metrics_history]
        
        # Plot 1: Elbow Angle
        self._plot_metric_with_smoothness(
            axes[0, 0], elbow_angles, 'Elbow Angle (degrees)', 
            smoothness_analysis['metric_smoothness']['elbow_angle'],
            self.colors['elbow']
        )
        
        # Plot 2: Spine Lean
        self._plot_metric_with_smoothness(
            axes[0, 1], spine_leans, 'Spine Lean (degrees)',
            smoothness_analysis['metric_smoothness']['spine_lean'],
            self.colors['spine']
        )
        
        # Plot 3: Head-Knee Distance
        self._plot_metric_with_smoothness(
            axes[1, 0], head_knee_dists, 'Head-Knee Distance (normalized)',
            smoothness_analysis['metric_smoothness']['head_knee_distance'],
            self.colors['head_knee']
        )
        
        # Plot 4: Foot Direction
        self._plot_metric_with_smoothness(
            axes[1, 1], foot_directions, 'Foot Direction (degrees)',
            smoothness_analysis['metric_smoothness']['foot_direction'],
            self.colors['foot']
        )
        
        # Add overall smoothness score
        overall_score = smoothness_analysis['overall_smoothness']
        fig.text(0.02, 0.02, f'Overall Smoothness Score: {overall_score:.2f}', 
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=self._get_score_color(overall_score), alpha=0.7))
        
        # Add problematic regions
        if smoothness_analysis['problematic_regions']:
            self._add_problematic_regions_annotation(fig, smoothness_analysis['problematic_regions'])
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _plot_metric_with_smoothness(self, ax, values: List[float], title: str,
                                   smoothness_data: Dict[str, Any], color: str):
        """Plot a metric with smoothness indicators."""
        frames = list(range(len(values)))
        
        # Plot main metric
        ax.plot(frames, values, color=color, linewidth=2, label='Metric Value')
        
        # Highlight outliers
        if 'outlier_frames' in smoothness_data:
            outlier_frames = smoothness_data['outlier_frames']
            outlier_values = [values[i] for i in outlier_frames if i < len(values)]
            ax.scatter(outlier_frames, outlier_values, color='red', s=50, 
                      label='Outliers', zorder=5)
        
        # Add smoothness score
        score = smoothness_data['overall_score']
        ax.text(0.02, 0.98, f'Smoothness: {score:.2f}', 
                transform=ax.transAxes, fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.2", facecolor=self._get_score_color(score), alpha=0.7),
                verticalalignment='top')
        
        # Styling
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def _get_score_color(self, score: float) -> str:
        """Get color based on score."""
        if score >= 0.8:
            return self.colors['good']
        elif score >= 0.6:
            return self.colors['warning']
        else:
            return self.colors['poor']
    
    def _add_problematic_regions_annotation(self, fig, problematic_regions: List[Dict[str, Any]]):
        """Add annotation for problematic regions."""
        if not problematic_regions:
            return
        
        # Create text box with problematic regions info
        text_lines = ["Problematic Regions:"]
        for i, region in enumerate(problematic_regions[:3]):  # Show first 3 regions
            text_lines.append(
                f"Frames {region['start_frame']}-{region['end_frame']}: "
                f"{', '.join(region['affected_metrics'])}"
            )
        
        if len(problematic_regions) > 3:
            text_lines.append(f"... and {len(problematic_regions) - 3} more regions")
        
        fig.text(0.98, 0.02, '\n'.join(text_lines), 
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7),
                verticalalignment='bottom', horizontalalignment='right')
    
    def create_end_slate(self, frame: np.ndarray, smoothness_analysis: Dict[str, Any]) -> np.ndarray:
        """Create end slate overlay with smoothness scores."""
        h, w = frame.shape[:2]
        
        # Create overlay
        overlay = frame.copy()
        
        # Semi-transparent background
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Title
        title = "Temporal Smoothness Analysis"
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        title_x = (w - title_size[0]) // 2
        cv2.putText(frame, title, (title_x, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        
        # Overall score
        overall_score = smoothness_analysis['overall_smoothness']
        score_text = f"Overall Smoothness: {overall_score:.2f}"
        score_color = self._get_cv2_score_color(overall_score)
        score_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
        score_x = (w - score_size[0]) // 2
        cv2.putText(frame, score_text, (score_x, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.2, score_color, 2)
        
        # Individual metric scores
        y_offset = 250
        for metric_name, scores in smoothness_analysis['metric_smoothness'].items():
            metric_score = scores['overall_score']
            metric_color = self._get_cv2_score_color(metric_score)
            
            # Format metric name
            display_name = metric_name.replace('_', ' ').title()
            metric_text = f"{display_name}: {metric_score:.2f}"
            
            cv2.putText(frame, metric_text, (100, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, metric_color, 2)
            y_offset += 50
        
        # Recommendations
        if 'recommendations' in smoothness_analysis:
            y_offset += 30
            cv2.putText(frame, "Recommendations:", (100, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 40
            
            for rec in smoothness_analysis['recommendations'][:3]:  # Show first 3
                cv2.putText(frame, f"• {rec}", (120, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                y_offset += 30
        
        return frame
    
    def _get_cv2_score_color(self, score: float) -> Tuple[int, int, int]:
        """Get OpenCV color based on score."""
        if score >= 0.8:
            return (0, 255, 0)  # Green
        elif score >= 0.6:
            return (0, 165, 255)  # Orange
        else:
            return (0, 0, 255)  # Red
