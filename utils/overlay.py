"""
Video overlay utilities for drawing pose skeleton and HUD with metrics.
Creates visual feedback on each frame of the analysis.
"""

import cv2
import numpy as np
from typing import Dict, Optional, Any, Tuple, List
import math

from .pose import KeyPoint
from .metrics import FrameMetrics


def draw_text_with_degree_symbol(frame, text, position, font, scale, color, thickness):
    """Draw text with proper degree symbol handling."""
    # Replace degree symbol with a more compatible approach
    if '°' in text:
        # Split text at degree symbol
        parts = text.split('°')
        if len(parts) == 2:
            # Draw the number part
            cv2.putText(frame, parts[0], position, font, scale, color, thickness)
            
            # Calculate position for the degree symbol
            (text_width, text_height), baseline = cv2.getTextSize(parts[0], font, scale, thickness)
            degree_pos = (position[0] + text_width, position[1])
            
            # Draw a small circle as degree symbol
            cv2.circle(frame, (degree_pos[0] + 3, degree_pos[1] - text_height + 3), 2, color, -1)
            
            # Draw the rest of the text if any
            if parts[1]:
                rest_pos = (degree_pos[0] + 8, position[1])
                cv2.putText(frame, parts[1], rest_pos, font, scale, color, thickness)
        else:
            cv2.putText(frame, text, position, font, scale, color, thickness)
    else:
        cv2.putText(frame, text, position, font, scale, color, thickness)


class VideoOverlay:
    """Handles drawing pose skeleton and metrics HUD on video frames."""
    
    def __init__(self):
        # Color scheme
        self.colors = {
            'skeleton': (0, 255, 0),      # Green for skeleton
            'joints': (0, 0, 255),        # Red for joints
            'good_metric': (0, 255, 0),   # Green for good values
            'warning_metric': (0, 165, 255),  # Orange for warnings
            'bad_metric': (0, 0, 255),    # Red for bad values
            'missing_metric': (128, 128, 128),  # Gray for missing
            'hud_bg': (0, 0, 0),          # Black background
            'hud_text': (255, 255, 255),  # White text
            'hud_border': (100, 100, 100)  # Gray border
        }
        
        # Pose connections for skeleton drawing
        self.pose_connections = [
            # Torso
            ('left_shoulder', 'right_shoulder'),
            ('left_shoulder', 'left_hip'),
            ('right_shoulder', 'right_hip'),
            ('left_hip', 'right_hip'),
            
            # Arms
            ('left_shoulder', 'left_elbow'),
            ('left_elbow', 'left_wrist'),
            ('right_shoulder', 'right_elbow'),
            ('right_elbow', 'right_wrist'),
            
            # Legs
            ('left_hip', 'left_knee'),
            ('left_knee', 'left_ankle'),
            ('right_hip', 'right_knee'),
            ('right_knee', 'right_ankle'),
            
            # Head
            ('nose', 'left_shoulder'),
            ('nose', 'right_shoulder'),
            
            # Feet
            ('left_ankle', 'left_heel'),
            ('left_heel', 'left_foot_index'),
            ('right_ankle', 'right_heel'),
            ('right_heel', 'right_foot_index')
        ]
        
        # Metric thresholds for color coding
        self.metric_thresholds = {
            'front_elbow_angle': {'good': (120, 160), 'warning': (100, 180)},
            'spine_lean_angle': {'good': (0, 15), 'warning': (0, 25)},
            'head_over_front_knee': {'good': (0, 0.15), 'warning': (0, 0.3)},
            'front_foot_direction': {'good': (0, 20), 'warning': (0, 35)}
        }
        
    def draw_overlay(self, frame: np.ndarray, 
                    pose_result: Optional[Dict[str, Any]],
                    metrics: FrameMetrics,
                    frame_number: int) -> np.ndarray:
        """Draw complete overlay including skeleton and HUD."""
        
        # Create copy of frame
        overlay_frame = frame.copy()
        
        # Draw pose skeleton if available
        if pose_result and pose_result.get('landmarks'):
            overlay_frame = self._draw_skeleton(overlay_frame, pose_result['landmarks'])
            
        # Draw metrics HUD
        overlay_frame = self._draw_hud(overlay_frame, metrics, frame_number)
        
        # Draw frame info
        overlay_frame = self._draw_frame_info(overlay_frame, frame_number, pose_result)
        
        return overlay_frame
    
    def _draw_skeleton(self, frame: np.ndarray, landmarks: Dict[str, KeyPoint]) -> np.ndarray:
        """Draw pose skeleton on frame."""
        
        # Draw connections
        for connection in self.pose_connections:
            start_name, end_name = connection
            
            if (start_name in landmarks and end_name in landmarks and
                landmarks[start_name].confidence > 0.3 and landmarks[end_name].confidence > 0.3):
                
                start_point = landmarks[start_name]
                end_point = landmarks[end_name]
                
                cv2.line(frame, 
                        (int(start_point.x), int(start_point.y)),
                        (int(end_point.x), int(end_point.y)),
                        self.colors['skeleton'], 2)
        
        # Draw joints
        for name, keypoint in landmarks.items():
            if keypoint.confidence > 0.3:
                cv2.circle(frame, 
                          (int(keypoint.x), int(keypoint.y)),
                          4, self.colors['joints'], -1)
                
                # Draw confidence indicator
                if keypoint.confidence < 0.6:
                    cv2.circle(frame,
                              (int(keypoint.x), int(keypoint.y)),
                              6, self.colors['warning_metric'], 2)
        
        return frame
    
    def _draw_hud(self, frame: np.ndarray, metrics: FrameMetrics, frame_number: int) -> np.ndarray:
        """Draw heads-up display with live metrics."""
    
        h, w = frame.shape[:2]
        
        # HUD dimensions and position - bottom-left corner
        hud_width = 320
        hud_height = 200
        hud_x = 10
        hud_y = h - hud_height - 10  # from bottom
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(
            overlay, 
            (hud_x, hud_y), 
            (hud_x + hud_width, hud_y + hud_height),
            self.colors['hud_bg'], 
            -1
        )
        
        # Apply semi-transparency
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        # Draw border
        cv2.rectangle(
            frame,
            (hud_x, hud_y),
            (hud_x + hud_width, hud_y + hud_height),
            self.colors['hud_border'], 
            2
        )
        
        # HUD title
        cv2.putText(
            frame, "Live Metrics", 
            (hud_x + 10, hud_y + 25),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            self.colors['hud_text'], 
            2
        )
        
        # Draw primary metrics
        primary_metrics = [
            ('Elbow Angle', 'front_elbow_angle', 'deg'),
            ('Spine Lean', 'spine_lean_angle', 'deg'),
            ('Head-Knee Dist', 'head_over_front_knee', ''),
            ('Foot Direction', 'front_foot_direction', 'deg')
        ]
        
        y_offset = 50
        for label, metric_name, unit in primary_metrics:
            value = getattr(metrics, metric_name)
            
            # Format value and determine color
            if value is not None:
                if unit == 'deg':
                    value_text = f"{value:.1f}°"  # Use degree symbol
                else:
                    value_text = f"{value:.3f}"
                color = self._get_metric_color(metric_name, value)
            else:
                value_text = "—"
                color = self.colors['missing_metric']
            
            # Draw metric label
            cv2.putText(
                frame, f"{label}:", 
                (hud_x + 10, hud_y + y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                self.colors['hud_text'], 
                1
            )
                         # Draw metric value
            draw_text_with_degree_symbol(
                frame, value_text,
                (hud_x + 180, hud_y + y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                color, 
                2
            )
            
            # Draw feedback cue
            if value is not None:
                feedback_cue = self._get_feedback_cue(metric_name, value)
                if feedback_cue:
                    cv2.putText(
                        frame, feedback_cue,
                        (hud_x + 280, hud_y + y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.4, 
                        color, 
                        1
                    )
        
        y_offset += 25
        
        # Draw additional info
        y_offset += 10
        cv2.putText(
            frame, f"Frame: {frame_number}",
            (hud_x + 10, hud_y + y_offset),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.4, 
            self.colors['hud_text'], 
            1
        )
    
        return frame

    
    def _draw_frame_info(self, frame: np.ndarray, frame_number: int, 
                        pose_result: Optional[Dict[str, Any]]) -> np.ndarray:
        """Draw frame information overlay."""
        
        h, w = frame.shape[:2]
        
        # Pose confidence indicator
        if pose_result:
            confidence = pose_result.get('confidence', 0.0)
            method = pose_result.get('method', 'unknown')
            
            confidence_text = f"Pose: {confidence:.2f} ({method})"
            confidence_color = (
                self.colors['good_metric'] if confidence > 0.7 else
                self.colors['warning_metric'] if confidence > 0.4 else
                self.colors['bad_metric']
            )
        else:
            confidence_text = "Pose: No Detection"
            confidence_color = self.colors['bad_metric']
        
        # Draw confidence info
        cv2.putText(frame, confidence_text,
                   (10, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, confidence_color, 1)
        
        return frame
    
    def _get_metric_color(self, metric_name: str, value: float) -> Tuple[int, int, int]:
        """Get color for metric value based on thresholds."""
        
        if metric_name not in self.metric_thresholds:
            return self.colors['hud_text']
        
        thresholds = self.metric_thresholds[metric_name]
        good_range = thresholds['good']
        warning_range = thresholds['warning']
        
        # Check if in good range
        if good_range[0] <= abs(value) <= good_range[1]:
            return self.colors['good_metric']
        # Check if in warning range
        elif warning_range[0] <= abs(value) <= warning_range[1]:
            return self.colors['warning_metric']
        # Outside acceptable ranges
        else:
            return self.colors['bad_metric']
    
    def _get_feedback_cue(self, metric_name: str, value: float) -> str:
        """Get feedback cue based on metric value and thresholds."""
        
        if metric_name not in self.metric_thresholds:
            return ""
        
        thresholds = self.metric_thresholds[metric_name]
        good_range = thresholds['good']
        warning_range = thresholds['warning']
        
        # Define feedback messages
        feedback_messages = {
            'front_elbow_angle': {
                'good': '✅ Good elbow',
                'warning': '⚠️ Elbow angle',
                'bad': '❌ Poor elbow'
            },
            'spine_lean_angle': {
                'good': '✅ Good posture',
                'warning': '⚠️ Spine lean',
                'bad': '❌ Poor posture'
            },
            'head_over_front_knee': {
                'good': '✅ Good head position',
                'warning': '⚠️ Head position',
                'bad': '❌ Head not over knee'
            },
            'front_foot_direction': {
                'good': '✅ Good foot direction',
                'warning': '⚠️ Foot direction',
                'bad': '❌ Poor foot direction'
            }
        }
        
        if metric_name not in feedback_messages:
            return ""
        
        messages = feedback_messages[metric_name]
        
        # Check ranges and return appropriate message
        if good_range[0] <= abs(value) <= good_range[1]:
            return messages['good']
        elif warning_range[0] <= abs(value) <= warning_range[1]:
            return messages['warning']
        else:
            return messages['bad']
    
    def draw_debug_overlay(self, frame: np.ndarray,
                          raw_landmarks: Optional[Dict[str, KeyPoint]],
                          smoothed_landmarks: Optional[Dict[str, KeyPoint]]) -> np.ndarray:
        """Draw debug overlay showing raw vs smoothed keypoints."""
        
        debug_frame = frame.copy()
        
        # Draw raw keypoints in red
        if raw_landmarks:
            for name, keypoint in raw_landmarks.items():
                if keypoint.confidence > 0.3:
                    cv2.circle(debug_frame,
                              (int(keypoint.x), int(keypoint.y)),
                              3, (0, 0, 255), -1)  # Red
        
        # Draw smoothed keypoints in green
        if smoothed_landmarks:
            for name, keypoint in smoothed_landmarks.items():
                if keypoint.confidence > 0.3:
                    cv2.circle(debug_frame,
                              (int(keypoint.x), int(keypoint.y)),
                              5, (0, 255, 0), 2)  # Green outline
        
        # Add legend
        cv2.putText(debug_frame, "Red: Raw | Green: Smoothed",
                   (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return debug_frame
    
    def create_analysis_visualization(self, frame: np.ndarray,
                                    metrics_history: List[FrameMetrics],
                                    current_frame: int) -> np.ndarray:
        """Create comprehensive analysis visualization with charts."""
        
        h, w = frame.shape[:2]
        
        # Create larger canvas for visualization
        viz_height = h + 200
        viz_frame = np.zeros((viz_height, w, 3), dtype=np.uint8)
        
        # Copy original frame to top
        viz_frame[:h, :w] = frame
        
        # Draw metric charts in bottom section
        chart_y = h + 10
        chart_height = 180
        
        self._draw_metric_charts(viz_frame, metrics_history, current_frame,
                               chart_y, chart_height, w)
        
        return viz_frame
    
    def _draw_metric_charts(self, frame: np.ndarray,
                          metrics_history: List[FrameMetrics],
                          current_frame: int,
                          chart_y: int, chart_height: int, chart_width: int):
        """Draw time-series charts for metrics."""
        
        if len(metrics_history) < 2:
            return
        
        # Chart configuration
        num_charts = 4
        chart_w = chart_width // num_charts
        margin = 10
        
        metrics_to_plot = [
            ('front_elbow_angle', 'Elbow', (0, 180)),
            ('spine_lean_angle', 'Spine', (0, 45)),
            ('head_over_front_knee', 'Head-Knee', (0, 1)),
            ('front_foot_direction', 'Foot', (0, 45))
        ]
        
        # Get data window (last 60 frames or all available)
        window_size = min(60, len(metrics_history))
        start_idx = max(0, len(metrics_history) - window_size)
        data_window = metrics_history[start_idx:]
        
        for i, (metric_name, title, y_range) in enumerate(metrics_to_plot):
            chart_x = i * chart_w + margin
            
            # Extract metric values
            values = []
            frame_indices = []
            
            for j, metrics in enumerate(data_window):
                value = getattr(metrics, metric_name)
                if value is not None:
                    values.append(value)
                    frame_indices.append(start_idx + j)
            
            if len(values) < 2:
                continue
            
            # Draw chart background
            cv2.rectangle(frame,
                         (chart_x, chart_y),
                         (chart_x + chart_w - margin, chart_y + chart_height),
                         (40, 40, 40), -1)
            cv2.rectangle(frame,
                         (chart_x, chart_y),
                         (chart_x + chart_w - margin, chart_y + chart_height),
                         (100, 100, 100), 1)
            
            # Draw title
            cv2.putText(frame, title,
                       (chart_x + 5, chart_y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Normalize values to chart coordinates
            y_min, y_max = y_range
            chart_inner_height = chart_height - 30
            
            normalized_values = []
            for value in values:
                norm_y = chart_y + 20 + chart_inner_height * (1 - (value - y_min) / (y_max - y_min))
                norm_y = max(chart_y + 20, min(chart_y + chart_height - 10, norm_y))
                normalized_values.append(norm_y)
            
            # Draw line chart
            points = []
            for j, (frame_idx, y_pos) in enumerate(zip(frame_indices, normalized_values)):
                x_pos = chart_x + 5 + (chart_w - margin - 10) * j / (len(values) - 1)
                points.append((int(x_pos), int(y_pos)))
            
            # Draw lines
            for j in range(len(points) - 1):
                cv2.line(frame, points[j], points[j + 1], (0, 255, 255), 1)
            
            # Highlight current point
            if points:
                current_point_idx = min(current_frame - start_idx, len(points) - 1)
                if 0 <= current_point_idx < len(points):
                    cv2.circle(frame, points[current_point_idx], 3, (0, 0, 255), -1)
            
            # Draw current value
            if values:
                current_val = values[-1]
                val_text = f"{current_val:.1f}"
                cv2.putText(frame, val_text,
                           (chart_x + 5, chart_y + chart_height - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


class AnalysisRenderer:
    """Renders comprehensive analysis results with multiple views."""
    
    def __init__(self):
        self.overlay = VideoOverlay()
        
    def create_summary_frame(self, frame: np.ndarray,
                           final_scores: Dict[str, Any],
                           stats: Dict[str, Any]) -> np.ndarray:
        """Create summary frame with final analysis results."""
        
        h, w = frame.shape[:2]
        
        # Create summary canvas
        summary_frame = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Dark background
        summary_frame[:] = (20, 20, 20)
        
        # Title
        title = "Cricket Cover Drive Analysis - Summary"
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        title_x = (w - title_size[0]) // 2
        cv2.putText(summary_frame, title,
                   (title_x, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Draw scores
        y_pos = 100
        for category, data in final_scores['summary'].items():
            score = data['score']
            feedback = data['feedback']
            
            # Score color
            if score >= 8:
                score_color = (0, 255, 0)  # Green
            elif score >= 6:
                score_color = (0, 165, 255)  # Orange
            else:
                score_color = (0, 0, 255)  # Red
            
            # Draw category
            cv2.putText(summary_frame, f"{category}:",
                       (50, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw score
            cv2.putText(summary_frame, f"{score}/10",
                       (250, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, score_color, 2)
            
            # Draw feedback
            cv2.putText(summary_frame, feedback,
                       (350, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            y_pos += 40
        
        # Draw global stats
        y_pos += 30
        cv2.putText(summary_frame, "Statistics:",
                   (50, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        y_pos += 30
        stats_text = [
            f"Frames Analyzed: {final_scores['global']['frames_analyzed']}",
            f"Missing Keypoints: {final_scores['global']['frames_with_missing_keypoints']}",
            f"Average FPS: {final_scores['global']['avg_fps']:.1f}",
            f"Handedness: {final_scores['global']['handedness']}"
        ]
        
        for stat_text in stats_text:
            cv2.putText(summary_frame, stat_text,
                       (70, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            y_pos += 25
        
        return summary_frame
    
    def create_comparison_view(self, original_frame: np.ndarray,
                             annotated_frame: np.ndarray) -> np.ndarray:
        """Create side-by-side comparison of original and annotated frames."""
        
        h, w = original_frame.shape[:2]
        
        # Create comparison canvas
        comparison_frame = np.zeros((h, w * 2 + 10, 3), dtype=np.uint8)
        
        # Place frames side by side
        comparison_frame[:h, :w] = original_frame
        comparison_frame[:h, w + 10:] = annotated_frame
        
        # Add labels
        cv2.putText(comparison_frame, "Original",
                   (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(comparison_frame, "Analysis",
                   (w + 20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Add separator line
        cv2.line(comparison_frame, (w + 5, 0), (w + 5, h), (100, 100, 100), 2)
        
        return comparison_frame