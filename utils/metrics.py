"""
Biomechanical metrics calculation for cricket cover drive analysis.
Computes per-frame angles, distances, and other cricket-specific metrics.
"""

import numpy as np
import math
from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass

from .pose import KeyPoint, PoseEstimator

MIN_CONFIDENCE = 0.05  # Lower threshold to allow more detections


def _is_valid_keypoint(landmarks: Dict[str, KeyPoint], name: str) -> bool:
    """Check if a keypoint exists and has sufficient confidence."""
    kp = landmarks.get(name)
    return kp is not None and kp.confidence >= MIN_CONFIDENCE


def _get_point(landmarks: Dict[str, KeyPoint], name: str) -> Optional[tuple]:
    """Return (x, y) tuple if keypoint is valid, else None."""
    if _is_valid_keypoint(landmarks, name):
        kp = landmarks[name]
        return (kp.x, kp.y)
    return None


def _angle_between_points(a, b, c) -> Optional[float]:
    """Return the angle at point b given three points."""
    if None in (a, b, c):
        return None
    ab = (a[0] - b[0], a[1] - b[1])
    cb = (c[0] - b[0], c[1] - b[1])
    dot = ab[0] * cb[0] + ab[1] * cb[1]
    mag_ab = math.sqrt(ab[0] ** 2 + ab[1] ** 2)
    mag_cb = math.sqrt(cb[0] ** 2 + cb[1] ** 2)
    if mag_ab == 0 or mag_cb == 0:
        return None
    cos_angle = max(-1, min(1, dot / (mag_ab * mag_cb)))
    return math.degrees(math.acos(cos_angle))


@dataclass
class FrameMetrics:
    """Container for all metrics computed in a single frame."""
    front_elbow_angle: Optional[float] = None
    spine_lean_angle: Optional[float] = None  
    head_over_front_knee: Optional[float] = None
    front_foot_direction: Optional[float] = None
    
    back_elbow_angle: Optional[float] = None
    knee_separation: Optional[float] = None
    shoulder_rotation: Optional[float] = None
    hip_rotation: Optional[float] = None
    
    front_elbow_velocity: Optional[float] = None
    spine_lean_velocity: Optional[float] = None
    
    frame_number: int = 0
    timestamp: float = 0.0


class MetricsCalculator:
    """Calculate biomechanical metrics from pose landmarks."""
    
    def __init__(self, handedness: str = "right"):
        self.handedness = handedness
        self.pose_estimator = PoseEstimator()
        self.prev_metrics: Optional[FrameMetrics] = None
        self.frame_count = 0
        
    def calculate_frame_metrics(self, 
                              landmarks: Dict[str, KeyPoint], 
                              frame_shape: Tuple[int, int],
                              timestamp: float = 0.0) -> FrameMetrics:
        """Calculate all metrics for a single frame."""
        
        h, w = frame_shape
        metrics = FrameMetrics(frame_number=self.frame_count, timestamp=timestamp)
        
        handedness_map = self.pose_estimator.get_handedness_keypoints(landmarks, self.handedness)
        
        metrics.front_elbow_angle = self._calculate_elbow_angle(
            landmarks, handedness_map['front_shoulder'], 
            handedness_map['front_elbow'], handedness_map['front_wrist']
        )
        
        metrics.spine_lean_angle = self._calculate_spine_lean(landmarks, frame_shape)
        
        metrics.head_over_front_knee = self._calculate_head_over_front_knee(
            landmarks, handedness_map['front_knee'], frame_shape
        )
        
        metrics.front_foot_direction = self._calculate_foot_direction(
            landmarks, handedness_map['front_heel'], 
            handedness_map['front_foot_index'], frame_shape
        )
        
        metrics.back_elbow_angle = self._calculate_elbow_angle(
            landmarks, handedness_map['back_shoulder'],
            handedness_map['back_elbow'], handedness_map['back_wrist']
        )
        
        metrics.knee_separation = self._calculate_knee_separation(
            landmarks, handedness_map['front_knee'], handedness_map['back_knee']
        )
        
        metrics.shoulder_rotation = self._calculate_shoulder_rotation(landmarks, frame_shape)
        metrics.hip_rotation = self._calculate_hip_rotation(landmarks, frame_shape)
        
        if self.prev_metrics is not None:
            dt = max(1e-6, timestamp - self.prev_metrics.timestamp)
            metrics.front_elbow_velocity = self._calculate_velocity(
                self.prev_metrics.front_elbow_angle, metrics.front_elbow_angle, dt
            )
            metrics.spine_lean_velocity = self._calculate_velocity(
                self.prev_metrics.spine_lean_angle, metrics.spine_lean_angle, dt
            )
        
        self.prev_metrics = metrics
        self.frame_count += 1
        
        return metrics
    
    def _calculate_elbow_angle(self, landmarks, shoulder_name, elbow_name, wrist_name):
        shoulder = _get_point(landmarks, shoulder_name)
        elbow = _get_point(landmarks, elbow_name)
        wrist = _get_point(landmarks, wrist_name)
        return _angle_between_points(shoulder, elbow, wrist)
    
    def _calculate_spine_lean(self, landmarks, frame_shape):
        left_shoulder = _get_point(landmarks, "left_shoulder")
        right_shoulder = _get_point(landmarks, "right_shoulder")
        left_hip = _get_point(landmarks, "left_hip")
        right_hip = _get_point(landmarks, "right_hip")
        if None in (left_shoulder, right_shoulder, left_hip, right_hip):
            return None
        mid_shoulder = ((left_shoulder[0] + right_shoulder[0]) / 2,
                        (left_shoulder[1] + right_shoulder[1]) / 2)
        mid_hip = ((left_hip[0] + right_hip[0]) / 2,
                   (left_hip[1] + right_hip[1]) / 2)
        angle = math.degrees(math.atan2(mid_hip[0] - mid_shoulder[0],
                                        mid_shoulder[1] - mid_hip[1]))
        return abs(angle)
    
    def _calculate_head_over_front_knee(self, landmarks, front_knee_name, frame_shape):
        nose = _get_point(landmarks, "nose")
        knee = _get_point(landmarks, front_knee_name)
        if None in (nose, knee):
            return None
        return abs(nose[0] - knee[0]) / max(1, abs(knee[1] - nose[1]))
    
    def _calculate_foot_direction(self, landmarks, heel_name, toe_name, frame_shape):
        heel = _get_point(landmarks, heel_name)
        toe = _get_point(landmarks, toe_name)
        return _angle_between_points(heel, toe, (toe[0] + 1, toe[1])) if heel and toe else None
    
    def _calculate_knee_separation(self, landmarks, front_knee_name, back_knee_name):
        front_knee = _get_point(landmarks, front_knee_name)
        back_knee = _get_point(landmarks, back_knee_name)
        if None in (front_knee, back_knee):
            return None
        dx = front_knee[0] - back_knee[0]
        dy = front_knee[1] - back_knee[1]
        return math.sqrt(dx*dx + dy*dy)
    
    def _calculate_shoulder_rotation(self, landmarks, frame_shape):
        left_shoulder = _get_point(landmarks, "left_shoulder")
        right_shoulder = _get_point(landmarks, "right_shoulder")
        right_hip = _get_point(landmarks, "right_hip")
        return _angle_between_points(left_shoulder, right_shoulder, right_hip)
    
    def _calculate_hip_rotation(self, landmarks, frame_shape):
        left_hip = _get_point(landmarks, "left_hip")
        right_hip = _get_point(landmarks, "right_hip")
        if None in (left_hip, right_hip):
            return None
        dx = right_hip[0] - left_hip[0]
        dy = right_hip[1] - left_hip[1]
        return math.degrees(math.atan2(dy, dx))
    
    def _calculate_velocity(self, prev_value, current_value, dt=1.0):
        if prev_value is None or current_value is None:
            return None
        return (current_value - prev_value) / dt
    
    def get_primary_metrics(self):
        return ['front_elbow_angle', 'spine_lean_angle', 'head_over_front_knee', 'front_foot_direction']
    
    def metrics_to_dict(self, metrics: FrameMetrics) -> Dict[str, Any]:
        return metrics.__dict__
