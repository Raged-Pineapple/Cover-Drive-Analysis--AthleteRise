"""
Pose estimation utilities with MediaPipe BlazePose and fallback models.
Includes keypoint normalization and temporal tracking.
"""

import cv2
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
import mediapipe as mp
from dataclasses import dataclass


@dataclass
class KeyPoint:
    """Represents a 2D keypoint with confidence."""
    x: float
    y: float
    confidence: float
    visibility: float = 1.0


class PoseEstimator:
    """Pose estimation with MediaPipe BlazePose and fallback refinement."""
    
    def __init__(self, 
                 model_complexity: int = 1,
                 min_detection_confidence: float = 0.05,  # Reduced from 0.15
                 min_tracking_confidence: float = 0.05,   # Reduced from 0.1
                 enable_segmentation: bool = False):
        
        # MediaPipe pose setup
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.pose = self.mp_pose.Pose(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            enable_segmentation=enable_segmentation,
            static_image_mode=False,  # For continuous tracking
            smooth_landmarks=True,    # For smoother tracking
            smooth_segmentation=True  # For smoother segmentation
        )
        
        # Keypoint indices for cricket-specific body parts
        self.cricket_keypoints = {
            # Upper body
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_elbow': 13,
            'right_elbow': 14,
            'left_wrist': 15,
            'right_wrist': 16,
            
            # Core
            'left_hip': 23,
            'right_hip': 24,
            'nose': 0,
            'mid_shoulder': None,  # Computed
            'mid_hip': None,      # Computed
            
            # Lower body  
            'left_knee': 25,
            'right_knee': 26,
            'left_ankle': 27,
            'right_ankle': 28,
            'left_heel': 29,
            'right_heel': 30,
            'left_foot_index': 31,
            'right_foot_index': 32
        }
        
        # For optical flow tracking
        self.prev_gray = None
        self.prev_keypoints = None
        
        # Fallback pose (placeholder for MoveNet/HRNet)
        self.has_fallback = False
        
    def estimate_pose(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """Estimate pose from frame with fallback and temporal refinement."""
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Primary pose estimation
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks:
            # Extract and normalize keypoints
            landmarks = self._extract_landmarks(results.pose_landmarks, frame.shape)
            confidence = self._calculate_pose_confidence(results.pose_landmarks)
            
            # Apply temporal refinement if we have previous frame
            if self.prev_keypoints is not None:
                landmarks = self._apply_temporal_refinement(frame, landmarks)
            
            self.prev_keypoints = landmarks
            self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            return {
                'landmarks': landmarks,
                'confidence': confidence,
                'raw_results': results,
                'method': 'mediapipe_primary'
            }
        
        # Fallback: try optical flow propagation if we have previous keypoints
        elif self.prev_keypoints is not None:
            refined_landmarks = self._optical_flow_fallback(frame)
            if refined_landmarks:
                return {
                    'landmarks': refined_landmarks,
                    'confidence': 0.3,  # Lower confidence for fallback
                    'raw_results': None,
                    'method': 'optical_flow_fallback'
                }
        
        # No pose detected
        return None
    
    def _extract_landmarks(self, pose_landmarks, frame_shape: Tuple[int, int]) -> Dict[str, KeyPoint]:
        """Extract and normalize keypoints to image coordinates."""
        h, w = frame_shape[:2]
        landmarks = {}
        
        # Extract raw keypoints
        for name, idx in self.cricket_keypoints.items():
            if idx is not None:  # Skip computed keypoints
                landmark = pose_landmarks.landmark[idx]
                landmarks[name] = KeyPoint(
                    x=landmark.x * w,
                    y=landmark.y * h,
                    confidence=landmark.visibility,
                    visibility=landmark.visibility
                )
        
        # Compute derived keypoints
        if 'left_shoulder' in landmarks and 'right_shoulder' in landmarks:
            ls, rs = landmarks['left_shoulder'], landmarks['right_shoulder']
            landmarks['mid_shoulder'] = KeyPoint(
                x=(ls.x + rs.x) / 2,
                y=(ls.y + rs.y) / 2,
                confidence=min(ls.confidence, rs.confidence),
                visibility=min(ls.visibility, rs.visibility)
            )
        
        if 'left_hip' in landmarks and 'right_hip' in landmarks:
            lh, rh = landmarks['left_hip'], landmarks['right_hip']
            landmarks['mid_hip'] = KeyPoint(
                x=(lh.x + rh.x) / 2,
                y=(lh.y + rh.y) / 2,
                confidence=min(lh.confidence, rh.confidence),
                visibility=min(lh.visibility, rh.visibility)
            )
        
        return landmarks
    
    def _calculate_pose_confidence(self, pose_landmarks) -> float:
        """Calculate overall pose confidence."""
        confidences = []
        
        for name, idx in self.cricket_keypoints.items():
            if idx is not None:
                landmark = pose_landmarks.landmark[idx]
                confidences.append(landmark.visibility)
        
        return np.mean(confidences) if confidences else 0.0
    
    def _apply_temporal_refinement(self, frame: np.ndarray, landmarks: Dict[str, KeyPoint]) -> Dict[str, KeyPoint]:
        """Apply optical flow and Kalman filtering for temporal consistency."""
        
        if self.prev_gray is None:
            return landmarks
        
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Prepare points for optical flow
        prev_points = []
        current_points = []
        point_names = []
        
        for name, kp in self.prev_keypoints.items():
            if kp.confidence > 0.1:  # Reduced from 0.3
                prev_points.append([kp.x, kp.y])
                if name in landmarks:
                    current_points.append([landmarks[name].x, landmarks[name].y])
                    point_names.append(name)
        
        if len(prev_points) < 3:  # Reduced from 5
            return landmarks
        
        prev_points = np.array(prev_points, dtype=np.float32)
        current_points = np.array(current_points, dtype=np.float32)
        
        try:
            # Calculate optical flow
            flow_points, status, error = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, current_gray, prev_points, current_points,
                winSize=(15, 15),
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            )
            
            # Apply temporal smoothing (simple weighted average)
            alpha = 0.7  # Weight for current detection
            
            for i, name in enumerate(point_names):
                if status[i] and error[i] < 50:  # Good optical flow
                    # Weighted average between detection and optical flow
                    flow_x, flow_y = flow_points[i]
                    det_x, det_y = landmarks[name].x, landmarks[name].y
                    
                    landmarks[name].x = alpha * det_x + (1 - alpha) * flow_x
                    landmarks[name].y = alpha * det_y + (1 - alpha) * flow_y
                    
        except cv2.error:
            # Optical flow failed, use raw landmarks
            pass
        
        return landmarks
    
    def _optical_flow_fallback(self, frame: np.ndarray) -> Optional[Dict[str, KeyPoint]]:
        """Use optical flow to propagate keypoints when pose detection fails."""
        
        if self.prev_gray is None or self.prev_keypoints is None:
            return None
        
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Prepare points
        prev_points = []
        point_names = []
        
        for name, kp in self.prev_keypoints.items():
            if kp.confidence > 0.2:  # Reduced from 0.4
                prev_points.append([kp.x, kp.y])
                point_names.append(name)
        
        if len(prev_points) < 5:  # Reduced from 8
            return None
        
        prev_points = np.array(prev_points, dtype=np.float32)
        
        try:
            # Calculate optical flow
            flow_points, status, error = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, current_gray, prev_points, None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
            )
            
            # Build new landmarks dict
            new_landmarks = {}
            
            for i, name in enumerate(point_names):
                if status[i] and error[i] < 30:  # Good flow
                    x, y = flow_points[i]
                    new_landmarks[name] = KeyPoint(
                        x=x, y=y, 
                        confidence=0.3,  # Lower confidence for propagated points
                        visibility=0.3
                    )
            
            # Only return if we have enough points
            if len(new_landmarks) >= 6:
                return new_landmarks
                
        except cv2.error:
            pass
        
        return None
    
    def get_keypoint_names(self) -> List[str]:
        """Get list of all keypoint names."""
        return list(self.cricket_keypoints.keys())
    
    def is_keypoint_valid(self, landmarks: Dict[str, KeyPoint], name: str, min_confidence: float = 0.3) -> bool:
        """Check if a keypoint is valid for metric calculation."""
        if name not in landmarks:
            return False
        
        kp = landmarks[name]
        return kp.confidence >= min_confidence and kp.visibility >= min_confidence
    
    def get_handedness_keypoints(self, landmarks: Dict[str, KeyPoint], handedness: str) -> Dict[str, str]:
        """Map front/back body parts based on handedness."""
        if handedness == "right":
            # Right-handed: front = left side, back = right side
            return {
                'front_shoulder': 'left_shoulder',
                'back_shoulder': 'right_shoulder',
                'front_elbow': 'left_elbow',
                'back_elbow': 'right_elbow',
                'front_wrist': 'left_wrist',
                'back_wrist': 'right_wrist',
                'front_hip': 'left_hip',
                'back_hip': 'right_hip',
                'front_knee': 'left_knee',
                'back_knee': 'right_knee',
                'front_ankle': 'left_ankle',
                'back_ankle': 'right_ankle',
                'front_heel': 'left_heel',
                'back_heel': 'right_heel',
                'front_foot_index': 'left_foot_index',
                'back_foot_index': 'right_foot_index'
            }
        else:
            # Left-handed: front = right side, back = left side  
            return {
                'front_shoulder': 'right_shoulder',
                'back_shoulder': 'left_shoulder', 
                'front_elbow': 'right_elbow',
                'back_elbow': 'left_elbow',
                'front_wrist': 'right_wrist',
                'back_wrist': 'left_wrist',
                'front_hip': 'right_hip',
                'back_hip': 'left_hip',
                'front_knee': 'right_knee',
                'back_knee': 'left_knee',
                'front_ankle': 'right_ankle',
                'back_ankle': 'left_ankle',
                'front_heel': 'right_heel',
                'back_heel': 'left_heel',
                'front_foot_index': 'right_foot_index',
                'back_foot_index': 'left_foot_index'
            }