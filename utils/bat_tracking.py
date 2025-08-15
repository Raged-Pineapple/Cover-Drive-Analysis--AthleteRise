"""
Bat detection and tracking for cricket cover drive analysis.
Uses computer vision techniques to detect and track the cricket bat.
"""

import numpy as np
import cv2
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from scipy.spatial.distance import euclidean

from .pose import KeyPoint


@dataclass
class BatDetection:
    """Represents a detected bat with its properties."""
    center: Tuple[int, int]
    angle: float
    length: float
    confidence: float
    frame_idx: int
    wrist_position: Optional[Tuple[int, int]] = None


@dataclass
class SwingPath:
    """Represents the swing path of the bat."""
    start_point: Tuple[int, int]
    end_point: Tuple[int, int]
    path_points: List[Tuple[int, int]]
    straightness: float
    angle_at_impact: Optional[float] = None


class BatDetector:
    """Detects and tracks cricket bat using computer vision."""
    
    def __init__(self):
        # Detection parameters
        self.detection_params = {
            'min_line_length': 50,
            'max_line_gap': 10,
            'hough_threshold': 50,
            'canny_low': 50,
            'canny_high': 150,
            'wrist_region_radius': 100,
            'min_bat_length': 30,
            'max_bat_length': 200
        }
        
        # Tracking parameters
        self.tracking_params = {
            'max_displacement': 50,
            'angle_tolerance': 30,
            'confidence_decay': 0.9,
            'min_tracking_confidence': 0.3
        }
        
        # State tracking
        self.previous_detections: List[BatDetection] = []
        self.swing_paths: List[SwingPath] = []
        self.current_swing_path: Optional[SwingPath] = None
        
    def detect_bat(self, frame: np.ndarray, wrist_keypoint: Optional[KeyPoint] = None,
                  frame_idx: int = 0) -> Optional[BatDetection]:
        """Detect bat in the current frame."""
        
        if wrist_keypoint is None:
            return None
        
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, self.detection_params['canny_low'], 
                         self.detection_params['canny_high'])
        
        # Define wrist region for bat detection
        wrist_region = self._define_wrist_region(edges, wrist_keypoint)
        
        # Detect lines in the wrist region
        lines = self._detect_lines_in_region(wrist_region, wrist_keypoint)
        
        # Filter and select the best bat line
        bat_line = self._select_bat_line(lines, wrist_keypoint)
        
        if bat_line is None:
            return None
        
        # Create bat detection object
        bat_detection = self._create_bat_detection(bat_line, wrist_keypoint, frame_idx)
        
        # Update tracking
        self._update_tracking(bat_detection)
        
        return bat_detection
    
    def _define_wrist_region(self, edges: np.ndarray, wrist_keypoint: KeyPoint) -> np.ndarray:
        """Define region around wrist for bat detection."""
        h, w = edges.shape
        
        # Create mask for wrist region
        mask = np.zeros_like(edges)
        center = (int(wrist_keypoint.x), int(wrist_keypoint.y))
        radius = self.detection_params['wrist_region_radius']
        
        cv2.circle(mask, center, radius, 255, -1)
        
        # Apply mask to edges
        wrist_region = cv2.bitwise_and(edges, mask)
        
        return wrist_region
    
    def _detect_lines_in_region(self, wrist_region: np.ndarray, 
                               wrist_keypoint: KeyPoint) -> List[Tuple[int, int, int, int]]:
        """Detect lines in the wrist region using Hough transform."""
        
        # Apply Hough line detection
        lines = cv2.HoughLinesP(
            wrist_region,
            rho=1,
            theta=np.pi/180,
            threshold=self.detection_params['hough_threshold'],
            minLineLength=self.detection_params['min_line_length'],
            maxLineGap=self.detection_params['max_line_gap']
        )
        
        if lines is None:
            return []
        
        # Convert to list of (x1, y1, x2, y2) tuples
        line_list = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            line_list.append((x1, y1, x2, y2))
        
        return line_list
    
    def _select_bat_line(self, lines: List[Tuple[int, int, int, int]], 
                        wrist_keypoint: KeyPoint) -> Optional[Tuple[int, int, int, int]]:
        """Select the best line that represents the bat."""
        
        if not lines:
            return None
        
        wrist_pos = (int(wrist_keypoint.x), int(wrist_keypoint.y))
        best_line = None
        best_score = 0.0
        
        for line in lines:
            x1, y1, x2, y2 = line
            
            # Calculate line properties
            line_length = euclidean((x1, y1), (x2, y2))
            line_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            distance_to_wrist = euclidean(line_center, wrist_pos)
            
            # Calculate line angle
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            angle = abs(angle) if angle < 0 else angle
            
            # Score the line based on multiple criteria
            score = self._score_bat_line(line_length, distance_to_wrist, angle)
            
            if score > best_score:
                best_score = score
                best_line = line
        
        return best_line if best_score > 0.5 else None
    
    def _score_bat_line(self, length: float, distance_to_wrist: float, angle: float) -> float:
        """Score a line based on how likely it is to be a bat."""
        
        # Length score (prefer lines within bat length range)
        min_len = self.detection_params['min_bat_length']
        max_len = self.detection_params['max_bat_length']
        
        if min_len <= length <= max_len:
            length_score = 1.0
        elif length < min_len:
            length_score = length / min_len
        else:
            length_score = max_len / length
        
        # Distance score (prefer lines close to wrist)
        max_distance = self.detection_params['wrist_region_radius']
        distance_score = 1.0 - (distance_to_wrist / max_distance)
        distance_score = max(0.0, distance_score)
        
        # Angle score (prefer horizontal-ish lines for cover drive)
        # Cover drive typically has bat at 0-45 degrees
        if 0 <= angle <= 45 or 135 <= angle <= 180:
            angle_score = 1.0
        elif 45 < angle < 135:
            angle_score = 0.5
        else:
            angle_score = 0.0
        
        # Combine scores
        total_score = (length_score * 0.4 + distance_score * 0.4 + angle_score * 0.2)
        
        return total_score
    
    def _create_bat_detection(self, line: Tuple[int, int, int, int], 
                            wrist_keypoint: KeyPoint, frame_idx: int) -> BatDetection:
        """Create a bat detection object from a line."""
        
        x1, y1, x2, y2 = line
        
        # Calculate bat properties
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        length = euclidean((x1, y1), (x2, y2))
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        
        # Calculate confidence based on line quality
        confidence = self._calculate_detection_confidence(line, wrist_keypoint)
        
        return BatDetection(
            center=center,
            angle=angle,
            length=length,
            confidence=confidence,
            frame_idx=frame_idx,
            wrist_position=(int(wrist_keypoint.x), int(wrist_keypoint.y))
        )
    
    def _calculate_detection_confidence(self, line: Tuple[int, int, int, int], 
                                      wrist_keypoint: KeyPoint) -> float:
        """Calculate confidence for bat detection."""
        
        x1, y1, x2, y2 = line
        wrist_pos = (int(wrist_keypoint.x), int(wrist_keypoint.y))
        
        # Base confidence from line scoring
        base_confidence = self._score_bat_line(
            euclidean((x1, y1), (x2, y2)),
            euclidean(((x1 + x2) // 2, (y1 + y2) // 2), wrist_pos),
            np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        )
        
        # Adjust confidence based on wrist confidence
        wrist_confidence = wrist_keypoint.confidence
        adjusted_confidence = base_confidence * wrist_confidence
        
        return min(1.0, adjusted_confidence)
    
    def _update_tracking(self, detection: BatDetection):
        """Update bat tracking state."""
        
        if detection is None:
            return
        
        # Add to previous detections
        self.previous_detections.append(detection)
        
        # Keep only recent detections
        max_history = 30
        if len(self.previous_detections) > max_history:
            self.previous_detections = self.previous_detections[-max_history:]
        
        # Update swing path tracking
        self._update_swing_path(detection)
    
    def _update_swing_path(self, detection: BatDetection):
        """Update swing path tracking."""
        
        if self.current_swing_path is None:
            # Start new swing path
            self.current_swing_path = SwingPath(
                start_point=detection.center,
                end_point=detection.center,
                path_points=[detection.center],
                straightness=1.0
            )
        else:
            # Continue current swing path
            self.current_swing_path.path_points.append(detection.center)
            self.current_swing_path.end_point = detection.center
            
            # Check if swing path should be finalized
            if self._should_finalize_swing_path(detection):
                self._finalize_swing_path()
    
    def _should_finalize_swing_path(self, detection: BatDetection) -> bool:
        """Determine if current swing path should be finalized."""
        
        if self.current_swing_path is None:
            return False
        
        # Finalize if path is long enough
        if len(self.current_swing_path.path_points) >= 10:
            return True
        
        # Finalize if bat angle changes significantly
        if len(self.previous_detections) >= 2:
            prev_angle = self.previous_detections[-2].angle
            angle_change = abs(detection.angle - prev_angle)
            if angle_change > 45:  # Significant angle change
                return True
        
        return False
    
    def _finalize_swing_path(self):
        """Finalize the current swing path."""
        
        if self.current_swing_path is None:
            return
        
        # Calculate swing path properties
        self.current_swing_path.straightness = self._calculate_path_straightness(
            self.current_swing_path.path_points
        )
        
        # Calculate angle at impact (if available)
        if len(self.current_swing_path.path_points) >= 3:
            impact_angle = self._calculate_impact_angle(self.current_swing_path.path_points)
            self.current_swing_path.angle_at_impact = impact_angle
        
        # Add to swing paths
        self.swing_paths.append(self.current_swing_path)
        self.current_swing_path = None
    
    def _calculate_path_straightness(self, path_points: List[Tuple[int, int]]) -> float:
        """Calculate how straight the swing path is."""
        
        if len(path_points) < 3:
            return 1.0
        
        # Calculate total path length
        total_length = 0.0
        for i in range(1, len(path_points)):
            total_length += euclidean(path_points[i], path_points[i-1])
        
        # Calculate straight-line distance
        straight_distance = euclidean(path_points[0], path_points[-1])
        
        # Straightness is ratio of straight distance to total path length
        if total_length > 0:
            straightness = straight_distance / total_length
        else:
            straightness = 1.0
        
        return min(1.0, straightness)
    
    def _calculate_impact_angle(self, path_points: List[Tuple[int, int]]) -> float:
        """Calculate the angle at impact (end of swing path)."""
        
        if len(path_points) < 3:
            return 0.0
        
        # Use last few points to calculate impact angle
        end_points = path_points[-3:]
        
        # Calculate average direction
        dx = end_points[-1][0] - end_points[0][0]
        dy = end_points[-1][1] - end_points[0][1]
        
        angle = np.arctan2(dy, dx) * 180 / np.pi
        return angle
    
    def get_swing_analysis(self) -> Dict[str, Any]:
        """Get analysis of detected swing paths."""
        
        if not self.swing_paths:
            return {
                'total_swings': 0,
                'avg_straightness': 0.0,
                'avg_impact_angle': 0.0,
                'swing_quality': 'No swings detected'
            }
        
        # Calculate statistics
        straightnesses = [path.straightness for path in self.swing_paths]
        impact_angles = [path.angle_at_impact for path in self.swing_paths if path.angle_at_impact is not None]
        
        avg_straightness = np.mean(straightnesses) if straightnesses else 0.0
        avg_impact_angle = np.mean(impact_angles) if impact_angles else 0.0
        
        # Assess swing quality
        if avg_straightness >= 0.8:
            swing_quality = "Excellent - Very straight swing path"
        elif avg_straightness >= 0.6:
            swing_quality = "Good - Relatively straight swing"
        elif avg_straightness >= 0.4:
            swing_quality = "Fair - Some swing path deviation"
        else:
            swing_quality = "Poor - Irregular swing path"
        
        return {
            'total_swings': len(self.swing_paths),
            'avg_straightness': avg_straightness,
            'avg_impact_angle': avg_impact_angle,
            'swing_quality': swing_quality,
            'straightness_range': {
                'min': min(straightnesses) if straightnesses else 0.0,
                'max': max(straightnesses) if straightnesses else 0.0
            }
        }
    
    def draw_bat_overlay(self, frame: np.ndarray, detection: Optional[BatDetection]) -> np.ndarray:
        """Draw bat detection overlay on frame."""
        
        if detection is None:
            return frame
        
        # Draw bat line
        center = detection.center
        length = detection.length
        angle_rad = detection.angle * np.pi / 180
        
        # Calculate line endpoints
        half_length = length / 2
        x1 = int(center[0] - half_length * np.cos(angle_rad))
        y1 = int(center[1] - half_length * np.sin(angle_rad))
        x2 = int(center[0] + half_length * np.cos(angle_rad))
        y2 = int(center[1] + half_length * np.sin(angle_rad))
        
        # Draw bat line
        color = (0, 255, 0) if detection.confidence > 0.7 else (0, 165, 255)
        thickness = 3 if detection.confidence > 0.7 else 2
        cv2.line(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Draw center point
        cv2.circle(frame, center, 5, (255, 0, 0), -1)
        
        # Draw confidence text
        confidence_text = f"Bat: {detection.confidence:.2f}"
        cv2.putText(frame, confidence_text, (center[0] + 10, center[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return frame
    
    def draw_swing_paths(self, frame: np.ndarray) -> np.ndarray:
        """Draw swing paths on frame."""
        
        # Draw completed swing paths
        for i, path in enumerate(self.swing_paths):
            if len(path.path_points) >= 2:
                # Draw path line
                color = (255, 255, 0)  # Yellow for swing paths
                thickness = 2
                
                for j in range(1, len(path.path_points)):
                    cv2.line(frame, path.path_points[j-1], path.path_points[j], 
                            color, thickness)
        
        # Draw current swing path
        if self.current_swing_path and len(self.current_swing_path.path_points) >= 2:
            color = (0, 255, 255)  # Cyan for current path
            thickness = 3
            
            for j in range(1, len(self.current_swing_path.path_points)):
                cv2.line(frame, self.current_swing_path.path_points[j-1], 
                        self.current_swing_path.path_points[j], color, thickness)
        
        return frame
