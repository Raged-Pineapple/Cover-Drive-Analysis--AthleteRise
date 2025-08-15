# metrics.py (robust, lenient)
import math
from typing import Dict, Optional
from .pose import KeyPoint

MIN_CONFIDENCE = 0.05  # be lenient

def _ok(ldm: Dict[str, KeyPoint], name: str) -> bool:
    kp = ldm.get(name)
    return kp is not None and kp.confidence >= MIN_CONFIDENCE

def _pt(ldm: Dict[str, KeyPoint], name: str) -> Optional[tuple]:
    if _ok(ldm, name):
        k = ldm[name]
        return (float(k.x), float(k.y))
    return None

def _angle(a, b, c) -> Optional[float]:
    """Return the angle at point b given three points."""
    if None in (a, b, c):
        return None
    ab = (a[0] - b[0], a[1] - b[1])
    cb = (c[0] - b[0], c[1] - b[1])
    nab = math.hypot(*ab)
    ncb = math.hypot(*cb)
    if nab == 0 or ncb == 0:
        return None
    cosv = max(-1.0, min(1.0, (ab[0] * cb[0] + ab[1] * cb[1]) / (nab * ncb)))
    return math.degrees(math.acos(cosv))

def front_elbow_angle(ldm: Dict[str, KeyPoint], handed: str) -> Optional[float]:
    """Calculate front elbow angle."""
    side = 'left' if handed == 'right' else 'right'
    return _angle(_pt(ldm, f"{side}_shoulder"), _pt(ldm, f"{side}_elbow"), _pt(ldm, f"{side}_wrist"))

def spine_lean_angle(ldm: Dict[str, KeyPoint]) -> Optional[float]:
    """Calculate spine lean angle relative to vertical."""
    ls, rs, lh, rh = _pt(ldm, "left_shoulder"), _pt(ldm, "right_shoulder"), _pt(ldm, "left_hip"), _pt(ldm, "right_hip")
    if None in (ls, rs, lh, rh):
        return None
    ms = ((ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0)
    mh = ((lh[0] + rh[0]) / 2.0, (lh[1] + rh[1]) / 2.0)
    # angle from vertical
    ang = math.degrees(math.atan2(mh[0] - ms[0], ms[1] - mh[1]))
    return abs(ang)

def head_over_front_knee(ldm: Dict[str, KeyPoint], handed: str) -> Optional[float]:
    """Return normalized horizontal distance from head to front knee."""
    side = 'left' if handed == 'right' else 'right'
    nose, knee = _pt(ldm, "nose"), _pt(ldm, f"{side}_knee")
    if None in (nose, knee):
        return None
    dy = max(1.0, abs(knee[1] - nose[1]))
    return abs(nose[0] - knee[0]) / dy

def front_foot_direction(ldm: Dict[str, KeyPoint], handed: str) -> Optional[float]:
    """Return angle of the front foot relative to horizontal axis."""
    side = 'left' if handed == 'right' else 'right'
    heel, toe = _pt(ldm, f"{side}_heel"), _pt(ldm, f"{side}_foot_index")
    if heel is None or toe is None:
        return None
    # angle relative to horizontal axis using a synthetic point to the right of toe
    return _angle(heel, toe, (toe[0] + 1.0, toe[1]))

def shoulder_rotation(ldm: Dict[str, KeyPoint]) -> Optional[float]:
    """Calculate shoulder rotation relative to horizontal."""
    ls, rs, rh = _pt(ldm, "left_shoulder"), _pt(ldm, "right_shoulder"), _pt(ldm, "right_hip")
    return _angle(ls, rs, rh)

def knee_separation(ldm: Dict[str, KeyPoint]) -> Optional[float]:
    """Calculate normalized distance between knees."""
    lk, rk = _pt(ldm, "left_knee"), _pt(ldm, "right_knee")
    if None in (lk, rk):
        return None
    dx = lk[0] - rk[0]
    dy = lk[1] - rk[1]
    return math.sqrt(dx * dx + dy * dy)

def get_metrics(ldm: Dict[str, KeyPoint], handed: str) -> Dict[str, Optional[float]]:
    """Calculate metrics from landmarks."""
    if not ldm:
        return {"front_elbow_angle": None, "spine_lean_angle": None, "head_over_front_knee": None,
                "front_foot_direction": None, "shoulder_rotation": None, "knee_separation": None}

    # Get relevant keypoints
    front_elbow = ldm.get("right_elbow" if handed == "right" else "left_elbow")
    front_wrist = ldm.get("right_wrist" if handed == "right" else "left_wrist")
    front_shoulder = ldm.get("right_shoulder" if handed == "right" else "left_shoulder")
    front_hip = ldm.get("right_hip" if handed == "right" else "left_hip")
    front_knee = ldm.get("right_knee" if handed == "right" else "left_knee")
    front_ankle = ldm.get("right_ankle" if handed == "right" else "left_ankle")
    nose = ldm.get("nose")
    opposite_knee = ldm.get("left_knee" if handed == "right" else "right_knee")
    opposite_ankle = ldm.get("left_ankle" if handed == "right" else "right_ankle")

    # Calculate metrics with lenient validation
    metrics = {
        "front_elbow_angle": float(front_elbow_angle(ldm, handed)) if front_shoulder and front_elbow and front_wrist else None,
        "spine_lean_angle": float(spine_lean_angle(ldm)) if front_hip and front_knee and front_ankle else None,
        "head_over_front_knee": float(head_over_front_knee(ldm, handed)) if nose and front_knee else None,
        "front_foot_direction": float(front_foot_direction(ldm, handed)) if front_ankle and front_knee else None,
        "shoulder_rotation": float(shoulder_rotation(ldm)) if front_shoulder and front_elbow else None,
        "knee_separation": float(knee_separation(ldm)) if front_knee and opposite_knee else None
    }

    # Convert any numpy values to Python floats
    for key, value in metrics.items():
        if isinstance(value, (np.float32, np.float64)):
            metrics[key] = float(value)

    return metrics

def is_frame_valid(metrics_dict: dict) -> bool:
    """A frame is valid if ANY metric is present (not None and not NaN)."""
    for v in metrics_dict.values():
        if v is None:
            continue
        if isinstance(v, float) and (not np.isnan(v)):
            return True
    return False

def interpolate_series(arr: List[Optional[float]]) -> List[float]:
    """Convert [None, 12.3, None, 15.0] -> interpolate linearly."""
    a = np.array([np.nan if x is None else float(x) for x in arr], dtype=float)
    if np.all(np.isnan(a)):
        return a.tolist()
    idx = np.arange(len(a))
    good = ~np.isnan(a)
    a[~good] = np.interp(idx[~good], idx[good], a[good])
    return a.tolist()

@dataclass
class FrameMetrics:
    """Container for all metrics computed in a single frame."""
    front_elbow_angle: Optional[float] = None
    spine_lean_angle: Optional[float] = None
    head_over_front_knee: Optional[float] = None
    front_foot_direction: Optional[float] = None
    shoulder_rotation: Optional[float] = None
    knee_separation: Optional[float] = None
    frame_number: int = 0
    timestamp: float = 0.0

class MetricsCalculator:
    """Calculate biomechanical metrics from pose landmarks."""
    
    def __init__(self, handedness: str = "right"):
        self.handedness = handedness
        self.frame_count = 0
        
    def calculate_frame_metrics(self, landmarks: Dict[str, KeyPoint], frame_shape: tuple) -> FrameMetrics:
        """Calculate all metrics for a single frame."""
        metrics = get_metrics(landmarks, self.handedness)
        
        frame_metrics = FrameMetrics(
            front_elbow_angle=metrics.get("front_elbow_angle"),
            spine_lean_angle=metrics.get("spine_lean_angle"),
            head_over_front_knee=metrics.get("head_over_front_knee"),
            front_foot_direction=metrics.get("front_foot_direction"),
            shoulder_rotation=metrics.get("shoulder_rotation"),
            knee_separation=metrics.get("knee_separation"),
            frame_number=self.frame_count
        )
        
        self.frame_count += 1
        return frame_metrics
    
    def get_empty_metrics(self) -> FrameMetrics:
        """Return empty metrics for frames with no pose detection."""
        return FrameMetrics(frame_number=self.frame_count)

def calculate_velocity(prev: KeyPoint, curr: KeyPoint) -> float:
    """Calculate velocity between two keypoints."""
    return math.sqrt((curr.x - prev.x) ** 2 + (curr.y - prev.y) ** 2)

def calculate_movement_smoothness(landmarks_list: List[Dict[str, KeyPoint]], min_frames: int = 10) -> Dict[str, Any]:
    """Calculate movement smoothness metrics.
    
    Args:
        landmarks_list: List of pose landmarks over time
        min_frames: Minimum number of frames required for analysis
        
    Returns:
        Dict containing smoothness score and feedback
    """
    # If insufficient frames, return N/A
    if len(landmarks_list) < min_frames:
        return {
            "smoothness_score": None,
            "feedback": "N/A due to limited data – try a clearer or longer video."
        }
    
    try:
        # Calculate velocity changes for key joints
        velocities = []
        for i in range(1, len(landmarks_list)):
            prev = landmarks_list[i-1]
            curr = landmarks_list[i]
            
            # Calculate velocity for key joints (shoulder, elbow, wrist)
            shoulder = calculate_velocity(prev['LEFT_SHOULDER'], curr['LEFT_SHOULDER'])
            elbow = calculate_velocity(prev['LEFT_ELBOW'], curr['LEFT_ELBOW'])
            wrist = calculate_velocity(prev['LEFT_WRIST'], curr['LEFT_WRIST'])
            
            velocities.append([shoulder, elbow, wrist])
        
        # Calculate standard deviation of velocity changes
        std_devs = np.std(velocities, axis=0)
        
        # Calculate smoothness score (lower is better)
        smoothness_score = np.mean(std_devs)
        
        # Generate feedback based on score
        if smoothness_score < 0.1:
            feedback = "Movement is very smooth and controlled."
        elif smoothness_score < 0.2:
            feedback = "Movement is generally smooth with some minor variations."
        else:
            feedback = "Movement shows significant variations. Focus on maintaining consistent motion."
        
        return {
            "smoothness_score": float(smoothness_score),
            "feedback": feedback
        }
    except Exception as e:
        logger.warning(f"Error calculating smoothness: {e}")
        return {
            "smoothness_score": None,
            "feedback": "N/A due to calculation error"
        }
