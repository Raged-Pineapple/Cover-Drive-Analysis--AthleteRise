# analysis.py
import os
import cv2
import json
import numpy as np
import yaml
from typing import Dict, Any, List, Optional
from utils.logging_setup import quiet_logs
quiet_logs()

import mediapipe as mp
from metrics import get_metrics, is_frame_valid, interpolate_series
from utils.report import build_report

# MediaPipe pose landmark indices to names mapping
IDX2NAME = {
    0: "nose",
    1: "left_eye_inner", 2: "left_eye", 3: "left_eye_outer",
    4: "right_eye_inner", 5: "right_eye", 6: "right_eye_outer",
    7: "left_ear", 8: "right_ear",
    9: "mouth_left", 10: "mouth_right",
    11: "left_shoulder", 12: "right_shoulder",
    13: "left_elbow", 14: "right_elbow",
    15: "left_wrist", 16: "right_wrist",
    17: "left_pinky", 18: "right_pinky",
    19: "left_index", 20: "right_index",
    21: "left_thumb", 22: "right_thumb",
    23: "left_hip", 24: "right_hip",
    25: "left_knee", 26: "right_knee",
    27: "left_ankle", 28: "right_ankle",
    29: "left_heel", 30: "right_heel",
    31: "left_foot_index", 32: "right_foot_index"
}

def load_yaml(path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_thresholds(path: str) -> Dict[str, Any]:
    """Load thresholds configuration."""
    return load_yaml(path)

def load_ideal(path: str) -> Dict[str, Any]:
    """Load ideal values configuration."""
    return load_yaml(path)

def analyze_video(input_path: str, output_dir: str = "output", handedness: str = "right") -> Dict[str, Any]:
    """Analyze cricket cover drive video with robust validation."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load configurations
    try:
        th = load_thresholds("configs/thresholds.yml")
        ideal = load_ideal("configs/ideal.yml")
    except FileNotFoundError:
        print("Warning: Configuration files not found, using defaults")
        th = {
            "front_elbow_angle": {"good": [120, 160], "warn": [100, 180]},
            "spine_lean_angle": {"good": [0, 15], "warn": [0, 25]},
            "head_over_front_knee": {"good": [0.00, 0.15], "warn": [0.00, 0.30]},
            "front_foot_direction": {"good": [0, 20], "warn": [0, 35]},
            "shoulder_rotation": {"good": [15, 45], "warn": [10, 60]},
            "knee_separation": {"good": [0.3, 0.6], "warn": [0.2, 0.7]}
        }
        ideal = {
            "front_elbow_angle": {"target": 135, "tol": 15, "weight": 0.3},
            "spine_lean_angle": {"target": 10, "tol": 5, "weight": 0.2},
            "head_over_front_knee": {"target": 0.12, "tol": 0.05, "weight": 0.2},
            "front_foot_direction": {"target": 12, "tol": 10, "weight": 0.3},
            "shoulder_rotation": {"target": 30, "tol": 15, "weight": 0.2},
            "knee_separation": {"target": 0.45, "tol": 0.15, "weight": 0.1}
        }

    # Open video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_path}")
    
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Processing video: {w}x{h} @ {fps:.1f}fps, {n_frames} frames")

    # Setup video writer for annotated output
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = os.path.join(output_dir, "annotated_video.mp4")
    writer = cv2.VideoWriter(out_path, fourcc, max(10.0, min(30.0, fps)), (w, h))

    # Initialize MediaPipe with lenient settings
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        model_complexity=1,
        enable_segmentation=False,
        smooth_landmarks=True,
        min_detection_confidence=0.15,
        min_tracking_confidence=0.10
    )

    all_metrics: List[Dict[str, Optional[float]]] = []
    valid_count = 0

    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        # Extract landmarks
        ldm = {}
        if res.pose_landmarks:
            for idx, lm in enumerate(res.pose_landmarks.landmark):
                name = IDX2NAME.get(idx)
                if name:
                    ldm[name] = type('KeyPoint', (), {
                        'x': lm.x * w,
                        'y': lm.y * h,
                        'confidence': lm.visibility
                    })()

        # Calculate metrics
        m = get_metrics(ldm, handedness)
        all_metrics.append(m)
        
        # Check if frame is valid (any metric present)
        if is_frame_valid(m):
            valid_count += 1

        # Draw progress HUD
        cv2.putText(frame, f"Frame {i+1}/{n_frames}", (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        # Write frame to output video
        writer.write(frame)
        i += 1

    # Clean up resources
    cap.release()
    writer.release()
    pose.close()
    
    print(f"Valid frames: {valid_count}/{len(all_metrics)}")

    # Prepare timeseries data (including interpolation)
    keys = ["front_elbow_angle", "spine_lean_angle", "head_over_front_knee", "front_foot_direction", "shoulder_rotation", "knee_separation"]
    series = {k: [d.get(k) for d in all_metrics] for k in keys}
    series_interp = {k: interpolate_series(v) for k, v in series.items()}

    # Calculate statistics for each metric
    def safe_stats(vals: List[Optional[float]]) -> Dict[str, float]:
        """Calculate statistics safely handling None values."""
        a = np.array([np.nan if v is None else v for v in vals], dtype=float)
        good = a[~np.isnan(a)]
        if good.size == 0: 
            return {"median": None, "p10": None, "p90": None, "pct_present": 0.0}
        return {
            "median": float(np.median(good)),
            "p10": float(np.percentile(good, 10)),
            "p90": float(np.percentile(good, 90)),
            "pct_present": float(good.size/len(a))
        }

    stats = {k: safe_stats(v) for k, v in series.items()}

    # Score each metric based on thresholds
    def score_metric(name: str, val: Optional[float]) -> int:
        """Score a metric based on thresholds."""
        if val is None: return None
        g0, g1 = th[name]["good"]
        w0, w1 = th[name]["warn"]
        v = abs(val)
        if g0 <= v <= g1: return 9  # Excellent
        if w0 <= v <= w1: return 7  # Good
        return 4  # Needs improvement

    # Calculate category scores
    cat = {
        "Footwork": score_metric("front_foot_direction", stats["front_foot_direction"]["median"]),
        "Head Position": score_metric("head_over_front_knee", stats["head_over_front_knee"]["median"]),
        "Swing Control": score_metric("front_elbow_angle", stats["front_elbow_angle"]["median"]),
        "Balance": score_metric("spine_lean_angle", stats["spine_lean_angle"]["median"]),
        "Follow-through": score_metric("shoulder_rotation", stats["shoulder_rotation"]["median"]),
        "Stance": score_metric("knee_separation", stats["knee_separation"]["median"])  # Added stance score
    }

    # Generate feedback text
    def generate_feedback(name: str, val: Optional[float]) -> str:
        """Generate feedback text for a metric."""
        if val is None: 
            return "N/A due to sparse detections."
        g0, g1 = th[name]["good"]
        w0, w1 = th[name]["warn"]
        v = abs(val)
        if g0 <= v <= g1: 
            return "Within target range."
        if w0 <= v <= w1: 
            return "Slightly outside ideal; improve consistency."
        return "Off target; prioritize technique drills."

    feedback = {
        "Footwork": generate_feedback("front_foot_direction", stats["front_foot_direction"]["median"]),
        "Head Position": generate_feedback("head_over_front_knee", stats["head_over_front_knee"]["median"]),
        "Swing Control": generate_feedback("front_elbow_angle", stats["front_elbow_angle"]["median"]),
        "Balance": generate_feedback("spine_lean_angle", stats["spine_lean_angle"]["median"]),
        "Follow-through": generate_feedback("shoulder_rotation", stats["shoulder_rotation"]["median"]),
        "Stance": generate_feedback("knee_separation", stats["knee_separation"]["median"])  # Added stance feedback
    }

    # Build evaluation JSON
    summary = {k: {"score": (None if v is None else int(v)), "feedback": feedback[k]} for k,v in cat.items()}
    evaluation = {
        "summary": summary,
        "global": {
            "frames_total": len(all_metrics),
            "frames_valid": valid_count,
            "valid_ratio": float(valid_count/max(1,len(all_metrics))),
            "notes": "Lenient validation & interpolation enabled for low-quality videos."
        },
        "stats": stats
    }

    # Save evaluation JSON
    json_path = os.path.join(output_dir, "evaluation.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(evaluation, f, indent=2)

    # Generate HTML report with charts
    build_report(output_dir, evaluation, series_interp)
    
    print(f"Analysis complete! Output saved to: {output_dir}")
    print(f"Files generated:")
    print(f"  - {out_path}")
    print(f"  - {os.path.join(output_dir, 'evaluation.json')}")
    print(f"  - {os.path.join(output_dir, 'report.html')}")
    print(f"  - Chart PNGs")
    
    return evaluation

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python analysis.py <video_path> [output_dir] [handedness]")
        sys.exit(1)
    
    video_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output"
    handedness = sys.argv[3] if len(sys.argv) > 3 else "right"
    
    try:
        result = analyze_video(video_path, output_dir, handedness)
        print(f"\n✅ Analysis successful!")
        print(f"Overall Score: {result['overall_score']:.1f}/10")
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
