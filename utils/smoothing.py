"""
Temporal smoothing utilities for cricket cover drive analysis.
Applies smoothing filters to reduce noise in pose estimation.
"""

import numpy as np
from typing import List, Optional, Dict
from dataclasses import dataclass

try:
    from scipy import signal  # Savitzky-Golay if available
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

from .metrics import FrameMetrics


def _interp_and_fill(arr: np.ndarray) -> np.ndarray:
    """Linear interpolate NaNs, edge-fill ends, fallback to zeros."""
    if np.all(np.isnan(arr)):
        return np.zeros_like(arr)
    idx = np.arange(len(arr))
    m = ~np.isnan(arr)
    if m.sum() >= 2:
        arr = np.interp(idx, idx[m], arr[m])
    else:
        # One or zero valid: fill with that value or 0
        val = arr[m][0] if m.sum() == 1 else 0.0
        arr = np.full_like(arr, val)
    return arr


def smooth_metric(values: List[Optional[float]]) -> List[float]:
    if not values:
        return []
    arr = np.array([np.nan if v is None else float(v) for v in values], dtype=float)
    arr = _interp_and_fill(arr)
    if len(arr) >= 5:
        if _HAS_SCIPY:
            # small window; quadratic
            win = min(9, len(arr) // 2 * 2 + 1)  # odd
            try:
                arr = signal.savgol_filter(arr, win, 2)
            except Exception:
                pass
        else:
            # simple moving average fallback
            k = min(5, max(3, len(arr)//8*2+1))
            pad = k // 2
            cumsum = np.cumsum(np.pad(arr, (pad, pad), mode='edge'))
            arr = (cumsum[k:] - cumsum[:-k]) / k
            # match original length
            arr = np.pad(arr, (pad, len(values) - len(arr) - pad), mode='edge')
    return arr.tolist()


class TemporalSmoother:
    """Apply temporal smoothing to pose metrics."""
    def __init__(self, window_size: int = 5, polyorder: int = 2):
        self.window_size = window_size
        self.polyorder = polyorder
        self.metric_history: Dict[str, List[Optional[float]]] = {}
        
    def smooth_metrics(self, metrics: FrameMetrics) -> FrameMetrics:
        if metrics is None:
            return metrics

        names = [
            'front_elbow_angle', 'spine_lean_angle', 'head_over_front_knee',
            'front_foot_direction', 'back_elbow_angle', 'knee_separation',
            'shoulder_rotation', 'hip_rotation', 'front_elbow_velocity', 'spine_lean_velocity'
        ]

        out = FrameMetrics(frame_number=metrics.frame_number, timestamp=metrics.timestamp)

        for name in names:
            self.metric_history.setdefault(name, [])
            self.metric_history[name].append(getattr(metrics, name, None))
            # limit history size
            if len(self.metric_history[name]) > 120:
                self.metric_history[name] = self.metric_history[name][-120:]
            smoothed = smooth_metric(self.metric_history[name])
            setattr(out, name, smoothed[-1] if smoothed else getattr(metrics, name, None))

        # Provide a synthetic wrist_velocity proxy for downstream modules
        # (not part of FrameMetrics dataclass—attach dynamically)
        try:
            d_elb = np.abs(np.diff(np.array([v if v is not None else 0.0 for v in self.metric_history['front_elbow_angle']], dtype=float)))
            d_spn = np.abs(np.diff(np.array([v if v is not None else 0.0 for v in self.metric_history['spine_lean_angle']], dtype=float)))
            est = 0.7 * (d_elb[-1] if d_elb.size else 0.0) + 0.3 * (d_spn[-1] if d_spn.size else 0.0)
            setattr(out, 'wrist_velocity', float(est))
        except Exception:
            setattr(out, 'wrist_velocity', 0.0)

        # Confidence proxy so temporal modules don’t crash
        if not hasattr(out, 'confidence') or getattr(out, 'confidence', None) is None:
            setattr(out, 'confidence', 0.8)

        return out
    
    def reset_history(self):
        self.metric_history.clear()
