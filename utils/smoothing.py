"""
Temporal smoothing utilities for cricket cover drive analysis.
Applies smoothing filters to reduce noise in pose estimation and
derives a 0–1 smoothness score from velocity stability.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional, Dict, Iterable
from dataclasses import dataclass

try:
    # Optional: used for Savitzky–Golay smoothing if available
    from scipy import signal  # type: ignore
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

from .metrics import FrameMetrics


# ---------- helpers ----------

def _interp_and_fill(arr: np.ndarray) -> np.ndarray:
    """Linear-interpolate NaNs and edge-fill ends. All-NaN -> zeros."""
    if np.all(np.isnan(arr)):
        return np.zeros_like(arr)
    idx = np.arange(len(arr))
    mask = ~np.isnan(arr)
    if mask.sum() >= 2:
        arr = np.interp(idx, idx[mask], arr[mask])
    else:
        # One or zero valid points -> constant fill with that value (or 0)
        val = arr[mask][0] if mask.sum() == 1 else 0.0
        arr = np.full_like(arr, val, dtype=float)
    return arr


def smooth_metric(values: List[Optional[float]]) -> List[float]:
    """Smooth a 1D metric series with optional SciPy S-G filter or MA fallback."""
    if not values:
        return []
    arr = np.array([np.nan if v is None else float(v) for v in values], dtype=float)
    arr = _interp_and_fill(arr)

    if len(arr) >= 5:
        if _HAS_SCIPY:
            # Odd window, quadratic polynomial
            win = min(9, (len(arr) // 2) * 2 + 1)
            try:
                arr = signal.savgol_filter(arr, win, 2)
            except Exception:
                # If SciPy fails (rare), fall back to moving average
                k = min(5, max(3, (len(arr) // 8) * 2 + 1))
                pad = k // 2
                csum = np.cumsum(np.pad(arr, (pad, pad), mode="edge"))
                arr = (csum[k:] - csum[:-k]) / k
                arr = np.pad(arr, (pad, len(values) - len(arr) - pad), mode="edge")
        else:
            k = min(5, max(3, (len(arr) // 8) * 2 + 1))
            pad = k // 2
            csum = np.cumsum(np.pad(arr, (pad, pad), mode="edge"))
            arr = (csum[k:] - csum[:-k]) / k
            arr = np.pad(arr, (pad, len(values) - len(arr) - pad), mode="edge")

    return arr.tolist()


def _safe_series(frames: Iterable[FrameMetrics], attr: str) -> np.ndarray:
    """Extract a numeric series for an attribute across frames (NaNs for missing)."""
    vals: List[float] = []
    for m in frames:
        v = getattr(m, attr, None)
        vals.append(np.nan if v is None else float(v))
    return np.array(vals, dtype=float)


def _vel_from_angle(frames: List[FrameMetrics], angle_attr: str) -> np.ndarray:
    """Compute velocity from an angle series using timestamps if available."""
    angles = _safe_series(frames, angle_attr)
    if angles.size < 2:
        return np.array([], dtype=float)

    # Timestamps (may be None)
    ts = np.array([getattr(m, "timestamp", np.nan) for m in frames], dtype=float)

    # If timestamps are mostly valid and monotonic, use them; else assume unit dt
    valid_ts = np.isfinite(ts)
    use_ts = valid_ts.sum() >= max(2, int(0.7 * ts.size)) and np.all(np.diff(ts[valid_ts]) > 0)

    if use_ts:
        dt = np.diff(ts)
        dt[dt <= 0] = np.nan
        vel = np.diff(angles) / dt
    else:
        vel = np.diff(angles)  # unit time step

    # Replace inf/nan with interpolated values then edge-fill
    vel = np.array([np.nan if not np.isfinite(x) else x for x in vel], dtype=float)
    vel = _interp_and_fill(vel)
    return vel


# ---------- main class ----------

class TemporalSmoother:
    """Apply temporal smoothing to per-frame pose metrics and score smoothness."""

    def __init__(self, window_size: int = 5, polyorder: int = 2):
        self.window_size = window_size
        self.polyorder = polyorder
        self.metric_history: Dict[str, List[Optional[float]]] = {}

    def smooth_metrics(self, metrics: FrameMetrics) -> FrameMetrics:
        """Update running history and return a smoothed FrameMetrics snapshot."""
        if metrics is None:
            return metrics

        names = [
            "front_elbow_angle",
            "spine_lean_angle",
            "head_over_front_knee",
            "front_foot_direction",
            "back_elbow_angle",
            "knee_separation",
            "shoulder_rotation",
            "hip_rotation",
            "front_elbow_velocity",
            "spine_lean_velocity",
        ]

        out = FrameMetrics(frame_number=metrics.frame_number, timestamp=metrics.timestamp)

        for name in names:
            self.metric_history.setdefault(name, [])
            self.metric_history[name].append(getattr(metrics, name, None))
            # limit memory
            if len(self.metric_history[name]) > 120:
                self.metric_history[name] = self.metric_history[name][-120:]
            smoothed = smooth_metric(self.metric_history[name])
            setattr(out, name, smoothed[-1] if smoothed else getattr(metrics, name, None))

        # Provide a synthetic wrist_velocity proxy for downstream modules
        try:
            elb = np.array(
                [0.0 if v is None else float(v) for v in self.metric_history.get("front_elbow_angle", [])],
                dtype=float,
            )
            spn = np.array(
                [0.0 if v is None else float(v) for v in self.metric_history.get("spine_lean_angle", [])],
                dtype=float,
            )
            d_elb = np.abs(np.diff(elb)) if elb.size >= 2 else np.array([0.0])
            d_spn = np.abs(np.diff(spn)) if spn.size >= 2 else np.array([0.0])
            est = 0.7 * (d_elb[-1] if d_elb.size else 0.0) + 0.3 * (d_spn[-1] if d_spn.size else 0.0)
            setattr(out, "wrist_velocity", float(est))
        except Exception:
            setattr(out, "wrist_velocity", 0.0)

        # Confidence proxy so temporal modules don’t crash
        if not hasattr(out, "confidence") or getattr(out, "confidence", None) is None:
            setattr(out, "confidence", 0.8)

        return out

    def smooth_sequence(self, frames: List[FrameMetrics]) -> List[FrameMetrics]:
        """Smooth a whole sequence (convenience)."""
        smoothed: List[FrameMetrics] = []
        for m in frames:
            smoothed.append(self.smooth_metrics(m))
        return smoothed

    def compute_smoothness(self, metrics_history: List[FrameMetrics]) -> float:
        """
        Compute a global smoothness score in [0, 1].
        Higher = smoother (lower variance / jerk of motion).
        Uses available velocity signals with robust fallbacks:
          1) wrist_velocity (if present)
          2) front_elbow_velocity / spine_lean_velocity
          3) finite-difference of front_elbow_angle / spine_lean_angle (+ timestamps if available)
        """
        if not metrics_history:
            return 0.0

        # 1) Preferred: wrist_velocity
        v_wrist = _safe_series(metrics_history, "wrist_velocity")
        v_wrist = v_wrist[np.isfinite(v_wrist)]
        candidates: List[np.ndarray] = []
        if v_wrist.size >= 3 and np.nanstd(v_wrist) > 0:
            candidates.append(v_wrist)

        # 2) Direct velocities if present
        for attr in ("front_elbow_velocity", "spine_lean_velocity"):
            v = _safe_series(metrics_history, attr)
            v = v[np.isfinite(v)]
            if v.size >= 3 and np.nanstd(v) > 0:
                candidates.append(v)

        # 3) Derive velocities from angles if needed
        if not candidates:
            for ang in ("front_elbow_angle", "spine_lean_angle"):
                v = _vel_from_angle(metrics_history, ang)
                if v.size >= 3 and np.nanstd(v) > 0:
                    candidates.append(v)

        if not candidates:
            return 0.0

        # Build a composite velocity signal by averaging normalized candidates
        normed: List[np.ndarray] = []
        for v in candidates:
            # robust scaling by median absolute deviation (avoid division by ~0)
            med = np.median(v)
            mad = np.median(np.abs(v - med)) or (np.std(v) or 1.0)
            normed.append((v - med) / mad)

        min_len = min(len(v) for v in normed)
        V = np.vstack([v[-min_len:] for v in normed]).mean(axis=0)

        # Variance (stability) + jerk (change in velocity) penalties
        var = float(np.var(V))
        jerk = np.diff(V)
        jerk_var = float(np.var(jerk)) if jerk.size else 0.0

        # Combine; weights can be tuned
        stability = 0.7 * var + 0.3 * jerk_var

        # Map to [0, 1]; lower stability penalty -> higher smoothness
        smoothness = 1.0 / (1.0 + stability)

        # Clamp for safety
        return float(max(0.0, min(1.0, smoothness)))

    def reset_history(self) -> None:
        self.metric_history.clear()


# Backwards-compat alias so `from utils.smoothing import Smoother` works
class Smoother(TemporalSmoother):
    """Alias for backward compatibility."""
    pass


__all__ = [
    "TemporalSmoother",
    "Smoother",
    "smooth_metric",
]
