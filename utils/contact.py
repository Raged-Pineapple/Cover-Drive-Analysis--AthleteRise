"""
Contact moment detection for cricket cover drive analysis.
Identifies bat-ball contact moments using wrist velocity peaks and motion heuristics.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

try:
    from scipy.signal import find_peaks
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

from .metrics import FrameMetrics


@dataclass
class ContactMoment:
    frame_idx: int
    confidence: float
    wrist_velocity: float
    elbow_angle: float
    spine_lean: float
    metrics: Dict[str, Any]


class ContactDetector:
    """Detects bat-ball contact moments from metrics and pose data."""
    
    def __init__(self):
        self.thresholds = {
            'wrist_velocity_peak_threshold': 0.6,
            'peak_prominence': 0.25,
            'peak_distance': 8,
            'elbow_angle_range': (80, 165),
            'spine_lean_max': 30.0,
            'confidence_threshold': 0.45,
            'contact_window': 5
        }
        self.detected_contacts: List[ContactMoment] = []
        
    def detect_contacts(self, metrics_history: List[FrameMetrics], fps: float = 30.0) -> List[ContactMoment]:
        n = len(metrics_history)
        if n < 8:
            return []

        self.detected_contacts = []
        wrist_velocities = self._calculate_wrist_velocities(metrics_history)

        # Peaks
        peaks = self._find_velocity_peaks(wrist_velocities)

        # If no peaks, pick the global max as a low-confidence fallback
        if not peaks and np.any(np.isfinite(wrist_velocities)):
            peaks = [int(np.nanargmax(wrist_velocities))]

        for idx in peaks:
            cm = self._analyze_contact_candidate(idx, metrics_history, wrist_velocities)
            if cm:
                self.detected_contacts.append(cm)

        # Filter: keep top unique by distance
        self.detected_contacts.sort(key=lambda x: x.confidence, reverse=True)
        filtered: List[ContactMoment] = []
        min_dist = max(6, self.thresholds['peak_distance'])
        for c in self.detected_contacts:
            if all(abs(c.frame_idx - f.frame_idx) >= min_dist for f in filtered):
                filtered.append(c)
        filtered.sort(key=lambda x: x.frame_idx)
        self.detected_contacts = filtered
        return self.detected_contacts
    
    # -------- Internals -------- #

    def _calculate_wrist_velocities(self, metrics_history: List[FrameMetrics]) -> np.ndarray:
        # Estimate from changes of elbow + spine (proxy)
        elbow = self._series(metrics_history, 'front_elbow_angle')
        spine = self._series(metrics_history, 'spine_lean_angle')
        d_elb = self._delta(elbow)
        d_spn = self._delta(spine)
        vel = 0.7 * d_elb + 0.3 * d_spn
        maxv = vel.max() if vel.size and np.isfinite(vel.max()) and vel.max() > 0 else 1.0
        return vel / maxv

    def _find_velocity_peaks(self, vel: np.ndarray) -> List[int]:
        if len(vel) < 5:
            return []
        if _HAS_SCIPY:
            peaks, _props = find_peaks(
                vel, height=self.thresholds['wrist_velocity_peak_threshold'],
                distance=self.thresholds['peak_distance']
            )
            return peaks.tolist()
        # Numpy fallback: simple local maxima with threshold
        peaks = []
        thr = self.thresholds['wrist_velocity_peak_threshold']
        for i in range(1, len(vel) - 1):
            if vel[i] >= thr and vel[i] > vel[i - 1] and vel[i] >= vel[i + 1]:
                peaks.append(i)
        # thin by distance
        out = []
        for p in peaks:
            if not out or p - out[-1] >= self.thresholds['peak_distance']:
                out.append(p)
        return out

    def _analyze_contact_candidate(self, idx: int,
                                   metrics_history: List[FrameMetrics],
                                   wrist_velocities: np.ndarray) -> Optional[ContactMoment]:
        if idx < 0 or idx >= len(metrics_history):
            return None
        m = metrics_history[idx]
        if m is None:
            return None

        wrist_velocity = float(wrist_velocities[idx])
        elbow_angle = float(getattr(m, 'front_elbow_angle', 0.0) or 0.0)
        spine_lean = float(getattr(m, 'spine_lean_angle', 0.0) or 0.0)

        # Factor scores
        vel_conf = min(1.0, wrist_velocity / max(1e-6, self.thresholds['wrist_velocity_peak_threshold']))
        lo, hi = self.thresholds['elbow_angle_range']
        if lo <= elbow_angle <= hi:
            elbow_conf = 1.0
        else:
            dist = (lo - elbow_angle) if elbow_angle < lo else (elbow_angle - hi)
            elbow_conf = max(0.0, 1.0 - (dist / 35.0))

        if abs(spine_lean) <= self.thresholds['spine_lean_max']:
            spine_conf = 1.0
        else:
            spine_conf = max(0.0, 1.0 - (abs(spine_lean) - self.thresholds['spine_lean_max']) / 25.0)

        # Temporal smoothness proxy: variance of last few deltas
        w = self.thresholds['contact_window']
        s = max(0, idx - w)
        e = min(len(metrics_history) - 1, idx + w)
        elbow_seg = self._delta(self._series(metrics_history[s:e+1], 'front_elbow_angle'))
        var = float(np.var(elbow_seg)) if elbow_seg.size else 0.0
        temporal_conf = float(np.clip(1.0 - var / 50.0, 0.0, 1.0))

        overall = float(np.mean([vel_conf, elbow_conf, spine_conf, temporal_conf]))

        return ContactMoment(
            frame_idx=idx,
            confidence=overall,
            wrist_velocity=wrist_velocity,
            elbow_angle=elbow_angle,
            spine_lean=spine_lean,
            metrics={
                'velocity_conf': vel_conf,
                'elbow_conf': elbow_conf,
                'spine_conf': spine_conf,
                'temporal_conf': temporal_conf
            }
        )

    def _series(self, metrics_history: List[FrameMetrics], name: str) -> np.ndarray:
        vals = []
        for m in metrics_history:
            v = getattr(m, name, None)
            vals.append(np.nan if v is None else float(v))
        arr = np.array(vals, dtype=float)
        if np.all(np.isnan(arr)):
            return np.zeros_like(arr)
        idx = np.arange(len(arr))
        mask = ~np.isnan(arr)
        if mask.sum() >= 2:
            arr = np.interp(idx, idx[mask], arr[mask])
        else:
            arr = np.nan_to_num(arr, nan=0.0)
        return arr

    def _delta(self, arr: np.ndarray) -> np.ndarray:
        d = np.zeros_like(arr)
        if len(arr) > 1:
            d[1:] = np.abs(np.diff(arr))
        return d

    # ---- Summaries / helpers ---- #

    def get_contact_summary(self, contacts: List[ContactMoment]) -> Dict[str, Any]:
        if not contacts:
            return {
                'total_contacts': 0,
                'avg_confidence': 0.0,
                'contact_timing': [],
                'contact_metrics': {}
            }
        return {
            'total_contacts': len(contacts),
            'avg_confidence': float(np.mean([c.confidence for c in contacts])),
            'contact_timing': [c.frame_idx for c in contacts],
            'contact_metrics': {
                'avg_wrist_velocity': float(np.mean([c.wrist_velocity for c in contacts])),
                'avg_elbow_angle': float(np.mean([c.elbow_angle for c in contacts])),
                'avg_spine_lean': float(np.mean([c.spine_lean for c in contacts])),
                'confidence_range': {
                    'min': float(min([c.confidence for c in contacts])),
                    'max': float(max([c.confidence for c in contacts]))
                }
            }
        }

    def get_contact_at_frame(self, frame_idx: int, contacts: List[ContactMoment]) -> Optional[ContactMoment]:
        for c in contacts:
            if abs(c.frame_idx - frame_idx) <= self.thresholds['contact_window']:
                return c
        return None

    def validate_contact_sequence(self, contacts: List[ContactMoment], total_frames: int) -> Dict[str, Any]:
        if not contacts:
            return {'valid': False, 'reason': 'No contacts detected', 'suggestions': ['Improve lighting/angle']}

        # Simple sanity checks
        if len(contacts) > max(1, total_frames // 10):
            return {'valid': False, 'reason': 'Too many contacts', 'suggestions': ['Increase peak distance/threshold']}
        return {'valid': True, 'reason': 'OK', 'suggestions': []}
