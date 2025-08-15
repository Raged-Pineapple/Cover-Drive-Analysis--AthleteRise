"""
Phase segmentation for cricket cover drive analysis.
Detects batting phases: stance, stride, downswing, impact, follow_through, recovery.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from .metrics import FrameMetrics


class BattingPhase(Enum):
    """Enumeration of batting phases."""
    STANCE = "stance"
    STRIDE = "stride"
    DOWNSWING = "downswing"
    IMPACT = "impact"
    FOLLOW_THROUGH = "follow_through"
    RECOVERY = "recovery"


@dataclass
class PhaseSegment:
    """Represents a phase segment with start/end frames and confidence."""
    phase: BattingPhase
    start_frame: int
    end_frame: int
    confidence: float
    metrics: Dict[str, Any]


class PhaseDetector:
    """Detects batting phases from per-frame metrics."""
    
    def __init__(self):
        # Phase detection thresholds (tolerant defaults)
        self.thresholds = {
            'wrist_speed_peak': 0.6,       # Normalized peak threshold (slightly lower)
            'elbow_angle_change': 10.0,    # Degrees (more tolerant)
            'spine_lean_threshold': 12.0,  # Degrees
            'phase_min_duration': 4,       # Minimum frames per phase
        }
        
        # State tracking
        self.current_phase = BattingPhase.STANCE
        self.phase_start_frame = 0

    # -------- Public API -------- #

    def detect_phases(self, metrics_history: List[FrameMetrics]) -> List[PhaseSegment]:
        """Detect phases from metrics history."""
        n = len(metrics_history)
        if n < 8:
            # Return a single STANCE phase to keep pipeline “available”
            return [PhaseSegment(
                phase=BattingPhase.STANCE,
                start_frame=0,
                end_frame=max(0, n - 1),
                confidence=0.5 if n > 0 else 0.0,
                metrics=self._extract_phase_metrics(0, max(0, n - 1), metrics_history) if n > 0 else {}
            )]

        # Derived signals
        wrist_speeds = self._calculate_wrist_speeds(metrics_history)
        elbow_changes = self._delta(self._series(metrics_history, 'front_elbow_angle'))
        spine_changes = self._delta(self._series(metrics_history, 'spine_lean_angle'))

        phases: List[PhaseSegment] = []
        self.current_phase = BattingPhase.STANCE
        self.phase_start_frame = 0

        for i in range(n):
            next_phase = self._classify_frame(
                i, n, wrist_speeds[i], elbow_changes[i], spine_changes[i]
            )
            if next_phase != self.current_phase:
                # close previous
                if i - 1 >= self.phase_start_frame:
                    seg = self._make_segment(self.current_phase, self.phase_start_frame, i - 1, metrics_history)
                    if seg:
                        phases.append(seg)
                # start new
                self.current_phase = next_phase
                self.phase_start_frame = i

        # close final phase
        if n - 1 >= self.phase_start_frame:
            seg = self._make_segment(self.current_phase, self.phase_start_frame, n - 1, metrics_history)
            if seg:
                phases.append(seg)

        # Merge tiny phases and ensure minimum durations
        phases = self._post_process_phases(phases)

        # Guarantee at least one phase
        if not phases:
            phases = [PhaseSegment(
                phase=BattingPhase.STANCE,
                start_frame=0,
                end_frame=n - 1,
                confidence=0.5,
                metrics=self._extract_phase_metrics(0, n - 1, metrics_history)
            )]
        return phases

    def get_phase_at_frame(self, frame_idx: int, phases: List[PhaseSegment]) -> Optional[BattingPhase]:
        for p in phases:
            if p.start_frame <= frame_idx <= p.end_frame:
                return p.phase
        return None

    def get_phase_summary(self, phases: List[PhaseSegment]) -> Dict[str, Any]:
        if not phases:
            return {}
        total_frames = sum(p.end_frame - p.start_frame + 1 for p in phases)
        counts: Dict[str, int] = {}
        totals: Dict[str, int] = {}
        confs: Dict[str, List[float]] = {}
        for p in phases:
            name = p.phase.value
            dur = p.end_frame - p.start_frame + 1
            counts[name] = counts.get(name, 0) + 1
            totals[name] = totals.get(name, 0) + dur
            confs.setdefault(name, []).append(p.confidence)
        dominant = max(totals, key=totals.get)
        return {
            'total_phases': len(phases),
            'total_frames': total_frames,
            'phase_durations': {
                k: {'count': counts[k], 'avg_duration': totals[k] / counts[k], 'total_duration': totals[k]}
                for k in counts
            },
            'phase_confidences': {k: float(np.mean(confs[k])) for k in confs},
            'dominant_phase': dominant
        }

    # -------- Internals -------- #

    def _series(self, metrics_history: List[FrameMetrics], name: str) -> np.ndarray:
        vals = []
        for m in metrics_history:
            v = getattr(m, name, None)
            vals.append(np.nan if v is None else float(v))
        arr = np.array(vals, dtype=float)
        # Fill NaNs with linear interpolation then edge-fill
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

    def _calculate_wrist_speeds(self, metrics_history: List[FrameMetrics]) -> np.ndarray:
        # Prefer a provided wrist velocity; otherwise estimate from elbow+spine deltas
        fv = self._series(metrics_history, 'front_elbow_angle')
        sv = self._series(metrics_history, 'spine_lean_angle')
        d_elb = self._delta(fv)
        d_spn = self._delta(sv)
        est = 0.7 * d_elb + 0.3 * d_spn
        # Normalize
        maxv = est.max() if est.size and np.isfinite(est.max()) and est.max() > 0 else 1.0
        return est / maxv

    def _classify_frame(self, i: int, total: int,
                        wrist_speed: float, elbow_change: float, spine_change: float) -> BattingPhase:
        # Coarse rules with tolerant thresholds
        if wrist_speed >= self.thresholds['wrist_speed_peak']:
            return BattingPhase.IMPACT
        if wrist_speed >= 0.35 and elbow_change >= self.thresholds['elbow_angle_change']:
            return BattingPhase.DOWNSWING
        if spine_change >= self.thresholds['spine_lean_threshold']:
            return BattingPhase.STRIDE
        if wrist_speed >= 0.15:
            return BattingPhase.FOLLOW_THROUGH
        if i >= int(0.85 * total):
            return BattingPhase.RECOVERY
        return BattingPhase.STANCE

    def _make_segment(self, phase: BattingPhase, s: int, e: int,
                      metrics_history: List[FrameMetrics]) -> Optional[PhaseSegment]:
        if e < s:
            return None
        seg_len = e - s + 1
        conf = self._phase_confidence(s, e, metrics_history)
        if seg_len < 2:
            conf *= 0.5
        return PhaseSegment(
            phase=phase, start_frame=s, end_frame=e,
            confidence=float(np.clip(conf, 0.0, 1.0)),
            metrics=self._extract_phase_metrics(s, e, metrics_history)
        )

    def _phase_confidence(self, s: int, e: int, metrics_history: List[FrameMetrics]) -> float:
        # Confidence from continuity (fewer NaNs => higher confidence)
        sub = metrics_history[s:e+1]
        total = len(sub)
        if total == 0:
            return 0.0
        valid = 0
        for m in sub:
            ok = 0
            ok += int(getattr(m, 'front_elbow_angle', None) is not None)
            ok += int(getattr(m, 'spine_lean_angle', None) is not None)
            ok += int(getattr(m, 'head_over_front_knee', None) is not None)
            valid += int(ok > 0)
        return valid / total

    def _extract_phase_metrics(self, s: int, e: int, metrics_history: List[FrameMetrics]) -> Dict[str, Any]:
        if e < s:
            return {}
        sub = metrics_history[s:e+1]
        def mean_of(name: str) -> float:
            arr = self._series(sub, name)
            return float(np.mean(arr)) if arr.size else 0.0
        return {
            'duration_frames': e - s + 1,
            'avg_elbow_angle': mean_of('front_elbow_angle'),
            'avg_spine_lean': mean_of('spine_lean_angle'),
            'avg_head_knee_dist': mean_of('head_over_front_knee'),
            'avg_foot_direction': mean_of('front_foot_direction'),
        }

    def _post_process_phases(self, phases: List[PhaseSegment]) -> List[PhaseSegment]:
        if not phases:
            return phases
        # Merge adjacent same-type short phases
        merged: List[PhaseSegment] = []
        cur = phases[0]
        for p in phases[1:]:
            if p.phase == cur.phase and (p.end_frame - cur.start_frame + 1) < self.thresholds['phase_min_duration'] + 2:
                cur.end_frame = p.end_frame
                cur.confidence = min(cur.confidence, p.confidence)
            else:
                if (cur.end_frame - cur.start_frame + 1) >= self.thresholds['phase_min_duration'] or not merged:
                    merged.append(cur)
                else:
                    # too small: merge into previous if exists
                    if merged:
                        merged[-1].end_frame = cur.end_frame
                cur = p
        # push last
        if (cur.end_frame - cur.start_frame + 1) >= self.thresholds['phase_min_duration'] or not merged:
            merged.append(cur)
        return merged
