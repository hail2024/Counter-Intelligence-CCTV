"""Per-wrist trajectory handling: smoothing, hand-proxy estimation and boundary-side tracking.

Boundary-crossing rule (per wrist)
----------------------------------
Each wrist has a *confirmed side*: HOME (its owner's side of the counter), FAR (the other side)
or UNKNOWN. The signed distance d to the boundary (positive = HOME) is compared with a hysteresis
margin m:

    d >  m  -> observation is on HOME
    d < -m  -> observation is on FAR
    |d| <= m -> inside the dead band: the confirmed side is kept (absorbs keypoint jitter)

The confirmed side only changes after ``persist_frames`` consecutive observations on the new side.
A crossing is reported only for a confirmed HOME -> FAR change. If the wrist is missing for longer
than ``max_gap_frames`` the confirmed side becomes UNKNOWN, so a wrist that disappears on one side
and reappears on the other is NOT counted as a crossing (the transition was not observed).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class EmaSmoother:
    """Exponential smoothing of a 2-D point that restarts after a gap (no smoothing across gaps)."""

    def __init__(self, alpha: float, max_gap_frames: int):
        self.alpha = alpha
        self.max_gap = max_gap_frames
        self.value: np.ndarray | None = None
        self.last_frame: int | None = None

    def update(self, frame_idx: int, point) -> np.ndarray | None:
        if point is None:
            return None
        p = np.asarray(point, float)
        if self.value is None or self.last_frame is None or frame_idx - self.last_frame > self.max_gap + 1:
            self.value = p
        else:
            self.value = self.alpha * p + (1 - self.alpha) * self.value
        self.last_frame = frame_idx
        return self.value.copy()


class Side(str, Enum):
    HOME = "HOME"
    FAR = "FAR"
    UNKNOWN = "UNKNOWN"


@dataclass
class SideTransition:
    kind: str            # "crossed" (HOME->FAR) | "returned" (FAR->HOME) | "appeared_far" | "appeared_home"
    pending_start_t: float  # time the wrist first entered the new side (start of the persistence run)


class WristSideTracker:
    def __init__(self, persist_frames: int, max_gap_frames: int):
        self.persist = max(1, persist_frames)
        self.max_gap = max_gap_frames
        self.confirmed = Side.UNKNOWN
        self._pending: Side | None = None
        self._pending_n = 0
        self._pending_t = 0.0
        self._gap = 0
        self.last_distance: float | None = None

    def update(self, t: float, home_distance: float | None, margin: float) -> SideTransition | None:
        """Feed one frame. ``home_distance`` is None when the wrist is missing/unreliable."""
        if home_distance is None:
            self._gap += 1
            self._pending, self._pending_n = None, 0
            if self._gap > self.max_gap:
                self.confirmed = Side.UNKNOWN
                self.last_distance = None
            return None
        self._gap = 0
        self.last_distance = home_distance
        if home_distance > margin:
            zone = Side.HOME
        elif home_distance < -margin:
            zone = Side.FAR
        else:  # dead band: keep the confirmed side, restart any pending change
            self._pending, self._pending_n = None, 0
            return None
        if zone == self.confirmed:
            self._pending, self._pending_n = None, 0
            return None
        if zone == self._pending:
            self._pending_n += 1
        else:
            self._pending, self._pending_n, self._pending_t = zone, 1, t
        if self._pending_n < self.persist:
            return None
        prev, self.confirmed = self.confirmed, zone
        self._pending, self._pending_n = None, 0
        if prev == Side.HOME and zone == Side.FAR:
            return SideTransition("crossed", self._pending_t)
        if prev == Side.FAR and zone == Side.HOME:
            return SideTransition("returned", self._pending_t)
        return SideTransition("appeared_far" if zone == Side.FAR else "appeared_home", self._pending_t)


@dataclass
class HandObs:
    """Usable evidence for one arm of one person in one frame (only built when the wrist is valid)."""
    side: str                    # "L" | "R"
    wrist: np.ndarray            # raw wrist keypoint
    wrist_s: np.ndarray          # smoothed wrist
    proxy: np.ndarray            # smoothed hand-centre estimate (see hand_proxy)
    conf: float                  # wrist keypoint confidence
    elbow: np.ndarray | None
    shoulder: np.ndarray | None
    forearm_px: float | None     # |wrist - elbow| this frame
    proxy_from_elbow: bool       # False when the elbow was missing and proxy == wrist
    extension: float | None = None  # |wrist - shoulder| / person scale
    tip: np.ndarray | None = None   # estimated fingertip of a straight hand (None without an elbow, see hand_tip)


def hand_tip(wrist_s: np.ndarray, proxy_s: np.ndarray, proxy_ratio: float, tip_ratio: float) -> np.ndarray:
    """Estimated fingertip of a straight hand: the smoothed wrist pushed ``tip_ratio`` forearm lengths along the
    smoothed forearm direction (recovered from the smoothed hand proxy). The hand is modelled as the segment
    wrist -> tip, so two hands whose FINGERS touch (palms still apart) are recognised as meeting."""
    return wrist_s + (tip_ratio / max(proxy_ratio, 1e-6)) * (proxy_s - wrist_s)


def hand_proxy(wrist: np.ndarray, elbow: np.ndarray | None, ratio: float) -> np.ndarray:
    """Estimated hand centre: the wrist pushed along the forearm direction by ``ratio`` forearm lengths.

    Pose models locate the wrist joint, and two clasped hands keep the wrists roughly a hand-length
    apart; comparing hand proxies is therefore a better proximity measure than wrist-to-wrist distance.
    It remains a 2-D estimate - it cannot see fingers or depth.
    """
    if elbow is None:
        return wrist.copy()
    return wrist + ratio * (wrist - elbow)


def count_reversals(values: np.ndarray, min_amplitude: float) -> int:
    """Count direction reversals of a 1-D signal whose swings exceed ``min_amplitude``.

    Implemented as a hysteresis extremum tracker: a new extremum is accepted only after the signal
    moves back by at least ``min_amplitude``, so small jitter does not count.
    """
    if len(values) < 3:
        return 0
    direction = 0  # +1 rising, -1 falling, 0 not yet determined
    hi = lo = ext = float(values[0])
    reversals = 0
    for v in map(float, values[1:]):
        if direction == 0:
            hi, lo = max(hi, v), min(lo, v)
            if v - lo >= min_amplitude:
                direction, ext = 1, v
            elif hi - v >= min_amplitude:
                direction, ext = -1, v
        elif direction == 1:
            if v > ext:
                ext = v
            elif ext - v >= min_amplitude:
                reversals, direction, ext = reversals + 1, -1, v
        else:
            if v < ext:
                ext = v
            elif v - ext >= min_amplitude:
                reversals, direction, ext = reversals + 1, 1, v
    return reversals


def detrended_axes(points: np.ndarray) -> list[np.ndarray]:
    """x and y of a 2-D path with the straight-line drift removed (one array per axis).

    The clasped hands usually drift slowly (arms settling, leaning) while pumping. A linear fit per
    axis removes the drift so only the back-and-forth component is left for reversal counting.
    Pumping is evaluated on each image axis separately because the principal axis of the whole path
    tends to follow the drift instead of the pump.
    """
    if len(points) < 3:
        return [np.zeros(len(points))] * 2
    t = np.arange(len(points), dtype=float)
    out = []
    for i in range(2):
        coef = np.polyfit(t, points[:, i], 1)
        out.append(points[:, i] - np.polyval(coef, t))
    return out
