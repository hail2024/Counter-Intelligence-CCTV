"""Boundary-crossing episodes for one tracked person.

Episode rule
* An episode starts when one of the person's wrists makes a confirmed HOME -> FAR transition
  (see trajectory.WristSideTracker for the hysteresis / persistence rule).
* While ANY wrist of the person stays on the far side the episode continues - the second wrist
  crossing, or the first wrist jittering at the line, does not create another event.
* The episode ends when no wrist is confirmed on the far side (returned home, or lost for longer
  than the gap tolerance), or after ``crossing.max_episode_s``.
* A new HOME -> FAR transition within ``crossing.rearm_s`` of the end re-opens the SAME event.
* A wrist first seen on the far side (transition not observed) never creates an event; it is
  recorded in the ledger's suppressed list for traceability.

Result: at most one BOUNDARY_CROSSING event per reach, however noisy the keypoints are.
"""
from __future__ import annotations

import numpy as np

from .calibration import CameraCalibration
from .config import AppConfig
from .event_logger import CROSSING, EventLedger
from .person_tracker import TrackedPerson
from .trajectory import Side, WristSideTracker


class PersonCrossingTracker:
    def __init__(self, track_id: int, role: str, cfg: AppConfig, fps: float, ledger: EventLedger):
        c = cfg.crossing
        self.cfg = c
        self.role = role
        self.track_id = track_id
        self.ledger = ledger
        self.in_interaction = False
        persist = max(1, int(round(c.persist_s * fps)))
        gap = max(1, int(round(c.max_gap_s * fps)))
        self.wrists = {s: WristSideTracker(persist, gap) for s in ("L", "R")}
        self.event_id: str | None = None
        self.active = False
        self.closed_at: float | None = None
        self.approaching = False
        self._approach_until = -1.0
        self._prev_d: dict[str, float] = {}
        self._confs: list[float] = []
        self._start_t = 0.0
        self._last_far_t = 0.0
        self._last_far_frame = 0
        self._max_pen = 0.0
        self._max_pen_ratio = 0.0
        self._far_hands: set[str] = set()
        self._notes: list[str] = []

    def _point(self, hand):
        """The point tested against the boundary: the estimated hand centre (default) or the wrist joint."""
        return hand.proxy if self.cfg.point == "hand" else hand.wrist_s

    @property
    def direction(self) -> str:
        return "customer_to_staff" if self.role == "customer" else "staff_to_customer"

    def far_sides(self) -> list[str]:
        return [s for s, w in self.wrists.items() if w.confirmed == Side.FAR]

    def update(self, frame_idx: int, t: float, person: TrackedPerson | None, cal: CameraCalibration,
               in_interaction: bool = False) -> str:
        """Feed one frame; returns 'crossed' | 'approaching' | 'idle'."""
        scale = None
        if person is not None and person.visible:
            scale, _ = person.scale()
        margin = max(self.cfg.hysteresis_min_px, self.cfg.hysteresis_ratio * (scale or 0.0))
        home_sign = 1.0 if self.role == "customer" else -1.0
        self.approaching = False
        self.in_interaction = in_interaction
        for side, tracker in self.wrists.items():
            hand = person.hands.get(side) if person is not None and person.visible else None
            d = home_sign * cal.customer_side_distance(self._point(hand)) if hand is not None else None
            tr = tracker.update(t, d, margin)
            if d is not None and scale:
                prev = self._prev_d.get(side)
                if (tracker.confirmed == Side.HOME and prev is not None and d < prev
                        and d <= self.cfg.approach_ratio * scale):
                    self.approaching = True
                self._prev_d[side] = d
            elif d is None:
                self._prev_d.pop(side, None)
            if tr is None:
                continue
            if tr.kind == "crossed":
                self._on_cross(side, frame_idx, t, tr.pending_start_t)
            elif tr.kind == "appeared_far":
                self.ledger.suppress(
                    kind="crossing", reason="far_side_without_observed_transition",
                    start_timestamp=t, end_timestamp=t,
                    customer_track_id=self.track_id if self.role == "customer" else None,
                    staff_track_id=self.track_id if self.role == "staff" else None,
                    detail=f"{side} wrist first observed on the far side; no crossing event created")

        far = self.far_sides()
        if self.active:
            if far:
                self._last_far_t, self._last_far_frame = t, frame_idx
                for side in far:
                    hand = person.hands.get(side) if person is not None and person.visible else None
                    if hand is None:
                        continue
                    pen = -home_sign * cal.customer_side_distance(self._point(hand))
                    if pen > self._max_pen:
                        self._max_pen = pen
                        self._max_pen_ratio = pen / scale if scale else 0.0
                    self._confs.append(hand.conf)
                    self._far_hands.add(side)
                if t - self._start_t > self.cfg.max_episode_s:
                    self._close("timeout (max_episode_s)")
            else:
                all_home = all(w.confirmed == Side.HOME for w in self.wrists.values())
                self._close("wrist_returned_home" if all_home else "wrist_lost_from_view")
        if self.active:
            return "crossed"
        if self.approaching:
            self._approach_until = t + self.cfg.approach_hold_s
        return "approaching" if self.approaching or t <= self._approach_until else "idle"

    def _on_cross(self, side: str, frame_idx: int, t: float, start_t: float) -> None:
        if self.active:
            self._far_hands.add(side)
            return
        rearm = self.cfg.rearm_s if not self.in_interaction else self.cfg.interaction_rearm_s
        if self.event_id is not None and self.closed_at is not None and t - self.closed_at <= rearm:
            self.ledger.reopen(self.event_id)  # re-crossing right after the end: same reach episode
            self._notes.append(f"re-crossed at {t:.2f}s within rearm window (merged)")
            self._far_hands.add(side)
            self.active = True
            return
        ev = self.ledger.open(
            CROSSING, start_t, frame_idx, t,
            customer_track_id=self.track_id if self.role == "customer" else None,
            staff_track_id=self.track_id if self.role == "staff" else None,
            crossing_direction=self.direction, interaction_state="BOUNDARY_CROSSED",
            start_reason=f"{self.role} {side} wrist: confirmed HOME->FAR transition")
        self.event_id = ev.event_id
        self.active = True
        self._start_t = start_t
        self._last_far_t, self._last_far_frame = t, frame_idx
        self._max_pen, self._max_pen_ratio = 0.0, 0.0
        self._confs = []
        self._far_hands = {side}
        self._notes = []

    def _close(self, reason: str) -> None:
        if not self.active or self.event_id is None:
            return
        ev = self.ledger.get(self.event_id)
        ev.max_boundary_penetration_px = self._max_pen
        ev.hands = sorted({f"{self.role}:{s}" for s in self._far_hands} | set(ev.hands))
        if self._confs:
            ev.keypoint_conf_mean = float(np.mean(self._confs))
            ev.keypoint_conf_min = float(np.min(self._confs))
        ev.evidence_summary = "; ".join(
            [f"{'+'.join(sorted(self._far_hands))} wrist(s) on far side, max penetration "
             f"{self._max_pen:.0f}px ({self._max_pen_ratio:.2f} forearms)"] + self._notes)
        who, side = (("Customer", "staff") if self.role == "customer" else ("Staff member", "customer"))
        ev.interpretation = (f"{who}'s hand reached across the counter into the {side} side "
                             f"({self._max_pen_ratio:.1f} forearm lengths past the line)")
        self.ledger.close(self.event_id, self._last_far_t, self._last_far_frame, reason)
        self.active = False
        self.closed_at = self._last_far_t

    def rekey(self, new_track_id: int) -> None:
        """Continue this person's crossing state under a new track ID (inferred ID switch)."""
        old = self.track_id
        self.track_id = new_track_id
        if self.event_id is not None and self.active:
            ev = self.ledger.get(self.event_id)
            ev.track_id_changes.append(f"{self.role} {old}->{new_track_id}")
            if "track_id_switch_inferred" not in ev.uncertainty_reason:
                ev.uncertainty_reason.append("track_id_switch_inferred")

    def finish(self, reason: str) -> None:
        self._close(reason)
