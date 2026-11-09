"""Per-track bookkeeping on top of the multi-object tracker.

Responsibilities
* Role assignment (customer / staff / none) from the configured regions, using the person's
  shoulder-centre anchor and a sticky majority vote so that one bad frame cannot flip a role.
* Person scale: the median forearm length of the track, used to normalise all distances.
* Hand evidence: valid wrists only (confidence, frame bounds, anatomical plausibility), smoothed.
* Possible track-ID switches: a new track that appears where a same-role track was just lost.

Track IDs are temporary tracker labels. They say nothing about real-world identity and may
change after occlusion; nothing here assumes they are permanent.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

import numpy as np

from .calibration import CameraCalibration
from .config import AppConfig
from .pose_estimator import ARM, KP, PersonDetection, keypoint_valid
from .trajectory import EmaSmoother, HandObs, hand_proxy, hand_tip


@dataclass
class IdSwitch:
    old_id: int | None   # None when ambiguous
    new_id: int
    role: str
    t: float
    candidates: list[int]

    @property
    def ambiguous(self) -> bool:
        return len(self.candidates) > 1


class TrackedPerson:
    def __init__(self, track_id: int, t: float, frame_idx: int, cfg: AppConfig, gap_frames: int):
        self.track_id = track_id
        self.role: str | None = None
        self.votes: deque = deque(maxlen=cfg.roles.window_frames)
        self.first_seen_t = t
        self.last_seen_t = t
        self.last_frame = frame_idx
        self.role_confirmed_t: float | None = None
        self.anchor: np.ndarray | None = None
        self.det: PersonDetection | None = None
        self.visible = False
        self.forearms: deque = deque(maxlen=cfg.keypoints.scale_history)
        self.upper_arms: deque = deque(maxlen=cfg.keypoints.scale_history)
        a = cfg.keypoints.smoothing_alpha
        self._wrist_s = {s: EmaSmoother(a, gap_frames) for s in ARM}
        self._proxy_s = {s: EmaSmoother(a, gap_frames) for s in ARM}
        self.hands: dict[str, HandObs] = {}
        self.trail = {s: deque(maxlen=max(1, cfg.output.trail_frames)) for s in ARM}
        self.rejected_wrists = 0   # wrists dropped by the plausibility check (diagnostics)
        # recent (t, {side: hand proxy}, shoulder anchor) samples: where were the hands BEFORE an interaction?
        self.history: deque = deque(maxlen=int(cfg.contact.engagement_window_s * 40) + 10)

    # ------------------------------------------------------------------ scale
    def scale(self) -> tuple[float | None, str]:
        """Characteristic length of this person in pixels (median forearm)."""
        if len(self.forearms) >= 3:
            return float(np.median(self.forearms)), "forearm"
        if len(self.upper_arms) >= 3:
            return 0.85 * float(np.median(self.upper_arms)), "upper_arm"
        if self.forearms:
            return float(np.median(self.forearms)), "forearm(few)"
        if self.det is not None:
            h = float(self.det.box[3] - self.det.box[1])
            return 0.15 * h, "box"
        return None, "none"

    @property
    def label(self) -> str:
        tag = {"customer": "C", "staff": "S"}.get(self.role or "", "?")
        return f"{tag}{self.track_id}"


def _anchor(det: PersonDetection, kc: float, size) -> np.ndarray:
    kp = det.keypoints
    pts = [kp[i, :2] for i in (KP["left_shoulder"], KP["right_shoulder"]) if keypoint_valid(kp, i, kc, size)]
    if not pts:
        pts = [kp[i, :2] for i in (KP["left_hip"], KP["right_hip"]) if keypoint_valid(kp, i, kc, size)]
    if pts:
        return np.mean(pts, axis=0)
    x1, y1, x2, y2 = det.box
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])


class PersonRegistry:
    def __init__(self, cfg: AppConfig, fps: float):
        self.cfg = cfg
        self.fps = fps
        self.gap_frames = max(1, int(round(cfg.crossing.max_gap_s * fps)))
        self.persons: dict[int, TrackedPerson] = {}
        self.untracked_detections = 0
        self.role_changes: list[tuple[float, int, str | None, str | None]] = []

    def reset(self) -> None:
        self.persons.clear()

    def update(self, frame_idx: int, t: float, detections: list[PersonDetection],
               cal: CameraCalibration, frame_size: tuple[int, int]) -> tuple[list[TrackedPerson], list[IdSwitch]]:
        cfg = self.cfg
        k = cfg.keypoints
        for p in self.persons.values():
            p.visible = False
            p.hands = {}
        newly_confirmed: list[TrackedPerson] = []
        for det in detections:
            if det.track_id is None:
                self.untracked_detections += 1  # no identity -> cannot be used for temporal reasoning
                continue
            p = self.persons.get(det.track_id)
            if p is None:
                p = self.persons[det.track_id] = TrackedPerson(det.track_id, t, frame_idx, cfg, self.gap_frames)
            p.visible, p.det, p.last_seen_t, p.last_frame = True, det, t, frame_idx
            p.anchor = _anchor(det, k.shoulder_conf, frame_size)

            # ---- role vote
            box_h = float(det.box[3] - det.box[1])
            min_frac = cfg.roles.min_person_height_frac
            min_frac = 0.0 if min_frac == "auto" else min_frac  # "auto" is resolved by the pipeline per video
            vote = None if box_h < min_frac * frame_size[1] else cal.region_of(p.anchor)
            p.votes.append(vote)
            counts = Counter(p.votes)
            old = p.role
            if p.role is None:
                for role in ("customer", "staff"):
                    if counts[role] >= cfg.roles.confirm_frames and counts[role] > counts[_other(role)]:
                        p.role = role
            else:
                other = _other(p.role)
                if counts[other] >= cfg.roles.confirm_frames and counts[other] > counts[p.role] + counts[None]:
                    p.role = other
                elif len(p.votes) == p.votes.maxlen and counts[None] > len(p.votes) // 2:
                    p.role = None  # left both regions
            if p.role != old:
                self.role_changes.append((t, p.track_id, old, p.role))
                if old is None and p.role is not None:
                    p.role_confirmed_t = t
                    newly_confirmed.append(p)

            # ---- hand evidence
            self._update_hands(p, frame_idx, frame_size)
            if p.hands:
                p.history.append((t, {s: h.proxy.copy() for s, h in p.hands.items()}, p.anchor.copy()))

        switches = [self._find_switch(p, t) for p in newly_confirmed
                    if p.role_confirmed_t - p.first_seen_t <= cfg.identity.switch_window_s]
        switches = [s for s in switches if s is not None]

        # forget tracks unseen for a long time (they cannot be re-associated by the tracker anymore)
        horizon = max(cfg.identity.switch_window_s, cfg.tracking.track_buffer_s) + 5.0
        for tid in [tid for tid, p in self.persons.items() if t - p.last_seen_t > horizon]:
            del self.persons[tid]
        visible = [p for p in self.persons.values() if p.visible]
        return visible, switches

    def _update_hands(self, p: TrackedPerson, frame_idx: int, size) -> None:
        k = self.cfg.keypoints
        kp = p.det.keypoints
        scale, src = p.scale()
        lo, hi = k.forearm_plausibility
        for side, (si, ei, wi) in ARM.items():
            elbow = kp[ei, :2].copy() if keypoint_valid(kp, ei, k.elbow_conf, size, k.edge_margin_px) else None
            shoulder = kp[si, :2].copy() if keypoint_valid(kp, si, k.shoulder_conf, size, k.edge_margin_px) else None
            if shoulder is not None and elbow is not None:
                p.upper_arms.append(float(np.hypot(*(elbow - shoulder))))
            if not keypoint_valid(kp, wi, k.wrist_conf, size, k.edge_margin_px):
                continue
            wrist = kp[wi, :2].copy()
            forearm = float(np.hypot(*(wrist - elbow))) if elbow is not None else None
            if forearm is not None and scale and src == "forearm" and len(p.forearms) >= 5:
                if not (lo * scale <= forearm <= hi * scale):
                    p.rejected_wrists += 1  # implausible arm: keypoint probably jumped to another person
                    continue
            if forearm is not None and forearm > 1:
                p.forearms.append(forearm)
            ws = p._wrist_s[side].update(frame_idx, wrist)
            proxy = p._proxy_s[side].update(frame_idx, hand_proxy(wrist, elbow, k.hand_extension_ratio))
            cur_scale, _ = p.scale()
            ext = (float(np.hypot(*(wrist - shoulder))) / cur_scale
                   if shoulder is not None and cur_scale else None)
            tip = hand_tip(ws, proxy, k.hand_extension_ratio, k.fingertip_ratio) if elbow is not None else None
            p.hands[side] = HandObs(side, wrist, ws, proxy, float(kp[wi, 2]), elbow, shoulder,
                                    forearm, elbow is not None, ext, tip)
            p.trail[side].append((frame_idx, ws))

    def _find_switch(self, new: TrackedPerson, t: float) -> IdSwitch | None:
        cfg = self.cfg.identity
        cands = []
        for old in self.persons.values():
            if old is new or old.visible or old.role != new.role or old.anchor is None:
                continue
            if old.last_seen_t < new.first_seen_t - cfg.switch_window_s:
                continue
            scale, _ = old.scale()
            if scale and np.hypot(*(new.anchor - old.anchor)) <= cfg.switch_radius_ratio * scale:
                cands.append(old.track_id)
        if not cands:
            return None
        return IdSwitch(cands[0] if len(cands) == 1 else None, new.track_id, new.role, t, cands)


def _other(role: str) -> str:
    return "staff" if role == "customer" else "customer"
