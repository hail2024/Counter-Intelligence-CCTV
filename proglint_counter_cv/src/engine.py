"""Per-frame orchestration of roles, crossings and customer-staff pair state machines.

Pair association rule: only (customer-role track, staff-role track) pairs are evaluated, with
roles coming from the calibrated regions. Wrists are never paired just because they are the two
closest points in the image. If one person's hand is close to the hands of two counterparts whose
distances are similar (within ``contact.ambiguity_ratio``), neither pair receives evidence for that
frame and the episode is flagged ``ambiguous_pairing``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np

from .calibration import CameraCalibration
from .config import AppConfig
from .crossing import PersonCrossingTracker
from .event_logger import EventLedger
from .geometry import segment_closest
from .interaction_state import STATE_PRIORITY, Evidence, PairFrame, PairStateMachine, State
from .classifier import InteractionClassifier
from .person_tracker import PersonRegistry, TrackedPerson
from .pose_estimator import KP, PersonDetection, keypoint_valid


@dataclass
class PersonView:
    person: TrackedPerson
    crossing: str = "idle"                 # idle | approaching | crossed
    crossing_event_id: str | None = None
    far_sides: list[str] = field(default_factory=list)


@dataclass
class PairView:
    customer_id: int
    staff_id: int
    state: State
    event_id: str | None
    event_type: str | None
    frame: PairFrame | None
    handshake_score: float | None = None   # live rule-based handshake evidence of the open episode (0..1)


@dataclass
class FrameResult:
    frame_idx: int
    t: float
    persons: list[PersonView]
    pairs: list[PairView]
    overall_state: State
    paused: bool = False
    pause_reason: str = ""


class InteractionEngine:
    def __init__(self, cfg: AppConfig, cal: CameraCalibration, fps: float, stride: int = 1,
                 ledger: EventLedger | None = None):
        self.cfg = cfg
        self.cal = cal
        self.fps = fps
        self.eff_fps = fps / max(1, stride)   # rate at which this engine actually sees frames
        self.ledger = ledger or EventLedger()
        self.ledger.join_gap_s = cfg.interactions.join_gap_s
        self.registry = PersonRegistry(cfg, self.eff_fps)
        self.crossers: dict[int, PersonCrossingTracker] = {}
        self.pairs: dict[tuple[int, int], PairStateMachine] = {}
        self.paused = False
        self.pause_reason = ""
        self.id_switches: list[dict] = []
        self.model = (InteractionClassifier.load(cfg.resolve_path(cfg.classifier.model_path))
                      if cfg.classifier.enabled else None)

    # ------------------------------------------------------------------ scene control
    def scene_cut(self, frame_idx: int, t: float, view_ok: bool, similarity: float | None) -> None:
        """A camera cut: close every episode, forget all tracks; pause if the new view does not
        match the calibration (its boundary/regions would be meaningless)."""
        self._close_all(frame_idx, t, "scene_cut")
        self.registry.reset()
        self.crossers.clear()
        self.pairs.clear()
        sim = "n/a" if similarity is None else f"{similarity:.2f}"
        if view_ok:
            self.paused, self.pause_reason = False, ""
            self.ledger.notes.append(f"{t:.2f}s camera cut: tracks reset (view similarity {sim})")
        else:
            self.paused = True
            self.pause_reason = f"camera view differs from calibration (similarity {sim})"
            self.ledger.notes.append(f"{t:.2f}s camera cut: ANALYSIS PAUSED - {self.pause_reason}")

    def set_paused(self, paused: bool, reason: str, t: float) -> None:
        if paused and not self.paused:
            self.ledger.notes.append(f"{t:.2f}s ANALYSIS PAUSED - {reason}")
        self.paused, self.pause_reason = paused, reason if paused else ""

    def finish(self, frame_idx: int, t: float) -> None:
        self._close_all(frame_idx, t, "video_end")

    def _close_all(self, frame_idx: int, t: float, reason: str) -> None:
        for cr in self.crossers.values():
            cr.finish(reason)
        for m in self.pairs.values():
            m.finish(frame_idx, t, reason)

    # ------------------------------------------------------------------ per frame
    def process(self, frame_idx: int, t: float, detections: list[PersonDetection],
                frame_size: tuple[int, int]) -> FrameResult:
        if self.paused:
            return FrameResult(frame_idx, t, [], [], State.NO_ACTIVE_INTERACTION, True, self.pause_reason)

        self._frame_size = frame_size
        persons, switches = self.registry.update(frame_idx, t, detections, self.cal, frame_size)
        for sw in switches:
            self._apply_switch(sw, t)

        # ---- crossings (one tracker per role-assigned track; invisible tracks get None so gaps age)
        views: dict[int, PersonView] = {}
        for p in persons:
            if p.role is None:
                continue
            cr = self.crossers.get(p.track_id)
            if cr is not None and cr.role != p.role:  # role flipped: close the old episode
                cr.finish("role_changed")
                cr = None
            if cr is None:
                cr = self.crossers[p.track_id] = PersonCrossingTracker(p.track_id, p.role, self.cfg,
                                                                       self.eff_fps, self.ledger)
        busy = {tid for key, m in self.pairs.items() if m.episode is not None for tid in key}
        for tid, cr in list(self.crossers.items()):
            p = self.registry.persons.get(tid)
            status = cr.update(frame_idx, t, p, self.cal, in_interaction=tid in busy)
            if p is None and not cr.active:
                del self.crossers[tid]
                continue
            if p is not None and p.visible:
                views[tid] = PersonView(p, status, cr.event_id if cr.active else None, cr.far_sides())
        for p in persons:
            views.setdefault(p.track_id, PersonView(p))

        # ---- pair evidence
        customers = [p for p in persons if p.role == "customer"]
        staff = [p for p in persons if p.role == "staff"]
        frames: dict[tuple[int, int], PairFrame] = {}
        for cp, sp in product(customers, staff):
            frames[(cp.track_id, sp.track_id)] = self._pair_frame(frame_idx, t, cp, sp, views)
        self._resolve_ambiguity(frames)

        # pairs with an open machine but a missing person still need a frame (their evidence is NONE)
        for key, m in self.pairs.items():
            if key not in frames:
                frames[key] = PairFrame(frame_idx, t, Evidence.NONE, "track_missing")

        out_pairs = []
        for key, pf in frames.items():
            m = self.pairs.get(key)
            if m is None:
                interesting = ((pf.evidence in (Evidence.BOTH, Evidence.NONE) and pf.d_ratio is not None
                                and pf.d_ratio <= self.cfg.contact.approach_ratio)
                               or pf.evidence == Evidence.PARTIAL_NEAR or pf.crossing_active)
                if not interesting:
                    continue
                m = self.pairs[key] = PairStateMachine(key[0], key[1], self.cfg, self.eff_fps, self.ledger, self.model)
            state = m.update(pf)
            eid = m.event_id
            ep = m.episode
            out_pairs.append(PairView(m.customer_id, m.staff_id, state, eid,
                                      self.ledger.get(eid).event_type if eid else None, pf,
                                      ep.hs_score if ep is not None and ep.close_frames else None))
            if eid:  # link overlapping crossing events of the same two people
                for tid in key:
                    cr = self.crossers.get(tid)
                    if cr is not None and cr.active and cr.event_id:
                        self.ledger.link(eid, cr.event_id)

        # two people reaching across at the same time with their hands close together are one exchange
        # (e.g. an item handed over without a contact assessment): their crossing rows share an interaction
        if self.cfg.interactions.link_paired_crossings:
            for (cid, sid), pf in frames.items():
                if pf.evidence != Evidence.BOTH or pf.d_ratio is None or pf.d_ratio > self.cfg.contact.approach_ratio:
                    continue
                crs = [self.crossers.get(cid), self.crossers.get(sid)]
                if all(cr is not None and cr.active and cr.event_id for cr in crs):
                    self.ledger.link(crs[0].event_id, crs[1].event_id)

        # forget idle machines whose people are gone
        for key in [k for k, m in self.pairs.items()
                    if m.idle and not all(tid in self.registry.persons for tid in k)]:
            del self.pairs[key]

        states = [pv.state for pv in out_pairs]
        states += [State.BOUNDARY_CROSSED if v.crossing == "crossed" else State.HAND_APPROACHING
                   for v in views.values() if v.crossing != "idle"]
        overall = min(states, key=STATE_PRIORITY.index) if states else State.NO_ACTIVE_INTERACTION
        return FrameResult(frame_idx, t, list(views.values()), out_pairs, overall)

    # ------------------------------------------------------------------ helpers
    def _pair_frame(self, frame_idx, t, cp: TrackedPerson, sp: TrackedPerson, views) -> PairFrame:
        crossing = any(views.get(tid) is not None and views[tid].crossing == "crossed"
                       for tid in (cp.track_id, sp.track_id))
        approaching = any(views.get(tid) is not None and views[tid].crossing == "approaching"
                          for tid in (cp.track_id, sp.track_id))
        sc, _ = cp.scale()
        ss, _ = sp.scale()
        scales = [s for s in (sc, ss) if s]
        if not scales:
            return PairFrame(frame_idx, t, Evidence.NONE, "person_scale_unknown",
                             crossing_active=crossing, approaching=approaching)
        scale = float(np.mean(scales))
        if cp.hands and sp.hands:
            best = None
            for ch, sh in product(cp.hands.values(), sp.hands.values()):
                d = float(np.hypot(*(ch.proxy - sh.proxy)))
                if best is None or d < best[0]:
                    best = (d, ch, sh)
            d, ch, sh = best
            pf = PairFrame(frame_idx, t, Evidence.BOTH, "", d / scale, d,
                           float(np.hypot(*(ch.wrist - sh.wrist))), min(ch.conf, sh.conf), scale,
                           ch, sh, crossing, approaching, tuple(cp.hands), tuple(sp.hands))
            touch = None
            for a, b in product(cp.hands.values(), sp.hands.values()):
                if a.tip is not None and b.tip is not None:
                    dd, s_at, t_at = segment_closest(a.wrist_s, a.tip, b.wrist_s, b.tip)
                    if touch is None or dd < touch[0]:
                        touch = (dd, s_at, t_at, a, b)
            if touch is not None:
                pf.touch_ratio, pf.touch_at = touch[0] / scale, (touch[1], touch[2])
                pf.t_c_hand, pf.t_s_hand = touch[3], touch[4]
                vals = self._reach(cp, sp, touch[3], touch[4], t)
                if vals is not None:
                    pf.t_c_reach, pf.t_s_reach, _, pf.t_c_base, pf.t_s_base = vals
            self._add_reach(pf, cp, sp, t)
            return pf
        if cp.hands or sp.hands:
            seen, hidden = (cp, sp) if cp.hands else (sp, cp)
            k = self.cfg.keypoints
            kp = hidden.det.keypoints
            elbows = [kp[i, :2] for i in (KP["left_elbow"], KP["right_elbow"])
                      if keypoint_valid(kp, i, k.elbow_conf, self._frame_size, k.edge_margin_px)]
            reason = f"{hidden.role}_wrist_not_visible"
            if not elbows:
                return PairFrame(frame_idx, t, Evidence.NONE, f"{hidden.role}_arm_not_visible",
                                 scale=scale, crossing_active=crossing, approaching=approaching)
            dmin = min(float(np.hypot(*(h.proxy - e))) for h in seen.hands.values() for e in elbows)
            # the hidden hand can be anywhere within (1 + hand_extension) forearms of its elbow
            kind = Evidence.PARTIAL_NEAR if dmin / scale <= self.cfg.contact.partial_near_ratio else Evidence.PARTIAL_FAR
            return PairFrame(frame_idx, t, kind, reason, scale=scale,
                             crossing_active=crossing, approaching=approaching)
        return PairFrame(frame_idx, t, Evidence.NONE, "no_wrists_visible", scale=scale,
                         crossing_active=crossing, approaching=approaching)

    def _add_reach(self, pf: PairFrame, cp: TrackedPerson, sp: TrackedPerson, t: float) -> None:
        """How far each person's hand is extended toward the other person, now and before.

        reach = projection of (hand - own shoulder centre) on the direction to the other person, divided by the
        distance between the two people (0 = at own body, 0.5 = halfway, 1 = at the other person).
        The baseline is the smallest reach of that same hand in the engagement window before this frame: a hand
        that has been resting/typing in place has reach ~= baseline (not engaged), a hand that came out to meet
        the other person has reach >> baseline.
        """
        vals = self._reach(cp, sp, pf.c_hand, pf.s_hand, t)
        if vals is not None:
            pf.c_reach, pf.s_reach, pf.meet, pf.c_base, pf.s_base = vals

    def _reach(self, cp: TrackedPerson, sp: TrackedPerson, ch, sh, t: float):
        """(customer reach, staff reach, meeting point, customer baseline, staff baseline) for hands ch / sh."""
        gap_v = sp.anchor - cp.anchor
        gap = float(np.hypot(*gap_v))
        if gap < 1:
            return None
        u = gap_v / gap
        c_reach = float((ch.proxy - cp.anchor) @ u) / gap
        s_reach = float((sh.proxy - sp.anchor) @ -u) / gap
        meet = float(((ch.proxy + sh.proxy) / 2 - cp.anchor) @ u) / gap
        win, lag = self.cfg.contact.engagement_window_s, self.cfg.contact.engagement_lag_s
        bases = []
        for person, side, sign in ((cp, ch.side, 1.0), (sp, sh.side, -1.0)):
            vals = [float((proxies[side] - anchor) @ (sign * u)) / gap for (te, proxies, anchor) in person.history
                    if t - win <= te <= t - lag and side in proxies]
            # a baseline needs a real look at the hand before the interaction; otherwise it is unknown
            seen_before = person.history and person.history[0][0] <= t - lag - self.cfg.contact.engagement_min_history_s
            bases.append(min(vals) if vals and seen_before else None)
        return c_reach, s_reach, meet, bases[0], bases[1]

    def _resolve_ambiguity(self, frames: dict[tuple[int, int], PairFrame]) -> None:
        """A person's hand may only be in contact with ONE counterpart at a time."""
        limit = self.cfg.contact.exit_ratio
        for idx in (0, 1):  # 0: group by customer, 1: group by staff
            groups: dict[int, list] = {}
            for key, pf in frames.items():
                if pf.evidence == Evidence.BOTH and pf.d_ratio <= limit:
                    groups.setdefault(key[idx], []).append((pf.d_ratio, key))
            for items in groups.values():
                if len(items) < 2:
                    continue
                items.sort()
                (d1, k1), (d2, _) = items[0], items[1]
                ambiguous = d2 < self.cfg.contact.ambiguity_ratio * max(d1, 1e-6)
                for d, k in items:
                    if ambiguous and k == k1:
                        # one episode (on the nearest pair) carries the ambiguity -> one UNCERTAIN record
                        frames[k] = PairFrame(frames[k].frame_idx, frames[k].t, Evidence.NONE,
                                              "ambiguous_pairing", d_ratio=d, scale=frames[k].scale)
                    elif ambiguous and d < self.cfg.contact.ambiguity_ratio * max(d1, 1e-6):
                        frames[k] = PairFrame(frames[k].frame_idx, frames[k].t, Evidence.NONE,
                                              "hand_shared_with_another_pair", scale=frames[k].scale)
                    elif k != k1:
                        frames[k] = PairFrame(frames[k].frame_idx, frames[k].t, Evidence.NONE,
                                              "hand_closer_to_another_person", scale=frames[k].scale)

    def _apply_switch(self, sw, t: float) -> None:
        rec = {"t": round(t, 3), "role": sw.role, "new": sw.new_id, "candidates": sw.candidates}
        if sw.ambiguous:
            rec["action"] = "ambiguous: interactions of the lost tracks marked identity_ambiguous"
            for m in self.pairs.values():
                if (m.customer_id if sw.role == "customer" else m.staff_id) in sw.candidates:
                    m.mark_identity_ambiguous()
        else:
            old, new = sw.old_id, sw.new_id
            moved = []
            cr = self.crossers.pop(old, None)
            if cr is not None and new not in self.crossers:
                cr.rekey(new)
                self.crossers[new] = cr
                moved.append("crossing")
            for key in [k for k in self.pairs if (k[0] if sw.role == "customer" else k[1]) == old]:
                new_key = (new, key[1]) if sw.role == "customer" else (key[0], new)
                if new_key in self.pairs:
                    continue
                m = self.pairs.pop(key)
                if m.idle:
                    continue
                m.rekey(sw.role, old, new)
                self.pairs[new_key] = m
                moved.append(f"pair{key}->{new_key}")
            rec["action"] = f"inferred {old}->{new}; continued: {moved or 'nothing active'}"
        self.id_switches.append(rec)
        self.ledger.notes.append(f"{t:.2f}s possible track-ID switch ({sw.role}): {rec['action']}")
