"""Episode-based customer-staff interaction state machine (one instance per customer/staff track pair).

States (from the assignment)
----------------------------
NO_ACTIVE_INTERACTION  nothing being evaluated for this pair
HAND_APPROACHING       a hand moves toward the boundary or toward the other person's hand
BOUNDARY_CROSSED       a wrist of either person is across the counter, no close hand evidence
POSSIBLE_HAND_CONTACT  a contact episode is open but not yet confirmed
LIKELY_HAND_CONTACT    proximity + persistence + slow relative motion confirmed
LIKELY_HANDSHAKE       likely contact that also meets the handshake motion criteria
UNCERTAIN              an open episode whose evidence is currently missing / one-sided / ambiguous
COOLDOWN               an episode just ended; renewed contact re-opens the same episode

Per-frame evidence (computed by the engine) is one of:
BOTH          both people have a usable hand; ``d_ratio`` = closest hand-proxy distance / forearm
PARTIAL_NEAR  one hand visible and near the other person's elbow while their wrist is hidden
PARTIAL_FAR   one hand visible and away from the other person's arm (counts as "apart")
NONE          no usable evidence (track missing, both wrists hidden, ambiguous pairing)

Episode lifecycle
-----------------
start      BOTH with d_ratio <= enter_ratio, or PARTIAL_NEAR
close      BOTH frames with d_ratio <= enter_ratio (or <= exit_ratio while already close: hysteresis)
confirm    LIKELY_HAND_CONTACT when close time >= min_contact_s AND close frames / jointly-visible
           frames >= min_close_ratio AND median relative hand speed <= max_relative_speed
upgrade    LIKELY_HANDSHAKE when additionally close time >= handshake.min_duration_s, both arms
           extended, hands move together, and (optionally) the clasped hands pump
end        apart for separation_s | no evidence for missing_tolerance_s | max_episode_s | video end
           | scene cut
after end  COOLDOWN for cooldown_s; contact again in cooldown re-opens the same episode/event

Only the episode END decides whether an unconfirmed episode becomes an UNCERTAIN event or is
suppressed (brief pass / fast relative motion); every decision is written to the ledger.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .config import AppConfig
from .event_logger import CONTACT, HANDSHAKE, UNCERTAIN, EventLedger
from .classifier import InteractionClassifier, episode_features
from .trajectory import HandObs, count_reversals, detrended_axes


class State(str, Enum):
    NO_ACTIVE_INTERACTION = "NO_ACTIVE_INTERACTION"
    HAND_APPROACHING = "HAND_APPROACHING"
    BOUNDARY_CROSSED = "BOUNDARY_CROSSED"
    POSSIBLE_HAND_CONTACT = "POSSIBLE_HAND_CONTACT"
    LIKELY_HAND_CONTACT = "LIKELY_HAND_CONTACT"
    LIKELY_HANDSHAKE = "LIKELY_HANDSHAKE"
    UNCERTAIN = "UNCERTAIN"
    COOLDOWN = "COOLDOWN"


# display priority when several pairs/people are active (highest first)
STATE_PRIORITY = [State.LIKELY_HANDSHAKE, State.LIKELY_HAND_CONTACT, State.UNCERTAIN,
                  State.POSSIBLE_HAND_CONTACT, State.BOUNDARY_CROSSED, State.HAND_APPROACHING,
                  State.COOLDOWN, State.NO_ACTIVE_INTERACTION]


class Evidence(str, Enum):
    BOTH = "BOTH"
    PARTIAL_NEAR = "PARTIAL_NEAR"
    PARTIAL_FAR = "PARTIAL_FAR"
    NONE = "NONE"


@dataclass
class PairFrame:
    frame_idx: int
    t: float
    evidence: Evidence
    reason: str = ""                   # why evidence is partial / missing
    d_ratio: float | None = None       # hand-proxy distance in forearm lengths
    d_px: float | None = None          # hand-proxy distance in pixels
    wrist_d_px: float | None = None    # raw wrist-to-wrist distance of the same hand pair
    conf: float | None = None          # min wrist confidence of the pair
    scale: float | None = None         # pair scale (px per forearm)
    c_hand: HandObs | None = None
    s_hand: HandObs | None = None
    crossing_active: bool = False
    approaching: bool = False
    c_sides: tuple = ()                # customer wrists visible this frame ("L", "R")
    s_sides: tuple = ()
    c_reach: float | None = None       # hand extension toward the other person (0 own body .. 1 other person)
    s_reach: float | None = None
    c_base: float | None = None        # that hand's smallest reach in the window before this frame
    s_base: float | None = None
    meet: float | None = None          # where the hands meet between the two people (0 customer .. 1 staff)
    touch_ratio: float | None = None   # shortest distance between the two hand SEGMENTS (wrist->fingertip), forearms
    touch_at: tuple | None = None      # where on (customer, staff) hand segment they are closest: 0 wrist .. 1 fingertip
    t_c_hand: HandObs | None = None    # the touching hand pair (may differ from the closest-palm pair c_hand/s_hand)
    t_s_hand: HandObs | None = None
    t_c_reach: float | None = None     # reach / baseline of the touching hands (see engine._reach)
    t_s_reach: float | None = None
    t_c_base: float | None = None
    t_s_base: float | None = None


LEVEL_NONE, LEVEL_CONTACT, LEVEL_HANDSHAKE = 0, 1, 2


@dataclass
class Episode:
    start_t: float
    start_frame: int
    last_t: float
    start_reason: str
    in_contact: bool = False
    close_frames: int = 0
    close_time: float = 0.0
    visible_frames: int = 0
    partial_time: float = 0.0
    last_close_t: float | None = None
    last_close_frame: int | None = None
    last_evidence_t: float = 0.0
    last_evidence_frame: int = 0
    apart_since: float | None = None
    missing_since: float | None = None
    level: int = LEVEL_NONE
    event_id: str | None = None
    reasons: list[str] = field(default_factory=list)       # uncertainty reasons (deduplicated)
    notes: list[str] = field(default_factory=list)
    min_d_ratio: float = np.inf
    min_d_px: float = np.inf
    min_wrist_px: float = np.inf
    confs: list[float] = field(default_factory=list)
    rel: list[np.ndarray] = field(default_factory=list)     # (c - s) / scale while close
    mids: list[np.ndarray] = field(default_factory=list)    # midpoint / scale while close
    speeds: list[float] = field(default_factory=list)       # relative speed (forearms/s) while close
    ext_both: int = 0                                       # close frames with both arms extended
    ext_known: int = 0                                      # close frames where both extensions were measurable
    hands: set = field(default_factory=set)
    last_rel: np.ndarray | None = None
    last_rel_t: float | None = None
    hs_score: float = 0.0
    hs_parts: dict = field(default_factory=dict)
    suppressed_ref: object = None
    engaged: tuple | None = None       # (customer side, staff side) of the last close frame
    max_reversals: int = 0
    max_clasp: float = 0.0             # peak clasp share (see PairStateMachine.clasp_frac)
    c_confs: list = field(default_factory=list)   # wrist confidence of each person's engaged hand, close frames
    s_confs: list = field(default_factory=list)
    p_ok_frames: int = 0
    p_contact: float | None = None     # trained-model probabilities (None without a model)
    p_hs: float | None = None
    reach_c: list = field(default_factory=list)
    reach_s: list = field(default_factory=list)
    meets: list = field(default_factory=list)
    d_close: list = field(default_factory=list)
    base_c: float | None = None        # frozen at the first close frame
    base_s: float | None = None
    ambiguous_time: float = 0.0

    def add_reason(self, r: str) -> None:
        if r and r not in self.reasons:
            self.reasons.append(r)

    @property
    def close_ratio(self) -> float:
        return self.close_frames / self.visible_frames if self.visible_frames else 0.0

    @property
    def median_speed(self) -> float | None:
        return float(np.median(self.speeds)) if self.speeds else None


@dataclass
class TouchEpisode:
    """Fingertip-touch evidence of one customer-staff pair while their palm points are NOT close (see
    contact.touch_*). Kept apart from Episode so the trained classifier's inputs are never changed by it."""
    start_t: float
    start_frame: int
    last_t: float
    in_touch: bool = True
    frames: int = 0
    visible: int = 0
    touch_time: float = 0.0
    last_touch_t: float | None = None
    last_touch_frame: int | None = None
    last_evidence_t: float = 0.0
    apart_since: float | None = None
    missing_since: float | None = None
    min_touch: float = np.inf
    confs: list[float] = field(default_factory=list)
    speeds: list[float] = field(default_factory=list)
    last_rel: np.ndarray | None = None
    last_rel_t: float | None = None
    reach_c: list = field(default_factory=list)     # same names as Episode so PairStateMachine.paired() applies
    reach_s: list = field(default_factory=list)
    base_c: float | None = None
    base_s: float | None = None
    hands: set = field(default_factory=set)
    reasons: list[str] = field(default_factory=list)
    level: int = LEVEL_NONE
    event_id: str | None = None
    at_c: list = field(default_factory=list)     # meeting point on the customer's / staff hand (1 = fingertip)
    at_s: list = field(default_factory=list)
    c_confs: list = field(default_factory=list)  # wrist confidence of each touching hand
    s_confs: list = field(default_factory=list)
    near_time: float = 0.0                       # time the hands were near (segments <= touch_near_ratio)
    min_near: float = np.inf
    near_at_c: list = field(default_factory=list)   # meeting points / relative speed while near (not touching)
    near_at_s: list = field(default_factory=list)
    near_speeds: list = field(default_factory=list)
    near_last_rel: np.ndarray | None = None
    near_last_t: float | None = None

    visible_at_last_touch: int = 0

    @property
    def ratio(self) -> float:
        """Touching share of the visible frames up to the last touching frame (the separation tail that ends the
        episode is not part of the touch)."""
        return self.frames / self.visible_at_last_touch if self.visible_at_last_touch else 0.0

    @property
    def median_speed(self) -> float | None:
        return float(np.median(self.speeds)) if self.speeds else None


class PairStateMachine:
    def __init__(self, customer_id: int, staff_id: int, cfg: AppConfig, fps: float, ledger: EventLedger,
                 model: InteractionClassifier | None = None):
        self.model = model
        self.k = cfg.classifier
        self.customer_id = customer_id
        self.staff_id = staff_id
        self.cfg = cfg
        self.c, self.h = cfg.contact, cfg.handshake
        self.frame_dt = 1.0 / fps
        self.ledger = ledger
        self.state = State.NO_ACTIVE_INTERACTION
        self.episode: Episode | None = None
        self.last_episode: Episode | None = None
        self.cooldown_until = -np.inf
        self.track_changes: list[str] = []
        self.last_frame: PairFrame | None = None
        self._prev_d: float | None = None
        self.touch: TouchEpisode | None = None
        self.last_touch: TouchEpisode | None = None
        self._cooldown_owner: str | None = None    # "palm" | "touch": which episode the current COOLDOWN follows

    @property
    def key(self) -> tuple[int, int]:
        return self.customer_id, self.staff_id

    @property
    def event_id(self) -> str | None:
        cooling = self.state == State.COOLDOWN
        ep = self.episode or (self.last_episode if cooling and self._cooldown_owner == "palm" else None)
        if ep is not None and ep.event_id:
            return ep.event_id
        # only the episode the current cooldown belongs to - never a stale record of an earlier episode
        tp = self.touch or (self.last_touch if cooling and self._cooldown_owner == "touch" else None)
        return tp.event_id if tp else None

    @property
    def idle(self) -> bool:
        return self.episode is None and self.touch is None and self.state != State.COOLDOWN

    # ------------------------------------------------------------------ main update
    def update(self, pf: PairFrame) -> State:
        self.last_frame = pf
        c = self.c
        if self.episode is None:
            if self.state == State.COOLDOWN and pf.t >= self.cooldown_until:
                self.state = State.NO_ACTIVE_INTERACTION
            near = pf.d_ratio is not None and pf.d_ratio <= c.enter_ratio
            starts = ((pf.evidence == Evidence.BOTH and near) or pf.evidence == Evidence.PARTIAL_NEAR
                      or (pf.evidence == Evidence.NONE and pf.reason == "ambiguous_pairing" and near))
            if not starts:
                touch_state = self._touch_update(pf)
                if touch_state is not None:
                    self.state = touch_state
                elif self.state != State.COOLDOWN:
                    self.state = self._idle_state(pf)
                return self.state
            handed = self._touch_absorb()   # palms close: the classifier-based episode takes over from a touch
            if self.state == State.COOLDOWN and self.last_episode is not None:
                self._reopen(pf)
            else:
                why = ("hand proxies within enter distance" if pf.evidence == Evidence.BOTH
                       else f"proximity with incomplete evidence ({pf.reason})")
                self.episode = Episode(pf.t, pf.frame_idx, pf.t, why, last_evidence_t=pf.t,
                                       last_evidence_frame=pf.frame_idx)
                if handed is not None:   # contact already confirmed by the touch: keep it (no drop back to POSSIBLE)
                    self.episode.level, self.episode.event_id = LEVEL_CONTACT, handed.event_id
                    self.episode.start_t, self.episode.start_frame = handed.start_t, handed.start_frame
                    self.ledger.reopen(handed.event_id)
        ep = self.episode
        dt = min(max(pf.t - ep.last_t, 0.0), 2 * self.frame_dt) or self.frame_dt
        ep.last_t = pf.t
        pf = self._engaged_hand_check(ep, pf)

        if pf.evidence == Evidence.BOTH:
            ep.missing_since = None
            ep.visible_frames += 1
            ep.last_evidence_t, ep.last_evidence_frame = pf.t, pf.frame_idx
            limit = c.exit_ratio if ep.in_contact else c.enter_ratio
            ep.in_contact = pf.d_ratio <= limit
            if ep.in_contact:
                ep.apart_since = None
                self._record_close(ep, pf, dt)
            elif self._apart(ep, pf):
                return self.state
        elif pf.evidence == Evidence.PARTIAL_NEAR:
            ep.missing_since, ep.apart_since = None, None
            ep.partial_time += dt
            ep.last_evidence_t, ep.last_evidence_frame = pf.t, pf.frame_idx
            ep.add_reason(pf.reason)
            if ep.event_id is None and ep.partial_time >= c.min_contact_s:
                self._open_event(ep, UNCERTAIN, pf)  # visible one-sided reach held long enough
        elif pf.evidence == Evidence.PARTIAL_FAR:
            ep.missing_since = None
            ep.last_evidence_t, ep.last_evidence_frame = pf.t, pf.frame_idx
            ep.in_contact = False
            if ep.close_frames:
                ep.add_reason(pf.reason)
            if self._apart(ep, pf):
                return self.state
        else:  # NONE
            if ep.missing_since is None:
                ep.missing_since = pf.t
            if ep.close_frames or ep.partial_time or ep.ambiguous_time:
                ep.add_reason(pf.reason)
            if pf.reason == "ambiguous_pairing":
                ep.ambiguous_time += dt
            if pf.t - ep.missing_since > c.missing_tolerance_s:
                self._end(pf, f"evidence_lost ({pf.reason})")
                return self.state

        self._classify(ep, pf)
        if pf.t - ep.start_t > c.max_episode_s:
            self._end(pf, "timeout (max_episode_s)")
            return self.state

        if pf.evidence in (Evidence.NONE, Evidence.PARTIAL_NEAR):
            self.state = State.UNCERTAIN
        elif ep.level == LEVEL_HANDSHAKE:
            self.state = State.LIKELY_HANDSHAKE
        elif ep.level == LEVEL_CONTACT:
            self.state = State.LIKELY_HAND_CONTACT
        else:
            self.state = State.POSSIBLE_HAND_CONTACT
        return self.state

    def _engaged_hand_check(self, ep: Episode, pf: PairFrame) -> PairFrame:
        """If a hand that was in contact disappears, the person's OTHER (visible) hand must not be
        read as "the hands separated" - the frame is missing evidence for the engaged hand."""
        if pf.evidence != Evidence.BOTH or ep.engaged is None or pf.d_ratio <= self.c.enter_ratio:
            return pf
        c_side, s_side = ep.engaged
        for side, visible, role in ((c_side, pf.c_sides, "customer"), (s_side, pf.s_sides, "staff")):
            if side not in visible:
                return PairFrame(pf.frame_idx, pf.t, Evidence.NONE, f"{role}_wrist_not_visible",
                                 scale=pf.scale, crossing_active=pf.crossing_active)
        return pf

    def _idle_state(self, pf: PairFrame) -> State:
        d = pf.d_ratio if pf.evidence == Evidence.BOTH else None
        closing = (d is not None and self._prev_d is not None and d < self._prev_d
                   and d <= self.c.approach_ratio)
        # hands that came close and now hover (not moving apart) stay "approaching": no flicker to NO_ACTIVE
        holding = (d is not None and self.state == State.HAND_APPROACHING and d <= self.c.approach_ratio
                   and (self._prev_d is None or d <= self._prev_d + 0.05))
        self._prev_d = d
        if pf.crossing_active:
            return State.BOUNDARY_CROSSED
        if pf.approaching or closing or holding:
            return State.HAND_APPROACHING
        return State.NO_ACTIVE_INTERACTION

    # ------------------------------------------------------------------ fingertip touch (palms apart)
    def _touch_update(self, pf: PairFrame) -> State | None:
        """Advance the fingertip-touch episode; returns the state it implies, or None when there is none."""
        c = self.c
        tp = self.touch
        both = pf.evidence == Evidence.BOTH and pf.touch_ratio is not None
        if tp is None:
            if self._cooldown_owner == "palm" and self.state == State.COOLDOWN and pf.t < self.cooldown_until:
                return None   # a palm episode just ended: renewed contact re-opens THAT episode (cooldown rule)
            if not (both and pf.touch_ratio <= c.touch_near_ratio):
                return None
            last = self.last_touch
            if last is not None and self.state == State.COOLDOWN and pf.t < self.cooldown_until:
                tp = self.touch = last            # re-touch inside the cooldown continues the same episode
                self.last_touch = None
                tp.apart_since = tp.missing_since = None
                if tp.event_id is not None:
                    self.ledger.reopen(tp.event_id)
            else:
                tp = self.touch = TouchEpisode(pf.t, pf.frame_idx, pf.t, last_evidence_t=pf.t)
        dt = min(max(pf.t - tp.last_t, 0.0), 2 * self.frame_dt) or self.frame_dt
        tp.last_t = pf.t
        if not both:
            if tp.missing_since is None:
                tp.missing_since = pf.t
            if pf.reason and tp.frames and pf.reason not in tp.reasons:
                tp.reasons.append(pf.reason)
            if pf.t - tp.missing_since > c.missing_tolerance_s:
                self._touch_end(pf, f"evidence_lost ({pf.reason})")
                return State.COOLDOWN
            return State.UNCERTAIN if tp.frames else State.POSSIBLE_HAND_CONTACT
        tp.missing_since = None
        tp.visible += 1
        tp.last_evidence_t = pf.t
        tp.in_touch = pf.touch_ratio <= (c.touch_exit_ratio if tp.in_touch else c.touch_enter_ratio)
        near = pf.touch_ratio <= c.touch_near_ratio
        if near:
            tp.near_time += dt
            tp.min_near = min(tp.min_near, pf.touch_ratio)
            if pf.t_c_reach is not None and not tp.in_touch:
                tp.reach_c.append(pf.t_c_reach)   # reach while near also shows whether both people reached out
                tp.reach_s.append(pf.t_s_reach)
                if not tp.frames and tp.base_c is None:
                    tp.base_c, tp.base_s = pf.t_c_base, pf.t_s_base
            if not tp.in_touch and pf.touch_at is not None:
                tp.near_at_c.append(pf.touch_at[0])
                tp.near_at_s.append(pf.touch_at[1])
                rel = (pf.t_c_hand.proxy - pf.t_s_hand.proxy) / pf.scale
                if tp.near_last_rel is not None and 0 < pf.t - tp.near_last_t <= 3 * self.frame_dt:
                    tp.near_speeds.append(float(np.hypot(*(rel - tp.near_last_rel))) / (pf.t - tp.near_last_t))
                tp.near_last_rel, tp.near_last_t = rel, pf.t
        if tp.in_touch:
            tp.apart_since = None
            if tp.frames == 0 and tp.base_c is None:
                tp.base_c, tp.base_s = pf.t_c_base, pf.t_s_base
            tp.frames += 1
            tp.touch_time += dt
            tp.last_touch_t, tp.last_touch_frame = pf.t, pf.frame_idx
            tp.visible_at_last_touch = tp.visible
            tp.min_touch = min(tp.min_touch, pf.touch_ratio)
            tp.confs.append(min(pf.t_c_hand.conf, pf.t_s_hand.conf))
            tp.c_confs.append(pf.t_c_hand.conf)
            tp.s_confs.append(pf.t_s_hand.conf)
            tp.hands.update({f"customer:{pf.t_c_hand.side}", f"staff:{pf.t_s_hand.side}"})
            if pf.t_c_reach is not None:
                tp.reach_c.append(pf.t_c_reach)
                tp.reach_s.append(pf.t_s_reach)
            if pf.touch_at is not None:
                tp.at_c.append(pf.touch_at[0])
                tp.at_s.append(pf.touch_at[1])
            rel = (pf.t_c_hand.proxy - pf.t_s_hand.proxy) / pf.scale
            if tp.last_rel is not None and tp.last_rel_t is not None and 0 < pf.t - tp.last_rel_t <= 3 * self.frame_dt:
                tp.speeds.append(float(np.hypot(*(rel - tp.last_rel))) / (pf.t - tp.last_rel_t))
            tp.last_rel, tp.last_rel_t = rel, pf.t
        elif near:
            tp.apart_since = None
        else:
            if tp.apart_since is None:
                tp.apart_since = pf.t
            if pf.t - tp.apart_since >= c.separation_s:
                self._touch_end(pf, "hands_separated")
                return State.COOLDOWN
        if tp.level == LEVEL_NONE:
            ok, _ = self._touch_ok(tp)
            if ok:
                tp.level = LEVEL_CONTACT
                self._touch_event(tp, CONTACT, pf)
        if tp.event_id is not None:
            self._touch_sync(tp)
        return State.LIKELY_HAND_CONTACT if tp.level == LEVEL_CONTACT else State.POSSIBLE_HAND_CONTACT

    def _touch_ok(self, tp: TouchEpisode) -> tuple[bool, list[str]]:
        c = self.c
        why = []
        if tp.touch_time < c.min_contact_s:
            why.append(f"fingertips close {tp.touch_time:.2f}s < {c.min_contact_s}s")
        if tp.ratio < c.min_close_ratio:
            why.append(f"touch ratio {tp.ratio:.2f} < {c.min_close_ratio}")
        ms = tp.median_speed
        if ms is not None and ms > c.max_relative_speed:
            why.append(f"relative speed {ms:.2f} > {c.max_relative_speed} forearms/s")
        paired, msg = self.touch_paired(tp)
        if not paired:
            why.append(msg)
        hidden = self.hands_hidden(tp.c_confs, tp.s_confs)
        if hidden:
            why.append(hidden)
        return not why, why

    def near_paired(self, tp: TouchEpisode) -> bool:
        """Both people offered their hands while near: each moved a hand out (``paired``), or both hands are held out
        toward each other (>= engaged_reach), fingertips facing (closest points >= touch_tip_min) and steady."""
        if self.paired(tp)[0]:
            return True
        c = self.c
        if not (tp.reach_c and tp.near_at_c and tp.near_speeds):
            return False
        ext = min(float(np.median(tp.reach_c)), float(np.median(tp.reach_s)))
        tip = min(float(np.median(tp.near_at_c)), float(np.median(tp.near_at_s)))
        return ext >= c.engaged_reach and tip >= c.touch_tip_min and float(np.median(tp.near_speeds)) <= c.touch_hold_max_speed

    def touch_paired(self, tp: TouchEpisode) -> tuple[bool, str]:
        """Paired-hand evidence for a fingertip touch: both people moved a hand out (``paired``), OR a HELD touch -
        both hands extended toward the other person (>= contact.engaged_reach), meeting tip to tip
        (>= contact.touch_tip_min along each hand) and steady together (<= contact.touch_hold_max_speed).
        A hand reaching over a resting/typing hand crosses it mid-hand and keeps moving, so it fails both."""
        ok, msg = self.paired(tp)
        if ok:
            return True, ""
        c = self.c
        if not (tp.reach_c and tp.at_c):
            return False, msg
        ext = min(float(np.median(tp.reach_c)), float(np.median(tp.reach_s)))
        tip = min(float(np.median(tp.at_c)), float(np.median(tp.at_s)))
        ms = tp.median_speed
        held = ext >= c.engaged_reach and tip >= c.touch_tip_min and ms is not None and ms <= c.touch_hold_max_speed
        if held:
            return True, ""
        return False, (f"{msg}; not a held fingertip touch (hands extended {ext:.2f}, meet at {tip:.2f} of the hand, "
                       f"relative speed {'n/a' if ms is None else f'{ms:.2f}'})")

    def _touch_event(self, tp: TouchEpisode, etype: str, pf: PairFrame) -> None:
        shared = self.ledger.find_concurrent_contact(self.customer_id, self.staff_id, tp.start_t, self.c.cooldown_s)
        if shared is not None:
            tp.event_id = shared.event_id
            if shared.status == "closed":
                self.ledger.reopen(shared.event_id)
            if self.ledger.rank(etype) > self.ledger.rank(shared.event_type):
                self.ledger.retype(shared.event_id, etype)
            return
        ev = self.ledger.open(etype, tp.start_t, tp.start_frame, pf.t, customer_track_id=self.customer_id,
                              staff_track_id=self.staff_id,
                              start_reason="fingertips met: hand segments within contact.touch_enter_ratio, palms apart",
                              interaction_state=(State.UNCERTAIN if etype == UNCERTAIN
                                                 else State.LIKELY_HAND_CONTACT).value)
        ev.track_id_changes = list(self.track_changes)
        tp.event_id = ev.event_id

    def _touch_sync(self, tp: TouchEpisode) -> None:
        ev = self.ledger.get(tp.event_id)
        if ev.event_type not in (CONTACT, UNCERTAIN):
            return   # an episode with clasped palms owns this record (handshake): keep its description
        if tp.confs:
            ev.keypoint_conf_mean = float(np.mean(tp.confs))
            ev.keypoint_conf_min = float(np.min(tp.confs))
            ev.evidence_score = ev.keypoint_conf_mean * tp.ratio
        if np.isfinite(min(tp.min_touch, tp.min_near)):
            ev.minimum_hand_distance_forearms = min(tp.min_touch, tp.min_near)
        ev.hands = sorted(tp.hands | set(ev.hands))
        ev.uncertainty_reason = list(tp.reasons)
        ms = tp.median_speed
        closest = min(tp.min_touch, tp.min_near)
        ev.evidence_summary = "; ".join(
            [f"fingertip touch {tp.touch_time:.2f}s over {tp.frames} frames", f"hands near {tp.near_time:.2f}s",
             f"touch ratio {tp.ratio:.2f}", f"closest hand segments {closest:.2f} forearms"]
            + ([f"median relative speed {ms:.2f} forearms/s"] if ms is not None else [])
            + [f"paired-hand evidence: {'moved out' if self.paired(tp)[0] else ('held tip-to-tip touch' if self.touch_paired(tp)[0] else 'no')}",
               "decided by rules (contact.touch_*), not by the trained classifier"])
        score = min(1.0, tp.touch_time / self.c.min_contact_s) * tp.ratio * (float(np.mean(tp.confs)) if tp.confs else 0)
        ev.contact_score = round(100 * score)
        ev.handshake_score = None
        if ev.event_type == CONTACT:
            ev.interpretation = (f"Likely hand contact ({tp.touch_time:.1f}s) - fingertips met and stayed together, "
                                 f"both hands held out to each other; palms did not clasp, so not a handshake "
                                 f"(contact {score:.0%})")
        elif tp.frames == 0:
            ev.interpretation = (f"Uncertain - possible hand contact not confirmed: both reached out and the hands came "
                                 f"within {tp.min_near:.2f} forearms for {tp.near_time:.1f}s, but never touched")
        else:
            ev.interpretation = (f"Uncertain: fingertips came close for {tp.touch_time:.1f}s - "
                                 f"{'; '.join(tp.reasons) or 'evidence insufficient'} (contact {score:.0%})")

    def _touch_end(self, pf: PairFrame, reason: str) -> None:
        tp = self.touch
        end_t = tp.last_touch_t if tp.last_touch_t is not None else tp.last_evidence_t
        end_f = tp.last_touch_frame if tp.last_touch_frame is not None else pf.frame_idx
        if tp.event_id is None:
            ok, why = self._touch_ok(tp)
            slow = tp.median_speed is None or tp.median_speed <= self.c.max_relative_speed
            if tp.touch_time >= self.c.uncertain_min_s and slow:
                for w in why:
                    if w not in tp.reasons:
                        tp.reasons.append(w)
                self._touch_event(tp, UNCERTAIN, pf)
            elif tp.frames == 0 and tp.near_time >= self.c.min_contact_s and self.near_paired(tp):
                # both reached out and held their hands near each other, but they never touched
                tp.reasons.append(f"hands came within {tp.min_near:.2f} forearms for {tp.near_time:.1f}s but never touched")
                self._touch_event(tp, UNCERTAIN, pf)
            else:
                self.ledger.suppress(kind="contact", reason="brief_fingertip_proximity" if slow else
                                     "fast_relative_motion (pass/wave)", start_timestamp=tp.start_t,
                                     end_timestamp=max(end_t, tp.start_t), customer_track_id=self.customer_id,
                                     staff_track_id=self.staff_id,
                                     detail=f"fingertips close {tp.touch_time:.2f}s, ended: {reason}")
        if tp.event_id is not None:
            ev = self.ledger.get(tp.event_id)
            if ev.event_type == CONTACT and not ev.shared:
                ok, why = self._touch_ok(tp)   # final decision on the whole episode (may correct the live one)
                if not ok:
                    for w in why:
                        if w not in tp.reasons:
                            tp.reasons.append(f"final evidence: {w}")
                    self.ledger.retype(tp.event_id, UNCERTAIN)
                    ev.interaction_state = State.UNCERTAIN.value
            self._touch_sync(tp)
            if ev.status == "open":
                self.ledger.close(tp.event_id, end_t, end_f, reason)
        self.ledger.episodes.append({
            "customer_track_id": self.customer_id, "staff_track_id": self.staff_id, "kind": "touch",
            "start": round(tp.start_t, 3), "end": round(max(end_t, tp.start_t), 3), "end_reason": reason,
            "event_id": tp.event_id,
            "outcome": self.ledger.get(tp.event_id).event_type if tp.event_id else "suppressed",
            "p_contact": None, "p_handshake": None, "features": {}})
        self.last_touch, self.touch = tp, None
        self.state = State.COOLDOWN
        self.cooldown_until = pf.t + self.c.cooldown_s
        self._cooldown_owner = "touch"

    def _touch_absorb(self) -> None:
        """Palms came close during a touch: the classifier-based Episode continues the interaction. A touch record
        already opened stays and is shared by that Episode (ledger.find_concurrent_contact)."""
        tp = self.touch
        handed = tp if tp is not None and tp.level >= LEVEL_CONTACT and tp.event_id is not None else None
        if tp is not None and tp.event_id is not None:
            self._touch_sync(tp)
            ev = self.ledger.get(tp.event_id)
            if ev.status == "open":   # closed now; the palm episode re-opens / shares it if it confirms contact
                end_t = tp.last_touch_t if tp.last_touch_t is not None else tp.last_t
                end_f = tp.last_touch_frame if tp.last_touch_frame is not None else tp.start_frame
                self.ledger.close(tp.event_id, end_t, end_f, "palms_came_close (continued by the palm episode)")
        self.touch = None
        return handed

    def _apart(self, ep: Episode, pf: PairFrame) -> bool:
        """Count a separated frame; end the episode once hands have been apart for separation_s."""
        if ep.apart_since is None:
            ep.apart_since = pf.t
        if pf.t - ep.apart_since >= self.c.separation_s:
            self._end(pf, "hands_separated")
            return True
        return False

    def _record_close(self, ep: Episode, pf: PairFrame, dt: float) -> None:
        if ep.close_frames == 0:
            ep.base_c, ep.base_s = pf.c_base, pf.s_base
        if pf.c_reach is not None:
            ep.reach_c.append(pf.c_reach)
            ep.reach_s.append(pf.s_reach)
            ep.meets.append(pf.meet)
        ep.d_close.append(pf.d_ratio)
        ep.close_frames += 1
        ep.close_time += dt
        ep.last_close_t, ep.last_close_frame = pf.t, pf.frame_idx
        ep.min_d_ratio = min(ep.min_d_ratio, pf.d_ratio)
        ep.min_d_px = min(ep.min_d_px, pf.d_px)
        if pf.wrist_d_px is not None:
            ep.min_wrist_px = min(ep.min_wrist_px, pf.wrist_d_px)
        ep.confs.append(pf.conf)
        ep.c_confs.append(pf.c_hand.conf)
        ep.s_confs.append(pf.s_hand.conf)
        ep.hands.update({f"customer:{pf.c_hand.side}", f"staff:{pf.s_hand.side}"})
        ep.engaged = (pf.c_hand.side, pf.s_hand.side)
        rel = (pf.c_hand.proxy - pf.s_hand.proxy) / pf.scale
        if pf.d_ratio <= self.c.enter_ratio:
            # handshake motion features use only frames where the hands are properly together, not the
            # arrival/departure frames inside the exit-hysteresis band
            ep.rel.append(rel)
            ep.mids.append((pf.c_hand.proxy + pf.s_hand.proxy) / 2 / pf.scale)
        if ep.last_rel is not None and ep.last_rel_t is not None and 0 < pf.t - ep.last_rel_t <= 3 * self.frame_dt:
            ep.speeds.append(float(np.hypot(*(rel - ep.last_rel))) / (pf.t - ep.last_rel_t))
        ep.last_rel, ep.last_rel_t = rel, pf.t
        ec, es = pf.c_hand.extension, pf.s_hand.extension
        if ec is not None and es is not None:
            ep.ext_known += 1
            if ec >= self.h.min_extension_ratio and es >= self.h.min_extension_ratio:
                ep.ext_both += 1

    # ------------------------------------------------------------------ classification
    def _contact_ok(self, ep: Episode) -> tuple[bool, list[str]]:
        c = self.c
        why = []
        if ep.close_time < c.min_contact_s:
            why.append(f"close {ep.close_time:.2f}s < {c.min_contact_s}s")
        if ep.close_ratio < c.min_close_ratio:
            why.append(f"close ratio {ep.close_ratio:.2f} < {c.min_close_ratio}")
        ms = ep.median_speed
        if ms is not None and ms > c.max_relative_speed:
            why.append(f"relative speed {ms:.2f} > {c.max_relative_speed} forearms/s")
        if "identity_ambiguous" in ep.reasons or "ambiguous_pairing" in ep.reasons:
            why.append("identity ambiguous")
        hidden = self.hands_hidden(ep.c_confs, ep.s_confs)
        if hidden:
            why.append(hidden)
        return not why, why

    def hands_hidden(self, c_confs: list, s_confs: list) -> str:
        """Non-empty reason when NEITHER person's hand is clearly visible during the contact (occlusion)."""
        if len(c_confs) < 3 or len(s_confs) < 3:
            return ""
        mc, ms = float(np.median(c_confs)), float(np.median(s_confs))
        if max(mc, ms) < self.c.visible_conf:
            return (f"hands hidden / low keypoint confidence (customer wrist {mc:.2f}, staff wrist {ms:.2f} "
                    f"< {self.c.visible_conf})")
        return ""

    def handshake_score(self, ep: Episode) -> tuple[float, dict]:
        """Weighted handshake evidence in [0, 1] and its components (see config `handshake`).

        duration     close time ramps from duration_start_s (0) to duration_full_s (1)
        pumping      (direction reversals of the detrended hand midpoint - 1) / (min_reversals - 1)
        comovement   1 - spread of the hands' relative offset / max_relative_std
        extension    fraction of close frames with both arms extended; 0.5 (neutral) when the arms
                     are not measurable in enough frames (top-down or close-up views)
        """
        h = self.h
        dur = float(np.clip((ep.close_time - h.duration_start_s) / (h.duration_full_s - h.duration_start_s), 0, 1))
        mids = np.asarray(ep.mids)
        rev = 0
        if len(mids) >= 5:
            kernel = np.ones(3) / 3  # 3-sample moving average removes single-frame keypoint jitter
            sm = np.column_stack([np.convolve(mids[:, i], kernel, mode="valid") for i in range(2)])
            rev = max(count_reversals(c, h.min_amplitude_ratio) for c in detrended_axes(sm))
        # one reversal is produced by the hands arriving and leaving; pumping credit starts at the second
        pump = float(np.clip((rev - 1) / max(h.min_reversals - 1, 1), 0, 1))
        rel = np.asarray(ep.rel)
        rel_std = float(np.sqrt(rel.var(axis=0).sum())) if len(rel) > 1 else 0.0
        comove = float(np.clip(1 - rel_std / h.max_relative_std, 0, 1))
        # "moving together" only counts motion of the joined hands: midpoint steps in which the hands did not
        # move relative to each other (converging/separating hands and a frozen pose earn nothing)
        travel = 0.0
        if len(mids) >= 2:
            d_mid = np.hypot(*np.diff(mids, axis=0).T)
            d_rel = np.hypot(*np.diff(rel, axis=0).T)
            travel = float(d_mid[d_rel < 0.5 * d_mid].sum())
        comove *= float(np.clip(travel / (2 * h.min_amplitude_ratio), 0, 1))
        observed = ep.ext_known / max(ep.close_frames, 1)
        ext = ep.ext_both / ep.ext_known if observed >= h.extension_min_observed and ep.ext_known else 0.5
        score = h.w_duration * dur + h.w_pumping * pump + h.w_comovement * comove + h.w_extension * ext
        return float(score), {"duration": dur, "pumping": pump, "reversals": rev, "comovement": comove,
                              "extension": ext, "extension_observed": observed}

    def contact_score(self, ep: Episode) -> float:
        """Evidence that hands were in contact, in [0, 1]: persistence x closeness ratio x keypoint confidence."""
        if not ep.close_frames:
            return 0.0
        persist = min(1.0, ep.close_time / self.c.min_contact_s)
        return float(persist * ep.close_ratio * float(np.mean(ep.confs)))

    def _update_scores(self, ep: Episode) -> None:
        score, parts = self.handshake_score(ep)
        # the motion pattern is the most pumping seen during the episode (the count fluctuates as frames are added)
        ep.max_reversals = max(ep.max_reversals, parts.get("reversals", 0))
        parts = dict(parts, reversals=ep.max_reversals)
        if score >= ep.hs_score or ep.level < LEVEL_HANDSHAKE:
            # keep the peak: the reported rule score is the one that justified the classification
            ep.hs_score, ep.hs_parts = max(score, ep.hs_score), (parts if score >= ep.hs_score else ep.hs_parts)
        if self.model is not None:
            # Shown percentages == decision values: the trained probabilities combined with the assignment's
            # explicit rules (paired-hand evidence; handshake needs a motion pattern), so a card never says
            # "handshake 91%" for an episode that is not a handshake.
            feats = episode_features(ep, parts, self.h.clasp_distance)
            mutual = 1.0 if self.paired(ep)[0] else self.k.one_sided_factor
            motion = min(1.0, parts.get("reversals", 0) / max(self.h.min_motion_reversals, 1))
            ep.p_contact = self.model.contact.prob(feats)
            ep.p_hs = self.model.handshake.prob(feats) * mutual * motion * self.clasp_factor(ep)
            ep.p_ok_frames = ep.p_ok_frames + 1 if ep.p_contact > self.k.contact_threshold else 0

    def paired(self, ep: Episode) -> tuple[bool, str]:
        """Mutual reach (required for a HANDSHAKE): both people brought a hand out toward the other.

        Touching someone's still hand is hand contact, but one cannot shake a passive hand. A hand is engaged if its
        median reach during the close frames exceeds its smallest reach before the episode by min_engagement, or if
        it was already held out before the episode (interaction already under way), or - when its earlier position
        is unknown - if it is extended at least engaged_reach toward the other person.
        """
        c = self.c
        out = []
        for role, reach, base in (("customer", ep.reach_c, ep.base_c), ("staff", ep.reach_s, ep.base_s)):
            if not reach:
                return False, "hand positions unknown"
            r = float(np.median(reach))
            ok = ((r - base >= c.min_engagement or base >= c.engaged_prior_reach) if base is not None
                  else r >= c.engaged_reach)
            if not ok:
                out.append(role)
        return (not out), ("" if not out else f"{' and '.join(out)} hand did not reach out (one-sided)")

    def _is_contact(self, ep: Episode, final: bool = False) -> bool:
        if self.model is None:
            return True
        if final:
            return ep.p_contact > self.k.contact_threshold
        # live: the probability must stay above the threshold for stable_s, so a one-frame spike opens no record
        return ep.p_ok_frames >= max(1, round(self.k.stable_s / self.frame_dt))

    def clasp_frac(self, ep: Episode) -> float:
        """Share of the close frames in which the two hand centres are clasped together (palm to palm) - the PEAK
        share reached once the hands had been together for handshake.clasp_window_s. Like pump reversals, the
        clasp is evidence that happened: frames added later (hands sliding or loosening before release) must not
        dilute an established grip below the threshold."""
        cur = float(np.mean(np.asarray(ep.d_close) < self.h.clasp_distance)) if ep.d_close else 0.0
        if ep.close_time >= self.h.clasp_window_s:
            ep.max_clasp = max(ep.max_clasp, cur)
        return max(cur if ep.close_time < self.h.clasp_window_s else 0.0, ep.max_clasp)

    def clasp_factor(self, ep: Episode) -> float:
        """1 when the hands were clasped often enough for a handshake, falling to 0 when they never were.

        A handshake grips palm to palm; giving or taking an item (bottle, coin, phone, paper) keeps the hands apart
        by the item's width even when both reach out and move together.
        """
        return min(1.0, self.clasp_frac(ep) / max(self.h.min_clasp_frac, 1e-6))

    def _is_handshake(self, ep: Episode) -> bool:
        if self.clasp_frac(ep) < self.h.min_clasp_frac:
            return False
        if self.model is None:
            return self.paired(ep)[0] and ep.hs_score >= self.h.score_threshold
        return ep.p_hs > self.k.handshake_threshold and ep.p_contact > self.k.contact_threshold

    def _classify(self, ep: Episode, pf: PairFrame) -> None:
        if ep.close_frames:
            self._update_scores(ep)
            self.clasp_frac(ep)   # keeps the peak clasp share up to date every frame
        if ep.level == LEVEL_NONE and ep.close_frames:
            ok, _ = self._contact_ok(ep)
            if ok and self._is_contact(ep):
                ep.level = LEVEL_CONTACT
                if ep.event_id is None:
                    self._open_event(ep, CONTACT, pf)
                else:
                    self.ledger.retype(ep.event_id, CONTACT)
                self.ledger.get(ep.event_id).interaction_state = State.LIKELY_HAND_CONTACT.value
        if ep.level == LEVEL_CONTACT:
            if self._is_handshake(ep):
                ep.level = LEVEL_HANDSHAKE
                self.ledger.retype(ep.event_id, HANDSHAKE) if self.ledger.rank(HANDSHAKE) >= self.ledger.rank(
                    self.ledger.get(ep.event_id).event_type) else None
                self.ledger.get(ep.event_id).interaction_state = State.LIKELY_HANDSHAKE.value
        if ep.event_id is not None:
            self._sync_event(ep)

    # ------------------------------------------------------------------ events
    def _open_event(self, ep: Episode, etype: str, pf: PairFrame) -> None:
        shared = self.ledger.find_concurrent_contact(self.customer_id, self.staff_id, ep.start_t, self.c.cooldown_s)
        if shared is not None:
            # the same person is already in an interaction record at this moment (e.g. the other person's track
            # was fragmented by occlusion): continue that ONE record instead of opening a duplicate
            ep.event_id = shared.event_id
            shared.shared = True
            if shared.status == "closed":
                self.ledger.reopen(shared.event_id)
            note = f"merged concurrent episode C{self.customer_id}/S{self.staff_id}"
            if note not in shared.track_id_changes:
                shared.track_id_changes.append(note)
            if self.ledger.rank(etype) > self.ledger.rank(shared.event_type):
                self.ledger.retype(shared.event_id, etype)
            return
        ev = self.ledger.open(etype, ep.start_t, ep.start_frame, pf.t,
                              customer_track_id=self.customer_id, staff_track_id=self.staff_id,
                              start_reason=ep.start_reason,
                              interaction_state=(State.UNCERTAIN if etype == UNCERTAIN
                                                 else State.LIKELY_HAND_CONTACT).value)
        ev.track_id_changes = list(self.track_changes)
        ep.event_id = ev.event_id

    def _sync_event(self, ep: Episode) -> None:
        ev = self.ledger.get(ep.event_id)
        if np.isfinite(ep.min_d_px):
            ev.minimum_hand_distance_px = ep.min_d_px
            ev.minimum_hand_distance_forearms = ep.min_d_ratio
        if np.isfinite(ep.min_wrist_px):
            ev.minimum_wrist_distance_px = ep.min_wrist_px
        if ep.confs:
            ev.keypoint_conf_mean = float(np.mean(ep.confs))
            ev.keypoint_conf_min = float(np.min(ep.confs))
            ev.evidence_score = ev.keypoint_conf_mean * ep.close_ratio
        ev.hands = sorted(ep.hands)
        ev.uncertainty_reason = list(ep.reasons)
        ms = ep.median_speed
        summary = [f"close {ep.close_time:.2f}s over {ep.close_frames} frames",
                   f"close ratio {ep.close_ratio:.2f}"]
        if ms is not None:
            summary.append(f"median relative speed {ms:.2f} forearms/s")
        if ep.partial_time:
            summary.append(f"one-sided evidence {ep.partial_time:.2f}s")
        if ep.hs_parts:
            p = ep.hs_parts
            summary.append(f"handshake score {ep.hs_score:.0%} (duration {p['duration']:.2f}, pumping {p['pumping']:.2f} "
                           f"[{p['reversals']} reversals], moving together {p['comovement']:.2f}, "
                           f"arms extended {p['extension']:.2f})")
        summary += ep.notes
        ev.evidence_summary = "; ".join(summary)
        hs, ct = self.display_scores(ep)
        ev.handshake_score = None if hs is None else round(100 * hs)
        ev.contact_score = None if ct is None else round(100 * ct)
        ev.interpretation = self.interpret(ep, ev.event_type)

    def display_scores(self, ep: Episode) -> tuple[float | None, float | None]:
        """(handshake, contact) in 0..1: trained-model probabilities when a model is loaded, else rule scores."""
        if not ep.close_frames:
            return None, None
        if self.model is not None and ep.p_contact is not None:
            return ep.p_hs, ep.p_contact
        return ep.hs_score, self.contact_score(ep)

    def interpret(self, ep: Episode, etype: str) -> str:
        """Plain-language reading of the episode (pose evidence only - objects are never detected)."""
        hs_v, ct_v = self.display_scores(ep)
        hs = f"handshake {hs_v:.0%}" if hs_v is not None else "handshake n/a"
        ct = f"contact {ct_v:.0%}" if ct_v is not None else "contact n/a"
        if etype == HANDSHAKE:
            cues = [c for c, k in (("pumping motion", "pumping"), ("hands moving together", "comovement"),
                                   ("both arms extended", "extension")) if ep.hs_parts.get(k, 0) >= 0.6]
            return (f"Likely handshake ({hs_v:.0%}): hands together {ep.close_time:.1f}s"
                    + (f", {', '.join(cues)}" if cues else ""))
        if etype == CONTACT:
            if ep.close_frames and self.clasp_frac(ep) < self.h.min_clasp_frac:
                return (f"Likely hand contact ({ep.close_time:.1f}s) - hands met but never clasped, consistent with "
                        f"giving/taking an item or a light touch; not a handshake ({hs}, {ct})")
            if ep.close_time < 1.2:
                return (f"Likely brief hand-to-hand contact ({ep.close_time:.1f}s) - consistent with giving/taking "
                        f"an item or a touch; not a handshake ({hs}, {ct})")
            return (f"Likely sustained hand contact ({ep.close_time:.1f}s) without handshake motion - e.g. holding "
                    f"or passing an item ({hs}, {ct})")
        if any("hidden" in r for r in ep.reasons):
            return (f"Uncertain: the hands met ({ep.close_time:.1f}s) where they cannot be seen clearly - both wrists have "
                    f"low keypoint confidence (occluded), so contact cannot be confirmed ({ct})")
        if any(r.startswith("final evidence") for r in ep.reasons):
            return (f"Uncertain: hands were close {ep.close_time:.1f}s but only one person reached - most likely a "
                    f"one-sided reach, not contact ({ct})")
        if any("not_visible" in r for r in ep.reasons) and not ep.close_frames:
            return f"Uncertain: one person's hand is hidden; possible reach toward the other person ({ct})"
        if "ambiguous_pairing" in ep.reasons:
            return f"Uncertain: hand near two people, cannot tell who it interacts with ({ct}, {hs})"
        if ep.close_frames:
            return (f"Uncertain: hands came close for only {ep.close_time:.1f}s - possibly a brief touch, reach or "
                    f"near-miss ({ct}, {hs})")
        return f"Uncertain interaction ({ct})"

    def _reopen(self, pf: PairFrame) -> None:
        ep = self.last_episode
        gap = pf.t - (ep.last_close_t if ep.last_close_t is not None else ep.last_evidence_t)
        ep.notes.append(f"re-contact after {gap:.2f}s gap merged (cooldown)")
        ep.apart_since = ep.missing_since = None
        ep.in_contact = False
        if ep.suppressed_ref is not None:  # it was suppressed at its end; now it continues instead
            self.ledger.unsuppress(ep.suppressed_ref)
            ep.suppressed_ref = None
        ep.last_t = pf.t
        if ep.event_id is not None:
            self.ledger.reopen(ep.event_id)
        self.episode, self.last_episode = ep, None

    def _end(self, pf: PairFrame, reason: str) -> None:
        ep = self.episode
        end_t = ep.last_close_t if ep.last_close_t is not None else ep.last_evidence_t
        end_f = ep.last_close_frame if ep.last_close_frame is not None else ep.last_evidence_frame
        c = self.c
        if ep.close_frames:
            self._update_scores(ep)
        if self.model is not None and ep.event_id is not None and ep.level >= LEVEL_CONTACT:
            # final decision on the complete episode evidence (may correct the live classification)
            hidden = self.hands_hidden(ep.c_confs, ep.s_confs)
            final = ((HANDSHAKE if self._is_handshake(ep) else CONTACT)
                     if self._is_contact(ep, final=True) and not hidden else UNCERTAIN)
            if final == UNCERTAIN:
                why = hidden or f"contact score {ep.p_contact:.0%}"
                ep.add_reason(f"final evidence does not support contact ({why})")
            ev_f = self.ledger.get(ep.event_id)
            if ev_f.shared and self.ledger.rank(final) < self.ledger.rank(ev_f.event_type):
                final = ev_f.event_type   # a shared record keeps the strongest outcome of its episodes
            self.ledger.retype(ep.event_id, final)
            self.ledger.get(ep.event_id).interaction_state = {
                HANDSHAKE: State.LIKELY_HANDSHAKE, CONTACT: State.LIKELY_HAND_CONTACT}.get(final, State.UNCERTAIN).value
        one_sided = (self.model is not None and ep.event_id is None and ep.close_frames and not ep.reasons
                     and ep.p_contact < self.k.uncertain_min_prob)
        if one_sided:
            # a hand near the other person's hand with no paired-hand evidence (e.g. reaching past a typing hand)
            ep.suppressed_ref = self.ledger.suppress(
                kind="contact", reason="no_paired_hand_evidence (one-sided reach)", start_timestamp=ep.start_t,
                end_timestamp=max(end_t, ep.start_t), customer_track_id=self.customer_id,
                staff_track_id=self.staff_id,
                detail=f"close {ep.close_time:.2f}s; hands never overlapped (no paired-hand contact evidence)"
                       + (f", contact score {ep.p_contact:.2f}" if ep.p_contact is not None else ""))
        elif ep.event_id is None:
            ms = ep.median_speed
            slow = ms is None or ms <= c.max_relative_speed
            if ep.close_time >= c.uncertain_min_s and slow:
                ok, why = self._contact_ok(ep)
                for w in why:
                    ep.add_reason(f"insufficient evidence: {w}")
                self._open_event(ep, UNCERTAIN, pf)
            elif ep.close_frames >= 2 and ep.reasons and slow:
                # close evidence interrupted by occlusion / identity problems: cannot decide
                self._open_event(ep, UNCERTAIN, pf)
            elif ep.ambiguous_time >= c.uncertain_min_s:
                ep.add_reason("ambiguous_pairing")
                self._open_event(ep, UNCERTAIN, pf)
            else:
                if not slow:
                    why = "fast_relative_motion (pass/wave)"
                elif ep.close_frames:
                    why = "brief_close_approach"
                else:
                    why = "brief_one_sided_proximity"
                ep.suppressed_ref = self.ledger.suppress(kind="contact", reason=why, start_timestamp=ep.start_t,
                                     end_timestamp=max(end_t, ep.start_t),
                                     customer_track_id=self.customer_id, staff_track_id=self.staff_id,
                                     detail=f"close {ep.close_time:.2f}s, ended: {reason}"
                                            + (f", speed {ms:.2f}" if ms is not None else ""))
        if ep.event_id is not None:
            self._sync_event(ep)
            self.ledger.close(ep.event_id, end_t, end_f, reason)
        self.ledger.episodes.append({
            "customer_track_id": self.customer_id, "staff_track_id": self.staff_id,
            "start": round(ep.start_t, 3), "end": round(max(end_t, ep.start_t), 3), "end_reason": reason,
            "event_id": ep.event_id,
            "outcome": self.ledger.get(ep.event_id).event_type if ep.event_id else "suppressed",
            "p_contact": None if ep.p_contact is None else round(ep.p_contact, 3),
            "p_handshake": None if ep.p_hs is None else round(ep.p_hs, 3),
            "features": {k: round(v, 4) for k, v in episode_features(ep, ep.hs_parts, self.h.clasp_distance).items()} if ep.close_frames else {},
        })
        self.last_episode = ep
        self.episode = None
        self.state = State.COOLDOWN
        self.cooldown_until = pf.t + c.cooldown_s
        self._cooldown_owner = "palm"

    # ------------------------------------------------------------------ external notifications
    def rekey(self, role: str, old_id: int, new_id: int) -> None:
        change = f"{role} {old_id}->{new_id}"
        self.track_changes.append(change)
        if role == "customer":
            self.customer_id = new_id
        else:
            self.staff_id = new_id
        for ep in (self.episode, self.last_episode):
            if ep is None:
                continue
            ep.add_reason("track_id_switch_inferred")
            if ep.event_id is not None:
                self.ledger.get(ep.event_id).track_id_changes.append(change)

    def mark_identity_ambiguous(self) -> None:
        if self.episode is not None:
            self.episode.add_reason("identity_ambiguous")

    def finish(self, frame_idx: int, t: float, reason: str) -> None:
        if self.episode is not None:
            self._end(PairFrame(frame_idx, t, Evidence.NONE, reason), reason)
        if self.touch is not None:
            self._touch_end(PairFrame(frame_idx, t, Evidence.NONE, reason), reason)
        self.state = State.NO_ACTIVE_INTERACTION
