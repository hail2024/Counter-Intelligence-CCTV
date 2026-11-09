"""The event ledger: the single authoritative record of interaction episodes.

* One row = one interaction EPISODE (never one row per frame, wrist or state transition).
* An episode's record is created once and then updated in place (end time, upgraded type, ...).
* Dashboard metrics, the CSV/JSON export and the event IDs drawn on the video all read this ledger.
* Candidate episodes that did NOT become events are kept in a separate diagnostics list with the
  reason they were suppressed, so every decision is traceable.

Event types
-----------
BOUNDARY_CROSSING    one person's wrist(s) crossed the counter boundary (single-person event)
LIKELY_HAND_CONTACT  customer-staff hand-contact episode (pose evidence only - not verified contact)
LIKELY_HANDSHAKE     a likely hand-contact episode that also met the handshake motion criteria
UNCERTAIN            a candidate customer-staff interaction whose evidence was insufficient or
                     degraded (occlusion, low confidence, identity ambiguity)

The last three are mutually exclusive: each customer-staff contact episode ends as exactly one of
them. BOUNDARY_CROSSING is a different family and may overlap in time with a contact episode;
the two are linked through ``related_event_ids``.

Interactions (the counting unit shown to users)
-----------------------------------------------
One customer-staff exchange often produces several rows: e.g. the customer's reach (a crossing), the staff
member's reach (another crossing) and the contact assessment of the clasp. Counting rows would report one
handshake as three "actions". Every row therefore also belongs to exactly one INTERACTION (``interaction_id``
I01, I02, ...), assigned online when the row opens:

* a row joins an existing interaction that involves one of the same people and that is still active or ended at
  most ``join_gap_s`` before the row started;
* ``link()`` (called by the engine for rows of the same exchange) merges two interactions into the older one;
* otherwise the row starts a new interaction.

An interaction's outcome is its strongest row: LIKELY_HANDSHAKE > LIKELY_HAND_CONTACT > UNCERTAIN >
BOUNDARY_CROSSING (a reach without a contact assessment). Different people are never merged by time alone.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

CROSSING = "BOUNDARY_CROSSING"
CONTACT = "LIKELY_HAND_CONTACT"
HANDSHAKE = "LIKELY_HANDSHAKE"
UNCERTAIN = "UNCERTAIN"
EVENT_TYPES = (CROSSING, CONTACT, HANDSHAKE, UNCERTAIN)
CONTACT_FAMILY = (CONTACT, HANDSHAKE, UNCERTAIN)
OUTCOME_RANK = {CROSSING: 0, UNCERTAIN: 1, CONTACT: 2, HANDSHAKE: 3}

CSV_FIELDS = [
    "event_id", "interaction_id", "event_type", "status", "start_timestamp", "end_timestamp", "duration_seconds",
    "confirmed_timestamp", "start_frame", "end_frame", "customer_track_id", "staff_track_id",
    "track_id_changes", "crossing_direction", "hands", "minimum_wrist_distance_px",
    "minimum_hand_distance_px", "minimum_hand_distance_forearms", "max_boundary_penetration_px",
    "keypoint_conf_mean", "keypoint_conf_min", "evidence_score", "handshake_score", "contact_score",
    "interpretation", "interaction_state",
    "uncertainty_reason", "start_reason", "end_reason", "evidence_summary", "related_event_ids",
]


@dataclass
class Event:
    event_id: str
    event_type: str
    start_timestamp: float
    start_frame: int
    confirmed_timestamp: float
    end_timestamp: float | None = None
    end_frame: int | None = None
    status: str = "open"                       # open | closed
    customer_track_id: int | None = None
    staff_track_id: int | None = None
    track_id_changes: list[str] = field(default_factory=list)
    crossing_direction: str | None = None      # customer_to_staff | staff_to_customer
    hands: list[str] = field(default_factory=list)  # e.g. ["customer:R", "staff:L"]
    minimum_wrist_distance_px: float | None = None
    minimum_hand_distance_px: float | None = None
    minimum_hand_distance_forearms: float | None = None
    max_boundary_penetration_px: float | None = None
    keypoint_conf_mean: float | None = None
    keypoint_conf_min: float | None = None
    evidence_score: float | None = None
    handshake_score: int | None = None         # % rule-based handshake evidence (not a trained probability)
    contact_score: int | None = None           # % rule-based contact evidence
    interpretation: str = ""                   # plain-language reading of the episode
    interaction_state: str | None = None       # highest state reached by the episode
    uncertainty_reason: list[str] = field(default_factory=list)
    start_reason: str = ""
    end_reason: str = ""
    evidence_summary: str = ""
    related_event_ids: list[str] = field(default_factory=list)
    shared: bool = False                       # several track-pair episodes merged into this one record
    interaction_id: str | None = None          # the customer-staff exchange this row belongs to (I01, ...)

    def participants(self) -> set[tuple[str, int]]:
        out = set()
        if self.customer_track_id is not None:
            out.add(("customer", self.customer_track_id))
        if self.staff_track_id is not None:
            out.add(("staff", self.staff_track_id))
        return out

    @property
    def duration_seconds(self) -> float | None:
        if self.end_timestamp is None:
            return None
        return max(0.0, self.end_timestamp - self.start_timestamp)

    def to_record(self) -> dict:
        d = asdict(self)
        d["duration_seconds"] = None if self.duration_seconds is None else round(self.duration_seconds, 3)
        for k in ("start_timestamp", "end_timestamp", "confirmed_timestamp"):
            if d[k] is not None:
                d[k] = round(d[k], 3)
        for k in ("minimum_wrist_distance_px", "minimum_hand_distance_px", "max_boundary_penetration_px"):
            if d[k] is not None:
                d[k] = round(d[k], 1)
        for k in ("minimum_hand_distance_forearms", "keypoint_conf_mean", "keypoint_conf_min", "evidence_score"):
            if d[k] is not None:
                d[k] = round(d[k], 3)
        return d


@dataclass
class SuppressedEpisode:
    kind: str                 # "contact" | "crossing"
    reason: str
    start_timestamp: float
    end_timestamp: float
    customer_track_id: int | None
    staff_track_id: int | None
    detail: str = ""


class EventLedger:
    def __init__(self, join_gap_s: float = 1.0):
        self.join_gap_s = join_gap_s
        self._interactions: dict[str, list[str]] = {}   # interaction id -> member event ids (in order)
        self._next_interaction = 1
        self._events: dict[str, Event] = {}
        self._order: list[str] = []
        self.suppressed: list[SuppressedEpisode] = []
        self.notes: list[str] = []          # global run notes (scene cuts, paused analysis, ...)
        self.episodes: list[dict] = []      # every candidate contact episode with its features (traceability)
        self.state_timeline: list[dict] = []  # overall interaction state over time: {state, start, end}
        self.pair_states: dict[str, list[dict]] = {}  # same, per customer-staff pair "C<id>-S<id>"
        self._next = 1

    # ------------------------------------------------------------------ mutation
    def open(self, event_type: str, t: float, frame: int, confirmed_t: float, **fields) -> Event:
        if event_type not in EVENT_TYPES:
            raise ValueError(f"unknown event type {event_type}")
        eid = f"E{self._next:03d}"
        self._next += 1
        ev = Event(event_id=eid, event_type=event_type, start_timestamp=t, start_frame=frame,
                   confirmed_timestamp=confirmed_t, **fields)
        self._events[eid] = ev
        self._order.append(eid)
        self._assign_interaction(ev)
        return ev

    # ------------------------------------------------------------------ interactions
    def _assign_interaction(self, ev: Event) -> None:
        who = ev.participants()
        best = None
        for iid, members in self._interactions.items():
            rows = [self._events[m] for m in members]
            if not who & set().union(*(r.participants() for r in rows)):
                continue
            active = any(r.status == "open" for r in rows)
            last_end = max((r.end_timestamp for r in rows if r.end_timestamp is not None), default=None)
            if active or (last_end is not None and ev.start_timestamp - last_end <= self.join_gap_s):
                if best is None or int(iid[1:]) > int(best[1:]):
                    best = iid   # the most recent matching interaction
        if best is None:
            best = f"I{self._next_interaction:02d}"
            self._next_interaction += 1
            self._interactions[best] = []
        self._interactions[best].append(ev.event_id)
        ev.interaction_id = best

    def _merge_interactions(self, a: str | None, b: str | None) -> None:
        if a is None or b is None or a == b:
            return
        keep, drop = sorted((a, b), key=lambda i: int(i[1:]))
        for eid in self._interactions.pop(drop):
            self._events[eid].interaction_id = keep
            self._interactions[keep].append(eid)
        self._interactions[keep].sort(key=self._order.index)

    def interactions(self) -> list[dict]:
        """One summary per interaction (the counting unit), in order of start time."""
        out = []
        for iid, members in self._interactions.items():
            rows = [self._events[m] for m in members]
            top = max(rows, key=lambda r: (OUTCOME_RANK[r.event_type], r.start_timestamp))
            ends = [r.end_timestamp for r in rows if r.end_timestamp is not None]
            crossings = [r for r in rows if r.event_type == CROSSING]
            out.append({
                "interaction_id": iid,
                "outcome": top.event_type,
                "start_timestamp": round(min(r.start_timestamp for r in rows), 3),
                "end_timestamp": round(max(ends), 3) if ends else None,
                "customer_track_ids": sorted({r.customer_track_id for r in rows if r.customer_track_id is not None}),
                "staff_track_ids": sorted({r.staff_track_id for r in rows if r.staff_track_id is not None}),
                "crossing_count": len(crossings),
                "crossing_directions": sorted({r.crossing_direction for r in crossings if r.crossing_direction}),
                "contact_assessment": next((r.event_type for r in rows if r.event_type in CONTACT_FAMILY), None),
                "handshake_score": top.handshake_score, "contact_score": top.contact_score,
                "event_ids": list(members),
                "interpretation": top.interpretation,
                "state_path": self.state_path(rows),
                "uncertainty_reason": sorted({u for r in rows for u in r.uncertainty_reason}),
                "status": "open" if any(r.status == "open" for r in rows) else "closed",
            })
        out.sort(key=lambda d: d["start_timestamp"])
        return out

    def record_states(self, t: float, overall: str, pairs: dict[str, str]) -> None:
        """Append one analysed frame's states (overall and per pair), merging runs of the same state."""
        def push(seq: list[dict], state: str) -> None:
            if seq and seq[-1]["state"] == state:
                seq[-1]["end"] = round(t, 3)
            else:
                seq.append({"state": state, "start": round(t, 3), "end": round(t, 3)})
        push(self.state_timeline, overall)
        for key, state in pairs.items():
            push(self.pair_states.setdefault(key, []), state)

    def state_path(self, rows: list[Event], lead_s: float = 2.0, tail_s: float = 1.5) -> list[str]:
        """States this interaction went through, in order (consecutive repeats collapsed) - the same states the video's
        state strip shows during it, e.g. HAND_APPROACHING > POSSIBLE_HAND_CONTACT > LIKELY_HAND_CONTACT > COOLDOWN.
        With several customer-staff pairs on screen, only this interaction's pair states are used."""
        t0 = min(r.start_timestamp for r in rows) - lead_s
        t1 = max((r.end_timestamp if r.end_timestamp is not None else r.start_timestamp) for r in rows) + tail_s
        cust = {r.customer_track_id for r in rows if r.customer_track_id is not None}
        staff = {r.staff_track_id for r in rows if r.staff_track_id is not None}
        segs = []
        if len(self.pair_states) > 1:   # several customer-staff pairs: this interaction's own pair(s)
            for key, seq in self.pair_states.items():
                c, s = (int(x[1:]) for x in key.split("-"))
                if (cust and c not in cust) or (staff and s not in staff):
                    continue
                segs += [x for x in seq if x["end"] >= t0 and x["start"] <= t1]
        if not segs:   # one pair (or none evaluated): exactly the states shown on the video
            segs = [x for x in self.state_timeline if x["end"] >= t0 and x["start"] <= t1]
        segs.sort(key=lambda x: x["start"])
        path: list[str] = []
        for x in segs:
            if x["state"] != "NO_ACTIVE_INTERACTION" and (not path or path[-1] != x["state"]):
                path.append(x["state"])
        return path

    def interaction_counts(self) -> dict[str, int]:
        c = {"total": 0, **{t: 0 for t in (HANDSHAKE, CONTACT, UNCERTAIN, CROSSING)}}
        for i in self.interactions():
            c["total"] += 1
            c[i["outcome"]] += 1
        return c

    def get(self, eid: str) -> Event:
        return self._events[eid]

    def retype(self, eid: str, new_type: str) -> None:
        """Upgrade an episode's classification in place (e.g. contact -> handshake). Never adds a row."""
        if new_type not in EVENT_TYPES:
            raise ValueError(new_type)
        ev = self._events[eid]
        if ev.event_type == CROSSING or new_type == CROSSING:
            raise ValueError("crossing and contact events are different families")
        ev.event_type = new_type

    @staticmethod
    def rank(event_type: str) -> int:
        return {UNCERTAIN: 0, CONTACT: 1, HANDSHAKE: 2}.get(event_type, -1)

    def find_concurrent_contact(self, customer_id, staff_id, t: float, window: float) -> Event | None:
        """An open (or just-closed) contact-family record involving either of these people around time t."""
        for ev in reversed(self.events):
            if ev.event_type not in CONTACT_FAMILY:
                continue
            if ev.customer_track_id != customer_id and ev.staff_track_id != staff_id:
                continue
            if ev.status == "open" or (ev.end_timestamp is not None and t - ev.end_timestamp <= window):
                return ev
        return None

    def close(self, eid: str, t_end: float, frame_end: int, reason: str) -> None:
        ev = self._events[eid]
        if ev.end_timestamp is not None and ev.shared:
            t_end = max(t_end, ev.end_timestamp)   # a shared record ends with its last episode
        ev.end_timestamp = max(t_end, ev.start_timestamp)
        ev.end_frame = max(frame_end, ev.start_frame)
        ev.end_reason = reason
        ev.status = "closed"

    def reopen(self, eid: str) -> None:
        ev = self._events[eid]
        ev.status = "open"

    def link(self, a: str, b: str) -> None:
        ea, eb = self._events[a], self._events[b]
        if b not in ea.related_event_ids:
            ea.related_event_ids.append(b)
        if a not in eb.related_event_ids:
            eb.related_event_ids.append(a)
        self._merge_interactions(ea.interaction_id, eb.interaction_id)

    def suppress(self, **kw) -> SuppressedEpisode:
        rec = SuppressedEpisode(**kw)
        self.suppressed.append(rec)
        return rec

    def unsuppress(self, rec: SuppressedEpisode) -> None:
        if rec in self.suppressed:
            self.suppressed.remove(rec)

    # ------------------------------------------------------------------ queries
    @property
    def events(self) -> list[Event]:
        return [self._events[e] for e in self._order]

    def open_events(self) -> list[Event]:
        return [e for e in self.events if e.status == "open"]

    def counts(self) -> dict[str, int]:
        c = {t: 0 for t in EVENT_TYPES}
        for e in self.events:
            c[e.event_type] += 1
        return c

    # ------------------------------------------------------------------ export
    def records(self) -> list[dict]:
        return [e.to_record() for e in self.events]

    def write_json(self, path: Path, meta: dict | None = None) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "meta": meta or {},
            "counting_rules": {
                "unit": "one row per observation episode (a crossing, or the contact assessment of a "
                        "customer-staff episode); rows of the same exchange share an interaction_id",
                "interaction_unit": "one interaction per customer-staff exchange; outcome = strongest row "
                                    "(LIKELY_HANDSHAKE > LIKELY_HAND_CONTACT > UNCERTAIN > BOUNDARY_CROSSING)",
                "mutually_exclusive": list(CONTACT_FAMILY),
                "separate_family": CROSSING,
                "distance_units": "pixels of the original video; *_forearms = multiples of forearm length",
            },
            "counts": self.counts(),
            "interaction_counts": self.interaction_counts(),
            "interactions": self.interactions(),
            "events": self.records(),
            "suppressed_candidates": [asdict(s) for s in self.suppressed],
            "candidate_episodes": self.episodes,
            "notes": self.notes,
            "state_timeline": self.state_timeline,
            "pair_states": self.pair_states,
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return path

    def write_csv(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
            w.writeheader()
            for rec in self.records():
                row = {}
                for k in CSV_FIELDS:
                    v = rec.get(k)
                    if isinstance(v, list):
                        v = "; ".join(map(str, v))
                    row[k] = "" if v is None else v  # empty cell = not observable / not applicable
                w.writerow(row)
        return path

    def write_interactions_csv(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = ["interaction_id", "outcome", "start_timestamp", "end_timestamp", "customer_track_ids",
                  "staff_track_ids", "crossing_count", "crossing_directions", "contact_assessment",
                  "handshake_score", "contact_score", "event_ids", "interpretation", "uncertainty_reason", "status"]
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for r in self.interactions():
                w.writerow({k: "; ".join(map(str, r[k])) if isinstance(r[k], list)
                            else ("" if r[k] is None else r[k]) for k in fields})
        return path
