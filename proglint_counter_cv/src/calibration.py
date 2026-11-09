"""Per-camera calibration: counter boundary, customer region and staff region.

A calibration belongs to ONE camera view. It stores the resolution it was drawn on and a small
fingerprint of the reference frame so that reusing it on a different view can be detected.
Nothing here guesses a boundary silently: ``suggest_calibration`` returns a proposal that the user
must review and save explicitly.
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from .geometry import (clip_polygon_halfplane, frame_rect, point_in_polygon, polygon_area,
                       polygon_centroid, scale_points, signed_distance_to_polyline)


class CalibrationError(ValueError):
    pass


@dataclass
class CameraCalibration:
    name: str
    reference_size: tuple[int, int]          # (width, height) the points were drawn on
    boundary: list[list[float]]              # polyline, >= 2 points
    customer_region: list[list[float]]       # polygon, >= 3 points
    staff_region: list[list[float]]          # polygon, >= 3 points
    fingerprint: list[int] = field(default_factory=list)
    source_video: str = ""
    created: str = ""
    notes: str = ""
    overrides: dict = field(default_factory=dict)  # optional per-camera threshold overrides

    # ------------------------------------------------------------------ derived geometry
    @property
    def customer_sign(self) -> float:
        """+1 if the customer region lies on the positive side of the boundary polyline, else -1."""
        s = signed_distance_to_polyline(polygon_centroid(self.customer_region), self.boundary)
        return 1.0 if s > 0 else -1.0

    def customer_side_distance(self, p) -> float:
        """Signed distance to the boundary, positive on the customer side, negative on the staff side."""
        return self.customer_sign * signed_distance_to_polyline(p, self.boundary)

    def region_of(self, p) -> str | None:
        in_c = point_in_polygon(p, self.customer_region)
        in_s = point_in_polygon(p, self.staff_region)
        if in_c and not in_s:
            return "customer"
        if in_s and not in_c:
            return "staff"
        return None  # outside both, or inside an overlap (ambiguous)

    # ------------------------------------------------------------------ validation / adaptation
    def validate(self) -> None:
        w, h = self.reference_size
        if w <= 0 or h <= 0:
            raise CalibrationError("reference_size must be positive")
        if len(self.boundary) < 2:
            raise CalibrationError("the counter boundary needs at least 2 points")
        for nm, poly in (("customer_region", self.customer_region), ("staff_region", self.staff_region)):
            if len(poly) < 3:
                raise CalibrationError(f"{nm} needs at least 3 points")
            if polygon_area(poly) < 0.001 * w * h:
                raise CalibrationError(f"{nm} is degenerate (area too small)")
        for nm, pts in (("boundary", self.boundary), ("customer_region", self.customer_region),
                        ("staff_region", self.staff_region)):
            for x, y in pts:
                if not (-1 <= x <= w and -1 <= y <= h):
                    raise CalibrationError(f"{nm} point ({x:.0f},{y:.0f}) is outside the {w}x{h} frame")
        b = np.asarray(self.boundary, float)
        if np.any(np.hypot(*np.diff(b, axis=0).T) < 1):
            raise CalibrationError("boundary has repeated/zero-length segments")
        sc = signed_distance_to_polyline(polygon_centroid(self.customer_region), self.boundary)
        ss = signed_distance_to_polyline(polygon_centroid(self.staff_region), self.boundary)
        if sc == 0 or ss == 0 or np.sign(sc) == np.sign(ss):
            raise CalibrationError(
                "customer and staff regions must lie on opposite sides of the counter boundary")

    def adapted_to(self, size: tuple[int, int]) -> "CameraCalibration":
        """Return a copy scaled to another resolution of the SAME view; refuse a different aspect ratio."""
        if tuple(size) == tuple(self.reference_size):
            return self
        rw, rh = self.reference_size
        if abs(size[0] / size[1] - rw / rh) > 0.01:
            raise CalibrationError(
                f"calibration '{self.name}' was drawn on {rw}x{rh} but the video is {size[0]}x{size[1]} "
                "(different aspect ratio) - recalibrate for this camera view")
        return CameraCalibration(
            name=self.name, reference_size=tuple(size),
            boundary=scale_points(self.boundary, self.reference_size, size),
            customer_region=scale_points(self.customer_region, self.reference_size, size),
            staff_region=scale_points(self.staff_region, self.reference_size, size),
            fingerprint=self.fingerprint, source_video=self.source_video, created=self.created,
            notes=self.notes, overrides=self.overrides)

    # ------------------------------------------------------------------ persistence
    def to_dict(self) -> dict:
        r = lambda pts: [[round(float(x), 1), round(float(y), 1)] for x, y in pts]  # noqa: E731
        return {
            "name": self.name,
            "reference_size": list(self.reference_size),
            "boundary": r(self.boundary),
            "customer_region": r(self.customer_region),
            "staff_region": r(self.staff_region),
            "source_video": self.source_video,
            "created": self.created,
            "notes": self.notes,
            "overrides": self.overrides,
            "fingerprint": list(self.fingerprint),
        }

    def save(self, path: str | Path) -> Path:
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = ("# Camera calibration - coordinates are pixels of reference_size (width, height).\n"
                  "# Valid ONLY for the camera view it was drawn on. Recalibrate if the camera moves.\n")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(header)
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False, default_flow_style=None, width=120)
        return path

    @staticmethod
    def from_dict(d: dict) -> "CameraCalibration":
        try:
            cal = CameraCalibration(
                name=str(d["name"]), reference_size=tuple(int(v) for v in d["reference_size"]),
                boundary=[list(map(float, p)) for p in d["boundary"]],
                customer_region=[list(map(float, p)) for p in d["customer_region"]],
                staff_region=[list(map(float, p)) for p in d["staff_region"]],
                fingerprint=[int(v) for v in d.get("fingerprint") or []],
                source_video=d.get("source_video", ""), created=d.get("created", ""),
                notes=d.get("notes", ""), overrides=d.get("overrides") or {})
        except (KeyError, TypeError, ValueError) as exc:
            raise CalibrationError(f"malformed calibration: {exc}") from exc
        cal.validate()
        return cal

    @staticmethod
    def load(path: str | Path) -> "CameraCalibration":
        path = Path(path)
        if not path.exists():
            raise CalibrationError(f"camera calibration not found: {path}")
        with open(path, encoding="utf-8") as fh:
            return CameraCalibration.from_dict(yaml.safe_load(fh) or {})


def now_iso() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def suggest_calibration(frame_size: tuple[int, int], anchors: list[tuple[float, float]],
                        name: str = "suggested", split: float = 0.5) -> CameraCalibration | None:
    """Propose a straight boundary halfway between the two main people, perpendicular to the line
    joining them, with each half of the frame as a region.

    ``anchors`` are shoulder-centre points of the two largest people. Returns None if fewer than two.
    The first anchor becomes the customer side; the UI offers a "swap roles" control because
    which person is staff cannot be inferred from geometry. This is a SUGGESTION to review.
    """
    if len(anchors) < 2:
        return None
    w, h = frame_size
    a, b = np.asarray(anchors[0], float), np.asarray(anchors[1], float)
    d = b - a
    mid = a + split * d   # the line crosses the a->b segment at this fraction (0.5 = halfway)
    if np.hypot(*d) < 1:
        return None
    perp = np.array([-d[1], d[0]]) / np.hypot(*d)
    # extend the perpendicular line until it leaves the frame
    span = np.hypot(w, h)
    p0, p1 = mid - perp * span, mid + perp * span
    rect = frame_rect(w, h)
    # clip the long line to the frame by intersecting with the rectangle polygon edges
    pts = _clip_segment_to_rect(p0, p1, w, h)
    if pts is None:
        return None
    q0, q1 = pts
    left = clip_polygon_halfplane(rect, q0, q1, keep_left=True)
    right = clip_polygon_halfplane(rect, q0, q1, keep_left=False)
    s_a = signed_distance_to_polyline(a, [q0, q1])
    cust, staff = (left, right) if s_a > 0 else (right, left)
    cal = CameraCalibration(name=name, reference_size=(w, h), boundary=[list(q0), list(q1)],
                            customer_region=cust, staff_region=staff, created=now_iso(),
                            notes="AUTO-SUGGESTED: review boundary position and customer/staff sides")
    cal.validate()
    return cal


def _clip_segment_to_rect(p0, p1, w, h):
    """Liang-Barsky clipping of segment p0->p1 to [0,w-1]x[0,h-1]."""
    x0, y0 = p0
    dx, dy = p1[0] - x0, p1[1] - y0
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x0), (dx, w - 1 - x0), (-dy, y0), (dy, h - 1 - y0)):
        if p == 0:
            if q < 0:
                return None
            continue
        r = q / p
        if p < 0:
            t0 = max(t0, r)
        else:
            t1 = min(t1, r)
        if t0 > t1:
            return None
    return (np.array([x0 + t0 * dx, y0 + t0 * dy]), np.array([x0 + t1 * dx, y0 + t1 * dy]))


def _role_cues(track: dict) -> tuple[float | None, float | None]:
    """(hand height, hand movement) of one person over the whole video, in forearm lengths.

    hand drop = median vertical distance of the wrists BELOW the shoulder centre. A staff member working at a desk
                keeps the hands up on the desk / keyboard (small drop); a customer's arms hang down or rest low.
    movement  = 75th percentile of wrist speed (forearms per second). A staff member typing keeps the hands still;
                the customer's hands move and reach.
    """
    drop = float(np.median(track["drop"])) if len(track["drop"]) >= 10 else None
    move = float(np.percentile(track["speed"], 75)) if len(track["speed"]) >= 10 else None
    return drop, move


def staff_side_score(first: dict, second: dict, drop_scale: float = 0.3, move_scale: float = 1.0) -> tuple[float | None, dict]:
    """> 0: the SECOND person looks like the staff member; < 0: the first one does. None: not measurable.
    Each cue is the first person's value minus the second's, divided by a fixed scale (no fitted weights)."""
    (d1, m1), (d2, m2) = _role_cues(first), _role_cues(second)
    parts = {}
    if d1 is not None and d2 is not None:
        parts["hand_drop"] = (d1 - d2) / drop_scale
    if m1 is not None and m2 is not None:
        parts["hand_movement"] = (m1 - m2) / move_scale
    cues = {"hand_drop_forearms": [None if d1 is None else round(d1, 2), None if d2 is None else round(d2, 2)],
            "hand_movement_forearms_per_s": [None if m1 is None else round(m1, 2), None if m2 is None else round(m2, 2)]}
    return (sum(parts.values()) if parts else None), cues


def auto_calibrate(frames: list[dict], frame_size: tuple[int, int], name: str, kp_conf: float = 0.4,
                   swap_roles: bool = False, background_ratio: float = 0.35, staff_rule: str = "activity",
                   staff_rule_min_margin: float = 0.5) -> tuple[CameraCalibration, dict]:
    """Derive the scene setup from the people tracked through the WHOLE video (no clicking needed).

    * The two main participants are the tracks with the largest (frames present x median box area)
      that stand in clearly different places.
    * Counter boundary = the line halfway between their median shoulder positions, perpendicular to
      the line joining them (a side-by-side view gives a vertical line, a top-down view a horizontal one).
    * Who is staff (``staff_rule="activity"``, side views): the person whose hands stay up at the desk and still
      (working / typing) is the staff member, the one whose hands hang lower and move (reaching) is the customer
      (``staff_side_score``). When the cues are not clear (|score| < ``staff_rule_min_margin``), in top-down views,
      or with ``staff_rule="position"``: customer = the left person (or the top person). ``swap_roles`` flips the
      result.
    * People shorter than ``background_ratio`` x the smaller main participant are background (no role).
      Others - e.g. a second customer standing farther back - keep a role and are paired normally.
    Raises CalibrationError when fewer than two people are tracked.
    """
    from .person_tracker import _anchor
    from .pose_estimator import PersonDetection
    w, h = frame_size
    tracks: dict[int, dict] = {}
    for f in frames:
        for d in f["p"]:
            if d["id"] is None:
                continue
            det = PersonDetection.from_json(d)
            t = tracks.setdefault(d["id"], {"n": 0, "h": [], "area": [], "anchor": [], "hands": [], "drop": [],
                                            "speed": [], "prev": {}})
            x1, y1, x2, y2 = det.box
            t["n"] += 1
            t["h"].append(y2 - y1)
            t["area"].append((x2 - x1) * (y2 - y1))
            t["anchor"].append(_anchor(det, kp_conf, frame_size))
            kp = det.keypoints
            sh = [kp[i, :2] for i in (5, 6) if kp[i, 2] >= kp_conf and (kp[i, 0] > 0 or kp[i, 1] > 0)]
            for wi, ei in ((9, 7), (10, 8)):   # wrist keypoints with usable confidence
                x, y, c = kp[wi]
                if c >= 0.5 and (x > 0 or y > 0):
                    t["hands"].append((float(x), float(y)))
                    fore = (float(np.hypot(x - kp[ei, 0], y - kp[ei, 1]))
                            if kp[ei, 2] >= kp_conf and (kp[ei, 0] > 0 or kp[ei, 1] > 0) else 0.0)
                    if sh and fore > 1.0:
                        t["drop"].append((float(y) - float(np.mean([q[1] for q in sh]))) / fore)
                        last = t["prev"].get(wi)
                        if last is not None and 0 < f["t"] - last[0] <= 0.2:
                            t["speed"].append(float(np.hypot(x - last[1], y - last[2])) / fore / (f["t"] - last[0]))
                    t["prev"][wi] = (f["t"], float(x), float(y))
                else:
                    t["prev"].pop(wi, None)
    ranked = sorted(tracks.items(), key=lambda kv: -kv[1]["n"] * float(np.median(kv[1]["area"])))
    if not ranked:
        raise CalibrationError("no people were detected in this video")
    a_id, a = ranked[0]
    a_pos = np.median(np.asarray(a["anchor"]), axis=0)
    diag = float(np.hypot(w, h))
    second = None
    for b_id, b in ranked[1:]:
        b_pos = np.median(np.asarray(b["anchor"]), axis=0)
        if np.hypot(*(b_pos - a_pos)) >= 0.12 * diag:   # a different person, not a fragment of the same track
            second = (b_id, b, b_pos)
            break
    if second is None:
        raise CalibrationError("only one person is visible for most of the video - a customer and a staff "
                               "member are both needed to place the counter line automatically")
    b_id, b, b_pos = second
    dx, dy = b_pos - a_pos
    side_view = abs(dx) >= abs(dy)
    first_is_customer = (a_pos[0] < b_pos[0]) if side_view else (a_pos[1] < b_pos[1])
    score, cues = staff_side_score(a, b)
    if staff_rule == "activity" and side_view and score is not None and abs(score) >= staff_rule_min_margin:
        first_is_customer = score > 0          # the second person looks like the staff member
        role_rule = (f"staff = the person working at the desk (hands kept up and still), score {abs(score):.1f}")
    elif staff_rule == "activity" and side_view:
        role_rule = ("roles UNCERTAIN (hand cues not clear"
                     + ("" if score is None else f", score {abs(score):.1f}") + "): customer = left person by default")
    else:
        role_rule = f"customer = {'left' if side_view else 'top'} person (position rule)"
    if swap_roles:
        first_is_customer = not first_is_customer
    cust, staff = (a_pos, b_pos) if first_is_customer else (b_pos, a_pos)
    # Place the line between the two people's usual HAND positions (their working zones), not their bodies:
    # a staff member's desk/laptop area often extends toward the customer. Each person's usual hand position is
    # the median of their wrists, projected on the line joining the two people.
    cust_tr, staff_tr = (a, b) if first_is_customer else (b, a)
    axis = staff - cust
    gap = float(np.hypot(*axis))
    split, split_src = 0.5, "midway between the two people"
    if gap > 1 and len(cust_tr["hands"]) >= 10 and len(staff_tr["hands"]) >= 10:
        u = axis / gap
        hc = float(np.median([(np.asarray(p) - cust) @ u for p in cust_tr["hands"]])) / gap
        hs = float(np.median([(np.asarray(p) - cust) @ u for p in staff_tr["hands"]])) / gap
        if 0.0 < hc < hs < 1.0:
            split, split_src = (hc + hs) / 2, "midway between the two people's usual hand positions"
    cal = suggest_calibration(frame_size, [tuple(cust), tuple(staff)], name, split)
    if cal is None:
        raise CalibrationError("could not place a counter line between the two main people")
    min_h = background_ratio * min(float(np.median(a["h"])), float(np.median(b["h"])))
    cal.overrides = {"roles": {"min_person_height_frac": round(min_h / h, 3)}}
    layout = "side by side" if side_view else "one above the other (top-down view)"
    cust_id, staff_id = (a_id, b_id) if first_is_customer else (b_id, a_id)
    cal.notes = (f"AUTO: counter line {split_src}; main participants are tracks {a_id} and {b_id} ({layout}); "
                 f"customer = track {cust_id}, staff = track {staff_id}: {role_rule}"
                 f"{' - then SWAPPED by the user' if swap_roles else ''}; "
                 f"people shorter than {min_h:.0f}px are treated as background")
    info = {"participants": [a_id, b_id], "layout": layout, "customer_anchor": [round(float(v), 1) for v in cust],
            "staff_anchor": [round(float(v), 1) for v in staff], "min_person_height_px": round(min_h, 1),
            "line_position": round(split, 3), "line_rule": split_src,
            "customer_track": cust_id, "staff_track": staff_id, "role_rule": role_rule,
            "role_score": None if score is None else round(score, 2), "role_cues": cues,
            "roles_uncertain": "UNCERTAIN" in role_rule, "swapped": swap_roles}
    return cal, info
