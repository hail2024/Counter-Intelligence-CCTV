"""Drawing of the annotated output video.

Colour legend (BGR), also drawn on every frame:
  purple  tracking / neutral information (boxes, IDs, keypoints, trails)
  blue    customer region and boundary-crossing wrists      magenta  staff region
  amber   possible hand contact / uncertain evidence
  green   LIKELY hand contact / LIKELY handshake (pose evidence, not verified physical contact)
  red     unreliable evidence (low-confidence keypoints) or paused analysis
"""
from __future__ import annotations

import cv2
import numpy as np

from .calibration import CameraCalibration
from .engine import FrameResult
from .event_logger import EventLedger
from .interaction_state import State
from .pose_estimator import KP

PURPLE = (200, 80, 160)
BLUE = (230, 140, 40)
MAGENTA = (180, 60, 220)
AMBER = (0, 180, 255)
GREEN = (80, 200, 60)
RED = (40, 40, 230)
WHITE = (255, 255, 255)
DARK = (25, 20, 25)
GREY = (150, 150, 150)

STATE_COLOR = {
    State.NO_ACTIVE_INTERACTION: GREY, State.HAND_APPROACHING: PURPLE, State.BOUNDARY_CROSSED: BLUE,
    State.POSSIBLE_HAND_CONTACT: AMBER, State.UNCERTAIN: AMBER, State.LIKELY_HAND_CONTACT: GREEN,
    State.LIKELY_HANDSHAKE: GREEN, State.COOLDOWN: GREY,
}
ARM_JOINTS = [KP[n] for n in ("left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                               "left_wrist", "right_wrist")]
ARM_LINKS = [(KP["left_shoulder"], KP["left_elbow"]), (KP["left_elbow"], KP["left_wrist"]),
             (KP["right_shoulder"], KP["right_elbow"]), (KP["right_elbow"], KP["right_wrist"]),
             (KP["left_shoulder"], KP["right_shoulder"])]


def _pt(p) -> tuple[int, int]:
    return int(round(float(p[0]))), int(round(float(p[1])))


def _label(img, text, org, color, scale, bg=DARK, thick=1):
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = int(org[0]), int(org[1])
    x = max(0, min(x, img.shape[1] - w - 4))
    y = max(h + 4, min(y, img.shape[0] - 2))
    cv2.rectangle(img, (x - 2, y - h - 4), (x + w + 2, y + base - 1), bg, -1)
    cv2.putText(img, text, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


class Renderer:
    def __init__(self, cal: CameraCalibration, frame_size: tuple[int, int], cfg, final: EventLedger | None = None):
        """``final``: the ledger of a completed logic pass. When given, the on-video counts, event types and
        descriptions are the FINAL ones - the same as the event log and the dashboard."""
        self.cal = cal
        self.final = final
        self._final_inter = final.interactions() if final is not None else None
        self._final_types = {e.event_id: e for e in final.events} if final is not None else {}
        self.w, self.h = frame_size
        self.cfg = cfg
        self.s = max(0.45, min(self.w, self.h) / 900)   # text scale relative to resolution
        self.th = max(1, int(round(self.s * 2)))
        self._static = self._build_static()
        self._mask = self._static.any(axis=2).astype(np.uint8)   # region pixels, computed once per video

    def _build_static(self) -> np.ndarray:
        overlay = np.zeros((self.h, self.w, 3), np.uint8)
        cv2.fillPoly(overlay, [np.int32(self.cal.customer_region)], BLUE)
        cv2.fillPoly(overlay, [np.int32(self.cal.staff_region)], MAGENTA)
        return overlay

    def draw(self, frame: np.ndarray, res: FrameResult, ledger: EventLedger, kp_conf: float) -> np.ndarray:
        img = frame  # drawn in place: the decoded frame is not reused after rendering
        cv2.copyTo(cv2.addWeighted(img, 0.86, self._static, 0.14, 0), self._mask, img)
        for poly, col, name in ((self.cal.customer_region, BLUE, "CUSTOMER SIDE"),
                                (self.cal.staff_region, MAGENTA, "STAFF SIDE")):
            cv2.polylines(img, [np.int32(poly)], True, col, 1, cv2.LINE_AA)
            top = min(poly, key=lambda p: p[1] + 0.001 * p[0])
            _label(img, name, (top[0] + 6, top[1] + 22 * self.s + 8), col, 0.5 * self.s)
        b = np.int32(self.cal.boundary)
        cv2.polylines(img, [b], False, DARK, self.th + 4, cv2.LINE_AA)
        cv2.polylines(img, [b], False, WHITE, self.th + 1, cv2.LINE_AA)
        _label(img, "COUNTER BOUNDARY", (b[0][0] + 6, b[0][1] + 40 * self.s), WHITE, 0.45 * self.s)

        if res.paused:
            self._banner(img, f"ANALYSIS PAUSED: {res.pause_reason}", RED)
        for pv in res.persons:
            self._person(img, pv, kp_conf)
        for pair in res.pairs:
            self._pair(img, pair)
        self._hud(img, res, ledger)
        self._state_strip(img, res.overall_state)
        self._legend(img)
        return img

    STRIP = [(State.NO_ACTIVE_INTERACTION, "NO ACTIVE"), (State.HAND_APPROACHING, "APPROACHING"),
             (State.BOUNDARY_CROSSED, "CROSSED"), (State.POSSIBLE_HAND_CONTACT, "POSSIBLE CONTACT"),
             (State.LIKELY_HAND_CONTACT, "LIKELY CONTACT"), (State.LIKELY_HANDSHAKE, "LIKELY HANDSHAKE"),
             (State.UNCERTAIN, "UNCERTAIN"), (State.COOLDOWN, "COOLDOWN")]

    def _state_strip(self, img, current: State) -> None:
        """All eight interaction states in a row above the legend; the current one is filled in its colour."""
        n = len(self.STRIP)
        margin = 6
        cell_w = (self.w - 2 * margin) / n
        cell_h = int(26 * self.s)
        y1 = self.h - int(30 * self.s) - cell_h
        sc = 0.36 * self.s
        for i, (state, name) in enumerate(self.STRIP):
            x0, x1 = int(margin + i * cell_w) + 2, int(margin + (i + 1) * cell_w) - 2
            on = state == current
            col = STATE_COLOR[state] if state not in (State.NO_ACTIVE_INTERACTION, State.COOLDOWN) else (
                (200, 200, 200) if state == State.NO_ACTIVE_INTERACTION else (170, 140, 120))
            if on:
                cv2.rectangle(img, (x0, y1), (x1, y1 + cell_h), col, -1)
                cv2.rectangle(img, (x0, y1), (x1, y1 + cell_h), WHITE, 2)
            else:
                cv2.rectangle(img, (x0, y1), (x1, y1 + cell_h), DARK, -1)
                cv2.rectangle(img, (x0, y1), (x1, y1 + cell_h), col, 1)
            (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, sc, 1)
            scale = sc * min(1.0, (x1 - x0 - 6) / max(tw, 1))
            (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
            cv2.putText(img, name, (x0 + (x1 - x0 - tw) // 2, y1 + (cell_h + th) // 2), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, DARK if on else col, 1 if not on else max(1, self.th - 1), cv2.LINE_AA)

    def _person(self, img, pv, kp_conf):
        p = pv.person
        det = p.det
        if det is None:
            return
        col = {"customer": BLUE, "staff": MAGENTA}.get(p.role or "", PURPLE)
        x1, y1, x2, y2 = map(int, det.box)
        cv2.rectangle(img, (x1, y1), (x2, y2), PURPLE, 1, cv2.LINE_AA)
        role = {"customer": "Customer", "staff": "Staff"}.get(p.role or "", "unassigned")
        _label(img, f"#{p.track_id} {role}", (x1 + 2, y1 + 2), col if p.role else GREY, 0.5 * self.s)
        kp = det.keypoints
        for a, c in ARM_LINKS:
            if kp[a, 2] >= kp_conf and kp[c, 2] >= kp_conf and kp[a, 0] > 0 and kp[c, 0] > 0:
                cv2.line(img, _pt(kp[a]), _pt(kp[c]), PURPLE, self.th, cv2.LINE_AA)
        for j in ARM_JOINTS:
            x, y, c = kp[j]
            if x <= 0 and y <= 0:
                continue  # undetected keypoint sentinel: nothing to draw
            if c >= kp_conf:
                cv2.circle(img, _pt((x, y)), 3 + self.th, PURPLE, -1, cv2.LINE_AA)
            elif c >= 0.15:  # detected but unreliable: hollow red, not used by the logic
                cv2.circle(img, _pt((x, y)), 3 + self.th, RED, 1, cv2.LINE_AA)
        for side, trail in p.trail.items():
            pts = [q for f, q in trail if f >= (p.last_frame - self.cfg.output.trail_frames)]
            if len(pts) > 1:
                cv2.polylines(img, [np.int32(pts)], False, BLUE if side in pv.far_sides else PURPLE, 1, cv2.LINE_AA)
        for side, hand in p.hands.items():
            far = side in pv.far_sides
            cv2.circle(img, _pt(hand.wrist_s), 7 + self.th, BLUE if far else WHITE, self.th, cv2.LINE_AA)
            hx, hy = _pt(hand.proxy)
            if hand.tip is not None:   # the hand segment wrist -> estimated fingertip used for touch evidence
                cv2.line(img, _pt(hand.wrist_s), _pt(hand.tip), WHITE, 1, cv2.LINE_AA)
            cv2.drawMarker(img, (hx, hy), WHITE, cv2.MARKER_DIAMOND, 8 + 2 * self.th, 1, cv2.LINE_AA)
            if far:
                arrow = "C->S" if p.role == "customer" else "S->C"
                tag = f"CROSSED {arrow}" + (f" {pv.crossing_event_id}" if pv.crossing_event_id else "")
                wx, wy = _pt(hand.wrist_s)
                _label(img, tag, (wx - 40 * self.s, wy + 40 * self.s), BLUE, 0.5 * self.s)

    def _pair(self, img, pair):
        pf = pair.frame
        if pf is None or pair.state in (State.NO_ACTIVE_INTERACTION, State.COOLDOWN):
            return
        col = STATE_COLOR[pair.state]
        if pf.c_hand is not None and pf.s_hand is not None:
            a, b = _pt(pf.c_hand.proxy), _pt(pf.s_hand.proxy)
            cv2.line(img, a, b, col, self.th + 1, cv2.LINE_AA)
            mid = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
            dist = f"palms {pf.d_ratio:.2f} fa ({pf.d_px:.0f}px)"
            if pf.touch_ratio is not None:
                dist += f" | fingertips {pf.touch_ratio:.2f} fa"
            dist += f" | conf {pf.conf:.2f}"
        else:
            mid, dist = None, pf.reason or "no paired-hand evidence"
        title = pair.state.value.replace("_", " ")
        if pair.event_id:
            title = f"{pair.event_id} {title}"
        final_ev = self._final_types.get(pair.event_id) if pair.event_id else None
        if final_ev is not None and pair.state in (State.LIKELY_HAND_CONTACT, State.LIKELY_HANDSHAKE):
            if final_ev.handshake_score is not None:   # the decided record's score, same as the event log / dashboard
                title += f" | handshake {final_ev.handshake_score}%"
        elif pair.handshake_score is not None and pair.state in (State.POSSIBLE_HAND_CONTACT, State.LIKELY_HAND_CONTACT,
                                                                 State.LIKELY_HANDSHAKE):
            title += f" | handshake {pair.handshake_score:.0%}"
        if mid is None:
            return
        off = int(70 * self.s)  # keep labels above the hands, never on top of the wrists
        _label(img, title, (mid[0] - 80 * self.s, mid[1] - off - 22 * self.s), col, 0.55 * self.s, thick=self.th)
        _label(img, dist, (mid[0] - 80 * self.s, mid[1] - off), WHITE, 0.42 * self.s)

    def _banner(self, img, text, color):
        cv2.rectangle(img, (0, self.h - int(40 * self.s)), (self.w, self.h), color, -1)
        cv2.putText(img, text, (10, self.h - int(12 * self.s)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * self.s,
                    WHITE, self.th, cv2.LINE_AA)

    def _hud(self, img, res: FrameResult, ledger: EventLedger):
        if self._final_inter is not None:   # interactions that have started by now, with their final outcome
            ic = {"total": 0, "LIKELY_HANDSHAKE": 0, "LIKELY_HAND_CONTACT": 0, "UNCERTAIN": 0, "BOUNDARY_CROSSING": 0}
            for it in self._final_inter:
                if it["start_timestamp"] <= res.t + 1e-6:
                    ic["total"] += 1
                    ic[it["outcome"]] += 1
        else:
            ic = ledger.interaction_counts()
        lines = [
            (f"t={res.t:6.2f}s  frame {res.frame_idx}", WHITE),
            (f"STATE: {res.overall_state.value}", STATE_COLOR[res.overall_state]),
            (f"INTERACTIONS {ic['total']}: handshake {ic['LIKELY_HANDSHAKE']} | contact {ic['LIKELY_HAND_CONTACT']} | "
             f"uncertain {ic['UNCERTAIN']} | reach only {ic['BOUNDARY_CROSSING']}", WHITE),
        ]
        ftype = lambda e: self._final_types[e.event_id].event_type if e.event_id in self._final_types else e.event_type  # noqa: E731
        active = [f"{e.interaction_id}/{e.event_id}:{ftype(e).replace('LIKELY_', '')}" for e in ledger.open_events()]
        if active:
            lines.append(("active: " + ", ".join(active), AMBER))
        done = [e for e in ledger.events if e.status == "closed" and e.end_timestamp is not None
                and 0 <= res.t - e.end_timestamp <= 3.0 and e.interpretation]
        if done:  # show what the episode that just ended most likely was
            last = self._final_types.get(done[-1].event_id, done[-1])
            text = f"{last.interaction_id}/{last.event_id}: {last.interpretation}"
            col = {"BOUNDARY_CROSSING": BLUE, "UNCERTAIN": AMBER}.get(last.event_type, GREEN)
            lines.append((text if len(text) <= 95 else text[:92] + "...", col))
        sc = 0.5 * self.s
        lh = int(24 * self.s)
        width = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, sc, 1)[0][0] for t, _ in lines) + 16
        box = img[0:lh * len(lines) + 10, 0:min(width, self.w)]
        box[:] = (box * 0.35).astype(np.uint8)
        for i, (t, c) in enumerate(lines):
            cv2.putText(img, t, (8, lh * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, sc, c, 1, cv2.LINE_AA)

    def _legend(self, img):
        items = [("tracking", PURPLE), ("customer", BLUE), ("staff", MAGENTA), ("possible/uncertain", AMBER),
                 ("likely contact", GREEN), ("unreliable", RED)]
        sc = 0.4 * self.s
        x = 8
        y = self.h - int(10 * self.s)
        for name, col in items:
            (w, _), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, sc, 1)
            cv2.rectangle(img, (x - 3, y - int(14 * self.s)), (x + w + 18, y + 4), DARK, -1)
            cv2.circle(img, (x + 5, y - int(5 * self.s)), 5, col, -1)
            cv2.putText(img, name, (x + 14, y), cv2.FONT_HERSHEY_SIMPLEX, sc, WHITE, 1, cv2.LINE_AA)
            x += w + 26


def draw_calibration(frame: np.ndarray, cal: CameraCalibration | None, pending: dict | None = None) -> np.ndarray:
    """Calibration preview (used by the UI and `main.py preview`)."""
    img = frame.copy()
    s = max(0.5, min(img.shape[:2]) / 900)
    if cal is not None:
        overlay = img.copy()
        cv2.fillPoly(overlay, [np.int32(cal.customer_region)], BLUE)
        cv2.fillPoly(overlay, [np.int32(cal.staff_region)], MAGENTA)
        img = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
        cv2.polylines(img, [np.int32(cal.boundary)], False, WHITE, 3, cv2.LINE_AA)
        for poly, name, col in ((cal.customer_region, "CUSTOMER", BLUE), (cal.staff_region, "STAFF", MAGENTA)):
            c = np.mean(np.asarray(poly), axis=0)
            _label(img, name, (c[0] - 40, c[1]), col, 0.7 * s, thick=2)
    for key, col in (("boundary", WHITE), ("customer", BLUE), ("staff", MAGENTA)):
        pts = (pending or {}).get(key) or []
        for i, p in enumerate(pts):
            cv2.circle(img, _pt(p), 6, col, -1, cv2.LINE_AA)
            cv2.putText(img, str(i + 1), (int(p[0]) + 8, int(p[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s, col, 2)
        if len(pts) > 1:
            cv2.polylines(img, [np.int32(pts)], key != "boundary", col, 2, cv2.LINE_AA)
    return img
