"""Synthetic pose sequences for deterministic tests (no model, no GPU).

Scene: 1000x600 frame, vertical counter boundary at x=500, customer region on the left, staff
region on the right. People are built with two-link arms (upper arm = forearm = 100 px), so the
person scale (median forearm length) is 100 px and every *_ratio threshold reads directly as
hundreds of pixels. Unused keypoints are left at the (0, 0) / conf 0 sentinel that Ultralytics
emits for undetected joints, so the tests also check that such keypoints are ignored.
"""
from __future__ import annotations

import math

import numpy as np

from src.calibration import CameraCalibration
from src.config import load_config
from src.engine import InteractionEngine
from src.event_logger import EventLedger
from src.pose_estimator import PersonDetection

W, H, FPS = 1000, 600, 30.0
LIMB = 100.0


def calibration() -> CameraCalibration:
    return CameraCalibration(
        name="synthetic", reference_size=(W, H), boundary=[[500, 0], [500, 599]],
        customer_region=[[0, 0], [480, 0], [480, 599], [0, 599]],
        staff_region=[[520, 0], [999, 0], [999, 599], [520, 599]])


def _elbow(shoulder, wrist, bend_down=True) -> np.ndarray:
    """Two-link inverse kinematics with equal 100 px segments."""
    s, w = np.asarray(shoulder, float), np.asarray(wrist, float)
    v = w - s
    L = float(np.hypot(*v))
    if L < 1e-6:
        return s + np.array([0.0, LIMB])
    u = v / L
    if L >= 2 * LIMB - 1e-6:
        return w - LIMB * u  # unreachable: keep forearm length, stretch upper arm
    a = L / 2
    h = math.sqrt(LIMB ** 2 - a ** 2)
    perp = np.array([-u[1], u[0]])
    if (perp[1] < 0) == bend_down:
        perp = -perp
    return s + a * u + h * perp


def person(tid: int | None, shoulder, wrist=None, conf: float = 0.98, wrist_conf: float | None = None,
           elbow_conf: float | None = None, facing: int = 1, left_wrist=None,
           left_wrist_conf: float | None = None) -> PersonDetection:
    """A person whose RIGHT arm reaches to ``wrist`` (None = resting); left arm rests (or ``left_wrist``).

    Default keypoint confidence 0.98 = what YOLO26s-pose gives clearly visible wrists on the project videos
    (median 0.97-0.99 during every visible contact); occlusion tests lower it explicitly."""
    sx, sy = shoulder
    kp = np.zeros((17, 3))
    kp[0] = (sx + 10 * facing, sy - 60, conf)                    # nose
    kp[5] = (sx - 15 * facing, sy, conf)                          # left shoulder
    kp[6] = (sx + 15 * facing, sy, conf)                          # right shoulder
    kp[11] = (sx - 10 * facing, sy + 200, conf)                   # hips
    kp[12] = (sx + 10 * facing, sy + 200, conf)
    rest_l = (sx - 15 * facing + 40 * facing, sy + 180)
    rest_r = (sx + 15 * facing + 40 * facing, sy + 180)
    for (si, ei, wi), target, wc in (((5, 7, 9), left_wrist or rest_l, left_wrist_conf),
                                     ((6, 8, 10), wrist or rest_r, wrist_conf)):
        s = kp[si, :2]
        e = _elbow(s, target)
        kp[ei] = (*e, conf if elbow_conf is None else elbow_conf)
        c = conf if wc is None else wc
        kp[wi] = (*target, c) if c > 0 else (0.0, 0.0, 0.0)
    valid = kp[kp[:, 2] > 0][:, :2]
    box = np.array([*(valid.min(axis=0) - 20), *(valid.max(axis=0) + 20)])
    return PersonDetection(tid, box, 0.9, kp)


class Run:
    """Feeds a list of per-frame detection lists through the engine and keeps the results."""

    def __init__(self, frames: list[list[PersonDetection]], cfg_overrides: dict | None = None,
                 fps: float = FPS, cal: CameraCalibration | None = None):
        self.cfg = load_config(overrides=cfg_overrides)
        self.ledger = EventLedger()
        self.engine = InteractionEngine(self.cfg, cal or calibration(), fps, ledger=self.ledger)
        self.results = []
        for i, dets in enumerate(frames):
            res = self.engine.process(i, i / fps, dets, (W, H))
            self.results.append(res)
            # same state recording as src/pipeline.run_video (state timeline + per-interaction state path)
            self.ledger.record_states(i / fps, res.overall_state.value,
                                      {f"C{pv.customer_id}-S{pv.staff_id}": pv.state.value for pv in res.pairs})
        self.engine.finish(len(frames) - 1, (len(frames) - 1) / fps)

    def of_type(self, t: str):
        return [e for e in self.ledger.events if e.event_type == t]

    @property
    def counts(self):
        return self.ledger.counts()

    def states(self) -> list[str]:
        return [r.overall_state.value for r in self.results]


def seconds(s: float) -> int:
    return int(round(s * FPS))


# --------------------------------------------------------------------------- canned motions
C_SHOULDER = (300, 250)
S_SHOULDER = (700, 250)


def idle(n: int, c_tid: int = 1, s_tid: int = 2) -> list[list[PersonDetection]]:
    return [[person(c_tid, C_SHOULDER), person(s_tid, S_SHOULDER, facing=-1)] for _ in range(n)]


def lerp(a, b, n: int):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return [tuple(a + (b - a) * (i / max(n - 1, 1))) for i in range(n)]


def handshake(n: int, pump_amp: float = 15.0, pump_hz: float = 2.5, c_tid: int = 1, s_tid: int = 2,
              y: float = 250.0, t0: int = 0, c_shoulder=C_SHOULDER, s_shoulder=S_SHOULDER, gap: float = 60.0, **kw):
    """Both hands clasp over the counter (hand proxies ~0.05 forearms apart, as in the real handshakes, which
    measured 0.00-0.12) and pump together. A larger wrist ``gap`` keeps the hands apart, as an item held between
    them does in a hand-over."""
    out = []
    for i in range(n):
        dy = pump_amp * math.sin(2 * math.pi * pump_hz * (t0 + i) / FPS)
        out.append([person(c_tid, c_shoulder, (500 - gap / 2, y + dy), **kw.get("c", {})),
                    person(s_tid, s_shoulder, (500 + gap / 2, y + dy), facing=-1, **kw.get("s", {}))])
    return out


def reach_in(n: int, c_tid=1, s_tid=2, y=250.0):
    """Both hands move from rest to the meeting position."""
    c_path = lerp((C_SHOULDER[0] + 55, C_SHOULDER[1] + 180), (470, y), n)
    s_path = lerp((S_SHOULDER[0] - 55, S_SHOULDER[1] + 180), (530, y), n)
    return [[person(c_tid, C_SHOULDER, c), person(s_tid, S_SHOULDER, s, facing=-1)]
            for c, s in zip(c_path, s_path)]


def reach_out(n: int, c_tid=1, s_tid=2, y=250.0):
    c_path = lerp((470, y), (C_SHOULDER[0] + 55, C_SHOULDER[1] + 180), n)
    s_path = lerp((530, y), (S_SHOULDER[0] - 55, S_SHOULDER[1] + 180), n)
    return [[person(c_tid, C_SHOULDER, c), person(s_tid, S_SHOULDER, s, facing=-1)]
            for c, s in zip(c_path, s_path)]
