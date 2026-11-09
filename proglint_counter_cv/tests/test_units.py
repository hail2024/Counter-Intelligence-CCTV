"""Unit tests: geometry, side tracking, config validation, calibration, ledger/logging, evaluation,
and a full pipeline run with a mocked pose model (no network, no GPU)."""
import csv
import json

import cv2
import numpy as np
import pytest
import synth

from src import pipeline as pipeline_mod
from src.calibration import CalibrationError, CameraCalibration, suggest_calibration
from src.config import ConfigError, load_config
from src.evaluation import Interval, evaluate
from src.event_logger import CONTACT, CROSSING, HANDSHAKE, CSV_FIELDS, EventLedger
from src.geometry import point_in_polygon, signed_distance_to_polyline
from src.pose_estimator import keypoint_valid
from src.trajectory import Side, WristSideTracker, count_reversals


# ------------------------------------------------------------------ geometry
def test_signed_distance_straight_line():
    line = [[0, 0], [0, 100]]
    a, b = signed_distance_to_polyline((10, 50), line), signed_distance_to_polyline((-10, 50), line)
    assert abs(abs(a) - 10) < 1e-9 and abs(abs(b) - 10) < 1e-9 and np.sign(a) != np.sign(b)


def test_signed_distance_extends_beyond_endpoints():
    line = [[0, 0], [0, 100]]
    assert np.sign(signed_distance_to_polyline((10, 500), line)) == np.sign(signed_distance_to_polyline((10, 50), line))


def test_polyline_side_is_consistent_around_a_corner():
    poly = [[0, 0], [100, 0], [100, 100]]  # L-shaped counter
    inside = [(50, 20), (80, 80), (95, 5)]
    outside = [(50, -20), (120, 50), (110, -10)]
    s_in = {np.sign(signed_distance_to_polyline(p, poly)) for p in inside}
    s_out = {np.sign(signed_distance_to_polyline(p, poly)) for p in outside}
    assert len(s_in) == 1 and len(s_out) == 1 and s_in != s_out


def test_point_in_polygon():
    sq = [[0, 0], [10, 0], [10, 10], [0, 10]]
    assert point_in_polygon((5, 5), sq) and not point_in_polygon((15, 5), sq)


# ------------------------------------------------------------------ keypoints / side tracker
def test_zero_sentinel_keypoint_is_invalid():
    kp = np.zeros((17, 3))
    kp[9] = (0, 0, 0.99)
    kp[10] = (50, 50, 0.99)
    assert not keypoint_valid(kp, 9, 0.5, (100, 100))
    assert keypoint_valid(kp, 10, 0.5, (100, 100))
    kp[10] = (50, 50, 0.3)
    assert not keypoint_valid(kp, 10, 0.5, (100, 100))


def test_side_tracker_requires_observed_transition_and_persistence():
    tr = WristSideTracker(persist_frames=3, max_gap_frames=2)
    events = [tr.update(i, 50, 10) for i in range(3)]
    assert events[-1].kind == "appeared_home"
    assert [tr.update(10 + i, -50, 10) for i in range(2)] == [None, None]  # not persisted yet
    assert tr.update(12, -50, 10).kind == "crossed"


def test_side_tracker_long_gap_forgets_side():
    tr = WristSideTracker(persist_frames=1, max_gap_frames=2)
    tr.update(0, 50, 10)
    for i in range(5):
        tr.update(1 + i, None, 10)
    assert tr.confirmed == Side.UNKNOWN
    assert tr.update(10, -50, 10).kind == "appeared_far"   # not a crossing


def test_count_reversals_ignores_small_jitter():
    assert count_reversals(np.array([0, 0.01, 0, 0.01, 0]), 0.05) == 0
    assert count_reversals(np.array([0, 1, 0, 1, 0]), 0.5) == 3


# ------------------------------------------------------------------ config
def test_default_config_loads():
    assert load_config().contact.exit_ratio >= load_config().contact.enter_ratio


@pytest.mark.parametrize("ov", [
    {"contact": {"enter_ratio": 1.0, "exit_ratio": 0.5}},
    {"contact": {"min_close_ratio": 1.5}},
    {"model": {"imgsz": 650}},
    {"contact": {"typo_key": 1}},
    {"handshake": {"min_duration_s": 0.1}},
])
def test_invalid_config_rejected(ov):
    with pytest.raises(ConfigError):
        load_config(overrides=ov)


# ------------------------------------------------------------------ calibration
def test_calibration_requires_opposite_sides():
    cal = synth.calibration()
    cal.staff_region = [[0, 0], [400, 0], [400, 599], [0, 599]]
    with pytest.raises(CalibrationError):
        cal.validate()


def test_calibration_scales_same_aspect_and_rejects_other(tmp_path):
    cal = synth.calibration()
    p = cal.save(tmp_path / "c.yaml")
    loaded = CameraCalibration.load(p)
    half = loaded.adapted_to((500, 300))
    assert half.boundary[0] == [250.0, 0.0]
    with pytest.raises(CalibrationError):
        loaded.adapted_to((800, 600))


def test_suggestion_puts_people_on_opposite_sides():
    cal = suggest_calibration((1000, 600), [(200, 300), (800, 300)])
    assert cal.region_of((200, 300)) == "customer" and cal.region_of((800, 300)) == "staff"


# ------------------------------------------------------------------ ledger
def test_ledger_retype_updates_in_place_and_exports(tmp_path):
    L = EventLedger()
    ev = L.open(CONTACT, 1.0, 30, 1.5, customer_track_id=1, staff_track_id=2)
    L.retype(ev.event_id, HANDSHAKE)
    L.close(ev.event_id, 2.5, 75, "hands_separated")
    L.open(CROSSING, 3.0, 90, 3.1, customer_track_id=1, crossing_direction="customer_to_staff")
    with pytest.raises(ValueError):
        L.retype(ev.event_id, CROSSING)
    assert L.counts() == {"BOUNDARY_CROSSING": 1, "LIKELY_HAND_CONTACT": 0, "LIKELY_HANDSHAKE": 1, "UNCERTAIN": 0}
    rows = list(csv.DictReader(open(L.write_csv(tmp_path / "e.csv"), encoding="utf-8")))
    assert [r["event_id"] for r in rows] == ["E001", "E002"] and list(rows[0]) == CSV_FIELDS
    assert rows[0]["duration_seconds"] == "1.5" and rows[1]["staff_track_id"] == ""  # missing -> empty, not 0
    data = json.loads(L.write_json(tmp_path / "e.json").read_text())
    assert data["counts"]["LIKELY_HANDSHAKE"] == 1 and data["events"][1]["staff_track_id"] is None


def test_one_row_per_episode_not_per_frame():
    r = synth.Run(synth.idle(15) + synth.reach_in(10) + synth.handshake(90) + synth.reach_out(10) + synth.idle(40))
    assert len(r.ledger.events) == 1                              # 90 contact frames -> 1 record
    shown = {p.event_id for res in r.results for p in res.pairs if p.event_id}
    assert shown == {r.ledger.events[0].event_id}                 # video annotations use the same ID


# ------------------------------------------------------------------ evaluation
def test_evaluation_matching_and_ambiguous_exclusion():
    events = [
        {"event_id": "E1", "event_type": "LIKELY_HANDSHAKE", "start_timestamp": 1.0, "end_timestamp": 2.0, "confirmed_timestamp": 1.4},
        {"event_id": "E2", "event_type": "LIKELY_HAND_CONTACT", "start_timestamp": 5.0, "end_timestamp": 5.5, "confirmed_timestamp": 5.4},
        {"event_id": "E3", "event_type": "LIKELY_HAND_CONTACT", "start_timestamp": 9.0, "end_timestamp": 9.5, "confirmed_timestamp": 9.4},
        {"event_id": "E4", "event_type": "UNCERTAIN", "start_timestamp": 12.0, "end_timestamp": 12.3, "confirmed_timestamp": 12.3},
    ]
    labels = [Interval("CONTACT", 1.1, 2.1), Interval("HANDSHAKE", 1.1, 2.1),
              Interval("AMBIGUOUS_CONTACT", 4.8, 5.6), Interval("CONTACT", 20, 21)]
    rep = evaluate(events, labels, 0.5)
    c = rep["categories"]["CONTACT"]
    assert (c["tp"], c["fp"], c["fn"], c["excluded_ambiguous"]) == (1, 1, 1, 1)
    assert rep["categories"]["HANDSHAKE"]["tp"] == 1
    assert "CROSSING" in rep["not_scored"] and rep["uncertain_predictions"][0]["pred"] == "E4"


# ------------------------------------------------------------------ full pipeline with a mocked model
class FakeTracker:
    """Stands in for PoseTracker: returns the synthetic handshake sequence frame by frame."""
    def __init__(self, *a, **k):
        self.frames = synth.idle(10) + synth.reach_in(10) + synth.handshake(45) + synth.reach_out(10) + synth.idle(15)
        self.i = 0

    def process(self, frame):
        d = self.frames[min(self.i, len(self.frames) - 1)]
        self.i += 1
        return d

    def reset(self):
        pass


def test_pipeline_end_to_end_with_mock_model(tmp_path, monkeypatch):
    video = tmp_path / "synthetic.mp4"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (synth.W, synth.H))
    for i in range(90):
        w.write(np.full((synth.H, synth.W, 3), 60 + (i % 3), np.uint8))
    w.release()
    monkeypatch.setattr(pipeline_mod, "PoseTracker", FakeTracker)
    cal = synth.calibration()
    out = {"annotated_dir": str(tmp_path / "a"), "events_dir": str(tmp_path / "e"),
           "reports_dir": str(tmp_path / "r"), "cache_dir": str(tmp_path / "c")}
    cfg = load_config(overrides={"output": out})
    labels = tmp_path / "gt.csv"
    labels.write_text("category,start_timestamp,end_timestamp,role,notes\nCONTACT,0.6,2.3,,\nHANDSHAKE,0.6,2.3,,\n")
    res = pipeline_mod.run_video(video, cal, cfg, labels_path=labels, use_cache=False)
    auto = pipeline_mod.run_video(video, None, cfg, run_name="auto", use_cache=True)   # no calibration given
    assert auto.auto_calibrated and auto.ledger.counts() == res.ledger.counts()
    assert res.annotated_path is not None and res.annotated_path.exists()
    cap = cv2.VideoCapture(str(res.annotated_path))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 90 and int(cap.get(3)) == synth.W
    cap.release()
    data = json.loads(res.events_json.read_text())
    assert data["counts"] == res.ledger.counts()                  # export == ledger
    assert data["counts"]["LIKELY_HANDSHAKE"] == 1
    assert res.evaluation["categories"]["HANDSHAKE"]["tp"] == 1
    assert res.report_md.exists() and res.evaluation_md.exists()


# ------------------------------------------------------------------ automatic scene setup
def _frames_from(seq):
    return [{"i": i, "t": i / 30, "cut": False, "p": [d.to_json() for d in dets]} for i, dets in enumerate(seq)]


def test_auto_calibration_places_line_between_main_people_and_ignores_background():
    from src.calibration import auto_calibrate
    seq = synth.idle(60)
    small = synth.person(9, (520, 120))                       # a small background person
    small.box = np.array([505, 100, 540, 160.0])
    seq = [f + [small] for f in seq]
    cal, info = auto_calibrate(_frames_from(seq), (synth.W, synth.H), "auto")
    assert set(info["participants"]) == {1, 2}
    assert cal.region_of(synth.C_SHOULDER) == "customer" and cal.region_of(synth.S_SHOULDER) == "staff"
    assert 450 < cal.boundary[0][0] < 550                     # vertical line midway between the two people
    assert cal.overrides["roles"]["min_person_height_frac"] * synth.H > 60   # background box (60 px) is filtered
    swapped, _ = auto_calibrate(_frames_from(seq), (synth.W, synth.H), "auto", swap_roles=True)
    assert swapped.region_of(synth.C_SHOULDER) == "staff"


def _working_vs_reaching(n=90, staff_left=True):
    """One person keeps both hands up on the desk, still (typing); the other's arms hang low and one hand moves."""
    import math
    seq = []
    for i in range(n):
        typing = (lambda sh, f: synth.person(1 if staff_left else 2, sh, (sh[0] + 90 * f, sh[1] + 60), facing=f,
                                             left_wrist=(sh[0] + 70 * f, sh[1] + 60)))
        moving = (lambda sh, f: synth.person(2 if staff_left else 1, sh,
                                             (sh[0] + f * (60 + 50 * math.sin(i / 3)), sh[1] + 150), facing=f))
        c, s = synth.C_SHOULDER, synth.S_SHOULDER
        seq.append([typing(c, 1), moving(s, -1)] if staff_left else [moving(c, 1), typing(s, -1)])
    return seq


@pytest.mark.parametrize("staff_left", [True, False])
def test_auto_calibration_staff_is_the_person_working_at_the_desk(staff_left):
    """Regression: the user's recordings had the seated staff member on the LEFT; position alone made them the
    customer. Staff = hands kept up at the desk and still; customer = hands lower and moving."""
    from src.calibration import auto_calibrate
    cal, info = auto_calibrate(_frames_from(_working_vs_reaching(staff_left=staff_left)), (synth.W, synth.H), "auto")
    left_role = cal.region_of(synth.C_SHOULDER)
    assert left_role == ("staff" if staff_left else "customer"), info["role_rule"]
    assert not info["roles_uncertain"]
    pos, _ = auto_calibrate(_frames_from(_working_vs_reaching(staff_left=staff_left)), (synth.W, synth.H), "auto",
                            staff_rule="position")
    assert pos.region_of(synth.C_SHOULDER) == "customer"     # the old position rule, still available


def test_auto_calibration_roles_uncertain_when_cues_are_equal():
    from src.calibration import auto_calibrate
    _, info = auto_calibrate(_frames_from(synth.idle(60)), (synth.W, synth.H), "auto")
    assert info["roles_uncertain"] and "left person by default" in info["role_rule"]


def test_auto_calibration_needs_two_people():
    from src.calibration import auto_calibrate
    seq = [[synth.person(1, synth.C_SHOULDER)] for _ in range(30)]
    with pytest.raises(CalibrationError):
        auto_calibrate(_frames_from(seq), (synth.W, synth.H), "auto")


def test_two_click_line_calibration():
    from src.pipeline import make_calibration_from_line
    frame = np.zeros((synth.H, synth.W, 3), np.uint8)
    cal = make_calibration_from_line("drawn", frame, (500, 100), (500, 400), "v.mp4")
    assert cal.region_of((200, 300)) == "customer" and cal.region_of((800, 300)) == "staff"
    flat = make_calibration_from_line("drawn", frame, (100, 300), (900, 300), "v.mp4")
    assert flat.region_of((500, 100)) == "customer"            # horizontal line: top side is the customer
    with pytest.raises(ValueError):
        make_calibration_from_line("drawn", frame, (500, 100), (502, 101), "v.mp4")


def test_contact_event_has_scores_and_interpretation():
    r = synth.Run(synth.idle(15) + synth.reach_in(10) + synth.handshake(45) + synth.reach_out(10) + synth.idle(40))
    ev = r.ledger.events[0]
    assert ev.event_type == HANDSHAKE and ev.handshake_score >= 60 and "Likely handshake" in ev.interpretation
