"""Regression tests for the interaction counting unit, interaction-level evaluation and the faster I/O path.

Background: on real footage one customer-staff exchange produced several ledger rows (the customer's
crossing, the staff member's crossing, the contact assessment), so a dashboard counting rows reported e.g.
7 "actions" for 4 real exchanges. Rows of one exchange now share an interaction_id and the dashboard counts
interactions. These tests pin that behaviour down (synthetic inputs, no model).
"""
import json

import cv2
import numpy as np
import pytest
from synth import C_SHOULDER, S_SHOULDER, Run, handshake, idle, lerp, person, reach_in, reach_out, seconds

from src.event_logger import CONTACT, CROSSING, HANDSHAKE, UNCERTAIN, EventLedger
from src.evaluation import Interval, evaluate, gt_interactions, load_labels
from src.pose_estimator import letterbox_shape
from src.video_io import AnnotatedVideoWriter


def _reach(c_tid=1, s_tid=2, hold_s=1.0):
    """Customer reaches across; staff does not respond (scenario 1 motion)."""
    out = [[person(c_tid, C_SHOULDER, w), person(s_tid, S_SHOULDER, facing=-1)]
           for w in lerp((355, 430), (620, 260), 12)]
    out += [[person(c_tid, C_SHOULDER, (620, 260)), person(s_tid, S_SHOULDER, facing=-1)]] * seconds(hold_s)
    out += [[person(c_tid, C_SHOULDER, w), person(s_tid, S_SHOULDER, facing=-1)]
            for w in lerp((620, 260), (355, 430), 12)]
    return out


# ------------------------------------------------------------------ one exchange = one interaction
def test_handshake_with_its_crossings_is_one_interaction():
    r = Run(idle(15) + reach_in(10) + handshake(seconds(1.5)) + reach_out(10) + idle(40))
    inter = r.ledger.interactions()
    assert len(inter) == 1, inter
    assert inter[0]["outcome"] == HANDSHAKE
    assert {e.interaction_id for e in r.ledger.events} == {inter[0]["interaction_id"]}
    assert r.ledger.interaction_counts()["total"] == 1


def test_reach_without_response_is_one_reach_only_interaction():
    r = Run(idle(15) + _reach() + idle(30))
    inter = r.ledger.interactions()
    assert len(inter) == 1 and inter[0]["outcome"] == CROSSING and inter[0]["crossing_count"] == 1
    assert inter[0]["contact_assessment"] is None


def test_two_separate_handshakes_are_two_interactions():
    one = reach_in(10) + handshake(seconds(1.2)) + reach_out(10)
    r = Run(idle(15) + one + idle(seconds(3)) + one + idle(40))
    inter = r.ledger.interactions()
    assert [i["outcome"] for i in inter] == [HANDSHAKE, HANDSHAKE]
    assert inter[0]["end_timestamp"] < inter[1]["start_timestamp"]


def test_reach_then_handshake_soon_after_is_one_interaction():
    """The customer reaches, pulls back briefly, then the two shake hands (< join_gap_s later)."""
    r = Run(idle(15) + _reach(hold_s=0.5) + idle(seconds(0.3)) + reach_in(10) + handshake(seconds(1.5))
            + reach_out(10) + idle(40))
    inter = r.ledger.interactions()
    assert len(inter) == 1 and inter[0]["outcome"] == HANDSHAKE
    assert inter[0]["crossing_count"] >= 1


def test_reaches_far_apart_in_time_are_separate_interactions():
    r = Run(idle(15) + _reach() + idle(seconds(3)) + _reach() + idle(30))
    assert [i["outcome"] for i in r.ledger.interactions()] == [CROSSING, CROSSING]


def test_different_people_are_never_merged_by_time_alone():
    """Pair A (C1, S2) shakes hands while customer C3 (another pair, lower) reaches across alone."""
    n = seconds(1.5)
    top = handshake(n, y=150, c_shoulder=(300, 150), s_shoulder=(700, 150))
    low_reach = lerp((355, 580), (620, 420), n)
    frames = [[person(1, (300, 150)), person(2, (700, 150), facing=-1),
               person(3, (300, 400)), person(4, (700, 400), facing=-1)] for _ in range(15)]
    for a, w in zip(top, low_reach):
        frames.append(a + [person(3, (300, 400), w), person(4, (700, 400), facing=-1)])
    frames += [[person(1, (300, 150)), person(2, (700, 150), facing=-1),
                person(3, (300, 400)), person(4, (700, 400), facing=-1)] for _ in range(40)]
    r = Run(frames)
    inter = r.ledger.interactions()
    by_customer = {tuple(i["customer_track_ids"]): i for i in inter}
    assert (1,) in by_customer and (3,) in by_customer
    assert by_customer[(1,)]["outcome"] == HANDSHAKE
    assert by_customer[(3,)]["outcome"] == CROSSING
    assert by_customer[(1,)]["interaction_id"] != by_customer[(3,)]["interaction_id"]


# ------------------------------------------------------------------ ledger rules
def test_ledger_joins_by_shared_person_and_gap_and_link_merges():
    led = EventLedger(join_gap_s=1.0)
    a = led.open(CROSSING, 1.0, 30, 1.1, customer_track_id=1)
    led.close(a.event_id, 2.0, 60, "returned")
    b = led.open(UNCERTAIN, 2.5, 75, 2.6, customer_track_id=1, staff_track_id=2)   # 0.5 s later, same customer
    c = led.open(CROSSING, 2.6, 78, 2.7, staff_track_id=9)                          # someone else at the same time
    assert a.interaction_id == b.interaction_id != c.interaction_id
    led.link(b.event_id, c.event_id)                                                # engine says: same exchange
    assert c.interaction_id == a.interaction_id
    led.close(b.event_id, 3.0, 90, "x")
    led.close(c.event_id, 3.0, 90, "x")
    d = led.open(CROSSING, 5.0, 150, 5.1, customer_track_id=1)                      # 2 s after the end
    assert d.interaction_id != a.interaction_id
    counts = led.interaction_counts()
    assert counts == {"total": 2, HANDSHAKE: 0, CONTACT: 0, UNCERTAIN: 1, CROSSING: 1}


def test_contact_retype_updates_interaction_outcome_in_place():
    led = EventLedger()
    x = led.open(CROSSING, 0.5, 15, 0.6, customer_track_id=1)
    y = led.open(CONTACT, 0.7, 21, 1.0, customer_track_id=1, staff_track_id=2)
    led.retype(y.event_id, HANDSHAKE)
    inter = led.interactions()
    assert len(inter) == 1 and inter[0]["outcome"] == HANDSHAKE
    assert inter[0]["event_ids"] == [x.event_id, y.event_id]


def test_exports_carry_interactions(tmp_path):
    led = EventLedger()
    e = led.open(CONTACT, 0.7, 21, 1.0, customer_track_id=1, staff_track_id=2)
    led.close(e.event_id, 2.0, 60, "hands_separated")
    data = json.loads(led.write_json(tmp_path / "e.json").read_text())
    assert data["interaction_counts"]["total"] == 1 and data["events"][0]["interaction_id"] == "I01"
    csv_text = led.write_interactions_csv(tmp_path / "i.csv").read_text()
    assert csv_text.splitlines()[0].startswith("interaction_id,outcome") and "I01" in csv_text
    assert "interaction_id" in led.write_csv(tmp_path / "e.csv").read_text().splitlines()[0]


# ------------------------------------------------------------------ interaction-level evaluation
def _ev(eid, iid, t0, t1, etype=CROSSING):
    return {"event_id": eid, "interaction_id": iid, "event_type": etype, "start_timestamp": t0,
            "end_timestamp": t1, "confirmed_timestamp": t0, "crossing_direction": None}


def test_gt_interactions_merge_overlaps_and_respect_explicit_groups():
    labels = [Interval("CONTACT", 1.0, 2.0, ref="a"), Interval("HANDSHAKE", 1.0, 2.0, ref="b"),
              Interval("CROSSING", 1.5, 2.5, ref="c", interaction="X"),        # explicit: someone else
              Interval("AMBIGUOUS_CONTACT", 5.0, 6.0, ref="d")]
    gts = gt_interactions(labels)
    assert len(gts) == 3
    assert [req for _, req in gts] == [True, True, False]


def test_interaction_evaluation_counts_and_ambiguous_handling():
    labels = [Interval("CONTACT", 1.0, 2.0, ref="a"), Interval("AMBIGUOUS_CONTACT", 5.0, 6.0, ref="b")]
    events = [_ev("E1", "I01", 0.9, 1.5), _ev("E2", "I01", 1.0, 2.1, CONTACT),   # one exchange, two rows
              _ev("E3", "I02", 5.2, 5.8, UNCERTAIN),                              # the ambiguous hand-over
              _ev("E4", "I03", 9.0, 9.5)]                                         # nothing happened here
    it = evaluate(events, labels)["categories"]["INTERACTION"]
    assert (it["tp"], it["fp"], it["fn"]) == (1, 1, 0)
    assert it["excluded_ambiguous"] == 1
    assert (it["count_predicted"], it["count_labelled"], it["count_error"]) == (3, 2, 1)


def test_label_file_optional_columns(tmp_path):
    p = tmp_path / "gt.csv"
    p.write_text("category,start_timestamp,end_timestamp,role,notes,interaction,expected_result,wrist_visibility\n"
                 "CROSSING,1,2,customer,reach,A,event,both_visible\n"
                 "CONTACT,1.5,2.5,,touch,B,uncertain,staff_wrist_occluded\n")
    labels = load_labels(p)
    assert [g.interaction for g in labels] == ["A", "B"]
    assert labels[1].extra == {"expected_result": "uncertain", "wrist_visibility": "staff_wrist_occluded"}
    assert len(gt_interactions(labels)) == 2                 # explicit groups are not merged by overlap
    old = tmp_path / "old.csv"                                # the original 5-column format still loads
    old.write_text("category,start_timestamp,end_timestamp,role,notes\nCONTACT,1,2,,\n")
    assert load_labels(old)[0].interaction is None


# ------------------------------------------------------------------ inference shape / video writer
@pytest.mark.parametrize("w,h,expected", [(1024, 576, (384, 640)), (1280, 720, (384, 640)),
                                          (1918, 970, (352, 640)), (576, 1024, (640, 384)), (640, 640, (640, 640))])
def test_letterbox_shape_matches_ultralytics_rect_inference(w, h, expected):
    assert letterbox_shape(w, h, 640) == expected


def test_writer_pipes_h264_and_handles_odd_frame_size(tmp_path):
    w, h, n = 321, 179, 25
    out = AnnotatedVideoWriter(tmp_path / "v.mp4", 25.0, (w, h), h264=True)
    for i in range(n):
        out.write(np.full((h, w, 3), i * 8 % 255, np.uint8))
    path = out.close()
    cap = cv2.VideoCapture(str(path))
    frames = 0
    while cap.grab():
        frames += 1
    cap.release()
    assert frames == n
    assert out.codec in ("h264", "mp4v")
