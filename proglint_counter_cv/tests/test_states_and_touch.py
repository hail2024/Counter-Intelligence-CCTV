"""All eight assignment states are produced, and fingertip-touch contact (palms apart) is handled by explicit rules.

Background: a real clip ("4.likely hand contact") showed two open hands meeting fingertip to fingertip; the palm
points stayed 0.78 forearms apart, so the palm-only logic never left HAND_APPROACHING. Each hand is now also modelled
as the segment wrist -> fingertip (keypoints.fingertip_ratio). A touch becomes LIKELY_HAND_CONTACT only with
paired-hand evidence; a hand passing over someone's resting hand must not.
"""
from synth import C_SHOULDER, S_SHOULDER, Run, handshake, idle, lerp, person, reach_in, reach_out, seconds

from src.event_logger import CONTACT, HANDSHAKE, UNCERTAIN
from src.geometry import segment_closest

ALL_STATES = {"NO_ACTIVE_INTERACTION", "HAND_APPROACHING", "BOUNDARY_CROSSED", "POSSIBLE_HAND_CONTACT",
              "LIKELY_HAND_CONTACT", "LIKELY_HANDSHAKE", "UNCERTAIN", "COOLDOWN"}


def test_all_eight_states_occur():
    """Each state comes from the scenario it belongs to; together the four scripted scenes produce all eight."""
    reach = [[person(1, C_SHOULDER, w), person(2, S_SHOULDER, facing=-1)] for w in lerp((355, 430), (620, 260), 12)]
    reach += [[person(1, C_SHOULDER, (620, 260)), person(2, S_SHOULDER, facing=-1)]] * seconds(1.0)
    reach += [[person(1, C_SHOULDER, w), person(2, S_SHOULDER, facing=-1)] for w in lerp((620, 260), (355, 430), 12)]
    crossing = Run(idle(15) + reach + idle(30))                                           # scenario 1
    still = Run(idle(15) + reach_in(10) + handshake(seconds(1.5), pump_amp=0) + reach_out(10) + idle(40))  # clasp
    shake = Run(idle(15) + reach_in(10) + handshake(seconds(1.5)) + reach_out(10) + idle(40))           # scenario 3
    occl = handshake(seconds(0.3)) + handshake(seconds(1.0), s={"wrist_conf": 0.0}) + reach_out(10)
    occluded = Run(idle(15) + reach_in(10) + occl + idle(40))                             # scenario 6
    assert "BOUNDARY_CROSSED" in crossing.states()
    assert "LIKELY_HAND_CONTACT" in still.states() and "LIKELY_HANDSHAKE" not in still.states()
    assert "UNCERTAIN" in occluded.states()
    seen = set().union(*(set(r.states()) for r in (crossing, still, shake, occluded)))
    assert seen == ALL_STATES, ALL_STATES - seen
    order = shake.states()
    first = {s: order.index(s) for s in set(order)}
    assert first["HAND_APPROACHING"] < first["POSSIBLE_HAND_CONTACT"] < first["LIKELY_HANDSHAKE"] < first["COOLDOWN"]
    inter = shake.ledger.interactions()
    assert inter[0]["state_path"][-1] == "COOLDOWN" and "LIKELY_HANDSHAKE" in inter[0]["state_path"]


# straight horizontal arms: shoulder -> wrist = 2 limbs, so forearm and hand point straight at the other person
C_SH, S_SH = (226, 250), (774, 250)
C_OUT, S_OUT = (426, 250), (574, 250)      # fingertips (wrist + 0.74 forearm) meet at x = 500; palms 0.78 apart


def _touch_frames(hold_s=1.0):
    frames = [[person(1, C_SH, C_OUT), person(2, S_SH, facing=-1)] for _ in range(seconds(1.0))]  # customer holds out
    frames += [[person(1, C_SH, C_OUT), person(2, S_SH, w, facing=-1)]
               for w in lerp((S_SH[0] - 40, S_SH[1] + 180), S_OUT, 12)]                         # staff reaches out
    frames += [[person(1, C_SH, C_OUT), person(2, S_SH, S_OUT, facing=-1)]] * seconds(hold_s)   # fingertips touch
    frames += [[person(1, C_SH, C_OUT), person(2, S_SH, w, facing=-1)]
               for w in lerp(S_OUT, (S_SH[0] - 40, S_SH[1] + 180), 12)]
    return frames + [[person(1, C_SH), person(2, S_SH, facing=-1)]] * 40


def test_held_fingertip_touch_is_likely_contact_not_handshake():
    r = Run(_touch_frames(1.0))
    fam = r.of_type(CONTACT) + r.of_type(HANDSHAKE) + r.of_type(UNCERTAIN)
    assert [e.event_type for e in fam] == [CONTACT], [(e.event_type, e.uncertainty_reason) for e in fam]
    assert "fingertips" in fam[0].interpretation
    assert fam[0].minimum_hand_distance_forearms < 0.2
    assert "POSSIBLE_HAND_CONTACT" in r.states() and "LIKELY_HAND_CONTACT" in r.states()


def test_brief_fingertip_touch_is_not_likely_contact():
    r = Run(_touch_frames(0.3))
    assert r.of_type(CONTACT) == [] and r.of_type(HANDSHAKE) == []


def test_hand_sweeping_over_a_resting_hand_is_not_contact():
    """5real pattern: the customer reaches across, over the staff member's resting (typing) hand, and keeps moving."""
    rest = (S_SHOULDER[0] - 120, 330.0)
    sweep = lerp((420, 330), (640, 330), seconds(0.5)) + lerp((640, 330), (420, 330), seconds(0.5))
    frames = [[person(1, C_SHOULDER), person(2, S_SHOULDER, rest, facing=-1)] for _ in range(seconds(1.0))]
    frames += [[person(1, C_SHOULDER, w), person(2, S_SHOULDER, rest, facing=-1)] for w in sweep * 2]
    frames += [[person(1, C_SHOULDER), person(2, S_SHOULDER, rest, facing=-1)] for _ in range(40)]
    r = Run(frames)
    assert r.of_type(CONTACT) == [] and r.of_type(HANDSHAKE) == []


def test_segment_closest_reports_where_segments_meet():
    d, s, t = segment_closest((0, 0), (10, 0), (20, 0), (12, 0))    # tip to tip, 2 apart
    assert abs(d - 2) < 1e-9 and s == 1.0 and t == 1.0
    d, s, t = segment_closest((0, 0), (10, 0), (5, -5), (5, 5))     # crossing mid-way
    assert d == 0.0 and abs(s - 0.5) < 1e-9 and abs(t - 0.5) < 1e-9


def test_contact_with_both_hands_hidden_is_uncertain():
    """6.uncertain pattern: the hands meet behind an occluder (a laptop). The pose model still places the wrists, but
    with lower confidence for BOTH people; the contact must not be confirmed (assignment: UNCERTAIN = occlusion / low
    keypoint confidence)."""
    hidden = {"c": {"wrist_conf": 0.88}, "s": {"wrist_conf": 0.92}}
    r = Run(idle(15) + reach_in(10) + handshake(seconds(1.5), pump_amp=0, **hidden) + reach_out(10) + idle(40))
    fam = r.of_type(CONTACT) + r.of_type(HANDSHAKE)
    assert fam == []
    unc = r.of_type(UNCERTAIN)
    assert len(unc) == 1 and any("hidden" in u for u in unc[0].uncertainty_reason)


def test_contact_with_one_hand_clearly_visible_is_still_confirmed():
    """mixkit_8909 pattern: one hand at the frame edge (lower confidence), the other clearly visible."""
    one = {"s": {"wrist_conf": 0.85}}
    r = Run(idle(15) + reach_in(10) + handshake(seconds(1.5), pump_amp=0, **one) + reach_out(10) + idle(40))
    assert len(r.of_type(CONTACT) + r.of_type(HANDSHAKE)) == 1


def test_hands_held_near_but_never_touching_is_possible_then_uncertain():
    """possible-hand-contact pattern: both reach out and hold their hands a few cm apart; they never touch."""
    near_c, near_s = (420, 250), (580, 250)    # fingertips 0.24 forearms apart: near, not touching
    frames = [[person(1, C_SH), person(2, S_SH, facing=-1)] for _ in range(seconds(1.0))]
    frames += [[person(1, C_SH, c), person(2, S_SH, s, facing=-1)]
               for c, s in zip(lerp((C_SH[0] + 40, C_SH[1] + 180), near_c, 15),
                               lerp((S_SH[0] - 40, S_SH[1] + 180), near_s, 15))]
    frames += [[person(1, C_SH, near_c), person(2, S_SH, near_s, facing=-1)]] * seconds(1.2)
    frames += [[person(1, C_SH), person(2, S_SH, facing=-1)]] * 40
    r = Run(frames)
    assert r.of_type(CONTACT) == [] and r.of_type(HANDSHAKE) == []
    assert "POSSIBLE_HAND_CONTACT" in r.states()
    unc = r.of_type(UNCERTAIN)
    assert len(unc) == 1 and "never touched" in unc[0].interpretation
