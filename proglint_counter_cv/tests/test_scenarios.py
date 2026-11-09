"""The eleven test scenarios of the assignment (section 5), on synthetic pose sequences.

These tests verify the LOGIC (geometry, temporal rules, state machine, event counting) with exact,
known inputs. They are not evidence of real-world CCTV accuracy - see docs/TEST_RESULTS.md for the
real-video results.
"""
import math

from synth import (C_SHOULDER, S_SHOULDER, Run, handshake, idle, lerp, person, reach_in, reach_out,
                   seconds)

from src.event_logger import CONTACT, CROSSING, HANDSHAKE, UNCERTAIN


# the assignment's jitter scenario is about WRIST keypoints: those tests use the wrist crossing rule
# (the hysteresis/persistence logic is identical for the default hand-point rule)
WRIST_RULE = {"crossing": {"point": "wrist"}}


def contact_family(run):
    return run.of_type(CONTACT) + run.of_type(HANDSHAKE)


# 1 -------------------------------------------------------------------------------------------
def test_s01_customer_reaches_across_staff_does_not_respond():
    reach = lerp((355, 430), (620, 260), 12)
    back = lerp((620, 260), (355, 430), 12)
    frames = idle(15)
    frames += [[person(1, C_SHOULDER, w), person(2, S_SHOULDER, facing=-1)] for w in reach]
    frames += [[person(1, C_SHOULDER, (620, 260)), person(2, S_SHOULDER, facing=-1)]] * seconds(1.0)
    frames += [[person(1, C_SHOULDER, w), person(2, S_SHOULDER, facing=-1)] for w in back]
    frames += idle(30)
    r = Run(frames)
    crossings = r.of_type(CROSSING)
    assert len(crossings) == 1
    assert crossings[0].crossing_direction == "customer_to_staff"
    assert crossings[0].customer_track_id == 1 and crossings[0].staff_track_id is None
    assert contact_family(r) == [] and r.of_type(UNCERTAIN) == []
    assert "BOUNDARY_CROSSED" in r.states()


# 2 -------------------------------------------------------------------------------------------
def test_s02_hands_pass_near_each_other_briefly():
    """Customer's hand sweeps past the staff's extended hand: close for only a few frames, fast."""
    sweep = lerp((380, 250), (700, 250), seconds(0.5))
    frames = idle(15)
    frames += [[person(1, C_SHOULDER, w), person(2, S_SHOULDER, (560, 262), facing=-1)] for w in sweep]
    frames += [[person(1, C_SHOULDER, (700, 250)), person(2, S_SHOULDER, (560, 262), facing=-1)]] * 10
    frames += idle(40)
    r = Run(frames)
    assert contact_family(r) == []
    assert r.of_type(UNCERTAIN) == []
    assert any(s.kind == "contact" for s in r.ledger.suppressed)  # the close pass is traceable


# 3 -------------------------------------------------------------------------------------------
def test_s03_handshake_over_several_frames():
    r = Run(idle(15) + reach_in(10) + handshake(seconds(1.5)) + reach_out(10) + idle(40))
    assert [e.event_type for e in r.ledger.events if e.event_type != CROSSING] == [HANDSHAKE]
    ev = r.of_type(HANDSHAKE)[0]
    assert (ev.customer_track_id, ev.staff_track_id) == (1, 2)
    assert 1.4 <= ev.duration_seconds <= 2.2
    assert "LIKELY_HANDSHAKE" in r.states()


def test_s03b_static_clasp_without_pumping_is_contact_not_handshake():
    r = Run(idle(15) + reach_in(10) + handshake(seconds(1.5), pump_amp=0) + reach_out(10) + idle(40))
    assert [e.event_type for e in contact_family(r)] == [CONTACT]
    ev = r.of_type(CONTACT)[0]
    assert "pumping 0.00" in ev.evidence_summary and ev.handshake_score < 60   # no motion evidence
    assert "giving/taking" in ev.interpretation or "sustained" in ev.interpretation


def test_s03d_pumping_hand_over_without_clasp_is_not_handshake():
    # both reach out and move together, but an item between the hands keeps them ~0.2-0.4 forearms apart
    # (the bottle hand-over measured min 0.17 / mean 0.40); without the clasp rule this scored handshake 84 %
    r = Run(idle(15) + reach_in(10) + handshake(seconds(1.5), gap=80) + reach_out(10) + idle(40))
    assert [e.event_type for e in contact_family(r)] == [CONTACT]
    ev = r.of_type(CONTACT)[0]
    assert ev.handshake_score < 50 and "never clasped" in ev.interpretation


def test_s03c_single_frame_close_is_never_contact():
    frames = idle(15) + reach_in(10)[:-1] + handshake(1) + reach_out(10)[1:] + idle(40)
    r = Run(frames)
    assert contact_family(r) == []


# 4 -------------------------------------------------------------------------------------------
def test_s04_hands_meet_briefly_and_separate_is_uncertain():
    r = Run(idle(15) + reach_in(8) + handshake(seconds(0.25), pump_amp=0) + reach_out(8) + idle(40))
    assert contact_family(r) == []
    unc = r.of_type(UNCERTAIN)
    assert len(unc) == 1
    assert any("insufficient evidence" in x for x in unc[0].uncertainty_reason)


def test_s04b_hands_meet_long_enough_is_likely_contact():
    r = Run(idle(15) + reach_in(8) + handshake(seconds(0.8), pump_amp=0) + reach_out(8) + idle(40))
    assert [e.event_type for e in contact_family(r)] == [CONTACT]
    assert r.of_type(UNCERTAIN) == []


# 5 -------------------------------------------------------------------------------------------
def test_s05_wave_near_the_other_persons_wrist():
    frames = idle(15)
    for i in range(seconds(2.0)):
        ph = 2 * math.pi * 3.0 * i / 30
        w = (430 + 70 * math.sin(ph), 250 + 30 * math.cos(ph))
        frames.append([person(1, C_SHOULDER, w), person(2, S_SHOULDER, (535, 250), facing=-1)])
    frames += idle(40)
    r = Run(frames)
    assert contact_family(r) == []
    assert r.of_type(UNCERTAIN) == []


# 6 -------------------------------------------------------------------------------------------
def test_s06_brief_wrist_occlusion_continues_the_same_episode():
    occluded = {"s": {"wrist_conf": 0.1}}  # below keypoints.wrist_conf -> missing
    frames = (idle(15) + reach_in(10) + handshake(seconds(0.6), pump_amp=0)
              + handshake(seconds(0.3), pump_amp=0, **occluded)
              + handshake(seconds(0.6), pump_amp=0) + reach_out(10) + idle(40))
    r = Run(frames)
    fam = contact_family(r) + r.of_type(UNCERTAIN)
    assert len(fam) == 1 and fam[0].event_type == CONTACT
    assert "staff_wrist_not_visible" in fam[0].uncertainty_reason  # occlusion is recorded, not hidden


def test_s06b_long_occlusion_before_confirmation_is_uncertain():
    occluded = {"s": {"wrist_conf": 0.1, "elbow_conf": 0.1}}
    frames = (idle(15) + reach_in(10) + handshake(seconds(0.3), pump_amp=0)
              + handshake(seconds(1.5), pump_amp=0, **occluded) + idle(40))
    r = Run(frames)
    assert contact_family(r) == []
    unc = r.of_type(UNCERTAIN)
    assert len(unc) == 1
    assert unc[0].end_reason.startswith("evidence_lost")


# 7 -------------------------------------------------------------------------------------------
def test_s07_multiple_customers_and_staff_stay_separate():
    """Pair A (C1, S2) shakes hands at the top; pair B (C3, S4) stands below, not interacting."""
    n = seconds(1.5)
    top = handshake(n, y=150, c_shoulder=(300, 150), s_shoulder=(700, 150))
    frames = []
    for i in range(15):
        frames.append([person(1, (300, 150)), person(2, (700, 150), facing=-1),
                       person(3, (300, 400)), person(4, (700, 400), facing=-1)])
    for a in top:
        frames.append(a + [person(3, (300, 400)), person(4, (700, 400), facing=-1)])
    for i in range(40):
        frames.append([person(1, (300, 150)), person(2, (700, 150), facing=-1),
                       person(3, (300, 400)), person(4, (700, 400), facing=-1)])
    r = Run(frames)
    fam = contact_family(r) + r.of_type(UNCERTAIN)
    assert len(fam) == 1
    assert (fam[0].customer_track_id, fam[0].staff_track_id) == (1, 2)


def test_s07b_hand_equidistant_to_two_staff_is_ambiguous_not_contact():
    """One customer hand between two staff hands at the same distance: pairing is ambiguous."""
    # straight arms (wrist > 2 limbs from shoulder) so the hand proxies are exactly mirror-symmetric
    frames = [[person(1, (300, 250)), person(2, (800, 170), facing=-1), person(3, (800, 330), facing=-1)]] * 10
    for i in range(seconds(1.5)):
        frames.append([person(1, (300, 250), (520, 250)),
                       person(2, (800, 170), (580, 215), facing=-1),
                       person(3, (800, 330), (580, 285), facing=-1)])
    frames += [[person(1, (300, 250)), person(2, (800, 170), facing=-1),
                person(3, (800, 330), facing=-1)]] * 40
    r = Run(frames)
    assert contact_family(r) == []                          # never assigned to either staff member
    unc = r.of_type(UNCERTAIN)
    assert len(unc) == 1 and "ambiguous_pairing" in unc[0].uncertainty_reason


# 8 -------------------------------------------------------------------------------------------
def test_s08_wrist_jitter_around_the_boundary_does_not_create_crossings():
    frames = idle(10)
    for i in range(seconds(4)):
        jitter = 8 * (1 if i % 2 else -1)              # +-8 px: inside the hysteresis band
        frames.append([person(1, C_SHOULDER, (500 + jitter, 300)), person(2, S_SHOULDER, facing=-1)])
    frames += idle(30)
    r = Run(frames, WRIST_RULE)
    assert r.of_type(CROSSING) == []


def test_s08b_large_alternating_jitter_is_debounced():
    frames = idle(10)
    for i in range(seconds(4)):
        jitter = 30 * (1 if i % 2 else -1)             # leaves the band but never persists
        frames.append([person(1, C_SHOULDER, (500 + jitter, 300)), person(2, S_SHOULDER, facing=-1)])
    frames += idle(30)
    assert Run(frames, WRIST_RULE).of_type(CROSSING) == []


def test_s08c_real_crossing_with_jitter_counts_once():
    frames = idle(10)
    path = lerp((420, 300), (600, 300), 20) + [(600, 300)] * 30 + lerp((600, 300), (420, 300), 20)
    for i, (x, y) in enumerate(path):
        frames.append([person(1, C_SHOULDER, (x + 10 * (-1) ** i, y)), person(2, S_SHOULDER, facing=-1)])
    frames += idle(30)
    assert len(Run(frames).of_type(CROSSING)) == 1


# 9 -------------------------------------------------------------------------------------------
def test_s09_long_interaction_is_one_episode_with_start_and_end():
    r = Run(idle(15) + reach_in(10) + handshake(seconds(5)) + reach_out(10) + idle(40))
    events = [e for e in r.ledger.events if e.event_type != CROSSING]
    assert len(events) == 1 and events[0].event_type == HANDSHAKE
    assert events[0].status == "closed"
    assert 4.8 <= events[0].duration_seconds <= 5.8


def test_s09b_brief_release_inside_a_long_handshake_is_not_a_new_event():
    gap = reach_out(4) + reach_in(4)  # ~0.27 s release, shorter than separation_s
    r = Run(idle(15) + reach_in(10) + handshake(seconds(2)) + gap + handshake(seconds(2), t0=60)
            + reach_out(10) + idle(40))
    assert len([e for e in r.ledger.events if e.event_type != CROSSING]) == 1


def test_s09c_regrip_within_cooldown_is_merged_into_the_same_event():
    gap = reach_out(10) + idle(5) + reach_in(10)  # apart ~0.8 s: episode ends, cooldown re-opens it
    r = Run(idle(15) + reach_in(10) + handshake(seconds(1.5)) + gap + handshake(seconds(1.5))
            + reach_out(10) + idle(40))
    events = [e for e in r.ledger.events if e.event_type != CROSSING]
    assert len(events) == 1
    assert "merged" in events[0].evidence_summary


def test_s09d_two_separate_handshakes_are_two_events():
    one = reach_in(10) + handshake(seconds(1.5)) + reach_out(10)
    r = Run(idle(15) + one + idle(seconds(3)) + one + idle(40))
    assert [e.event_type for e in r.ledger.events if e.event_type != CROSSING] == [HANDSHAKE, HANDSHAKE]


# 10 ------------------------------------------------------------------------------------------
def test_s10_track_id_switch_during_brief_occlusion():
    """Staff track 2 vanishes for 5 frames mid-handshake and comes back as track 7."""
    hs1 = handshake(seconds(1.0))
    gap = [[f[0]] for f in handshake(5, t0=30)]           # staff not detected at all
    hs2 = handshake(seconds(1.0), s_tid=7, t0=35)
    out = [[person(1, C_SHOULDER), person(7, S_SHOULDER, facing=-1)] for _ in range(40)]
    r = Run(idle(15) + reach_in(10) + hs1 + gap + hs2 + out)
    events = [e for e in r.ledger.events if e.event_type != CROSSING]
    assert len(events) == 1                               # not double counted
    ev = events[0]
    assert ev.customer_track_id == 1 and ev.staff_track_id == 2   # original identity kept...
    assert ev.track_id_changes == ["staff 2->7"]                   # ...and the switch is explicit
    assert "track_id_switch_inferred" in ev.uncertainty_reason


def test_s10b_ambiguous_id_switch_is_not_silently_resolved():
    """Two staff tracks are lost at the same place; a new one appears: identity is ambiguous."""
    frames = idle(15, s_tid=2)
    frames += [[person(1, C_SHOULDER, (465, 250)), person(2, S_SHOULDER, (535, 250), facing=-1),
                person(3, (720, 260), facing=-1)] for _ in range(seconds(0.3))]
    frames += [[person(1, C_SHOULDER, (465, 250))] for _ in range(4)]
    frames += [[person(1, C_SHOULDER, (465, 250)), person(9, S_SHOULDER, (535, 250), facing=-1)]
               for _ in range(seconds(1.5))]
    frames += idle(40, s_tid=9)
    r = Run(frames)
    assert any("ambiguous" in n for n in r.ledger.notes)
    flagged = [e for e in r.ledger.events if "identity_ambiguous" in e.uncertainty_reason]
    unflagged_contact = [e for e in contact_family(r) if e.staff_track_id in (2, 3)]
    assert flagged or not unflagged_contact


# 11 ------------------------------------------------------------------------------------------
def test_s11_only_one_persons_hand_visible():
    """Customer reaches across; staff's wrist is hidden (elbow visible near the customer's hand)."""
    reach = lerp((355, 430), (590, 250), 12)
    hidden = {"wrist_conf": 0.0}
    frames = idle(15)
    frames += [[person(1, C_SHOULDER, w), person(2, S_SHOULDER, (560, 260), facing=-1, **hidden)]
               for w in reach]
    frames += [[person(1, C_SHOULDER, (590, 250)), person(2, S_SHOULDER, (560, 260), facing=-1, **hidden)]
               for _ in range(seconds(1.5))]
    frames += [[person(1, C_SHOULDER, w), person(2, S_SHOULDER, facing=-1)] for w in reversed(reach)]
    frames += idle(40)
    r = Run(frames)
    assert len(r.of_type(CROSSING)) == 1                  # the observable crossing is reported
    assert contact_family(r) == []                        # no two-person contact claimed
    for e in r.of_type(UNCERTAIN):
        assert "staff_wrist_not_visible" in e.uncertainty_reason
