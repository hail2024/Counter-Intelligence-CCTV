# Test results

Run date 2026-10-26, Windows 11, Python 3.12.9, CPU only (12 logical cores, no CUDA), Ultralytics 8.4.33,
YOLO26s-pose (imgsz 640) via OpenVINO 2026.4. Everything below was produced by running the code; raw outputs are in
`outputs/`. Baseline before this round: docs/BASELINE_AUDIT.md and `outputs/reports/benchmark_baseline_2026-10-26.json`.

Reproduce:
```bash
python -m pytest                       # Track A (synthetic), ~20 s
python tools/train_classifier.py       # leave-one-video-out validation + final classifier (~1 min with cached poses)
python tools/benchmark.py --render     # every labelled video with the shipped classifier (+ annotated videos)
```

Two kinds of evidence are kept apart throughout:
- **SYNTHETIC**: exact keypoint sequences fed to the logic (no model). Proves the rules, not real-world accuracy.
- **REAL VIDEO**: the full pipeline on recorded footage, scored against hand-made labels (data/labels).

## Track A - synthetic tests (`python -m pytest` → **77 passed**)

| Group | Tests | What they pin down |
|---|---|---|
| 11 assignment scenarios | `tests/test_scenarios.py`, 23 cases | the expected result of each scenario (table below) |
| Interaction counting & evaluation (new) | `tests/test_interactions.py`, 18 cases | one exchange = one interaction; different people never merged; ledger join/merge rules; interaction-level scoring; optional label columns; letterbox shape for OpenVINO; H.264 writer incl. odd frame sizes |
| States, fingertip touch, occlusion (new) | `tests/test_states_and_touch.py`, 8 cases | all eight assignment states are produced (each by its own scenario) in lifecycle order; a held fingertip touch with palms apart → LIKELY_HAND_CONTACT, never a handshake; a 0.3 s touch → no contact; a hand sweeping over a resting hand → no contact; both wrists hidden → UNCERTAIN; one wrist hidden, the other clear → still contact; hands held near but never touching → POSSIBLE then UNCERTAIN; segment geometry |
| Units & pipeline | `tests/test_units.py`, 28 cases | geometry, hysteresis, config validation, calibration, ledger, evaluation, auto setup incl. who-is-staff (staff on either side; uncertain fallback), end-to-end pipeline with a mocked model |

## Track B - real video (15 labelled clips)

Clips: the user's 11 test cases (`2real_testcase` … `12real_testcase`) and `real_test`, the user's `test`, and two
Mixkit stock clips. 31 labelled exchanges: 19 required (11 handshakes, 1 other contact, the rest reaches) and 12
ambiguous ones made only of AMBIGUOUS_* rows (object hand-overs, shared paperwork). Label rows: 12 CONTACT,
11 HANDSHAKE, 8 CROSSING, 11 AMBIGUOUS_CONTACT, 3 AMBIGUOUS_CROSSING. One annotator; labels in `data/labels/`.

### Honest accuracy: leave-one-video-out (each video scored by a classifier trained without it)

| Category (TP / FP / FN) | Baseline: PyTorch inference | **Now: OpenVINO inference (shipped)** |
|---|---|---|
| Crossing | 7 / 1 / 1 | **7 / 0 / 1** |
| Contact (incl. handshakes) | 11 / 0 / 1 | **11 / 0 / 1** |
| Handshake | 9 / 1 / 2 | **8 / 1 / 3** |

Same protocol, same labels, same logistic model type; only the inference runtime differs. The re-run on PyTorch
reproduced the documented baseline exactly. Differences: the false crossing in `7real_testcase` is gone; the
top-down handshake in `mixkit_5537` is lost (1 pump reversal instead of 2 at the `min_motion_reversals` threshold).
Reports: `outputs/reports/training_report.md` (OpenVINO) and `training_report_model_torch.md` (PyTorch).

### Shipped model on all videos (in-sample; the classifier saw these videos in training)

`python tools/benchmark.py --render` → `outputs/reports/benchmark.json`

| Video | Labelled exchanges | Predicted interactions | Interaction TP/FP/FN | What the dashboard shows |
|---|---|---|---|---|
| 10real_testcase | 2 | 2 | 1/0/0 | I01 handshake, I02 reach (paper exchange, ambiguous) |
| 11real_testcase | 3 | 3 | 2/0/0 | handshake, reach, reach |
| 12real_testcase | 2 | 2 | 1/0/0 | handshake, bottle hand-over as contact (ambiguous) |
| 2real_testcase | 1 | 1 | 1/0/0 | uncertain - "most likely a one-sided reach" (no contact claimed) |
| 3real_testcase | 4 | 3 | 2/0/0 | handshake, small-item contact, phone hand-over; the first phone hand-over produced no event |
| 4real_testcase | 2 | 1 | 1/0/0 | handshake; the pen hand-over produced no event |
| 5real_testcase | 1 | 1 | 1/0/0 | uncertain one-sided reach (crossing missed, see errors) |
| 6real_testcase | 1 | 1 | 1/0/0 | reach, no contact |
| 7real_testcase | 2 | 2 | 2/0/0 | the man's reach; the hidden woman's handshake as uncertain |
| 8real_testcases | 1 | 1 | 1/0/0 | contact (fingers on the keyboard beside the staff's, ambiguous) |
| 9real_testcase | 1 | 1 | 1/0/0 | reach, no contact |
| mixkit_5537 | 2 | 2 | 1/0/0 | handshake reported as contact; clipboard push uncertain |
| mixkit_8909 | 1 | 1 | 1/0/0 | one 13 s handshake = one interaction |
| real_test | 4 | 4 | 1/0/0 | handshake reported as contact; 3 object hand-overs uncertain |
| test | 4 | 4 | 2/0/0 | 2 handshakes, clipboard push uncertain, clipboard push as reach |
| **Total** | **31** | **29** | **19 / 0 / 0** | count exact on **13 / 15** videos; sum of count errors 2 |

Category totals in-sample: crossing 7/0/1, contact 11/0/1, handshake 8/0/3.
Interaction TP counts required exchanges only; ambiguous exchanges (object hand-overs) are optional (TECHNICAL_DESIGN,
"Evaluation method"), so "Labelled exchanges" > required labels on several rows.

Before the interaction grouping the dashboard counted rows: `test.mp4` showed 7, `mixkit_5537` 4, `11real` 4,
`7real` 3 for 4, 2, 3 and 2 real exchanges.

### The user's scenario clips (REAL VIDEO, named after the expected result; no label files)

| Clip | Expected (name) | Output | State path (= the video's state strip) |
|---|---|---|---|
| 1.boundary crossed | crossing | BOUNDARY_CROSSING (customer → staff) | HAND_APPROACHING → BOUNDARY_CROSSED |
| 4.likely hand contact | likely contact | LIKELY_HAND_CONTACT (fingertips) | HAND_APPROACHING → POSSIBLE → LIKELY_HAND_CONTACT |
| 5.hand_approaching or possible hand contact | approaching / possible | BOUNDARY_CROSSING (reach, no contact) | HAND_APPROACHING → BOUNDARY_CROSSED |
| 6.uncertain | uncertain | UNCERTAIN - hands met behind the laptop (both wrists low confidence) | HAND_APPROACHING → BOUNDARY_CROSSED → POSSIBLE → UNCERTAIN |
| Likely handshake | handshake | LIKELY_HANDSHAKE (81 %) | HAND_APPROACHING → POSSIBLE → LIKELY_HANDSHAKE |
| possible hand contact | possible contact | UNCERTAIN - both reached out, hands within 0.23 forearms for 1.2 s, never touched | HAND_APPROACHING → POSSIBLE → UNCERTAIN |

6/6 match their names. Three fixes were needed (all checked against the 15 labelled videos: scores unchanged,
leave-one-video-out unchanged, shipped classifier file unchanged):
- **Handshake demoted at the end** (Likely handshake): the clasp share fell from >30 % to 22 % as more frames were
  added. The clasp rule now uses the peak share after 0.5 s together (like the pump count).
- **Occluded hands confirmed as contact** (6.uncertain): contact now needs at least one wrist clearly visible (median
  confidence ≥ 0.95). Hands behind the laptop: 0.88 / 0.93; every visible contact in the data: one hand ≥ 0.97.
- **Near but never touching gave nothing** (possible hand contact): hands within 0.35 forearms start the validation
  (POSSIBLE_HAND_CONTACT); held ≥ 0.5 s by two people offering their hands → UNCERTAIN "never touched".

**Video = dashboard.** Decisions are made on each episode's complete evidence when it ends, but the video used to be
drawn live, so it could show LIKELY_HANDSHAKE for an episode the log then reported as contact. The pipeline now runs the
decision logic first (≈ 3 ms/frame) and draws the video in a second pass with the final result of every episode: state
strip, labels, counts and the recorded states all come from the final ledger. Checked on all 21 videos: no frame shows a
decided state that the event log does not contain.

### Fingertip touch (REAL VIDEO, 26 Oct)

`4.likely hand contact.mp4` (user upload, 3.3 s, no label file): two open hands meet fingertip to fingertip. Before:
0 interactions, the state never left HAND_APPROACHING - the palm points stayed 0.78 forearms apart, above the 0.60
"close" threshold, because the pose model only gives wrists and the palm estimate cannot see fingers. The keypoints
themselves were correct (checked on the frames).

Now each hand is also the segment wrist → fingertip (fingertip = wrist + 0.74 forearm, the adult hand/forearm length
ratio). Result: **1 interaction, LIKELY_HAND_CONTACT 2.23-3.27 s**, states HAND_APPROACHING → POSSIBLE_HAND_CONTACT →
LIKELY_HAND_CONTACT. Not scored (no labels), but it matches the clip's name and the frames.

The touch rule was checked against every touch-only moment in the 15 labelled videos (segments within 0.20 forearms
while palms are not close):

| Touch-only episode | relative speed (forearms/s) | meet point on the hands (1 = fingertip) | outcome |
|---|---|---|---|
| 4.likely hand contact (true touch) | 0.30 | 1.00 / 1.00 | LIKELY_HAND_CONTACT |
| 5real (customer reaches OVER the typing hands, no touch) | 0.86 / 2.78 | 0.73 / 0.77 | UNCERTAIN (unchanged) |
| 11real, 12real, 10real, 3real (paper / bottle / phone hand-overs, labelled ambiguous) | 0.46-1.52 | mixed | contact or uncertain, not scored |
| real_test, test (brief, < 0.25 s) | 0.86-2.12 | mixed | no change |

All 15 labelled videos give the same scores as before the change (crossing 7/0/1, contact 11/0/1, handshake 8/0/3,
interaction 19/0/0, count exact on 13/15); leave-one-video-out also unchanged (handshake 8/1/3). Only ONE true fingertip
touch exists in the data, so the touch thresholds are provisional (LIMITATIONS 15).

Two defects found while doing this and fixed with regression coverage: a touch record could stay "open" after the
palm logic took over (it absorbed a later, separate reach into the same interaction in 11real), and a pair in
COOLDOWN could report a stale record of an earlier episode.

### The eight states, as produced on real video

`<video>_events.json` now holds `state_timeline` (overall state over time) and `pair_states` (per customer-staff pair);
each interaction has a `state_path`. Over the 17 user videos all eight states occur:

| State | Example (video, time) |
|---|---|
| NO_ACTIVE_INTERACTION | every video before the first approach |
| HAND_APPROACHING | 4.likely hand contact 1.07-2.20 s |
| BOUNDARY_CROSSED | 11real 7.95-8.80 s (customer reaches to the laptop) |
| POSSIBLE_HAND_CONTACT | 4.likely hand contact 2.23-2.67 s (validation in progress) |
| LIKELY_HAND_CONTACT | 4.likely hand contact 2.70-3.27 s |
| LIKELY_HANDSHAKE | 3real 2.41-3.64 s, 11real 1.81-2.82 s |
| UNCERTAIN | 11real 9.85-10.01 s, then single frames to 10.70 s (one hand's evidence drops in and out) |
| COOLDOWN | 11real 2.85-3.80 s (after the handshake) |

A state that does not occur in a video is the correct output for that video (e.g. no LIKELY_HANDSHAKE without a
handshake). Display smoothing added: a hand hovering close stays HAND_APPROACHING, and a person's "approaching the line"
flag is held 0.3 s, so the displayed state no longer flickers frame to frame.

### Who is staff (role assignment, REAL VIDEO)

Truth read from the frames for the 10 side-view clips where it is unambiguous (the person working at the laptop).
Staff sits in 8 of them and stands in 2 (2real, 3real); in 5 the customer also sits.

| Rule | Correct | Wrong |
|---|---|---|
| Before: left person = customer | 7 / 10 | 3real, 7real, `5.hand_approaching or possible hand contact` (staff seated on the LEFT) |
| **Now: staff = hands kept up at the desk and still** | **10 / 10** | - (closest call 3real, score 0.81 vs margin 0.5) |

The rule was chosen on these same 10 clips (no held-out set). Not verifiable from the frames, so not scored:
12real (both standing), the top-down clips and the stock clips. Event accuracy is unchanged by the role fix
(labels do not record roles; leave-one-video-out after retraining: crossing 7/0/1, contact 11/0/1, handshake 8/1/3).
The new upload `5.hand_approaching or possible hand contact.mp4` has no labels: it now reports 1 interaction,
"customer's hand reached across into the staff side" (1.4-3.6 s); NOT scored.

### Speed (`real_test.mp4`, 521 frames, 17.4 s of video, this laptop's CPU)

| | Before (24 Oct) | Now |
|---|---|---|
| Pose inference | 320 ms/frame (PyTorch) | **49 ms/frame** (OpenVINO, same checkpoint) |
| First analysis of the video, end to end | ~200 s | **54 s** (+ ~7 s once per new video shape for the OpenVINO export) |
| Re-analysis with cached poses incl. the annotated video (e.g. after swapping roles) | 29 s | **10 s** (overlay mask computed once, frames piped straight to H.264) |
| "Fast" mode in the UI (every 2nd frame), first analysis | - | **32 s**, but it found **3 of the 4** exchanges in real_test (one uncertain hand-over missed). Measured on this one video only; the accuracy tables above are for the default mode |

Still not real-time on CPU (≈ 3× slower than real time on a first run).

## The 11 assignment scenarios

| # | Scenario → expected | SYNTHETIC result | REAL VIDEO evidence | Status |
|---|---|---|---|---|
| 1 | Customer reaches across, staff does not respond → crossing, no contact | PASS (1 crossing, no contact) | 9real, 6real: reach, no contact ✔. 2real: crossing + "uncertain, most likely one-sided reach", no contact ✔. 5real: no contact ✔ but the crossing itself is missed ✘ | REAL-TESTED, 1 miss |
| 2 | Hands pass near each other briefly → no contact from proximity alone | PASS (logged as suppressed) | 6real: reach to the mouse beside the staff's hand, no contact ✔. Close passes suppressed as "fast relative motion" in 11real and real_test ✔ | REAL-TESTED |
| 3 | Handshake over several frames → likely handshake if criteria met | PASS (+ static clasp → contact, pumping hand-over → contact, 1 close frame → nothing) | 11 labelled handshakes: 8 found as LIKELY_HANDSHAKE (LOVO); 2 reported as contact (real_test sleeve, mixkit_5537 top-down), 1 uncertain (7real occlusion). 1 phone hand-over called a handshake (3real) | REAL-TESTED, 3 FN / 1 FP |
| 4 | Hands meet briefly and separate → likely contact only if evidence suffices, else uncertain | PASS (0.25 s → uncertain, 0.8 s → contact) | 3real small-item touch → LIKELY_HAND_CONTACT ✔ (the only labelled plain contact). Object hand-overs in real_test → UNCERTAIN ✔ | REAL-TESTED (1 example) |
| 5 | Wave near the other person's wrist → no contact unless paired evidence | PASS | 9real: customer waves toward the typing staff member → reach only, no contact ✔ | REAL-TESTED (1 example) |
| 6 | Wrist occluded → continue only if reliable, else uncertain | PASS (short gap: same event; long gap before confirmation: uncertain) | 7real: woman hidden behind another person during the handshake → UNCERTAIN, no contact claimed ✔. mixkit_8909: customer wrist briefly lost inside a 13 s handshake → still one handshake ✔ | INCIDENTAL real evidence (no dedicated clip) |
| 7 | Multiple customers and staff → pairs kept separate | PASS (+ equidistant hand → 1 uncertain "ambiguous pairing") | 7real (two customers): 2 separate interactions with the right people ✔. real_test: 2 background people ignored ✔ | REAL-TESTED (1 clip) |
| 8 | Wrist jitter at the boundary → no repeated crossings | PASS (jitter → 0, real crossing with jitter → 1) | No clip films a hand resting on the line. Incidental: 3real re-crossings merged into one row; 7 far-side appearances suppressed in mixkit_8909; no video produced repeated crossing rows | **NEEDS FOOTAGE** (hand resting on the counter line) |
| 9 | Contact lasting several seconds → one episode with start/end | PASS (+ brief release, re-grip, two separate handshakes → 2) | mixkit_8909: 13.5 s handshake → exactly 1 interaction / 1 contact row ✔ | REAL-TESTED |
| 10 | Track ID changes during brief occlusion → no wrong attribution; uncertain if needed | PASS (unambiguous switch carried over and recorded; ambiguous switch flagged) | 4real (customer 2→4) and 7real (staff 3→6): switches inferred, crossing continued, recorded in the run notes. Neither happened during a contact episode | INCIDENTAL; **NEEDS FOOTAGE** (ID switch in the middle of a handshake) |
| 11 | Only one person's hand visible → no two-person contact claim | PASS (crossing reported, no contact) | 7real, test.mp4: "staff wrist not visible" episodes ended UNCERTAIN, never contact ✔ | INCIDENTAL; **NEEDS FOOTAGE** (counter or monitor hiding one hand) |

Footage still needed to test scenarios 8, 10 and 11 directly: a hand resting on the counter line for several
seconds; a customer walking behind a pillar mid-interaction; a monitor or counter edge hiding the staff member's hand
while the customer reaches. Label them with `expected_result` and `wrist_visibility` (data/README.md).

## Remaining errors (all real, not hidden)

| Video | Truth | Output | Stage | Why |
|---|---|---|---|---|
| real_test 1.4 s | handshake | LIKELY_HAND_CONTACT | pose | wrist keypoints on the black sleeve: hands look 0.44 forearms apart, so the clasp rule fails |
| mixkit_5537 0.8 s | handshake (top-down) | LIKELY_HAND_CONTACT | missing evidence | pumping is barely visible from above: 1 pump reversal, rule needs 2 |
| 7real_testcase | handshake, woman hidden behind another person | UNCERTAIN | occlusion | she is not detected for most of the handshake; the assignment says mark uncertain |
| 3real_testcase 12.6 s | phone hand-over (ambiguous) | LIKELY_HANDSHAKE when held out | 2-D pose limit | hands wrap around the phone (clasped 55 % of frames) |
| 5real_testcase | reach to the power bank | no crossing | calibration | the hand stops ~15 px past the automatic line, inside the jitter band; draw the line in that case |
| 3real 4.0 s, 4real 4.9 s | phone / pen hand-over (ambiguous) | no event | evidence | the item keeps the hands apart; these exchanges are not counted (count error 1 each) |

## NOT TESTED / limits of these numbers

- Pose keypoint accuracy (no keypoint ground truth). OpenVINO vs PyTorch agreement was measured instead (0.5 px median).
- One annotator, no agreement study. Only one plain (non-handshake) contact example: the contact vs handshake
  boundary is the weakest part.
- There is no untouched test set: the 15 clips are used for training and leave-one-video-out validation. The LOVO
  numbers are the honest estimate; the in-sample table is optimistic.
- 15 short clips from a few rooms. **Not** enough to claim accuracy on arbitrary new cameras. The system is not
  declared fully validated.
- "Fast" (every 2nd frame) mode beyond the single real_test check above; GPU inference; videos longer than 25 s; camera views other than desk
  side/top-down.
