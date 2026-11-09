# Requirements traceability

Source: `usecase_05.pdf` (Computer Vision Interview Assignment, sections 1-7). Status: **Done** = implemented and
tested; **Partial** = implemented, with the limitation noted.

## Objectives (section 2) and functional requirements (section 3)

| # | Requirement | Implementation | Verified by | Status |
|---|---|---|---|---|
| R1 | Detect and track people near the counter; keep track IDs where possible | `pose_estimator.PoseTracker` (YOLO26s-pose via OpenVINO + ByteTrack), `person_tracker.PersonRegistry` | real-video runs; `test_pipeline_end_to_end_with_mock_model` | Done |
| R2 | Estimate wrist, elbow, shoulder keypoints | COCO-17 indices in `pose_estimator.KP/ARM`; validity in `keypoint_valid` | `test_zero_sentinel_keypoint_is_invalid` | Done |
| R3 | Configure counter boundary and customer/staff regions per camera | automatic per video (`calibration.auto_calibrate`), optional two-click line (UI) or saved `config/cameras/*.yaml` | `test_auto_calibration_*`, `test_two_click_line_calibration`, `test_calibration_*` | Done |
| R4 | Associate customer and staff sides using regions / scene rules | automatic setup picks staff = the person working at the desk (`calibration.staff_side_score`), regions follow; sticky role voting on shoulder anchor in `person_tracker`; background-size rule | `test_auto_calibration_staff_is_the_person_working_at_the_desk`, S07; real: staff correct in 10/10 side-view clips | Done |
| R5 | Extract wrist keypoints **and confidences** per tracked person | `HandObs.conf`; logged as `keypoint_conf_mean/min` | event logs | Done |
| R6 | Wrist crossing via a temporally consistent trajectory | `trajectory.WristSideTracker` (hysteresis + persistence + gap rule), `crossing.PersonCrossingTracker` | S01, S08a-c, `test_side_tracker_*`; 8 labelled real crossings | Done (real video: 7/8 found, see TEST_RESULTS) |
| R7b | Hand contact when only the fingertips touch | hand segments wrist → fingertip, `TouchEpisode` rules | `test_held_fingertip_touch_*`, `test_hand_sweeping_over_*`; real: 4.likely hand contact | Done (1 real example) |
| R7 | Measure customer-staff wrist distance over time | `engine._pair_frame` (hand-proxy and raw wrist distance, px and forearm units) | all contact scenarios | Done |
| R8 | Configurable proximity, confidence and duration criteria | `config/default.yaml` sections `keypoints`, `contact`, `handshake`, `classifier`, `interactions` (validated at start-up) | `test_invalid_config_rejected` | Done |
| R9 | No single-frame handshake/contact | `contact.min_contact_s`, `min_close_ratio` | S02, S03c | Done |
| R10 | Likely handshake only with multi-frame motion support | episode classifier P(handshake) > 0.5 AND ≥ 2 pump reversals AND both people reaching out AND palms clasped in ≥ 30 % of close frames (rule-based score as fallback) | S03, S03b, S03d, S09; real: 8/11 handshakes leave-one-video-out | Partial (top-down pumping is hard to see) |
| R11 | Distinguish crossing/reach from likely contact; ambiguous -> uncertain | Separate event families; `UNCERTAIN` outcomes with reasons | S01, S04, S06b, S07b, S11 | Done |
| R12 | End episode on separation or timeout; suppress duplicates | `separation_s`, `missing_tolerance_s`, `max_episode_s`, cooldown re-open, crossing `rearm_s` | S09a-d, S08c | Done |
| R13 | Handle missing keypoints, occlusion, overlapping hands, ID switches without fabricating contact | engaged-hand check, `ambiguous_pairing`, `IdSwitch` handling, `PARTIAL_*` evidence | S06, S06b, S07b, S10, S10b, S11 | Done |
| R14 | No product/package/object movement in event logic | No object model is loaded; only person class (`classes=[0]`) | code review | Done |
| R15 | One event per interaction episode, timestamped log | `event_logger.EventLedger`: one row per observation episode, rows of one exchange grouped into one INTERACTION (`interaction_id`, `<video>_interactions.csv`) | `test_one_row_per_episode_not_per_frame`, S09, `tests/test_interactions.py`; real: interaction count exact on most videos (TEST_RESULTS) | Done |
| R16 | Do not inflate the number of unique interaction episodes (crossing + contact of one exchange = 1) | interaction grouping rules in TECHNICAL_DESIGN "Interactions" | `test_handshake_with_its_crossings_is_one_interaction`, `test_reach_then_handshake_soon_after_is_one_interaction`, `test_different_people_are_never_merged_by_time_alone` | Done |
| R17 | Usable speed on a CPU laptop | OpenVINO runtime (same checkpoint), cached detections, in-place rendering, direct H.264 pipe | measured in TEST_RESULTS (speed table) | Done (not real-time) |

## State machine (section 4)

All eight states are in `interaction_state.State`; transitions are documented in TECHNICAL_DESIGN.md and
exercised by the scenario tests (state sequence asserted in S01 and S03; all eight produced in
`test_all_eight_states_occur`). They are visible on the video (state strip), in the JSON (`state_timeline`,
`pair_states`, per-interaction `state_path`) and on the dashboard (state chips, "Interaction states" tab).
All eight occur on the user's real videos (TEST_RESULTS "The eight states").

## Test scenarios (section 5)

| Scenario | Test(s) in `tests/test_scenarios.py` |
|---|---|
| 1 Customer reaches across, staff does not respond | `test_s01_*` |
| 2 Hands pass near each other briefly | `test_s02_*`, `test_s03c_*` |
| 3 Handshake over several frames | `test_s03_*`, `test_s03b_*` |
| 4 Hands meet briefly and separate | `test_s04_*`, `test_s04b_*` |
| 5 Wave near the other person's wrist | `test_s05_*` |
| 6 Wrist occluded during interaction | `test_s06_*`, `test_s06b_*` |
| 7 Multiple customers and staff | `test_s07_*`, `test_s07b_*` |
| 8 Wrist jitter around boundary | `test_s08_*`, `test_s08b_*`, `test_s08c_*` |
| 9 Episode lasting several seconds | `test_s09_*` (a-d) |
| 10 Track ID change during brief occlusion | `test_s10_*`, `test_s10b_*` |
| 11 Only one person's hand visible | `test_s11_*` |

## Deliverables (section 7)

| Deliverable | Where |
|---|---|
| Runnable source + setup | `main.py`, `app.py`, `src/`, README.md |
| Configuration (boundary, regions, keypoint confidence, proximity, duration) | `config/default.yaml`, `config/cameras/*.yaml` |
| Annotated video | `outputs/annotated/*_annotated.mp4` (H.264; plays inside the dashboard) |
| CSV/JSON event log | `outputs/events/*_events.csv/.json`, `outputs/events/*_interactions.csv` |
| Baseline audit | docs/BASELINE_AUDIT.md |
| Annotation format | data/README.md |
| Explanation of rules, transitions, suppression, identity limits | docs/TECHNICAL_DESIGN.md, docs/LIMITATIONS.md |
| Test results incl. FP/FN/uncertain | docs/TEST_RESULTS.md, `outputs/reports/` |

## Design decisions not fixed by the assignment (assumptions)

- Crossing is tested on the estimated **hand point** (wrist + 0.35 forearm) by default, because a reaching hand
  passes the line before the wrist joint does; `crossing.point: wrist` applies the assignment's literal wrist rule
  (the jitter tests use it). Both use the same hysteresis/persistence logic.
- Distances are normalised by the person's median forearm length so that thresholds transfer across resolutions.
- Proximity uses a "hand proxy" extrapolated from the forearm because pose models locate wrists, not hands.
- Handshake = likely contact that the trained classifier and the fixed motion/clasp rules accept (see TECHNICAL_DESIGN.md).
- The dashboard counts interactions (exchanges), not ledger rows.
- The counter line is placed automatically between the two main participants; the left/top person is the customer unless swapped.
- An object hand-over where both hands hold the object is labelled `AMBIGUOUS_CONTACT` in ground truth.
