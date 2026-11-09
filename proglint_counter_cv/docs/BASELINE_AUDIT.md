# Baseline audit (2026-10-26, before this round of changes)

Purpose: establish where the remaining errors come from (pose estimation, tracking, calibration, event logic or
missing evidence) **before** changing anything, so that changes are targeted and measurable. Everything below was
checked in the code and by running it; the baseline numbers are kept in
`outputs/reports/benchmark_baseline_2026-10-26.json`.

## 1. What the project actually uses

| Question | Finding (source) |
|---|---|
| Model and checkpoint | `models/yolo26s-pose.pt` (Ultralytics YOLO26s-pose, COCO-17 keypoints), `model.path` in `config/default.yaml`. Loaded by `src/pose_estimator.PoseTracker`, which refuses non-pose models. Other checkpoints in `models/` (yolo11n/s/m-pose) are not used. |
| Pretrained or custom-trained? | **Pretrained, unchanged.** No pose fine-tuning was ever done; there are no keypoint labels for these videos. |
| Something custom-trained? | Yes, a small **episode classifier** (`models/interaction_model.json`, two logistic regressions: contact vs no-contact, handshake vs other contact) trained by `tools/train_classifier.py` on candidate episodes from 15 labelled videos (14 handshake, 1 contact, 9 ambiguous, 5 none). |
| Inference runtime | PyTorch on CPU, `imgsz 640`, `classes=[0]` (person only). Measured 320 ms/frame on `real_test.mp4`. |
| Wrist keypoints / confidence | `res.keypoints.xy` + `.conf` → `PersonDetection.keypoints (17,3)`. A wrist is used only if conf ≥ 0.50, not the (0,0) sentinel, inside the frame, and its forearm length is plausible for that person (`keypoints.forearm_plausibility`). EMA smoothing (α 0.6). The "hand point" is extrapolated 0.35 forearm past the wrist. |
| Person tracking | Ultralytics ByteTrack (`tracking.*`, 1 s buffer). `person_tracker.PersonRegistry` adds role voting, forearm scale and an ID-switch detector (new track appearing near where a same-role track was lost). |
| Customer/staff pairing | Automatic per video (`calibration.auto_calibrate`): the two people present most and largest are the participants; counter line midway between their usual hand positions; left/top = customer (swappable). People shorter than 35 % of the smaller participant are background and are never assigned a role. Only (customer-role, staff-role) track pairs are evaluated; a hand equally close to two counterparts is `ambiguous_pairing`. |
| Boundary crossing | `trajectory.WristSideTracker`: hysteresis band (0.15 forearm, ≥ 4 px), side change must persist 0.10 s, gaps ≤ 0.30 s keep the side, a wrist first seen on the far side never counts. `crossing.PersonCrossingTracker`: one episode per person while any wrist stays across; re-crossing within 1 s (10 s during a contact episode) re-opens the same row. |
| Contact / handshake scores | The state machine computes episode features (close time, close ratio, closest/mean hand distance, engagement of each hand, co-movement, pump reversals). The classifier turns them into P(contact) and P(handshake); fixed rules on top: a handshake needs ≥ 2 pump reversals, both people reaching out, and the palms clasped (hand centres < 0.20 forearm) in ≥ 30 % of close frames. Without the model file, rule-based scores are used. |
| Are the displayed percentages calibrated probabilities? | **No.** They are logistic-regression outputs fitted on 16-20 examples with balanced class weights; nobody measured their calibration. They are decision scores. The README and the uncertainty reasons called them "P(contact)"; the UI now says "evidence score". |
| Counting unit | **Rows of the event ledger.** One row per crossing episode per person, plus one row per customer-staff contact episode. Rows are never per frame (verified: `test_one_row_per_episode_not_per_frame`). |

## 2. Why the user saw "8-9 events for 4 actions"

The screenshot in the request came from the **previous version** of the project (now in
`proglint_counter_cv_previous_backup`). There, `real_test.mp4` (a side view) was analysed with a default
horizontal counter line meant for top-down footage. Consequences visible in the screenshot: every person was labelled
staff, the two background people were tracked, and typing on the laptops was counted as hand crossings. The rebuilt
version (24 Oct) replaced that with automatic per-video setup and a background-size filter; on the same video it
gives 4 rows for the 4 real exchanges.

A second, still-present cause of inflated counts: **one exchange = several rows.** Measured on the baseline runs:

| Video | Real exchanges (labels) | Ledger rows shown to the user |
|---|---|---|
| test.mp4 | 4 (2 handshakes, 2 clipboard pushes) | 7 (4 crossings + 2 handshakes + 1 uncertain) |
| 11real_testcase | 3 | 4 |
| mixkit_5537 | 2 | 4 |
| 7real_testcase | 2 | 3 |

The rows are correct observations (a crossing is not a contact), but the dashboard headline counted rows, so a
handshake with both hands crossing the line looked like three "actions". **Root cause: missing interaction-level
grouping, not detection errors.**

## 3. Where the remaining errors come from (baseline, leave-one-video-out)

| Error | Video | Stage responsible | Evidence |
|---|---|---|---|
| Handshake reported as plain contact | real_test 1.4 s | **pose** | wrist keypoints sit on the black sleeve; the hands look 0.44 forearms apart while clasped, so the clasp rule fails |
| Phone hand-over reported as handshake | 3real_testcase 12.6 s | **missing evidence** (2-D pose) | the hands wrap around the phone; clasped in 55 % of frames; indistinguishable from a handshake with wrists only |
| Handshake reported as UNCERTAIN | 7real_testcase | **pose/tracking (occlusion)** | the woman is hidden behind another person for most of the handshake: correct behaviour per the assignment |
| Reach not counted | 5real_testcase | **calibration** | the hand stops ~15 px past the automatic line, inside the jitter band |
| Extra crossing | 7real_testcase | **event logic / labels** | the woman's hand also crossed during her handshake; the labels only had the man's reach |

None of these is a pose-model capacity problem that fine-tuning on generic data would fix: two are the physical
limits of 2-D wrist keypoints, one is occlusion, one is line placement, one is labelling. **Decision: no pose
retraining.** A fine-tuned pose model would need keypoint-labelled frames from these cameras (wrists under
sleeves and around objects) and a keypoint-error benchmark to prove an improvement; neither exists.

## 4. Speed (the user asked for "very fast")

Profiled on `real_test.mp4` (521 frames): pose inference 320 ms/frame (≈ 166 s), annotated-video rendering
52 ms/frame (≈ 27 s, of which 20 s in `Renderer.draw`: a full-frame boolean mask was rebuilt every frame), writing
mp4v then transcoding to H.264 ≈ 3 s. Interaction logic itself: 4 ms/frame. Whole run ≈ 200 s for 17 s of video.

## 5. Test coverage at baseline

| Kind | What | Result |
|---|---|---|
| Synthetic (no model) | 11 assignment scenarios as 23 test cases + unit tests, `python -m pytest` | 48 passed |
| Real video | 15 labelled clips, `tools/benchmark.py` and leave-one-video-out in `tools/train_classifier.py` | LOVO: crossing 7/1/1, contact 11/0/1, handshake 9/1/2 (TP/FP/FN) |
| Scenario ↔ real footage | Scenarios 1, 2, 3, 4, 5, 7, 9 have real footage. 6, 8, 10, 11 occur only incidentally (see TEST_RESULTS) | — |
| Interaction count | not measured | — |

## 6. Defects found in the tooling during the audit

- `tools/train_classifier.py` defaulted to a decision **tree** while the shipped model and README say **logistic**:
  re-running the documented command silently changed the model type. Fixed (default logistic, `--model tree` opt-in).
- `real_test` and others: CROSSING predictions outside any label are false positives, but crossing rows inside
  labelled contact intervals are excluded from scoring (documented in `evaluation.py`). Kept, documented.
- The ground truth of `7real_testcase` merged two different people's actions when grouped by time overlap; an
  explicit `interaction` column now separates them (`data/labels/7real_testcase_gt.csv`).
