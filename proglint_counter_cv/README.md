# Proglint Counter Intelligence
**Counter-Crossing & Customer–Staff Hand-Contact Analytics** (Computer Vision Intern use case 05)

Detects when a wrist crosses a configured counter boundary and classifies likely customer–staff hand contact and
likely handshakes from CCTV, using person tracking and pose keypoints (wrists, elbows, shoulders) over time.
Ambiguous situations are reported as `UNCERTAIN`, and each interaction episode produces exactly one event.
No product, package or object detection is used.

> Pose keypoints show a *likely* contact pattern. They cannot prove physical contact. See docs/LIMITATIONS.md.

## Pipeline
```
video → YOLO26s-pose (OpenVINO) + ByteTrack → roles from calibrated regions → per-person crossing episodes
                                                                  → per customer×staff pair state machine
      → one event ledger (rows grouped into interactions) → annotated MP4 · CSV/JSON log · dashboard · evaluation
```
Details, thresholds and the full state-transition table: [docs/TECHNICAL_DESIGN.md](docs/TECHNICAL_DESIGN.md).

## Requirements
Python 3.12 (tested 3.12.9 on Windows 11, CPU). A GPU is optional (`model.device: "0"`).

```bash
pip install -r requirements.txt
python main.py setup-model          # downloads models/yolo26s-pose.pt (official Ultralytics release)
```

## Quick start - dashboard
```bash
streamlit run app.py                # http://localhost:8501
```
1. **Upload** a video (or pick one from `data/input/`).
2. Press **▶ Analyse video**. Nothing else is required.
3. **Results**: the annotated video plays in the page, next to one card per **interaction** (one customer–staff
   exchange: a reach and the handshake that follows are ONE interaction). Each card gives the outcome, the time,
   a plain-language reading, the evidence scores and the observation rows it is made of; ▶ play jumps the video
   there. The headline counts interactions by outcome: likely handshake, likely hand contact, uncertain, reach
   without contact.
   *Tables & timeline*: interactions and observation rows, a timeline, click a row to jump.
4. **Speed**: "Every frame" analyses each frame (≈ 3× the video's duration on this laptop's CPU, first run);
   "Fast" every 2nd frame (≈ 40 % quicker, may miss brief exchanges). Re-analysing a video takes seconds.

**The eight interaction states** of the assignment (NO_ACTIVE_INTERACTION, HAND_APPROACHING, BOUNDARY_CROSSED,
POSSIBLE_HAND_CONTACT, LIKELY_HAND_CONTACT, LIKELY_HANDSHAKE, UNCERTAIN, COOLDOWN) are shown: as a strip on the video
(current state lit), as chips under the headline (lit = occurred in this video), on every interaction card as its
state path, and in the *Interaction states* tab (meaning, when it occurred, 8-lane timeline).

**The video and the dashboard always agree**: every episode is decided on its complete evidence first, then the video
is drawn with those final results (state strip, labels and counts).

**Hand contact** is detected both when palms clasp (handshake, hand-over) and when only the **fingertips** touch: each
hand is modelled as the segment wrist → fingertip; a touch counts only if it lasts ≥ 0.5 s and both hands reach out
to each other (a hand passing over someone's typing hand does not count).

**How the scene is set up automatically.**
- The two people present most and largest through the whole video are the customer and the staff member.
- The counter line is placed halfway between them.
- People shorter than 35% of the smaller participant (background, passers-by) are ignored: they get no role, so
  they can never produce an event.
- **Staff = the person working at the desk** (hands kept up on the desk/keyboard and still); **customer = the person
  whose hands hang lower and reach across**. Checked on 10 side-view videos, seated or standing, staff on the left or
  right: 10/10. If the hands do not make it clear (or in top-down views) the left/top person is taken as the customer
  and the app warns you. The sidebar's *Swap customer and staff* setting corrects it.

## Command line
```bash
# any video: the counter line and roles are set up automatically (+ evaluation when labels are given)
python main.py run --video data/input/real_test.mp4 --labels data/labels/real_test_gt.csv
python main.py run --video my_new_clip.mp4 --swap-roles          # right/bottom person is the customer

# optional: a saved camera calibration for a fixed CCTV view, and a preview of it
python main.py run --video cam1.mp4 --camera config/cameras/real_test.yaml
python main.py preview --video data/input/real_test.mp4 --camera config/cameras/real_test.yaml --frame 70

# score an existing log
python main.py evaluate --events outputs/events/real_test_events.json --labels data/labels/real_test_gt.csv
```
Options: `--stride 2` (faster), `--max-frames N`, `--no-video`, `--no-cache`.
The first run of a video performs pose inference (~50 ms/frame on CPU with the OpenVINO runtime; the first video
of a new frame shape also spends ~7 s exporting the model once). Its detections are cached in
`outputs/cache/`, so re-running with new thresholds takes seconds.

## Saved camera calibration (optional)
For a fixed CCTV camera you can keep a calibration. `config/cameras/<camera>.yaml` holds the boundary polyline, the two region polygons, the reference resolution and a
view fingerprint. It is valid only for the camera view it was drawn on:
- a different resolution with the same aspect ratio is scaled automatically;
- a different aspect ratio is refused;
- a view that doesn't match the fingerprint produces a warning, and analysis pauses after a camera cut.

An optional `overrides:` block holds per-camera threshold changes (e.g. `roles.min_person_height_frac` to ignore
distant background people).

## Outputs
| File | Content |
|---|---|
| `outputs/annotated/<video>_annotated.mp4` | H.264, same FPS and size as the input: tracks, roles, keypoints, trails, boundary, regions, state, event IDs, distance/confidence, legend |
| `outputs/events/<video>_interactions.csv` | one row per interaction (customer–staff exchange): outcome, times, people, crossings, member rows |
| `outputs/events/<video>_events.csv/.json` | one row per observation episode, with its `interaction_id` (JSON also has the interactions, suppressed candidates, notes and counting rules) |
| `outputs/reports/<video>_run.md/.json` | run statistics, counts, warnings, scene cuts, ID switches |
| `outputs/reports/<video>_evaluation.md/.json` | precision, recall, F1, FP and FN lists, detection delay |

### Reading the event log
`event_type`: `BOUNDARY_CROSSING` (single person, with `crossing_direction`), or exactly one of
`LIKELY_HAND_CONTACT` / `LIKELY_HANDSHAKE` / `UNCERTAIN` for a customer–staff episode.
- `interpretation`: plain-language reading (handshake / giving-taking / brief touch / uncertain + why).
- `interaction_id`: the exchange the row belongs to. **Count interactions, not rows**: a handshake whose two
  hands both crossed the line has 3 rows (2 crossings + 1 contact assessment) but is 1 interaction.
- `handshake_score`, `contact_score`: % output of the trained classifier (below), combined with the fixed handshake
  rules. They are decision scores, **not calibrated probabilities**. Without the model file, rule-based scores are used.

- Distances: `*_px` are original-video pixels; `*_forearms` are multiples of the person's forearm length.
- `evidence_score` is mean keypoint confidence × close ratio. It is a heuristic, **not a probability**.
- `uncertainty_reason`, `start_reason`, `end_reason` and `evidence_summary` explain each decision.
- Empty cells mean "not observable".

## Trained classifier
`models/interaction_model.json` holds two small logistic-regression models trained on labelled episodes from the
project videos: a contact score and a handshake score. They use pose features only:
- how close the hands get (strongest cue);
- how long and how steadily they stay together;
- whether both people reached out;
- pumping motion and hands moving together.

Two assignment rules are applied on top:
- a handshake needs a pumping motion pattern;
- a handshake needs both people to reach out.

The shown percentages are exactly the decision values. To add new labelled videos (`data/labels/<video>_gt.csv`)
and retrain with leave-one-video-out validation:
```bash
python tools/train_classifier.py      # writes models/interaction_model.json + outputs/reports/training_report.md
python tools/benchmark.py             # scores every labelled video
```

## Tests and evaluation
```bash
python -m pytest            # 77 tests: 11 assignment scenarios (23) + interaction counting (18) + states, touch & occlusion (8) + unit/pipeline/auto setup (28)
```
Real-video results, false positives and negatives, and what is NOT TESTED: [docs/TEST_RESULTS.md](docs/TEST_RESULTS.md).
Label format: [data/README.md](data/README.md).

## Project layout
```
app.py  main.py  config/default.yaml  config/cameras/*.yaml
src/  config · geometry · video_io · pose_estimator · person_tracker · trajectory · crossing ·
      interaction_state · engine · event_logger · visualization · evaluation · calibration · pipeline
tests/  synth.py (synthetic scenes) · test_scenarios.py · test_units.py
docs/   REQUIREMENTS · TECHNICAL_DESIGN · TEST_RESULTS · LIMITATIONS · PRESENTATION
```

## Known limitations (summary)
- Image-space (2-D) distance only.
- Object hand-overs look like contact.
- Handshake pumping is invisible from top-down cameras and in close-ups.
- Crossing is judged by the wrist joint.
- Track IDs are temporary.
- Not real-time on CPU.

Full list: [docs/LIMITATIONS.md](docs/LIMITATIONS.md).
