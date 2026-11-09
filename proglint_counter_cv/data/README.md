# Data

## input/ (not committed)
| File | Source | Content | Role in testing |
|---|---|---|---|
| real_test.mp4 | Provided by the user (own recording) | 1024x576, 29.94 FPS, 17.4 s, side view of a desk, 2 foreground + 2 background people | Development clip |
| test.mp4 | Provided by the user (screen recording of stock footage) | 1918x970, 30 FPS, 20 s, top-down desk, edited | Development (used when designing the handshake score) |
| mixkit_8909.mp4 | Mixkit free stock video 8909 (Mixkit Stock Video Free License) | 1280x720, 30 FPS, 14.4 s close-up handshake, first frame black | Development (used when designing the handshake score) |
| mixkit_5537.mp4 | Mixkit free stock video 5537 | 1280x720, 23.98 FPS, top-down desk | Processed as a NEW upload after the handshake score was fixed; similar footage to test.mp4 |
| uploads/2real_testcase … 12real_testcase | Provided by the user (own recordings, 11 clips) | short side-view desk clips, one or two scenarios each | Real-video test cases; scored leave-one-video-out |

Uploads from the UI are stored in input/uploads/.

## Three kinds of data (do not confuse them)

| Kind | What is labelled | Used here for | Status in this project |
|---|---|---|---|
| A. Pose-estimation training data | body keypoints per person (e.g. COCO-Keypoints) | training the pose model | **Not created.** The pretrained YOLO26s-pose checkpoint (COCO-17) is used unchanged. No keypoint labels exist for these videos, so pose accuracy is not measured. |
| B. Interaction-classification data | episodes: crossing, contact, handshake, ambiguous, none | training the small episode classifier (`tools/train_classifier.py`) | The label files below. Candidate episodes produced by the state machine are labelled by overlap with them. |
| C. Evaluation data | the same interval format | measuring the whole system | The same 15 files, used **leave-one-video-out**: each video is scored by a classifier that never saw it. There is no separate untouched test set; see docs/TEST_RESULTS.md. |

A generic pose dataset contains no contact-event labels: contact ground truth has to be labelled on video of the
target camera.

## labels/ - annotation format

`data/labels/<video stem>_gt.csv`, read by `src/evaluation.py`. Lines starting with `#` are comments.

Required columns: `category,start_timestamp,end_timestamp,role,notes`

| Column | Values | Meaning |
|---|---|---|
| category | `CROSSING` | one person's hand crosses the counter boundary (a reach) |
| | `CONTACT` | likely customer-staff hand contact (handshakes included) |
| | `HANDSHAKE` | a handshake (also add a `CONTACT` row for it) |
| | `AMBIGUOUS_CONTACT` | the annotator cannot tell whether hands touched (e.g. an object hand-over): contact predictions here are not scored |
| | `AMBIGUOUS_CROSSING` | the annotator cannot tell whether the hand passed the line: crossing predictions here are not scored |
| start_timestamp / end_timestamp | seconds | interval on the video timeline |
| role | `customer`, `staff` or empty | for CROSSING: whose hand crossed (checked against the predicted direction) |
| notes | text | what happened, in words |

Optional columns (any subset, any order; empty = unknown). Older files without them stay valid.

| Column | Values | Meaning |
|---|---|---|
| video_id | text | video identifier, if the file is used outside its `<stem>_gt.csv` name |
| interaction | any id (A, B, ...) | rows with the same id are ONE customer-staff exchange. Without it, overlapping rows are merged into one exchange. Use it when two different people interact at overlapping times (e.g. `7real_testcase`). |
| customer_track_id / staff_track_id | integer | tracker IDs, when known from a previous run (informational) |
| direction | `customer_to_staff` / `staff_to_customer` | crossing direction (informational; `role` is what is scored) |
| wrist_visibility | e.g. `both_visible`, `customer_wrist_occluded`, `staff_wrist_occluded`, `both_occluded` | occlusion during the interval |
| expected_result | `event`, `uncertain`, `no_event` | what the correct system output is, e.g. `uncertain` for scenario 11 (one hand hidden) |

How the labels are scored (event level, ±0.5 s tolerance) is described in `src/evaluation.py` and
docs/TECHNICAL_DESIGN.md. In short: CROSSING, CONTACT and HANDSHAKE are matched one-to-one; INTERACTION groups the
rows into exchanges as described above, and an exchange made only of AMBIGUOUS rows is optional (matching it is
neither a hit nor a false alarm).

Labels were made by one annotator from frame contact sheets at ~0.15-0.25 s steps, before looking at the
per-event comparison. To label new footage, export frames (any video player) and write the CSV; for footage that
should test a specific assignment scenario, fill `expected_result` and `wrist_visibility`.

Privacy: use only footage you are authorised to process; the system does not identify people (track IDs are
temporary tracker labels).
