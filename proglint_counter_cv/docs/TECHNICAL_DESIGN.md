# Technical design

## Pipeline

```
video ─► VideoReader (container timestamps, cut detection)
      ─► YOLO26s-pose (OpenVINO runtime) + ByteTrack  (boxes, track IDs, 17 keypoints + confidences)   [cached per video]
      ─► PersonRegistry   roles from calibrated regions, person scale, valid/smoothed hand evidence, ID-switch detection
      ─► PersonCrossingTracker (per person)      ─► BOUNDARY_CROSSING episodes
      ─► PairStateMachine (per customer×staff)   ─► LIKELY_HAND_CONTACT / LIKELY_HANDSHAKE / UNCERTAIN episodes
      ─► EventLedger (single source of truth): observation rows grouped into INTERACTIONS
      ─► CSV / JSON / interactions CSV / dashboard / on-video IDs / evaluation
```

Responsibilities are separated: the **pose model** only estimates keypoints; the **tracker** only associates
detections over time; **geometry** (`calibration`, `trajectory`, `crossing`) decides boundary crossings; the
**interaction logic** (`engine`, `interaction_state`, `classifier`) classifies likely contact, handshake-like motion
or uncertainty; the **event manager** (`event_logger.EventLedger`) owns episodes and interactions; the
**evaluator** (`evaluation`) compares the ledger with labels. No module detects products or objects.

## Model and tracker choice

- **YOLO26s-pose** (Ultralytics 8.4.33, COCO-17 keypoints). Compared on frames of the development clip with
  YOLO11n/s/m-pose: YOLO11n misplaced the dark-sleeved customer's wrist during the handshake; YOLO11m and
  YOLO26s agreed and YOLO26s was faster on this CPU (~0.3 s/frame at imgsz 640 including tracking). Benchmark
  accuracy of the pose model is **not** the accuracy of this application.
- **Runtime: OpenVINO** (`model.backend: openvino`, default on CPU). The same checkpoint is exported once per input
  shape (`models/openvino/yolo26s-pose_<h>x<w>_openvino_model`, ~7 s) with the exact rectangular letterbox shape
  PyTorch would use (`pose_estimator.letterbox_shape`, e.g. 384×640 for 16:9), so preprocessing is identical.
  Measured on `real_test.mp4`: wrist keypoints agree with PyTorch to 0.5 px median (p95 1.4 px, max 11 px), pose
  time 320 → 46 ms/frame in the pipeline. Leave-one-video-out on all 15 labelled videos (logistic classifier
  retrained on each backend's features): crossing 7/1/1 → 7/0/1, contact 11/0/1 → 11/0/1, handshake 9/1/2 → 8/1/3.
  The one handshake lost (`mixkit_5537`, top-down) has 2 pump reversals under PyTorch and 1 under OpenVINO: it sits
  on the `min_motion_reversals: 2` threshold. If export fails the tracker falls back to PyTorch and the run report
  says so. `model.backend: torch` restores PyTorch (use it on a CUDA GPU).
- **ByteTrack** (Ultralytics built-in). Fixed camera, no appearance model needed for a baseline. The track buffer is
  configured in seconds. A new tracker is created per video and reset at camera cuts.

## Geometry

- Boundary = polyline; `signed_distance_to_polyline` gives distance to the nearest segment, signed by side, with end
  segments extended to infinity and averaged normals at vertices (no sign flip at concave corners).
- Roles: a track's shoulder-centre anchor must lie in the customer or staff polygon; a role needs 3 consistent
  votes and only changes if the other role wins the 15-frame window. People outside both regions are ignored.
- **Person scale** = median forearm (elbow→wrist) length of the track. All thresholds named `*_ratio` are in forearm
  lengths, so the same config works at 576p and 1080p.
- **Hand proxy** = wrist + 0.35 × (wrist − elbow). Clasped hands keep the wrist joints about one hand apart; on
  the development clip the wrist distance during the handshake was ~0.9 forearms while the hand-proxy distance
  was ~0.4.

## Boundary-crossing rule (per wrist, then per person)

| Observation | Effect |
|---|---|
| signed distance to home side `d > m` | observation HOME; `d < −m` FAR; `|d| ≤ m` dead band keeps the confirmed side |
| `m` | max(4 px, 0.15 forearms) (hysteresis) |
| side change | confirmed only after 0.10 s of consecutive observations on the new side (debounce) |
| wrist missing > 0.30 s | confirmed side → UNKNOWN; reappearing on the far side is **not** a crossing |
| HOME → FAR | starts a person crossing episode (one event, even if both wrists cross) |
| no wrist on FAR | episode ends (`wrist_returned_home` / `wrist_lost_from_view`); 60 s hard timeout |
| re-cross within 1.0 s of the end | same event re-opened (`rearm_s`) |

## Contact evidence per frame (per customer×staff pair)

`BOTH` (both have a valid hand: closest hand-proxy pair is used), `PARTIAL_NEAR` (one wrist hidden but the visible
hand is within 2.25 forearms of the hidden hand's elbow, i.e. could be touching), `PARTIAL_FAR`, `NONE` (no usable
evidence, track missing, or `ambiguous_pairing`). A hand close to two counterparts with distances within 1.25×
is ambiguous: only one episode carries it and it can only end as `UNCERTAIN`.

**Engaged-hand rule:** once a hand pair has been close, if one of those hands disappears the frame is *missing
evidence*, never "separated" (the person's other, resting hand is not used as proof of separation).

## State machine (one per customer×staff pair)

| From | Condition | To |
|---|---|---|
| NO_ACTIVE_INTERACTION | a wrist of either person in an active crossing episode | BOUNDARY_CROSSED |
| NO_ACTIVE_INTERACTION / BOUNDARY_CROSSED | wrist approaching boundary, or hand distance ≤ 2.0 forearms and decreasing | HAND_APPROACHING |
| any idle state | hand distance ≤ enter 0.6 forearms, or PARTIAL_NEAR, or ambiguous close | POSSIBLE_HAND_CONTACT (episode opens) |
| POSSIBLE_HAND_CONTACT | close time ≥ 0.5 s **and** close/visible frames ≥ 0.7 **and** median relative speed ≤ 3 forearms/s **and** no identity ambiguity | LIKELY_HAND_CONTACT (event created, start back-dated to episode start) |
| LIKELY_HAND_CONTACT | handshake score ≥ 60 % (see below) | LIKELY_HANDSHAKE (same event retyped; the peak score is reported) |
| POSSIBLE / LIKELY_* | evidence NONE or PARTIAL_NEAR this frame | UNCERTAIN (display state; the episode continues) |
| UNCERTAIN | evidence returns within 0.7 s | back to the episode's level |
| any open episode | apart (> exit 0.9 forearms) for 0.4 s / no evidence for 0.7 s / 120 s / scene cut / video end | COOLDOWN (episode ends) |
| COOLDOWN | close again within 1.0 s | same episode re-opened (same event ID) |
| COOLDOWN | 1.0 s elapsed | NO_ACTIVE_INTERACTION (re-armed) |

The machine is not linear: a crossing can occur without contact, uncertainty can appear at any open stage, and a
brief reach never becomes a contact event.

### What an episode becomes when it ends

| Episode evidence | Ledger outcome |
|---|---|
| reached LIKELY_HAND_CONTACT / LIKELY_HANDSHAKE | that event (with any occlusion / ID notes in `uncertainty_reason`) |
| slow close time ≥ 0.2 s but criteria unmet | `UNCERTAIN` ("insufficient evidence: ...") |
| close evidence interrupted by occlusion / identity problems | `UNCERTAIN` (reason listed) |
| one-sided proximity ≥ 0.5 s (counterpart wrist hidden) | `UNCERTAIN` ("staff_wrist_not_visible") |
| ambiguous pairing ≥ 0.2 s | `UNCERTAIN` ("ambiguous_pairing") |
| shorter / fast (pass, wave) | no event; logged in `suppressed_candidates` with the reason |

## Handshake score and plain-language interpretation

The score combines four components, each in 0..1. It is computed only from frames where the hands are properly
together (inside the enter distance):

| Component | Weight | Definition |
|---|---|---|
| duration | 0.30 | close time, 0 at 0.5 s rising to 1 at 1.0 s |
| pumping | 0.35 | (direction reversals of the drift-removed hand midpoint − 1) / (2 − 1); one reversal comes from arriving/leaving |
| moving together | 0.15 | (1 − spread of the hands' relative offset / 0.4), × the share of motion in which the hands did not move relative to each other |
| arms extended | 0.20 | fraction of frames with both shoulder→wrist ≥ 1.3 forearms; 0.5 (neutral) when measurable in < 30 % of frames |

Contact score = min(1, close time / 0.5 s) × close ratio × mean wrist confidence.

The `interpretation` text maps the outcome to a reading:
- handshake (with the cues that fired);
- brief contact < 1.2 s ("consistent with giving/taking an item or a touch");
- sustained contact without handshake motion;
- uncertain with its cause: hand hidden / ambiguous pairing / too brief.

This is a pose-only reading; objects are never detected. The weights are design choices, checked on 3 clips
and then 1 held-out clip (see TEST_RESULTS.md). They were not fitted by training.

## Automatic scene setup

Pass 1 tracks people through the whole video (this needs no counter line). Then:
1. The two main participants are the tracks with the largest (frames present × median box area) that stand at
   least 12 % of the frame diagonal apart.
2. The counter line is the perpendicular bisector of their median shoulder positions: vertical for a side view,
   horizontal for top-down.
3. **Who is staff** (`roles.staff_rule: activity`, side views). The staff member is the person *working at the
   desk*; the customer is the one who comes to it and reaches. Two pose cues over the whole video, each the
   difference between the two people divided by a fixed scale (no fitted weights):
   - hand drop: median height of the wrists below the shoulders, in forearm lengths (scale 0.3). Hands kept up on the
     desk / keyboard → small; arms hanging (standing customer) or resting low → large;
   - hand movement: 75th percentile of wrist speed, forearms/s (scale 1.0). Typing hands are still, reaching hands move.
   The person with the higher-and-stiller hands is staff if |score| ≥ `staff_rule_min_margin` (0.5). Otherwise, and
   in top-down views where wrist height means nothing, customer = left (or top) person and the run report and UI say
   the roles are UNCERTAIN. The swap option flips the result.
   Posture alone ("seated = staff") was rejected: in the project videos staff sits in 8 clips and stands in 2 (2real, 3real),
   customers sit in 5, and knees (the usual seated/standing cue) are hidden by the desk in almost every frame.
   Clothing and ID cards were not used: the customer wears the ID card in 2real and both people wear one in 3real,
   and detecting them would need an object detector, which the assignment excludes.
4. People shorter than 0.35 × the smaller participant get no role (background).

A saved or hand-drawn line is still supported; the same background filter is applied to it.

## Duplicate suppression - why counts do not explode

1. Events are episodes; a record is created once and updated in place (type upgrade, end time).
2. One pair machine per customer×staff pair; left/right wrists are alternatives inside one episode, not separate events.
3. Hysteresis on distance (enter 0.6 / exit 0.9) and on boundary side; side changes need persistence.
4. Episodes end only after sustained separation; cooldown/rearm merge immediate re-contacts and re-crossings.
5. ID switches: if exactly one same-role track was lost within 1 s near where a new track appears, the episode is
   carried over and the change recorded (`track_id_changes`, `track_id_switch_inferred`); if ambiguous, the episode
   is flagged `identity_ambiguous` and cannot be confirmed.
6. Dashboard metrics, exports and on-video IDs all read the same ledger.
7. **Interactions.** Rows are observations, interactions are what the dashboard counts (next section).

## Fingertip touch (palms apart)

Pose models give wrists, not fingers. The palm point (wrist + 0.35 forearm) is right for a clasp, but two open hands
touching only at the fingertips keep their palm points a hand-length apart (0.78 forearms in the user's clip). Each
hand is therefore also modelled as the segment wrist → fingertip, fingertip = wrist + `keypoints.fingertip_ratio`
(0.74, adult hand length / forearm length) along the smoothed forearm direction. `geometry.segment_closest` gives the
shortest distance between the two people's hand segments and where on each hand it occurs.

A separate `TouchEpisode` inside the pair machine runs only while the palm-based Episode is idle, so the trained
classifier's inputs are never changed by it:

| Step | Rule |
|---|---|
| start / continue | segments within `contact.touch_enter_ratio` 0.20 forearms (stay while ≤ `touch_exit_ratio` 0.30) → POSSIBLE_HAND_CONTACT |
| confirm → LIKELY_HAND_CONTACT | touch ≥ `min_contact_s` 0.5 s, ≥ 70 % of the visible frames up to the last touch, relative speed ≤ 3 forearms/s, AND paired-hand evidence: both hands moved out (as for contact), or a **held** touch - both hands extended ≥ 0.30 toward the other person, meeting tip to tip (≥ `touch_tip_min` 0.85 along each hand) and steady (≤ `touch_hold_max_speed` 0.5 forearms/s) |
| end | apart for `separation_s` or no evidence for `missing_tolerance_s`; the whole episode is re-checked and demoted to UNCERTAIN if it no longer meets the rule |
| otherwise | slow touch ≥ `uncertain_min_s` → UNCERTAIN with the failed criteria as reasons; shorter → suppressed (logged) |
| palms come close | the classifier-based Episode takes over; the touch record is closed and shared with it (one record) |

Never a handshake: a handshake needs palms clasped. Why "held tip to tip": a customer reaching *over* the staff
member's typing hand also brings the segments together in the image, but crosses the hand mid-way (meet point
0.73-0.77) while moving (0.86-2.78 forearms/s); the true touch met at 1.00/1.00 at 0.30 forearms/s.

## Occlusion and "near, never touched"

- Contact (palm or fingertip path, live and final decision) is confirmed only if at least one of the two engaged wrists is
  clearly visible: median wrist confidence ≥ `contact.visible_conf` (0.95) over the contact. Both below → UNCERTAIN
  "hands hidden / low keypoint confidence" (the assignment's definition of UNCERTAIN). One clear hand is enough, so a
  close-up with one hand at the frame edge still counts.
- Hand segments within `contact.touch_near_ratio` (0.35) start the touch validation (POSSIBLE_HAND_CONTACT). If they never
  touch but both people offered their hands (moved out, or held out tip-facing and steady) for ≥ `min_contact_s`, the
  episode is recorded as UNCERTAIN "possible hand contact not confirmed - never touched"; otherwise it is only logged.

## Final decisions on the video (two passes)

`pipeline.run_video` first runs the interaction logic without drawing (`_logic_pass`) to learn every episode's final
classification, then runs it again while drawing. `apply_final` replaces decided states on screen with the episode's
final type, and shows UNCERTAIN (instead of POSSIBLE) once the validation window of an episode that ends UNCERTAIN has
passed. The renderer's counts, active-event types and descriptions come from the final ledger. The states recorded for
the dashboard (`state_timeline`, `state_path`) are the displayed ones, so the dashboard and the video always agree.

## The eight states on screen

All eight states of the assignment are computed per customer-staff pair (`interaction_state.State`) and per person
for crossings. The overall state of a frame is the most advanced one (`STATE_PRIORITY`). It is:
- drawn on the video as a strip of all eight states with the current one filled in its colour;
- recorded per frame (`EventLedger.record_states`) as `state_timeline` and `pair_states` in the JSON;
- summarised per interaction as `state_path` (e.g. HAND_APPROACHING → POSSIBLE_HAND_CONTACT → LIKELY_HAND_CONTACT →
  COOLDOWN), shown on the dashboard cards, with an 8-lane state timeline and the assignment's meaning of each state.

## Interactions - the counting unit

One customer-staff exchange produces several ledger rows: the customer's crossing, the staff member's crossing, and
the single contact assessment of the episode (LIKELY_HANDSHAKE, LIKELY_HAND_CONTACT or UNCERTAIN). Counting rows
reported e.g. 7 "actions" for the 4 exchanges in `test.mp4`. Every row therefore belongs to exactly one interaction
(`interaction_id` I01, I02, ...), assigned online by `EventLedger` when the row opens:

| Rule | Why |
|---|---|
| A row joins the most recent interaction that involves one of the same track IDs and is still active or ended ≤ `interactions.join_gap_s` (1.0 s, = crossing re-arm time) before the row started | a reach followed by a handshake is one exchange |
| `EventLedger.link(a, b)` merges two interactions into the older one. The engine links a contact row with the crossing rows of its two people, and (`interactions.link_paired_crossings`) two people's simultaneous crossings while their hands are within `contact.approach_ratio` | an item handed over with both arms across the line, but no contact assessment, is one exchange |
| Rows of different people are never merged by time alone | scenario 7: one customer's reach next to another customer's handshake stays two interactions (test `test_different_people_are_never_merged_by_time_alone`) |
| Outcome = strongest row: LIKELY_HANDSHAKE > LIKELY_HAND_CONTACT > UNCERTAIN > BOUNDARY_CROSSING ("reach, no contact") | a crossing is never reported as contact; an uncertain assessment is never reported as confirmed contact |

Outputs: `interaction_id` column in the event CSV/JSON, `<video>_interactions.csv`, `interactions` and
`interaction_counts` in the JSON, the dashboard cards and headline count, and the run report. All are computed from
the ledger, so they cannot disagree.

## Uncertainty and identity limitations

Track IDs are tracker labels, not identities. ID continuity across an occlusion is *inferred* and always written
into the event. Camera cuts (edited footage) close all episodes and reset tracking; if the new view no longer
matches the calibration fingerprint, analysis pauses and the video shows a red banner.

## Evaluation method

`src/evaluation.py`: categories CROSSING / CONTACT / HANDSHAKE; a prediction matches a label if they overlap after
widening the prediction by ±0.5 s (and roles agree for crossings); greedy one-to-one matching by overlap. UNCERTAIN
events are listed separately, never counted as TP or FP. AMBIGUOUS_CONTACT label intervals exclude overlapping
contact predictions from scoring. Detection delay = confirmation time − label start.

**INTERACTION** (what the dashboard counts): ground-truth interactions are label rows grouped by their optional
`interaction` column, else by time overlap. A group containing a CROSSING / CONTACT / HANDSHAKE row is required; a
group of only AMBIGUOUS_* rows (e.g. an object hand-over) is optional: a predicted interaction matching it is
neither TP nor FP, and missing it is not an FN. Predicted interactions are the ledger's interactions (span of
their rows). The **count error** |predicted − labelled (required + optional)| is reported per video because it is
the number a user compares with what they saw. Pose-keypoint accuracy is a separate question and is not measured
(no keypoint labels).
