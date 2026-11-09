# Interview presentation outline (≈10 minutes)

1. **Problem and requirements.** CCTV at a counter: detect wrists crossing a configured boundary and likely
   customer-staff hand contact / handshakes, with uncertainty, one event per episode, no product analysis.
   Show the requirement traceability table (docs/REQUIREMENTS.md).
2. **Proposed solution.** Pretrained pose + tracker give keypoints; all interaction logic is my own rule-based,
   episode-based state machine. Demo in the dashboard: upload a video → Analyse (counter line placed automatically)
   → event cards with plain-language readings and percentages → timeline → evaluation.
3. **Architecture.** video_io → pose + ByteTrack (cached) → person registry (roles, scale, hand evidence) →
   crossing trackers + pair state machines → one event ledger → video / CSV / JSON / dashboard / evaluation.
4. **Model and algorithm choices.** YOLO26s-pose chosen after comparing YOLO11n/s/m on real frames. Distances in
   forearm lengths (resolution independent). Hand proxy because pose gives wrists, not hands. Hysteresis +
   persistence for crossings.
5. **Event state machine.** The 8 states, the transition table, and how an episode ends; why counts cannot explode
   (episodes, hysteresis, cooldown merge, ID-switch carry-over, single ledger).
6. **Test scenarios and results.** 11 scenarios as 23 synthetic tests (48 tests in total, all pass). Real video:
   4 clips with automatic setup, every interaction one episode. Contact 5/5 with 0 false contacts; handshake 5/5.
   Hand-overs are read as giving/taking at 15–39 % handshake score. Uncertain near-misses are shown on test.mp4.
   Be explicit that the score was designed on 3 clips and checked on 1 new one.
7. **Failure cases and limitations.**
   - A quick handshake filmed from above, with no visible pump, can fall below 60 %.
   - Object hand-overs look like contact (read as giving/taking).
   - 2-D overlap ≠ contact.
   - Automatic setup assumes the two main people are customer and staff.
   - Crossing not validated on real footage yet.
8. **Future improvements.** Collect and label more CCTV from the target camera; hand keypoint model (e.g. a hand
   landmark model on wrist crops) for finger contact; view-aware handshake criteria; re-identification; GPU / ONNX
   export for real-time; inter-annotator agreement for labels.

## Likely questions and truthful answers

- **How do you avoid counting one handshake many times?** Events are episodes with hysteresis on distance, 0.4 s
  separation to end, and a 1 s cooldown that re-opens the same event. A 13.5 s handshake in mixkit_8909 gave one event.
- **Why not just threshold wrist distance in pixels?** Pixels change with resolution and distance from camera, and
  clasped wrists are a hand-length apart. I measure hand-proxy distance in forearm lengths.
- **Does a short close approach count?** No. It needs 0.5 s of close evidence, 70 % close ratio and slow relative
  motion; a fast pass or wave is suppressed and logged with its reason.
- **How do you know it's a handshake?** A transparent score: time together 30 %, pumping 35 %, hands moving
  together 15 %, arms extended 20 % (neutral when not measurable); ≥ 60 % = likely handshake. Real handshakes scored
  64–97 %, hand-overs 15–39 %. My first all-criteria-must-pass rule found only 1 of 4 handshakes, because arm
  extension isn't measurable from above.
- **What if the tracker switches IDs?** If exactly one same-role track was lost nearby within 1 s, the episode is
  carried over and the switch is written into the event; otherwise it is flagged identity_ambiguous.
- **What is the accuracy?** On 4 clips with 5 labelled contacts: 5/5 contacts and 5/5 handshakes, 0 false
  contacts. The score was designed on 3 of those clips, so only 1 clip is truly held-out. This is preliminary.
- **Did you train a model?** No. The pose model is pretrained. There is not enough labelled interaction data to
  train a contact classifier honestly; with only a handful of labelled episodes it would overfit. The rules are
  explicit and every decision is explained in the log.
- **Who sets up the counter line?** It is placed automatically from the two main people in the video and can be
  corrected with two clicks.
- **Is the pose model's mAP your accuracy?** No. I report the interaction classifier separately; pose accuracy was
  not measured.
- **Why does the bottle hand-over count as contact?** Pose cannot see whether skin touched; I label those
  ambiguous in ground truth and exclude them from scoring rather than claim either way.
