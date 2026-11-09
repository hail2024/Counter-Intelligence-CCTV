# Limitations

1. **2-D projection.** Hand distance is measured in the image. Hands that overlap in the picture may be far apart
   in depth. All contact outcomes are therefore "LIKELY", never verified contact.
2. **No finger/hand model.** COCO-17 pose has wrists, not hands. The hand position is extrapolated from the forearm
   (hand proxy). Holding the same object (bottle hand-over) looks like hand contact to the system.
3. **The classifier is trained on little data.** 15 videos, with 1 plain-contact example. Its percentages are
   decision scores, not calibrated probabilities (fitted on 16-20 examples with balanced class weights). A fixed clasp rule
   (hands palm to palm in >= 30 % of close frames) stops item hand-overs being called handshakes, but a hand-over
   where the hands wrap around a small item (phone, video 3) still looks clasped; see TEST_RESULTS.md.
   A handshake with no visible pumping (e.g. a quick shake filmed from above) fails the "≥ 2 pump reversals" rule
   and is reported as plain contact (mixkit_5537).
4. **Crossing point and line position.** By default the estimated hand centre must pass the line
   (`crossing.point: hand`; `wrist` is the stricter option). The automatic line sits midway between the two
   people's usual hand zones. A reach that stops just past it (inside the jitter margin) is not counted
   (video 5); a saved camera calibration can adjust the line for a fixed view.
5. **Keypoint jitter and dark clothing.** Wrists on dark sleeves are less reliable (seen with the black hoodie).
   Hysteresis, persistence, smoothing and the forearm plausibility check reduce, not remove, the effect.
6. **Track IDs are temporary.** ByteTrack may switch IDs after occlusion. Unambiguous switches are inferred and
   recorded in the event; ambiguous ones make the episode uncertain. No re-identification across long absences.
7. **Automatic setup assumptions.** It assumes the two people present most and largest are the customer and the
   staff member, facing each other, with the counter between them. Staff = the person whose hands stay up at the
   desk and still; this was checked on 10 side-view clips (10/10) that were also used to choose the rule, so a
   different working style (e.g. a staff member who stands back with hands at the sides) can fool it - the
   sidebar's "Swap customer and staff" setting fixes reversed roles. Top-down views fall back to
   top = customer. If the line lands in the wrong place, use a saved camera calibration. Scenes with several
   customers queueing at once may need a saved calibration.
8. **Edited footage.** Camera cuts reset tracking and close episodes, so an interaction spanning a cut is split.
   CCTV is assumed to be a fixed, uncut view.
9. **Thresholds are provisional.** They were designed on three clips and checked on one new clip (5 contact
   labels in total). That is too little data for a generalisation claim.
10. **CPU speed.** With the OpenVINO runtime ~50 ms/frame of pose inference on the test laptop; a first analysis
    takes ~3× the video's duration, so it is not real-time on CPU. "Fast" mode (every 2nd frame) is ~40 % quicker but
    missed 1 of 4 exchanges on real_test. Detections are cached, so re-analysis takes seconds.
11. **Licences.** Ultralytics models and code are AGPL-3.0 (commercial use needs an Ultralytics licence). Mixkit
    clips are under the Mixkit free licence. Use only footage you are authorised to process.
12. **Runtime trade-off.** OpenVINO and PyTorch give near-identical keypoints (0.5 px median) but not bit-identical
    ones: on a borderline case (a top-down handshake with 2 vs 1 pump reversals) the outcome flipped. The classifier
    is trained on OpenVINO features; switching `model.backend` should be followed by `python tools/train_classifier.py
    --backend torch`.
13. **Interaction grouping is by person and time.** Rows join an exchange when they share a tracked person within
    1 s (or are linked by a contact episode / simultaneous close reaches). A track-ID switch that is not inferred
    splits one exchange into two; two exchanges by the same person less than 1 s apart are counted as one.
14. **Scenarios 8, 10 and 11 have only incidental real-video evidence** (TEST_RESULTS.md lists the footage needed).
15. **Fingertip touch rests on one real example.** The fingertip is extrapolated along the forearm (a bent wrist or
    curled fingers move the real fingertip). The held-touch thresholds (tip ≥ 0.85, speed ≤ 0.5) separate the one
    true touch from the measured non-touches, but more fingertip-touch footage is needed before calling them calibrated.
16. **Occlusion rule rests on one real example.** "Both wrists below 0.95 confidence = hidden" separated the hands behind the
    laptop (0.88 / 0.93) from every visible contact (≥ 0.97 for at least one hand); a different camera or lighting can shift
    confidences. A hand occluded with confident (wrong) keypoints is not detected.
17. **Two passes.** The video is drawn after the decisions, so a live CCTV stream would show decisions with the delay
    of one episode (the offline video shows them from the start of the episode).
