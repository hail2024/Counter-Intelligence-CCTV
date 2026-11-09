# Run report - 12real_testcase.mp4

- Video: 848x478, 30.02 FPS, 235 frames (7.8 s)
- Scene setup: automatic - AUTO: counter line midway between the two people's usual hand positions; main participants are tracks 2 and 1 (side by side); customer = track 2, staff = track 1: staff = the person working at the desk (hands kept up and still), score 1.3; people shorter than 164px are treated as background
- Frames analysed: 235 (stride 1), wall time 25.45 s, pose 69.9 ms/frame
- Inference backend: openvino
- Annotated video: D:\proglint_counter_cv\proglint_counter_cv\outputs\annotated\12real_testcase_annotated.mp4

**2 interaction(s)**: 1 likely handshake, 1 likely hand contact, 0 uncertain, 0 reach without contact.

## Interactions (counting unit)

| ID | Outcome | Start | End | Crossings | Rows | What it looks like |
|---|---|---|---|---|---|---|
| I01 | LIKELY_HANDSHAKE | 0.63 | 3.20 | 1 | E001, E002 | Likely handshake (68%): hands together 2.5s, pumping motion, both arms extended |
| I02 | LIKELY_HAND_CONTACT | 4.56 | 6.93 | 0 | E003 | Likely hand contact (1.8s) - hands met but never clasped, consistent with giving/taking an item or a light touch; not a handshake (handshake 15%, contact 84%) |

## Observation rows (event log)

| BOUNDARY_CROSSING | LIKELY_HAND_CONTACT | LIKELY_HANDSHAKE | UNCERTAIN | suppressed candidates |
|---|---|---|---|---|
| 1 | 1 | 1 | 0 | 0 |

| ID | Interaction | Type | Start | End | Handshake % | What it looks like |
|---|---|---|---|---|---|---|
| E001 | I01 | BOUNDARY_CROSSING | 0.70 | 3.20 | - | Customer's hand reached across the counter into the staff side (0.4 forearm lengths past the line) |
| E002 | I01 | LIKELY_HANDSHAKE | 0.63 | 3.06 | 68 | Likely handshake (68%): hands together 2.5s, pumping motion, both arms extended |
| E003 | I02 | LIKELY_HAND_CONTACT | 4.56 | 6.93 | 15 | Likely hand contact (1.8s) - hands met but never clasped, consistent with giving/taking an item or a light touch; not a handshake (handshake 15%, contact 84%) |
