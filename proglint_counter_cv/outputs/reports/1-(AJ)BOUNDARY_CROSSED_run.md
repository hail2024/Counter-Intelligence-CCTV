# Run report - 1-(AJ)BOUNDARY_CROSSED.mp4

- Video: 1024x576, 29.83 FPS, 176 frames (5.9 s)
- Scene setup: automatic - AUTO: counter line midway between the two people's usual hand positions; main participants are tracks 1 and 2 (side by side); customer = track 1, staff = track 2: staff = the person working at the desk (hands kept up and still), score 5.7; people shorter than 104px are treated as background
- Frames analysed: 176 (stride 1), wall time 18.2 s, pose 55.6 ms/frame
- Inference backend: openvino
- Annotated video: D:\proglint_counter_cv\proglint_counter_cv\outputs\annotated\1-(AJ)BOUNDARY_CROSSED_annotated.mp4

**1 interaction(s)**: 0 likely handshake, 0 likely hand contact, 0 uncertain, 1 reach without contact.

## Interactions (counting unit)

| ID | Outcome | Start | End | Crossings | Rows | What it looks like |
|---|---|---|---|---|---|---|
| I01 | BOUNDARY_CROSSING | 2.00 | 5.77 | 1 | E001 | Customer's hand reached across the counter into the staff side (0.3 forearm lengths past the line) |

## Observation rows (event log)

| BOUNDARY_CROSSING | LIKELY_HAND_CONTACT | LIKELY_HANDSHAKE | UNCERTAIN | suppressed candidates |
|---|---|---|---|---|
| 1 | 0 | 0 | 0 | 0 |

| ID | Interaction | Type | Start | End | Handshake % | What it looks like |
|---|---|---|---|---|---|---|
| E001 | I01 | BOUNDARY_CROSSING | 2.00 | 5.77 | - | Customer's hand reached across the counter into the staff side (0.3 forearm lengths past the line) |
