# Run report - possible hand contact.mp4

- Video: 1024x576, 29.69 FPS, 95 frames (3.2 s)
- Scene setup: automatic - AUTO: counter line midway between the two people's usual hand positions; main participants are tracks 2 and 1 (side by side); customer = track 2, staff = track 1: staff = the person working at the desk (hands kept up and still), score 6.9; people shorter than 118px are treated as background
- Frames analysed: 48 (stride 2), wall time 34.74 s, pose 231.8 ms/frame
- Inference backend: openvino
- Annotated video: D:\proglint_counter_cv\proglint_counter_cv\outputs\annotated\possible hand contact_annotated.mp4

**1 interaction(s)**: 0 likely handshake, 0 likely hand contact, 1 uncertain, 0 reach without contact.

## Interactions (counting unit)

| ID | Outcome | Start | End | Crossings | Rows | What it looks like |
|---|---|---|---|---|---|---|
| I01 | UNCERTAIN | 1.87 | 3.17 | 0 | E001 | Uncertain - possible hand contact not confirmed: both reached out and the hands came within 0.24 forearms for 1.2s, but never touched |

## Observation rows (event log)

| BOUNDARY_CROSSING | LIKELY_HAND_CONTACT | LIKELY_HANDSHAKE | UNCERTAIN | suppressed candidates |
|---|---|---|---|---|
| 0 | 0 | 0 | 1 | 0 |

| ID | Interaction | Type | Start | End | Handshake % | What it looks like |
|---|---|---|---|---|---|---|
| E001 | I01 | UNCERTAIN | 1.87 | 3.17 | - | Uncertain - possible hand contact not confirmed: both reached out and the hands came within 0.24 forearms for 1.2s, but never touched |
