# Run report - 4-Akash and Harani likely hand contact.mp4

- Video: 1024x576, 29.76 FPS, 122 frames (4.1 s)
- Scene setup: automatic - AUTO: counter line midway between the two people's usual hand positions; main participants are tracks 1 and 2 (side by side); customer = track 2, staff = track 1: staff = the person working at the desk (hands kept up and still), score 0.5; people shorter than 142px are treated as background
- Frames analysed: 122 (stride 1), wall time 9.91 s, pose 63.3 ms/frame
- Inference backend: openvino
- Annotated video: D:\proglint_counter_cv\proglint_counter_cv\outputs\annotated\4-Akash and Harani likely hand contact_annotated.mp4

**1 interaction(s)**: 0 likely handshake, 1 likely hand contact, 0 uncertain, 0 reach without contact.

## Interactions (counting unit)

| ID | Outcome | Start | End | Crossings | Rows | What it looks like |
|---|---|---|---|---|---|---|
| I01 | LIKELY_HAND_CONTACT | 3.10 | 3.60 | 0 | E001 | Likely hand contact (0.5s) - hands met but never clasped, consistent with giving/taking an item or a light touch; not a handshake (handshake 0%, contact 88%) |

## Observation rows (event log)

| BOUNDARY_CROSSING | LIKELY_HAND_CONTACT | LIKELY_HANDSHAKE | UNCERTAIN | suppressed candidates |
|---|---|---|---|---|
| 0 | 1 | 0 | 0 | 0 |

| ID | Interaction | Type | Start | End | Handshake % | What it looks like |
|---|---|---|---|---|---|---|
| E001 | I01 | LIKELY_HAND_CONTACT | 3.10 | 3.60 | 0 | Likely hand contact (0.5s) - hands met but never clasped, consistent with giving/taking an item or a light touch; not a handshake (handshake 0%, contact 88%) |
