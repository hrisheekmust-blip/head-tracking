# Head tracking (robot cameras)

Part of the Head-Tracked Mobile Stereo Audio System capstone. Owner: Hrisheek Mustyala.

Each speaker robot has a Pi Camera Module 3 on the speaker head. This code finds the listener's
face in that camera and reports how far it is from the center of the image. Since the camera
rides on the speaker, driving that error to zero aims the speaker at the listener.

## What it outputs

| Field | For | Meaning |
|---|---|---|
| `yaw_error_deg` | pan servo | head is right (+) / left (−) of the camera axis |
| `pitch_error_deg` | tilt servo | head is above (+) / below (−) the camera axis |
| `head_height_m` | audio | listener eye/ear height above the floor (needs a distance) |
| `detected` | everyone | face seen in this frame |

Head height uses `height = camera_height + floor_distance × tan(elevation)`. The elevation comes
from the image and the camera tilt; the floor distance has to come from the overhead camera
(or `--distance` for testing).

## How it works

1. Grab a 640×360 frame (scaled down by the Pi's camera hardware, not the CPU), at most 15 fps.
2. Find faces with **YuNet**, a small pre-trained face detector that runs in OpenCV. No training data needed.
3. Once the listener is found, only search a small square around them next frame. That's much
   cheaper than searching the whole frame. The whole frame is searched if the face is lost, and every 30 frames as a check.
4. Keep following the same person; if there's no one near the last position, take the largest (closest) face.
5. Use the point between the eyes (≈ ear height), convert its pixel position to angles, and send the result.

## Install

First, after cloning, download the face detector models (~230 KB each) into `models/`:
```bash
python3 download_models.py
```

Laptop:
```bash
pip install opencv-python numpy
python run_head_tracker.py --source 0 --show --distance 2.0
```
The red cross should follow your eyes, and yaw/pitch should go to about 0 when you're centered.

Raspberry Pi (Pi 4 or 5, Raspberry Pi OS Bookworm):
```bash
sudo apt install -y python3-picamera2
pip install --break-system-packages opencv-python-headless numpy   # apt's OpenCV 4.6 is too old, need 4.8+
python3 run_head_tracker.py --source picam --robot-id 1 --udp 127.0.0.1:5005 --listen 5006 --stats
```
`--stats` prints fps and CPU use every 5 s, so we can check how much of the Pi it takes.

## Interface (UDP + JSON)

The tracker sends one message per frame to `--udp host:port`:
```json
{"robot_id": 1, "detected": true, "yaw_error_deg": -12.4, "pitch_error_deg": 1.8,
 "elevation_deg": 6.8, "head_height_m": 1.19, "distance_m": 2.0, "confidence": 0.91,
 "head_px": [251.2, 176.0], "bbox": [221.6, 140.1, 60.3, 76.9], "t": 1793308555.04, "mode": "roi"}
```
It listens on `--listen port` for either of these (any subset of keys):
```json
{"robot_id": 1, "distance_m": 2.1}   // overhead CV: floor distance from THIS robot's camera to the listener
{"camera_pitch_deg": 5.0}             // servo code: camera tilt up from level
```
A distance older than 1 s is dropped. When the face is missed, the last position is held for
0.5 s with `detected: false`, then all angles go to `null`.

`examples/servo_follow.py` is a starting point for the PCA9685 servo driver. It reads these
messages, nudges pan/tilt toward zero error, and sends the tilt back. Run it with `--dry-run` to test without servos.

## Calibration (once per robot camera)

Needed for accurate head height. Without it the code uses the datasheet 66° field of view,
which is fine for aiming but not for the 5 cm target.
```bash
python3 calibrate.py --source picam --out calib/robot1.json --square 0.025
python3 run_head_tracker.py --source picam --calib calib/robot1.json ...
```
Autofocus is locked (`--lens-position 0.5`, focused at about 2 m). Refocusing changes the focal
length and breaks the calibration, so calibrate and run with the same value.

## Tests
```bash
pip install pytest && python -m pytest -q
```

## Status

Tested on a laptop/server with OpenCV 4.8 and 5.0 using recorded video. **Not yet tested on the
Pi camera, with real servos, or for real head-height accuracy.**

Known limits:
- Range is about 3 m at 640×360. Use `--width 960 --height 540` for more (costs more CPU).
- Follows one person. Multiple listeners isn't handled.
- Hobby servos have no position feedback, so the tilt sent back is the commanded angle.
- Rolling shutter: measure while the robot is stopped.
