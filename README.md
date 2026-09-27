# Three-Sphere VR Tracker

A calibrated OpenCV and IMU fusion tracker for one rigid controller carrying
red, blue, and white spheres. DroidCam provides camera-relative position while
the ESP32 controller provides timestamped quaternion and acceleration samples.
An error-state Kalman filter predicts at IMU rate and corrects drift from the
camera.

## Tracking model

The controller model contains exactly three non-collinear sphere centers and
their physical diameters. Measurements are in millimetres, with the origin at
the IMU center and axes matching the body axes configured in the controller
firmware:

```json
{
  "spheres": [
    {"label": "red",   "center_mm": [-70, 0, 0], "diameter_mm": 20},
    {"label": "blue",  "center_mm": [70, 0, 0],  "diameter_mm": 20},
    {"label": "white", "center_mm": [0, 45, 30], "diameter_mm": 20}
  ]
}
```

The values in `config/controller.example.json` are examples only. Measure all
three centers and diameters on the real rigid assembly. Legacy four-LED model
files are intentionally unsupported.

Tracking states:

- `CALIBRATING_STILL`: hold the rig still with all spheres visible.
- `CALIBRATING_DELAY`: rotate it smoothly to estimate DroidCam latency.
- `FULL`: three-sphere camera correction plus IMU prediction.
- `DEGRADED_2`: two-sphere position correction using IMU orientation.
- `IMU_ONLY`: camera is briefly occluded; prediction is limited to 250 ms.
- `CAMERA_ONLY`: all spheres are visible but IMU data is stale.
- `LOST`: neither source can provide a safe pose.

## Install

Python 3.11 or newer is required:

```bash
source .venv/bin/activate
python -m pip install -e '.[test]'
```

Flash the matching `../vrrecv` firmware before running the tracker. Its normal
USB output is now a CRC-protected timestamped fusion stream and is not
compatible with the old HadesVR serial driver. Pairing order is unchanged: the
first controller after receiver boot is `right`, the second is `left`.

## 1. Configure the sphere rig

```bash
cp config/controller.example.json config/controller.json
```

Replace every example center and diameter. The sphere edges must be clearly
visible; exposure-dependent glowing halos do not provide reliable diameter
measurements.

## 2. Calibrate DroidCam

Print `assets/checkerboard-9x6-25mm.svg` at 100% scale and verify its square
size. Capture tilted views across the full image:

```bash
vr-led-tracker calibrate-camera \
  --device /dev/video0 \
  --columns 9 --rows 6 --square-mm 25 \
  --output config/camera.json
```

Keep DroidCam zoom, orientation, and aspect ratio unchanged afterward.

## 3. Calibrate sphere colors

```bash
vr-led-tracker calibrate-colors \
  --device /dev/video0 \
  --model config/controller.json \
  --samples-per-color 10 \
  --output config/colors.json
```

Click inside each prompted sphere in ten different frames and vary its angle
and position slightly so the samples include realistic lighting changes. Red uses wrapped
hue ranges, blue uses its sampled hue, and white uses low saturation plus high
brightness. Press `R` to redo a color or `Q` to cancel.

## 4. Run fused tracking

Close SteamVR and serial monitors, then run:

```bash
vr-led-tracker track \
  --device /dev/video0 \
  --serial-device /dev/ttyACM0 \
  --baud 230400 \
  --imu-slot right \
  --model config/controller.json \
  --camera config/camera.json \
  --colors config/colors.json
```

At startup, hold the controller still with all three spheres visible for about
one second. Then rotate it smoothly while keeping the spheres visible. The
tracker correlates camera and IMU angular motion to estimate DroidCam latency
over a 0–500 ms range before entering normal tracking.

The window shows the annotated DroidCam image and synthetic fused 3D pose side
by side. Controls:

- `M`: show or hide color masks.
- `R`: discard alignment, latency, and Kalman state and recalibrate.
- `Q` or Escape: quit.

For serial-only diagnostics, use:

```bash
vr-led-tracker inertial-preview --device /dev/ttyACM0 --baud 230400
```

## Troubleshooting

- **No serial data:** flash the updated receiver, select the correct pairing
  slot, and close other programs using `/dev/ttyACM0`.
- **Still calibration restarts:** keep all three spheres visible and prevent
  both translation and rotation for a full second.
- **Latency calibration does not finish:** make several smooth, distinctive
  rotations without hiding a sphere.
- **White false detections:** reduce reflections and recalibrate colors under
  the intended room lighting.
- **Wrong depth:** verify camera calibration, physical sphere diameters, and
  sphere-center coordinates.
- **Pose jumps:** confirm model axes match the firmware IMU body axes and the
  correct `--imu-slot` is selected.

## Tests

```bash
pytest
```

The suite covers synthetic sphere projection, P3P and two-sphere translation,
CRC framing, timestamp rollover, Kalman behavior, and latency estimation. Final
validation still requires the real DroidCam, receiver, IMU, and sphere rig.
