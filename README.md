# Stereo Three White Sphere VR Tracker

A calibrated OpenCV and IMU fusion tracker for one rigid controller carrying
three white spheres. Two low-latency V4L2 streams provide stereoscopic
position while the ESP32 controller provides timestamped quaternion and
acceleration samples. When either camera loses the controller, tracking can
temporarily fall back to the remaining camera.
An error-state Kalman filter predicts at IMU rate and corrects drift from the
camera.

## Tracking model

The controller model contains exactly three non-collinear, asymmetrically
spaced sphere centers and their physical diameters. Measurements are in
millimetres, with the origin at the IMU center and axes matching the body axes
configured in the controller firmware:

```json
{
  "spheres": [
    {"label": "sphere_0", "center_mm": [-70, 0, 0],  "diameter_mm": 20},
    {"label": "sphere_1", "center_mm": [70, 0, 0],   "diameter_mm": 20},
    {"label": "sphere_2", "center_mm": [15, 45, 30], "diameter_mm": 20}
  ]
}
```

The values in `config/controller.example.json` are examples only. Measure all
three centers and diameters on the real rigid assembly. The three pairwise
distances must differ by at least 5%, allowing identical white markers to be
assigned to geometry identities. Legacy LED model files are unsupported.

Tracking states:

- `CALIBRATING_STILL`: hold the rig still with all spheres visible.
- `FULL`: three-sphere camera correction plus IMU prediction.
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

## 2. Mount and calibrate the cameras

Mount the cameras rigidly with overlapping views. They must not move relative
to each other after calibration. The left camera defines the fixed tracking
coordinate frame: +X right, +Y down, and +Z forward as viewed by that camera.

Start both adb/scrcpy V4L2 streams and identify their device paths. The streams
do not need hardware synchronization; frames are paired by their local receipt
timestamps. Use matching frame rates when possible.

Print `assets/checkerboard-9x6-25mm.svg` at 100% scale and verify its square
size. Capture tilted views across the full image:

```bash
vr-led-tracker calibrate-stereo \
  --left-device /dev/video0 \
  --right-device /dev/video2 \
  --columns 9 --rows 6 --square-mm 25 \
  --frames 20 \
  --max-frame-skew-ms 20 \
  --output config/stereo_camera.json
```

The checkerboard must be detected in both views before a sample can be
accepted. Capture varied positions, distances, and tilts spanning the shared
field of view. Keep both cameras' zoom, focus, orientation, resolution, and
aspect ratio unchanged afterward. The original `calibrate-camera` command
remains available for single-camera diagnostics but its output cannot be used
for stereo tracking.

## 3. Run fused tracking

Close SteamVR and serial monitors, then run:

```bash
vr-led-tracker track \
  --left-device /dev/video0 \
  --right-device /dev/video2 \
  --serial-device /dev/ttyACM0 \
  --baud 230400 \
  --imu-slot right \
  --max-frame-skew-ms 20 \
  --camera-latency-ms 0 \
  --model config/controller.json \
  --stereo-camera config/stereo_camera.json
```

At startup, hold the controller still with all three spheres visible for about
one second. Tracking starts immediately after that calibration. Camera delay is
a fixed value rather than an estimated value; leave `--camera-latency-ms` at
zero for a low-latency scrcpy stream, or provide a measured value from 0 to 500.

White spheres are segmented independently in both cameras using an adaptive
per-frame brightness threshold. Epipolar geometry pairs detections across the
views, and the asymmetric model assigns the unordered blobs to `sphere_0`,
`sphere_1`, and `sphere_2`. Valid pairs are triangulated before the result is
fused with the IMU. The preview reports `STEREO`, `LEFT_MONO`, `RIGHT_MONO`, or
`NONE` as the active camera mode.

The window shows both annotated camera images and the synthetic fused 3D pose.
Gray circles are unassigned white candidates; colored labels identify the
three candidates accepted by the pose solver. Controls:

- `M`: show or hide the adaptive white mask.
- `R`: discard alignment and Kalman state and recalibrate.
- `Q` or Escape: quit.

For serial-only diagnostics, use:

```bash
vr-led-tracker inertial-preview --device /dev/ttyACM0 --baud 230400
```

## Troubleshooting

- **No serial data:** flash the updated receiver, select the correct pairing
  slot, and close other programs using `/dev/ttyACM0`.
- **No stereo updates:** ensure both devices are producing frames, both views
  see the spheres, and increase `--max-frame-skew-ms` slightly if the displayed
  skew frequently exceeds 20 ms. The accepted range is 1–100 ms.
- **Poor depth or sideways offset:** recalibrate after checking that the camera
  mount is rigid and the checkerboard square measurement is correct.
- **Still calibration restarts:** keep all three spheres visible and prevent
  both translation and rotation for a full second.
- **Camera and IMU motion are offset:** set a measured fixed delay with
  `--camera-latency-ms`; scrcpy streams should normally start at zero.
- **White false detections:** avoid bright white background objects and strong
  reflections. The geometric assignment rejects candidates that do not match
  the measured asymmetric rig.
- **Wrong depth:** verify stereo calibration, physical sphere diameters, and
  sphere-center coordinates.
- **Pose jumps:** confirm model axes match the firmware IMU body axes and the
  correct `--imu-slot` is selected.

## Tests

```bash
pytest
```

The suite covers adaptive white detection, frame synchronization, stereo
triangulation, unordered identity assignment, monocular fallback, synthetic
sphere projection, P3P, CRC framing, timestamp rollover, fixed-delay replay,
and Kalman behavior. Final validation still requires both real cameras, the
receiver, IMU, and sphere rig.
