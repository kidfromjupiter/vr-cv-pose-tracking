# Three White Sphere VR Tracker

A calibrated OpenCV and IMU fusion tracker for one rigid controller carrying
three white spheres. A low-latency V4L2 stream, such as an adb/scrcpy camera
feed, provides camera-relative position while the ESP32 controller provides
timestamped quaternion and acceleration samples. Position fusion uses the
adaptive camera/accelerometer filter structure and defaults from HadesVR.
Orientation remains on the receiver's IMU quaternion path after a one-time
camera-to-IMU alignment at startup.

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
- `IMU_ONLY`: the camera is occluded; IMU prediction continues while serial
  samples remain live.
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

## 2. Calibrate the camera

Print `assets/checkerboard-9x6-25mm.svg` at 100% scale and verify its square
size. Capture tilted views across the full image:

```bash
vr-led-tracker calibrate-camera \
  --device /dev/video0 \
  --columns 9 --rows 6 --square-mm 25 \
  --output config/camera.json
```

Keep the camera zoom, orientation, and aspect ratio unchanged afterward.

For stereo tracking, mount two cameras rigidly with an overlapping view of the
whole tracking volume, then capture at least 20 checkerboard poses visible in
both cameras:

```bash
vr-led-tracker calibrate-stereo \
  --left-device /dev/video0 --right-device /dev/video2 \
  --columns 9 --rows 6 --square-mm 25 \
  --frames 20 --output config/stereo_camera.json
```

The saved transform maps the left-camera coordinate frame into the right
camera. The left camera remains the permanent world/tracking frame.

## 3. Run fused tracking

Close SteamVR and serial monitors, then run:

```bash
vr-led-tracker track \
  --device /dev/video0 \
  --serial-device /dev/ttyACM0 \
  --baud 230400 \
  --imu-slot right \
  --model config/controller.json \
  --camera config/camera.json \
  --fusion-settings config/hades_fusion.json
```

At startup, hold the controller still with all three spheres visible for about
one second. This measures accelerometer bias and the fixed transform between
the camera and receiver quaternion. Tracking starts immediately afterward.
Camera measurements are applied when received; there is no camera-latency
estimation or delayed replay path.

To track with both cameras while keeping the monocular command available, run:

```bash
vr-led-tracker track-stereo \
  --left-device /dev/video0 --right-device /dev/video2 \
  --serial-device /dev/ttyACM0 --imu-slot right \
  --model config/controller.json \
  --stereo-camera config/stereo_camera.json \
  --fusion-settings config/hades_fusion.json
```

The capture threads pair frames by monotonic receipt time within 20 ms. A
matched pair uses epipolar correspondence and triangulation; an unmatched or
geometrically invalid pair can fall back to either camera's monocular P3P
solution. Right-camera poses are transformed into the left-camera frame before
fusion. Orientation still follows the IMU quaternion after startup alignment.

`config/hades_fusion.json` exposes the Hades-style camera and IMU measurement
uncertainty, estimation uncertainty, process noise, and per-sample velocity
damping. The defaults are the HadesVR controller values. Its experimental
camera-velocity yaw correction is also available under
`yaw_drift_correction`, but is disabled by default. When disabled, optical
rotation is used only for startup alignment and subsequent orientation comes
entirely from the receiver quaternion.

The same file's `identity_tracking` section controls temporal sphere identity.
Each sphere is predicted from its last two accepted image positions. An
assignment implying more than `max_speed_px_s` for any sphere is rejected; if
no assignment remains, IMU-only tracking is used until the tracks can be
continued or `reacquire_timeout_s` expires.

The `stereo_tracking` section controls epipolar, rigid-fit, and minimum
triangulation-angle gates. Optical confidence is reduced continuously for high
reprojection error, a narrow triangulation angle, small sphere image area, or
large frame skew. Mono fallback starts with lower confidence than a sound
stereo solution. These per-frame confidence values scale the Hades camera
measurement uncertainty rather than altering the IMU orientation path.

White spheres are segmented automatically from low-saturation pixels using an
adaptive per-frame brightness threshold. The asymmetric model and predicted
pose assign the unordered white blobs to `sphere_0`, `sphere_1`, and
`sphere_2`.

The window shows the annotated camera image and synthetic fused 3D pose side
by side. Gray circles are unassigned white candidates; colored labels identify
the three candidates accepted by the pose solver. Controls:

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
- **Still calibration restarts:** keep all three spheres visible and prevent
  both translation and rotation for a full second.
- **Position coasts or drifts during occlusion:** this branch deliberately
  keeps integrating while the IMU is live. Increase damping or reduce IMU
  process noise in `config/hades_fusion.json`; absolute position is corrected
  again when all three spheres return.
- **Yaw slowly drifts:** correct the IMU/magnetometer calibration first. The
  optional HadesVR camera-velocity yaw correction can be enabled in the fusion
  settings, but it only operates inside its configured speed range.
- **White false detections:** avoid bright white background objects and strong
  reflections. The geometric assignment rejects candidates that do not match
  the measured asymmetric rig.
- **Wrong depth:** verify camera calibration, physical sphere diameters, and
  sphere-center coordinates.
- **Pose jumps:** confirm model axes match the firmware IMU body axes and the
  correct `--imu-slot` is selected. If real high-speed movement is rejected by
  the identity gate, raise `identity_tracking.max_speed_px_s` gradually.

## Tests

```bash
pytest
```

The suite covers adaptive white detection, unordered identity assignment,
synthetic sphere projection, P3P, CRC framing, timestamp rollover, Hades-style
adaptive fusion, stereo synchronization, triangulation, mono fallback,
confidence-weighted camera gain, device-timestamp prediction, static optical
alignment, and camera dropout/reacquisition. Final validation still requires
the real cameras, receiver, IMU, and sphere rig.

## HadesVR attribution

The position filter, tuning defaults, velocity damping, and optional
camera-velocity yaw correction are adapted from
[HadesVR at commit 0a6de3c](https://github.com/HadesVR/HadesVR/tree/0a6de3c19e22978dbd2a17e23fcb2e041b0cee48).
HadesVR is MIT licensed; the required notice is retained in
`THIRD_PARTY_NOTICES.md`.
