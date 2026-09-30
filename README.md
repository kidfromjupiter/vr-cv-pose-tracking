# Blue-Ball Camera Position Tracker

This application combines two independent measurements:

- A calibrated camera tracks the X, Y, and Z position of one blue ball.
- The controller's serial IMU supplies orientation relative to its startup pose.

The ball's known physical diameter provides monocular depth. The reported position is the
center of the ball in camera coordinates; no ball-to-controller offset is applied. IMU
acceleration is not used to dead-reckon position.

## Install

Python 3.11 or newer is required:

```bash
source .venv/bin/activate
python -m pip install -e '.[test]'
```

The receiver must provide the CRC-protected timestamped fusion stream expected by
`vr-led-tracker`. The first controller paired after receiver boot is `right`; the second is
`left`.

## 1. Configure the ball

Copy the example and enter the ball's measured edge-to-edge diameter in millimetres:

```bash
cp config/controller.example.json config/controller.json
```

```json
{
  "ball": {
    "label": "blue",
    "diameter_mm": 20.0
  }
}
```

Use a matte, uniformly colored sphere with a clearly visible edge. An inaccurate diameter
produces the same proportional error in the estimated distance.

## 2. Calibrate the camera

Print `assets/checkerboard-9x6-25mm.svg` at 100% scale and verify its square size. Capture
tilted views across the whole frame:

```bash
vr-led-tracker calibrate-camera \
  --device /dev/video0 \
  --columns 9 --rows 6 --square-mm 25 \
  --output config/camera.json
```

Keep the camera zoom, orientation, resolution aspect ratio, and focus unchanged afterward.

## 3. Calibrate the blue color

```bash
vr-led-tracker calibrate-color \
  --device /dev/video0 \
  --samples 10 \
  --output config/color.json
```

Click inside the blue ball in ten different frames. Move it around the intended tracking
area and vary its angle slightly so the saved HSV range covers realistic lighting. Press
`R` to restart sampling or `Q`/Escape to cancel. Repeat this step when the lighting or camera
exposure changes materially.

## 4. Track position and orientation

```bash
vr-led-tracker track \
  --device /dev/video0 \
  --serial-device /dev/ttyACM0 \
  --baud 230400 \
  --imu-slot right \
  --model config/controller.json \
  --camera config/camera.json \
  --color config/color.json
```

At startup, hold the controller still for about one second. That pose becomes zero
orientation. A single uniform sphere cannot determine absolute IMU-to-camera rotational
alignment, so displayed rotation is relative to this startup pose.

Tracking states:

- `CALIBRATING_IMU`: waiting for a stable startup orientation.
- `FULL`: camera XYZ and IMU orientation are both current.
- `CAMERA_ONLY`: XYZ is current but IMU orientation is unavailable or stale.
- `IMU_ONLY`: orientation is current but the ball is not detected.
- `LOST`: neither measurement is current.

When the ball disappears, the last position is shown as stale for up to 250 ms and is then
invalidated. It is never extrapolated from accelerometer data.

Preview controls:

- `M`: show or hide the calibrated blue mask.
- `R`: reset position history and recalibrate the startup IMU orientation.
- `Q` or Escape: quit.

For serial-only diagnostics:

```bash
vr-led-tracker inertial-preview --device /dev/ttyACM0 --baud 230400
```

## Troubleshooting

- **Ball is not detected:** recalibrate its color under the current lighting and inspect the
  mask with `M`.
- **Wrong depth:** verify the checkerboard camera calibration and measured ball diameter.
- **Depth changes as the ball moves across the image:** keep camera focus/zoom fixed and
  recalibrate the camera with views covering the full frame.
- **A different blue object is selected:** remove similarly colored objects; after initial
  acquisition, temporal continuity favors the existing target.
- **No orientation:** close other serial monitors, verify the selected IMU slot, and hold the
  controller still during startup calibration.
- **Orientation axes do not match camera axes:** startup recentering provides relative
  orientation only; absolute camera-frame alignment is intentionally not inferred.

## Tests

```bash
pytest
```

The suite covers camera calibration scaling, HSV calibration and segmentation, synthetic
known-sphere position recovery, lens distortion, serial framing, startup-relative IMU
orientation, filtering, and camera/IMU availability states. Final validation still requires
the real camera, ball, receiver, and IMU.
