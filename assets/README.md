# Calibration checkerboard

`checkerboard-9x6-25mm.svg` is an A4-landscape, print-ready OpenCV
checkerboard with 9×6 inner corners and 25 mm squares.

Print it using **Actual size** or **100% scale**. Disable options such as
“Fit to page” or “Shrink oversized pages”. Measure several printed squares
with a ruler before calibration and pass the measured size to
`--square-mm`. Mount the page on a rigid, flat surface.

The matching command is:

```bash
vr-led-tracker calibrate-camera \
  --columns 9 --rows 6 --square-mm 25 \
  --output config/camera.json
```

