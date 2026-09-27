from __future__ import annotations

from pathlib import Path

import cv2

from .errors import TrackerError


def normalize_device(device: str) -> int | str:
    if device.isdecimal():
        return int(device)
    return device


def open_camera(device: str) -> cv2.VideoCapture:
    source = normalize_device(device)
    if isinstance(source, str) and source.startswith("/dev/") and not Path(source).exists():
        raise TrackerError(
            f"Camera device {source} does not exist. Start DroidCam and verify the V4L2 device first."
        )
    capture = cv2.VideoCapture(source, cv2.CAP_V4L2)
    if not capture.isOpened():
        capture.release()
        raise TrackerError(f"Could not open DroidCam camera {device}")
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def read_frame(capture: cv2.VideoCapture):
    ok, frame = capture.read()
    if not ok or frame is None:
        raise TrackerError("DroidCam stopped providing frames")
    return frame

