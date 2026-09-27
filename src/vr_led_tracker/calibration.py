from __future__ import annotations

import time

import cv2
import numpy as np

from .camera import open_camera, read_frame
from .config import CameraCalibration
from .errors import TrackerError


def _draw_text(frame: np.ndarray, lines: list[str]) -> None:
    y = 28
    for line in lines:
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 240, 20), 2, cv2.LINE_AA)
        y += 27


def run_camera_calibration(
    device: str,
    output: str,
    columns: int,
    rows: int,
    square_mm: float,
    required_frames: int = 15,
) -> CameraCalibration:
    if columns < 3 or rows < 3 or square_mm <= 0 or required_frames < 8:
        raise TrackerError("Invalid checkerboard dimensions, square size, or frame count")
    capture = open_camera(device)
    pattern_size = (columns, rows)
    object_template = np.zeros((columns * rows, 3), np.float32)
    object_template[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2) * square_mm
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    descriptors: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None
    window = "Camera calibration"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    try:
        while len(image_points) < required_frames:
            frame = read_frame(capture)
            height, width = frame.shape[:2]
            image_size = (width, height)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCornersSB(
                gray,
                pattern_size,
                flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY,
            )
            display = frame.copy()
            novel = False
            descriptor = None
            if found:
                corners2 = corners.reshape(-1, 2)
                minimum = corners2.min(axis=0)
                maximum = corners2.max(axis=0)
                center = (minimum + maximum) / 2.0 / np.array([width, height])
                area = np.prod((maximum - minimum) / np.array([width, height]))
                direction = corners2[columns - 1] - corners2[0]
                angle = np.arctan2(direction[1], direction[0]) / np.pi
                descriptor = np.array([center[0], center[1], area * 2.0, angle * 0.5])
                novel = not descriptors or min(np.linalg.norm(descriptor - old) for old in descriptors) > 0.075
                cv2.drawChessboardCorners(display, pattern_size, corners, found)
            _draw_text(
                display,
                [
                    f"Accepted views: {len(image_points)}/{required_frames}",
                    "Move/tilt board to cover the frame",
                    "SPACE capture | Q cancel",
                    "Ready" if found and novel else ("Try a new viewpoint" if found else "Checkerboard not found"),
                ],
            )
            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                raise TrackerError("Camera calibration cancelled")
            if key == ord(" ") and found and novel and descriptor is not None:
                object_points.append(object_template.copy())
                image_points.append(corners.astype(np.float32))
                descriptors.append(descriptor)
                time.sleep(0.08)
    finally:
        capture.release()
        cv2.destroyWindow(window)
    if image_size is None:
        raise TrackerError("No calibration frames were captured")
    rms, matrix, distortion, _, _ = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    calibration = CameraCalibration(
        image_size,
        matrix,
        distortion,
        float(rms),
        {"columns": columns, "rows": rows, "square_mm": square_mm, "views": len(image_points)},
    )
    calibration.save(output)
    return calibration
