from __future__ import annotations

import time

import cv2
import numpy as np

from .camera import StereoCameraCapture, open_camera, read_frame
from .config import CameraCalibration, StereoCalibration
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


def _checkerboard_observation(frame, pattern_size):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCornersSB(
        gray, pattern_size, flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    )
    display = frame.copy()
    if not found:
        return None, display, None
    cv2.drawChessboardCorners(display, pattern_size, corners, True)
    points = corners.reshape(-1, 2)
    height, width = frame.shape[:2]
    low, high = points.min(axis=0), points.max(axis=0)
    center = (low + high) / (2.0 * np.array([width, height]))
    area = np.prod((high - low) / np.array([width, height]))
    direction = points[pattern_size[0] - 1] - points[0]
    angle = np.arctan2(direction[1], direction[0]) / np.pi
    return corners.astype(np.float32), display, np.array([center[0], center[1], area * 2, angle * 0.5])


def run_stereo_calibration(
    left_device: str,
    right_device: str,
    output: str,
    columns: int,
    rows: int,
    square_mm: float,
    required_frames: int = 20,
    max_frame_skew_ms: float = 20.0,
) -> StereoCalibration:
    if columns < 3 or rows < 3 or square_mm <= 0 or required_frames < 10:
        raise TrackerError("Invalid checkerboard dimensions, square size, or frame count")
    capture = StereoCameraCapture(left_device, right_device, max_frame_skew_ms / 1000.0)
    pattern = (columns, rows)
    object_template = np.zeros((columns * rows, 3), np.float32)
    object_template[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2) * square_mm
    objects, left_points, right_points, descriptors = [], [], [], []
    left_size = right_size = None
    window = "Stereo camera calibration"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    try:
        while len(objects) < required_frames:
            observation = capture.next_observation()
            if observation.left is None or observation.right is None:
                continue
            left_frame, right_frame = observation.left.frame, observation.right.frame
            left_size = (left_frame.shape[1], left_frame.shape[0])
            right_size = (right_frame.shape[1], right_frame.shape[0])
            lc, ld, ldesc = _checkerboard_observation(left_frame, pattern)
            rc, rd, rdesc = _checkerboard_observation(right_frame, pattern)
            descriptor = None if ldesc is None or rdesc is None else np.r_[ldesc, rdesc]
            novel = descriptor is not None and (
                not descriptors or min(np.linalg.norm(descriptor - old) for old in descriptors) > 0.09
            )
            height = min(ld.shape[0], rd.shape[0])
            panels = [cv2.resize(item, None, fx=height/item.shape[0], fy=height/item.shape[0]) for item in (ld, rd)]
            display = np.hstack(panels)
            _draw_text(display, [
                f"Accepted stereo views: {len(objects)}/{required_frames}",
                f"Frame skew: {(observation.skew_s or 0) * 1000:.1f} ms",
                "Move and tilt the board through both views",
                "SPACE capture | Q cancel",
                "Ready" if novel else ("Try a new viewpoint" if descriptor is not None else "Board needed in both views"),
            ])
            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                raise TrackerError("Stereo calibration cancelled")
            if key == ord(" ") and novel:
                objects.append(object_template.copy())
                left_points.append(lc)
                right_points.append(rc)
                descriptors.append(descriptor)
                time.sleep(0.08)
    finally:
        capture.close()
        cv2.destroyWindow(window)
    if left_size is None or right_size is None:
        raise TrackerError("No stereo calibration frames were captured")
    if left_size != right_size:
        raise TrackerError(
            "Stereo calibration requires both cameras to use the same image resolution"
        )
    left_rms, left_matrix, left_dist, _, _ = cv2.calibrateCamera(objects, left_points, left_size, None, None)
    right_rms, right_matrix, right_dist, _, _ = cv2.calibrateCamera(objects, right_points, right_size, None, None)
    result = cv2.stereoCalibrate(
        objects, left_points, right_points, left_matrix, left_dist, right_matrix, right_dist,
        left_size, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-7),
        flags=cv2.CALIB_FIX_INTRINSIC,
    )
    stereo_rms, left_matrix, left_dist, right_matrix, right_dist, rotation, translation = result[:7]
    board = {"columns": columns, "rows": rows, "square_mm": square_mm, "views": len(objects)}
    calibration = StereoCalibration(
        CameraCalibration(left_size, left_matrix, left_dist, float(left_rms), board),
        CameraCalibration(right_size, right_matrix, right_dist, float(right_rms), board),
        rotation, translation.reshape(3), float(stereo_rms), board,
    )
    calibration.save(output)
    return calibration
