from __future__ import annotations

import math
import time
from collections import deque

import cv2
import numpy as np

from .errors import TrackerError
from .inertial import InertialPose, InertialPoseTracker
from .serial_pose import FusionFrameParser, FusionSample


WINDOW_SIZE = (1280, 720)
BOX_VERTICES = np.array(
    [
        [-0.08, -0.03, -0.02],
        [0.08, -0.03, -0.02],
        [0.08, 0.03, -0.02],
        [-0.08, 0.03, -0.02],
        [-0.08, -0.03, 0.02],
        [0.08, -0.03, 0.02],
        [0.08, 0.03, 0.02],
        [-0.08, 0.03, 0.02],
    ],
    dtype=np.float64,
)
BOX_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)
AXIS_COLORS = ((0, 0, 255), (0, 220, 0), (255, 120, 30))
STATE_COLORS = {
    "WAITING": (130, 130, 130),
    "CALIBRATING": (0, 210, 255),
    "LIVE": (60, 230, 60),
    "STALE": (50, 50, 255),
    "CALIBRATING_STILL": (0, 210, 255),
    "CALIBRATING_IMU": (0, 210, 255),
    "FULL": (60, 230, 60),
    "IMU_ONLY": (0, 160, 255),
    "CAMERA_ONLY": (200, 160, 40),
    "LOST": (50, 50, 255),
}


def _project(
    points: np.ndarray,
    width: int,
    height: int,
    eye: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    relative = np.asarray(points, dtype=np.float64) - eye
    depth = relative @ forward
    focal = min(width, height) * 0.88
    pixels = np.column_stack(
        [
            width / 2.0 + focal * (relative @ right) / depth,
            height / 2.0 - focal * (relative @ up) / depth,
        ]
    )
    return pixels, depth


def _line_3d(
    image: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    eye: np.ndarray,
    target: np.ndarray,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    projected, depth = _project(
        np.vstack([first, second]), image.shape[1], image.shape[0], eye, target
    )
    if np.all(depth > 0.01):
        cv2.line(
            image,
            tuple(np.rint(projected[0]).astype(int)),
            tuple(np.rint(projected[1]).astype(int)),
            color,
            thickness,
            cv2.LINE_AA,
        )


def _draw_grid(image: np.ndarray, span: float, eye: np.ndarray, target: np.ndarray) -> None:
    step = span / 5.0
    for index in range(-5, 6):
        offset = index * step
        color = (66, 66, 72) if index else (105, 105, 112)
        _line_3d(image, np.array([-span, offset, 0.0]), np.array([span, offset, 0.0]), eye, target, color)
        _line_3d(image, np.array([offset, -span, 0.0]), np.array([offset, span, 0.0]), eye, target, color)


def _put_text(
    image: np.ndarray,
    text: str,
    position: tuple[int, int],
    color: tuple[int, int, int] = (225, 225, 225),
    scale: float = 0.48,
) -> None:
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _draw_panel(
    label: str,
    pose: InertialPose,
    box_color: tuple[int, int, int],
    trail: deque[np.ndarray],
    size: tuple[int, int],
    model_points_m: np.ndarray | None = None,
    diameters_m: np.ndarray | None = None,
    sphere_colors: tuple[tuple[int, int, int], ...] | None = None,
) -> np.ndarray:
    width, height = size
    image = np.full((height, width, 3), (24, 24, 29), dtype=np.uint8)
    displacement = float(np.linalg.norm(pose.position_m))
    span = max(0.6, min(8.0, displacement * 1.5 + 0.5))
    target = np.array([0.0, 0.0, 0.2])
    eye = np.array([1.55, -2.1, 1.35]) * span + target
    _draw_grid(image, span, eye, target)

    anchor = np.array([0.0, 0.0, 0.24])
    center = anchor + pose.position_m
    if pose.state != "WAITING":
        if len(trail) > 1:
            points = np.asarray(trail) + anchor
            projected, depth = _project(points, width, height, eye, target)
            for index in range(1, len(points)):
                if depth[index - 1] > 0.01 and depth[index] > 0.01:
                    cv2.line(
                        image,
                        tuple(np.rint(projected[index - 1]).astype(int)),
                        tuple(np.rint(projected[index]).astype(int)),
                        (95, 95, 105),
                        1,
                        cv2.LINE_AA,
                    )

        if model_points_m is None:
            vertices = (pose.rotation_matrix @ BOX_VERTICES.T).T + center
            projected, depth = _project(vertices, width, height, eye, target)
            for first, second in BOX_EDGES:
                if depth[first] > 0.01 and depth[second] > 0.01:
                    cv2.line(
                        image,
                        tuple(np.rint(projected[first]).astype(int)),
                        tuple(np.rint(projected[second]).astype(int)),
                        box_color,
                        2,
                        cv2.LINE_AA,
                    )

            nose_local = np.array(
                [[0.08, -0.025, 0.0], [0.125, 0.0, 0.0], [0.08, 0.025, 0.0]]
            )
            nose = (pose.rotation_matrix @ nose_local.T).T + center
            nose_pixels, nose_depth = _project(nose, width, height, eye, target)
            if np.all(nose_depth > 0.01):
                cv2.polylines(
                    image,
                    [np.rint(nose_pixels).astype(np.int32)],
                    True,
                    box_color,
                    2,
                    cv2.LINE_AA,
                )
        else:
            rig_points = (pose.rotation_matrix @ np.asarray(model_points_m).T).T + center
            pixels, depths = _project(rig_points, width, height, eye, target)
            colors = sphere_colors or tuple([box_color] * len(rig_points))
            diameters = np.ones(len(rig_points)) * 0.03 if diameters_m is None else diameters_m
            for point, pixel, depth, diameter, color in zip(
                rig_points, pixels, depths, diameters, colors
            ):
                if depth <= 0.01:
                    continue
                edge_pixel, _ = _project(
                    np.vstack([point, point + np.array([diameter * 0.5, 0.0, 0.0])]),
                    width,
                    height,
                    eye,
                    target,
                )
                radius_px = max(4, int(round(np.linalg.norm(edge_pixel[1] - edge_pixel[0]))))
                cv2.circle(
                    image,
                    tuple(np.rint(pixel).astype(int)),
                    radius_px,
                    color,
                    -1,
                    cv2.LINE_AA,
                )
                cv2.circle(
                    image,
                    tuple(np.rint(pixel).astype(int)),
                    radius_px,
                    (30, 30, 35),
                    1,
                    cv2.LINE_AA,
                )

        for axis, color in enumerate(AXIS_COLORS):
            endpoint = center + pose.rotation_matrix[:, axis] * 0.13
            _line_3d(image, center, endpoint, eye, target, color, 2)
        accel = pose.acceleration_m_s2
        accel_norm = float(np.linalg.norm(accel))
        if accel_norm > 0.01:
            endpoint = center + accel / accel_norm * min(0.3, accel_norm * 0.05)
            _line_3d(image, center, endpoint, eye, target, (0, 235, 255), 2)

    state_color = STATE_COLORS[pose.state]
    _put_text(image, f"{label}  {pose.state}", (14, 27), state_color, 0.62)
    if pose.state == "CALIBRATING":
        _put_text(image, "Hold still", (14, 53), state_color, 0.52)
        bar_width = int((width - 28) * pose.calibration_progress)
        cv2.rectangle(image, (14, 63), (width - 14, 72), (65, 65, 70), -1)
        cv2.rectangle(image, (14, 63), (14 + bar_width, 72), state_color, -1)
    elif pose.state == "WAITING":
        _put_text(image, "Waiting for controller data", (14, 53), state_color)

    x, y, z = pose.position_m
    vx, vy, vz = pose.velocity_m_s
    ax, ay, az = pose.acceleration_m_s2
    yaw, pitch, roll = pose.euler_zyx_deg
    bottom = height - 82
    _put_text(image, f"Displacement m  {x:+.3f} {y:+.3f} {z:+.3f}", (14, bottom))
    _put_text(image, f"Velocity m/s    {vx:+.3f} {vy:+.3f} {vz:+.3f}", (14, bottom + 20))
    _put_text(image, f"Linear m/s2     {ax:+.2f} {ay:+.2f} {az:+.2f}", (14, bottom + 40))
    _put_text(image, f"YPR deg          {yaw:+.1f} {pitch:+.1f} {roll:+.1f}", (14, bottom + 60))
    return image


def run_inertial_preview(device: str = "/dev/ttyACM0", baud: int = 230400) -> None:
    try:
        import serial
    except ImportError as exc:
        raise TrackerError("pyserial is required for inertial-preview") from exc
    try:
        port = serial.Serial(device, baudrate=baud, timeout=0)
    except (OSError, serial.SerialException) as exc:
        raise TrackerError(f"Cannot open serial device {device}: {exc}") from exc

    parser = FusionFrameParser()
    trackers = {"RIGHT": InertialPoseTracker(), "LEFT": InertialPoseTracker()}
    samples: dict[str, FusionSample] = {}
    trails = {"RIGHT": deque(maxlen=120), "LEFT": deque(maxlen=120)}
    last_frame_time: float | None = None
    previous_render_time: float | None = None
    fps = 0.0
    window = "VR inertial pose"
    try:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, *WINDOW_SIZE)
        while True:
            now = time.monotonic()
            try:
                waiting = port.in_waiting
                data = port.read(waiting) if waiting else b""
            except (OSError, serial.SerialException) as exc:
                raise TrackerError(f"Cannot read serial device {device}: {exc}") from exc
            frames = parser.feed(data, now)
            if frames:
                for latest in frames:
                    samples[latest.slot.upper()] = latest
                last_frame_time = now

            poses: dict[str, InertialPose] = {}
            stream_live = last_frame_time is not None and now - last_frame_time <= 0.25
            for label, tracker in trackers.items():
                if stream_live and label in samples:
                    poses[label] = tracker.update(samples[label], now)
                else:
                    tracker.mark_stale(now)
                    poses[label] = tracker.snapshot()
                if poses[label].state == "LIVE":
                    trail = trails[label]
                    if not trail or np.linalg.norm(trail[-1] - poses[label].position_m) > 0.002:
                        trail.append(poses[label].position_m.copy())

            if previous_render_time is not None:
                instant_fps = 1.0 / max(now - previous_render_time, 1e-6)
                fps = instant_fps if fps == 0.0 else 0.1 * instant_fps + 0.9 * fps
            previous_render_time = now
            panel_width = WINDOW_SIZE[0] // 2
            panel_size = (panel_width, WINDOW_SIZE[1])
            right_panel = _draw_panel(
                "RIGHT", poses["RIGHT"], (60, 155, 255), trails["RIGHT"], panel_size
            )
            left_panel = _draw_panel(
                "LEFT", poses["LEFT"], (255, 155, 65), trails["LEFT"], panel_size
            )
            display = np.hstack([right_panel, left_panel])
            cv2.line(display, (panel_width, 0), (panel_width, WINDOW_SIZE[1]), (90, 90, 95), 1)
            status = (
                f"{device}  {baud} baud  FPS {fps:.1f}  frames {parser.frames_decoded}  "
                f"drops {sum(parser.dropped_frames.values())}  discarded {parser.invalid_bytes} B"
            )
            text_size = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
            cv2.rectangle(display, (0, 0), (text_size[0] + 16, 23), (15, 15, 18), -1)
            _put_text(display, status, (8, 17), (205, 205, 210), 0.45)
            _put_text(
                display,
                "R recenter | C calibrate | Q quit",
                (WINDOW_SIZE[0] - 275, 17),
                (205, 205, 210),
                0.43,
            )
            cv2.imshow(window, display)
            render_time_ms = (time.monotonic() - now) * 1000.0
            key = cv2.waitKey(max(1, int(round(1000.0 / 60.0 - render_time_ms)))) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("r"):
                for label, tracker in trackers.items():
                    tracker.recenter()
                    trails[label].clear()
            elif key == ord("c"):
                for label, tracker in trackers.items():
                    tracker.recalibrate()
                    trails[label].clear()
    finally:
        port.close()
        cv2.destroyAllWindows()
