"""Receiver-only orientation diagnostics; no camera or acceleration integration."""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from .errors import TrackerError
from .filtering import rotation_matrix_to_euler_zyx
from .inertial import wxyz_to_rotation_matrix
from .inertial_preview import (
    AXIS_COLORS,
    BOX_EDGES,
    BOX_VERTICES,
    STATE_COLORS,
    WINDOW_SIZE,
    _draw_grid,
    _line_3d,
    _project,
    _put_text,
)
from .serial_pose import FusionFrameParser, FusionSample


@dataclass
class ControllerOrientation:
    sample: FusionSample | None = None
    rotation: np.ndarray | None = None
    reference: np.ndarray | None = None
    last_valid_time: float | None = None

    def update(self, sample: FusionSample, now: float) -> bool:
        rotation = wxyz_to_rotation_matrix(sample.quaternion_wxyz)
        if rotation is None:
            return False
        self.sample = sample
        self.rotation = rotation
        self.last_valid_time = now
        return True

    def state(self, now: float) -> str:
        if self.last_valid_time is None:
            return "WAITING"
        return "LIVE" if now - self.last_valid_time <= 0.25 else "STALE"

    def displayed_rotation(self) -> np.ndarray:
        if self.rotation is None:
            return np.eye(3)
        if self.reference is None:
            return self.rotation.copy()
        return self.reference.T @ self.rotation

    def recenter(self, now: float) -> None:
        if self.state(now) == "LIVE" and self.rotation is not None:
            self.reference = self.rotation.copy()


def draw_orientation_panel(
    label: str,
    controller: ControllerOrientation,
    now: float,
    size: tuple[int, int],
) -> np.ndarray:
    width, height = size
    image = np.full((height, width, 3), (24, 24, 29), dtype=np.uint8)
    center = np.array([0.0, 0.0, 0.16])
    eye = center + np.array([0.42, -0.56, 0.36])
    _draw_grid(image, 0.25, eye, center)
    state = controller.state(now)
    _put_text(image, f"{label}  {state}", (14, 27), STATE_COLORS[state], 0.7)
    mode = "RECENTERED" if controller.reference is not None else "RECEIVER ORIENTATION"
    _put_text(image, mode, (14, 53))

    if controller.sample is None:
        _put_text(image, "Waiting for a valid quaternion", (14, 82))
        return image

    rotation = controller.displayed_rotation()
    vertices = (rotation @ BOX_VERTICES.T).T + center
    box_color = (255, 155, 65) if label == "LEFT" else (60, 155, 255)
    for first, second in BOX_EDGES:
        _line_3d(image, vertices[first], vertices[second], eye, center, box_color, 2)
    nose = np.array([[0.08, -0.025, 0.0], [0.125, 0.0, 0.0], [0.08, 0.025, 0.0]])
    nose = (rotation @ nose.T).T + center
    for first, second in ((0, 1), (1, 2), (2, 0)):
        _line_3d(image, nose[first], nose[second], eye, center, box_color, 2)
    for axis, (name, color) in enumerate(zip("XYZ", AXIS_COLORS)):
        endpoint = center + rotation[:, axis] * 0.15
        _line_3d(image, center, endpoint, eye, center, color, 2)
        pixel, depth = _project(endpoint[None, :], width, height, eye, center)
        if depth[0] > 0.01:
            x, y = np.rint(pixel[0]).astype(int)
            _put_text(image, name, (int(x) + 5, int(y) - 5), color, 0.6)

    w, x, y, z = controller.sample.quaternion_wxyz
    yaw, pitch, roll = rotation_matrix_to_euler_zyx(rotation)
    age_ms = (now - controller.last_valid_time) * 1000.0
    bottom = height - 100
    _put_text(image, f"Received w,x,y,z: {w:+.4f} {x:+.4f} {y:+.4f} {z:+.4f}", (14, bottom))
    _put_text(image, f"Displayed YPR deg: {yaw:+.1f} {pitch:+.1f} {roll:+.1f}", (14, bottom + 25))
    _put_text(image, f"Sample age: {age_ms:.0f} ms  Sequence: {controller.sample.sequence}", (14, bottom + 50))
    _put_text(image, "Axes: X red | Y green | Z blue", (14, bottom + 75))
    return image


def run_orientation_preview(device: str = "/dev/ttyACM0", baud: int = 230400) -> None:
    if baud <= 0:
        raise TrackerError("--baud must be positive")
    try:
        import serial
    except ImportError as exc:
        raise TrackerError("pyserial is required for the orientation viewer") from exc
    try:
        port = serial.Serial(device, baudrate=baud, timeout=0)
    except (OSError, serial.SerialException) as exc:
        raise TrackerError(f"Cannot open serial device {device}: {exc}") from exc

    parser = FusionFrameParser()
    controllers = {"left": ControllerOrientation(), "right": ControllerOrientation()}
    window = "Controller quaternion orientation"
    header_height = 52
    panel_size = (WINDOW_SIZE[0] // 2, WINDOW_SIZE[1] - header_height)
    try:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, *WINDOW_SIZE)
        while True:
            started = time.monotonic()
            try:
                waiting = port.in_waiting
                data = port.read(waiting) if waiting else b""
            except (OSError, serial.SerialException) as exc:
                raise TrackerError(f"Cannot read serial device {device}: {exc}") from exc
            now = time.monotonic()
            for sample in parser.feed(data, now):
                controllers[sample.slot].update(sample, now)

            panels = np.hstack([
                draw_orientation_panel(slot.upper(), controller, now, panel_size)
                for slot, controller in controllers.items()
            ])
            header = np.full((header_height, WINDOW_SIZE[0], 3), (15, 15, 18), dtype=np.uint8)
            _put_text(header, f"{device}  {baud} baud  Frames: {parser.frames_decoded}  "
                      f"CRC errors: {parser.crc_errors}  Drops: {sum(parser.dropped_frames.values())}", (14, 20))
            _put_text(header, "R recenter live controllers | C restore receiver orientation | Q / Esc quit", (14, 42))
            display = np.vstack([header, panels])
            cv2.line(display, (panel_size[0], header_height), (panel_size[0], WINDOW_SIZE[1]), (90, 90, 95), 1)
            cv2.imshow(window, display)
            remaining_ms = 1000.0 / 60.0 - (time.monotonic() - started) * 1000.0
            key = cv2.waitKey(max(1, int(round(remaining_ms)))) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("r"):
                for controller in controllers.values():
                    controller.recenter(time.monotonic())
            elif key == ord("c"):
                for controller in controllers.values():
                    controller.reference = None
    except cv2.error as exc:
        raise TrackerError(f"Cannot display orientation viewer: {exc}") from exc
    finally:
        try:
            port.close()
        finally:
            cv2.destroyAllWindows()
