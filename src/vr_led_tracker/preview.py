from __future__ import annotations

import queue
import threading
import time
from collections import deque

import cv2
import numpy as np

from .camera import open_camera, read_frame
from .config import CameraCalibration, ControllerModel, load_color_profiles
from .detection import SphereDetection, SphereDetector
from .errors import TrackerError
from .filtering import rotation_matrix_to_euler_zyx
from .fusion import FusionResult, FusionTracker
from .inertial import InertialPose
from .inertial_preview import _draw_panel
from .serial_pose import FusionFrameParser, FusionSample


PALETTE = {"red": (20, 20, 245), "blue": (245, 100, 20), "white": (245, 245, 245)}
STATE_COLORS = {
    "CALIBRATING_STILL": (0, 210, 255),
    "CALIBRATING_DELAY": (0, 170, 255),
    "FULL": (40, 230, 40),
    "DEGRADED_2": (0, 200, 255),
    "IMU_ONLY": (0, 160, 255),
    "CAMERA_ONLY": (220, 170, 30),
    "LOST": (40, 40, 255),
}


class _SerialReader:
    def __init__(self, device: str, baud: int) -> None:
        try:
            import serial
        except ImportError as exc:
            raise TrackerError("pyserial is required for fused tracking") from exc
        try:
            self.port = serial.Serial(device, baudrate=baud, timeout=0.05)
        except (OSError, serial.SerialException) as exc:
            raise TrackerError(f"Cannot open serial device {device}: {exc}") from exc
        self.serial_module = serial
        self.parser = FusionFrameParser()
        self.samples: queue.Queue[FusionSample] = queue.Queue()
        self.error: Exception | None = None
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="fusion-serial", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        try:
            while not self.stop_event.is_set():
                count = max(1, self.port.in_waiting)
                data = self.port.read(count)
                if not data:
                    continue
                arrival = time.monotonic()
                for sample in self.parser.feed(data, arrival):
                    self.samples.put(sample)
        except (OSError, self.serial_module.SerialException) as exc:
            self.error = exc

    def drain(self) -> list[FusionSample]:
        values = []
        while True:
            try:
                values.append(self.samples.get_nowait())
            except queue.Empty:
                return values

    def close(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=0.2)
        self.port.close()


def _draw_detections(
    frame: np.ndarray,
    model: ControllerModel,
    detections: dict[str, SphereDetection],
) -> None:
    for sphere in model.spheres:
        detection = detections.get(sphere.label)
        if detection is None:
            continue
        center = tuple(np.rint(detection.center).astype(int))
        color = PALETTE[sphere.label]
        cv2.circle(frame, center, max(3, int(round(detection.radius))), color, 2, cv2.LINE_AA)
        cv2.drawMarker(frame, center, color, cv2.MARKER_CROSS, 10, 1)
        cv2.putText(
            frame,
            f"{sphere.label} {2 * detection.radius:.1f}px",
            (center[0] + 7, center[1] - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )


def _draw_pose(frame: np.ndarray, result: FusionResult, calibration: CameraCalibration) -> None:
    if result.pose is None:
        return
    rvec, _ = cv2.Rodrigues(result.pose.rotation_matrix)
    cv2.drawFrameAxes(
        frame,
        calibration.camera_matrix,
        calibration.distortion,
        rvec,
        result.pose.position_m.reshape(3, 1) * 1000.0,
        60.0,
        2,
    )


def _camera_panel(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    scale = min(width / frame.shape[1], height / frame.shape[0])
    resized = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    output = np.full((height, width, 3), (18, 18, 22), dtype=np.uint8)
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    output[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return output


def _put_lines(frame: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]]) -> None:
    y = 28
    for line, color in lines:
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)
        y += 24


def run_preview(
    device: str,
    serial_device: str,
    baud: int,
    imu_slot: str,
    model_path: str,
    camera_path: str,
    colors_path: str,
    full_error_limit_px: float = 5.0,
) -> None:
    model = ControllerModel.load(model_path)
    source_calibration = CameraCalibration.load(camera_path)
    profiles = load_color_profiles(colors_path, model.labels)
    capture = open_camera(device)
    serial_reader = _SerialReader(serial_device, baud)
    detector = SphereDetector(profiles)
    calibration: CameraCalibration | None = None
    tracker: FusionTracker | None = None
    previous_pixels: dict[str, np.ndarray] = {}
    trail: deque[np.ndarray] = deque(maxlen=180)
    fps = 0.0
    previous_frame_time: float | None = None
    show_masks = False
    window = "Three-sphere camera/IMU tracker"
    try:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 1440, 720)
        while True:
            frame = read_frame(capture)
            now = time.monotonic()
            if serial_reader.error is not None:
                raise TrackerError(f"Cannot read serial device {serial_device}: {serial_reader.error}")
            height, width = frame.shape[:2]
            if calibration is None:
                calibration = source_calibration.for_image_size((width, height))
                tracker = FusionTracker(model, calibration, imu_slot)
                tracker.pose_estimator.center_error_limit_px = full_error_limit_px
            assert tracker is not None and calibration is not None
            for sample in serial_reader.drain():
                tracker.add_imu(sample, sample.arrival_time or now)

            detections, masks = detector.detect(frame, previous_pixels)
            previous_pixels.update({label: value.center for label, value in detections.items()})
            result = tracker.process_camera(detections, now)
            display_camera = frame.copy()
            _draw_detections(display_camera, model, detections)
            _draw_pose(display_camera, result, calibration)

            if result.pose is not None:
                if not trail or np.linalg.norm(trail[-1] - result.pose.position_m) > 0.002:
                    trail.append(result.pose.position_m.copy())
                inertial_pose = InertialPose(
                    result.state,
                    result.pose.rotation_matrix,
                    result.pose.position_m,
                    result.pose.velocity_m_s,
                    np.zeros(3),
                    rotation_matrix_to_euler_zyx(result.pose.rotation_matrix),
                    result.calibration_progress,
                )
            else:
                inertial_pose = InertialPose(
                    result.state,
                    np.eye(3),
                    np.zeros(3),
                    np.zeros(3),
                    np.zeros(3),
                    np.zeros(3),
                    result.calibration_progress,
                )
            synthetic = _draw_panel(
                "FUSED",
                inertial_pose,
                (80, 210, 255),
                trail,
                (720, 720),
                model.object_points / 1000.0,
                model.diameters_mm / 1000.0,
                tuple(PALETTE[label] for label in model.labels),
            )

            if previous_frame_time is not None:
                instant = 1.0 / max(now - previous_frame_time, 1e-6)
                fps = instant if fps == 0.0 else 0.1 * instant + 0.9 * fps
            previous_frame_time = now
            latency = "estimating" if result.latency_s is None else f"{result.latency_s * 1000:.0f} ms"
            state_color = STATE_COLORS[result.state]
            lines = [
                (f"{result.state}  spheres {len(detections)}/3  FPS {fps:.1f}", state_color),
                (f"DroidCam latency: {latency}", (235, 235, 235)),
            ]
            if result.state == "CALIBRATING_STILL":
                lines.append((result.calibration_detail, (0, 210, 255)))
            elif result.state == "CALIBRATING_DELAY":
                lines.append((result.calibration_detail, (0, 190, 255)))
            lines.append(("M masks | R recalibrate | Q quit", (210, 210, 210)))
            camera_view = _camera_panel(display_camera, (720, 720))
            _put_lines(camera_view, lines)
            display = np.hstack([camera_view, synthetic])
            cv2.imshow(window, display)
            if show_masks:
                cv2.imshow("Sphere masks", np.hstack([masks[label] for label in model.labels]))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("m"):
                show_masks = not show_masks
                if not show_masks:
                    cv2.destroyWindow("Sphere masks")
            elif key == ord("r"):
                tracker.reset()
                previous_pixels.clear()
                trail.clear()
    finally:
        capture.release()
        serial_reader.close()
        cv2.destroyAllWindows()
