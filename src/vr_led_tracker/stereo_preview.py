from __future__ import annotations

import time
from collections import deque
from dataclasses import replace

import cv2
import numpy as np

from .camera import StereoCameraCapture
from .config import ControllerModel, StereoCalibration
from .detection import SphereDetection, SphereDetector
from .errors import TrackerError
from .filtering import rotation_matrix_to_euler_zyx
from .fusion import StereoFusionTracker
from .hades_fusion import HadesFusionSettings
from .inertial import InertialPose
from .inertial_preview import _draw_panel
from .preview import STATE_COLORS, _SerialReader, _camera_panel, _draw_detections, _draw_pose, _put_lines


def run_stereo_preview(
    left_device: str, right_device: str, serial_device: str, baud: int, imu_slot: str,
    model_path: str, stereo_camera_path: str, fusion_settings_path: str,
    full_error_limit_px: float = 5.0, max_frame_skew_ms: float = 20.0,
) -> None:
    model = ControllerModel.load(model_path)
    source_calibration = StereoCalibration.load(stereo_camera_path)
    settings = HadesFusionSettings.load(fusion_settings_path)
    capture = StereoCameraCapture(left_device, right_device, max_frame_skew_ms / 1000.0)
    try:
        serial_reader = _SerialReader(serial_device, baud)
    except Exception:
        capture.close()
        raise
    detectors = {"left": SphereDetector(), "right": SphereDetector()}
    latest_frames: dict[str, np.ndarray | None] = {"left": None, "right": None}
    latest_candidates: dict[str, list[SphereDetection]] = {"left": [], "right": []}
    latest_masks: dict[str, np.ndarray | None] = {"left": None, "right": None}
    latest_thresholds = {"left": 0, "right": 0}
    calibration = None
    tracker = None
    trail: deque[np.ndarray] = deque(maxlen=180)
    show_masks = False
    fps = 0.0
    previous_time = None
    window = "Stereo Hades-style three-sphere tracker"
    try:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 1920, 640)
        while True:
            observation = capture.next_observation()
            now = time.monotonic()
            if serial_reader.error is not None:
                raise TrackerError(f"Cannot read serial device {serial_device}: {serial_reader.error}")
            current = {"left": None, "right": None}
            for side, packet in (("left", observation.left), ("right", observation.right)):
                if packet is None:
                    continue
                latest_frames[side] = packet.frame
                detections, mask, threshold = detectors[side].detect(packet.frame)
                latest_candidates[side], latest_masks[side], latest_thresholds[side] = detections, mask, threshold
                current[side] = detections
            if latest_frames["left"] is None or latest_frames["right"] is None:
                continue
            if calibration is None:
                lf, rf = latest_frames["left"], latest_frames["right"]
                calibration = source_calibration.for_image_sizes(
                    (lf.shape[1], lf.shape[0]), (rf.shape[1], rf.shape[0])
                )
                tracker = StereoFusionTracker(model, calibration, imu_slot, settings, max_frame_skew_ms / 1000.0)
                tracker.pose_estimator.center_error_limit_px = full_error_limit_px
                tracker.pose_estimator.left_mono.center_error_limit_px = full_error_limit_px
                tracker.pose_estimator.right_mono.center_error_limit_px = full_error_limit_px
            for sample in serial_reader.drain():
                tracker.add_imu(sample, sample.arrival_time or now)
            result = tracker.process_stereo(
                current["left"], current["right"], observation.timestamp, observation.skew_s or 0.0
            )
            left_display, right_display = latest_frames["left"].copy(), latest_frames["right"].copy()
            _draw_detections(left_display, latest_candidates["left"], tracker.pose_estimator.last_left_assignment)
            _draw_detections(right_display, latest_candidates["right"], tracker.pose_estimator.last_right_assignment)
            _draw_pose(left_display, result, calibration.left)
            if result.pose is not None:
                right_rotation = calibration.right_from_left_rotation @ result.pose.rotation_matrix
                right_position = (
                    calibration.right_from_left_rotation @ (result.pose.position_m * 1000.0)
                    + calibration.right_from_left_translation_mm
                ) / 1000.0
                right_result = replace(
                    result,
                    pose=replace(
                        result.pose,
                        rotation_matrix=right_rotation,
                        position_m=right_position,
                    ),
                )
                _draw_pose(right_display, right_result, calibration.right)
            if result.pose is not None:
                if not trail or np.linalg.norm(trail[-1] - result.pose.position_m) > 0.002:
                    trail.append(result.pose.position_m.copy())
                pose = InertialPose(result.state, result.pose.rotation_matrix, result.pose.position_m,
                                    result.pose.velocity_m_s, np.zeros(3),
                                    rotation_matrix_to_euler_zyx(result.pose.rotation_matrix),
                                    result.calibration_progress)
            else:
                pose = InertialPose(result.state, np.eye(3), np.zeros(3), np.zeros(3),
                                    np.zeros(3), np.zeros(3), result.calibration_progress)
            synthetic = _draw_panel("FUSED", pose, (80, 210, 255), trail, (640, 640),
                                    model.object_points / 1000.0, model.diameters_mm / 1000.0,
                                    tuple([(245, 245, 245)] * 3))
            if previous_time is not None:
                instant = 1.0 / max(now - previous_time, 1e-6)
                fps = instant if fps == 0 else 0.1 * instant + 0.9 * fps
            previous_time = now
            estimate = result.camera_estimate
            mode = estimate.source if estimate is not None else "NO_OPTICAL_POSE"
            skew_text = "mono" if observation.skew_s is None else f"{observation.skew_s*1000:.1f} ms"
            lines = [
                (f"{result.state}  {mode}  FPS {fps:.1f}  skew {skew_text}", STATE_COLORS[result.state]),
                (f"Candidates L/R {len(latest_candidates['left'])}/{len(latest_candidates['right'])}  thresholds {latest_thresholds['left']}/{latest_thresholds['right']}", (235,235,235)),
            ]
            if estimate is not None:
                angle = "n/a" if estimate.triangulation_angle_deg is None else f"{estimate.triangulation_angle_deg:.2f} deg"
                lines.append((f"Reprojection {estimate.reprojection_error_px:.2f}px  angle {angle}  confidence scale {estimate.camera_measurement_scale:.2f}x", (235,235,235)))
            if result.state == "CALIBRATING_STILL":
                lines.append((result.calibration_detail, (0,210,255)))
            lines.append(("M masks | R recalibrate | Q quit", (210,210,210)))
            left_panel, right_panel = _camera_panel(left_display, (640,640)), _camera_panel(right_display, (640,640))
            _put_lines(left_panel, lines)
            cv2.imshow(window, np.hstack([left_panel, right_panel, synthetic]))
            if show_masks:
                if latest_masks["left"] is not None: cv2.imshow("Left sphere mask", latest_masks["left"])
                if latest_masks["right"] is not None: cv2.imshow("Right sphere mask", latest_masks["right"])
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1: break
            if key == ord("m"):
                show_masks = not show_masks
                if not show_masks:
                    cv2.destroyWindow("Left sphere mask"); cv2.destroyWindow("Right sphere mask")
            elif key == ord("r"):
                tracker.reset(); trail.clear()
    finally:
        capture.close()
        serial_reader.close()
        cv2.destroyAllWindows()
