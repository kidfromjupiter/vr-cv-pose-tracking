from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from .config import CameraCalibration, ControllerModel, StereoCalibration
from .detection import SphereDetection
from .hades_fusion import HadesFusionSettings, HadesMotionFilter, HadesPose
from .inertial import WORLD_GRAVITY_G, _rotation_distance, wxyz_to_rotation_matrix
from .pose import PoseEstimate, PoseEstimator, StereoPoseEstimator
from .serial_pose import FusionSample, TimestampMapper


@dataclass(frozen=True)
class FusionResult:
    state: str
    pose: HadesPose | None
    camera_estimate: PoseEstimate | None
    visible_spheres: int
    calibration_progress: float
    calibration_detail: str


class FusionTracker:
    def __init__(
        self,
        model: ControllerModel,
        calibration: CameraCalibration,
        slot: str,
        settings: HadesFusionSettings | None = None,
    ) -> None:
        self.model = model
        self.slot = slot
        self.settings = settings or HadesFusionSettings()
        self.pose_estimator = PoseEstimator(
            model,
            calibration,
            identity_max_speed_px_s=self.settings.identity_max_speed_px_s,
            identity_reacquire_timeout_s=self.settings.identity_reacquire_timeout_s,
        )
        self.filter = HadesMotionFilter(self.settings)
        self.timestamp_mapper = TimestampMapper()
        self.state = "CALIBRATING_STILL"
        self.latest_sample: FusionSample | None = None
        self.latest_imu_rotation: np.ndarray | None = None
        self.latest_imu_time: float | None = None
        self.latest_arrival_time: float | None = None
        self.previous_imu_rotation: np.ndarray | None = None
        self.previous_imu_time: float | None = None
        self.latest_angular_speed = 0.0
        self.still_started: float | None = None
        self.bias_samples: list[np.ndarray] = []
        self.still_camera_poses: deque[tuple[float, np.ndarray, np.ndarray]] = deque()
        self.calibration_detail = "waiting for camera and IMU data"
        self.last_camera_update: float | None = None
        self.last_camera_estimate: PoseEstimate | None = None
        self.last_camera_read_time: float | None = None

    def reset(self) -> None:
        center_error_limit_px = self.pose_estimator.center_error_limit_px
        fresh = FusionTracker(
            self.model,
            self.pose_estimator.calibration,
            self.slot,
            self.settings,
        )
        fresh.pose_estimator.center_error_limit_px = center_error_limit_px
        self.__dict__.update(fresh.__dict__)

    def add_imu(self, sample: FusionSample, arrival_time: float) -> None:
        if sample.slot != self.slot:
            return
        rotation = wxyz_to_rotation_matrix(sample.quaternion_wxyz)
        if rotation is None:
            return
        timestamp = self.timestamp_mapper.map(sample.timestamp_us, arrival_time)
        if self.previous_imu_rotation is not None and self.previous_imu_time is not None:
            dt = timestamp - self.previous_imu_time
            if 0.0 < dt < 0.2:
                self.latest_angular_speed = _rotation_distance(
                    rotation, self.previous_imu_rotation
                ) / dt
        self.latest_sample = sample
        self.latest_imu_rotation = rotation
        self.latest_imu_time = timestamp
        self.latest_arrival_time = arrival_time

        if self.filter.initialized and self.previous_imu_time is not None:
            dt = timestamp - self.previous_imu_time
            if 0.0 < dt < 0.2:
                self.filter.predict(rotation, sample.acceleration_g, dt)
        self.previous_imu_rotation = rotation.copy()
        self.previous_imu_time = timestamp

    def process_camera(
        self,
        detections: dict[str, SphereDetection] | list[SphereDetection],
        read_time: float,
    ) -> FusionResult:
        visible = len(detections)
        self.last_camera_read_time = read_time
        prediction_fresh = (
            self.filter.initialized
            and self.last_camera_update is not None
            and read_time - self.last_camera_update <= 0.25
        )
        estimate = self.pose_estimator.estimate_camera_pose(
            detections,
            self.filter.rotation if prediction_fresh else None,
            self.filter.position_filter.current_estimate * 1000.0 if prediction_fresh else None,
            frame_time=read_time,
        )
        return self._process_estimate(estimate, visible, read_time)

    def _process_estimate(
        self, estimate: PoseEstimate | None, visible: int, read_time: float
    ) -> FusionResult:
        if not self.filter.initialized:
            return self._calibrate_still(estimate, visible, read_time)

        imu_fresh = self.latest_arrival_time is not None and read_time - self.latest_arrival_time < 0.15
        if estimate is not None:
            self.filter.update_camera(
                estimate.tvec.reshape(3) / 1000.0,
                read_time,
                estimate.camera_measurement_scale,
            )
            self.last_camera_update = read_time
            self.last_camera_estimate = estimate
            self.state = "FULL" if imu_fresh else "CAMERA_ONLY"
        else:
            self.filter.mark_camera_missing()
            self.state = "IMU_ONLY" if imu_fresh else "LOST"
        return self.result(visible)

    def _calibrate_still(
        self, estimate: PoseEstimate | None, visible: int, read_time: float
    ) -> FusionResult:
        if estimate is None:
            self._reset_still_window("need a stable assignment of all three white spheres")
            return self.result(visible)
        if self.latest_sample is None or self.latest_imu_rotation is None:
            self._reset_still_window(f"waiting for {self.slot} IMU packets")
            return self.result(visible)
        if self.latest_arrival_time is None or read_time - self.latest_arrival_time >= 0.25:
            self._reset_still_window(f"{self.slot} IMU packets are stale")
            return self.result(visible)

        acceleration_norm = float(np.linalg.norm(self.latest_sample.acceleration_g))
        if abs(acceleration_norm - 1.0) >= 0.08:
            self._reset_still_window(f"acceleration is {acceleration_norm:.2f} g; need 0.92-1.08 g")
            return self.result(visible)

        camera_position = estimate.tvec.reshape(3) / 1000.0
        camera_rotation = self._rotation(estimate)
        if self.still_camera_poses:
            _, reference_position, reference_rotation = self.still_camera_poses[0]
            translation_change = float(np.linalg.norm(camera_position - reference_position))
            rotation_change = math.degrees(_rotation_distance(camera_rotation, reference_rotation))
            if translation_change > 0.015 or rotation_change > 5.0:
                self._reset_still_window(
                    f"camera saw motion: {translation_change * 1000.0:.0f} mm, "
                    f"{rotation_change:.1f} deg"
                )

        if not self.still_camera_poses:
            self.still_started = read_time
        self.still_camera_poses.append((read_time, camera_position.copy(), camera_rotation.copy()))
        expected_body_gravity = self.latest_imu_rotation.T @ WORLD_GRAVITY_G
        self.bias_samples.append(self.latest_sample.acceleration_g - expected_body_gravity)
        assert self.still_started is not None
        elapsed = read_time - self.still_started
        self.calibration_detail = (
            f"camera stable {elapsed:.1f}/1.0 s; "
            f"IMU drift {math.degrees(self.latest_angular_speed):.1f} deg/s"
        )
        if elapsed >= 1.0 and len(self.bias_samples) >= 20:
            self.filter.initialize(
                camera_position,
                camera_rotation,
                self.latest_imu_rotation,
                np.mean(self.bias_samples, axis=0),
            )
            self.filter.camera_tracking = True
            self.filter.last_camera_time = read_time
            self.filter.last_camera_position = camera_position.copy()
            self.filter.last_camera_estimate = camera_position.copy()
            self.last_camera_update = read_time
            self.state = "FULL"
            self.calibration_detail = "calibration complete"
        self.last_camera_estimate = estimate
        return self.result(visible)

    def _reset_still_window(self, detail: str) -> None:
        self.still_started = None
        self.bias_samples.clear()
        self.still_camera_poses.clear()
        self.calibration_detail = detail

    @staticmethod
    def _rotation(estimate: PoseEstimate) -> np.ndarray:
        return cv2.Rodrigues(estimate.rvec)[0]

    def result(self, visible: int) -> FusionResult:
        progress = 0.0
        if self.state == "CALIBRATING_STILL" and self.still_started is not None:
            progress = min(
                1.0,
                ((self.last_camera_read_time or self.still_started) - self.still_started),
            )
        return FusionResult(
            self.state,
            self.filter.snapshot() if self.filter.initialized else None,
            self.last_camera_estimate,
            visible,
            progress,
            self.calibration_detail,
        )


class StereoFusionTracker(FusionTracker):
    def __init__(
        self, model: ControllerModel, calibration: StereoCalibration, slot: str,
        settings: HadesFusionSettings | None = None, max_frame_skew_s: float = 0.02,
    ) -> None:
        active_settings = settings or HadesFusionSettings()
        super().__init__(model, calibration.left, slot, active_settings)
        self.stereo_calibration = calibration
        self.max_frame_skew_s = float(max_frame_skew_s)
        self.pose_estimator = StereoPoseEstimator(
            model, calibration,
            epipolar_error_limit_px=active_settings.stereo_epipolar_error_limit_px,
            rigid_error_limit_mm=active_settings.stereo_rigid_error_limit_mm,
            min_triangulation_angle_deg=active_settings.stereo_min_triangulation_angle_deg,
            quality_reference_area_px2=active_settings.stereo_quality_reference_area_px2,
            quality_reference_angle_deg=active_settings.stereo_quality_reference_angle_deg,
            max_stereo_measurement_scale=active_settings.stereo_max_measurement_scale,
            mono_base_measurement_scale=active_settings.stereo_mono_base_measurement_scale,
            max_mono_measurement_scale=active_settings.stereo_max_mono_measurement_scale,
            identity_max_speed_px_s=active_settings.identity_max_speed_px_s,
            identity_reacquire_timeout_s=active_settings.identity_reacquire_timeout_s,
            allow_mono_fallback=active_settings.stereo_allow_mono_fallback,
        )

    def reset(self) -> None:
        error_limit = self.pose_estimator.center_error_limit_px
        fresh = StereoFusionTracker(
            self.model, self.stereo_calibration, self.slot, self.settings, self.max_frame_skew_s
        )
        fresh.pose_estimator.center_error_limit_px = error_limit
        fresh.pose_estimator.left_mono.center_error_limit_px = error_limit
        fresh.pose_estimator.right_mono.center_error_limit_px = error_limit
        self.__dict__.update(fresh.__dict__)

    def process_stereo(
        self, left_detections: list[SphereDetection] | None,
        right_detections: list[SphereDetection] | None, read_time: float,
        frame_skew_s: float = 0.0,
    ) -> FusionResult:
        visible = max(len(left_detections or []), len(right_detections or []))
        self.last_camera_read_time = read_time
        fresh = self.filter.initialized and self.last_camera_update is not None and read_time - self.last_camera_update <= 0.25
        estimate = self.pose_estimator.estimate_camera_pose(
            left_detections, right_detections,
            self.filter.rotation if fresh else None,
            self.filter.position_filter.current_estimate * 1000.0 if fresh else None,
            read_time, frame_skew_s, self.max_frame_skew_s,
        )
        return self._process_estimate(estimate, visible, read_time)
