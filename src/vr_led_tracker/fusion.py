from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from .config import CameraCalibration, ControllerModel
from .detection import SphereDetection
from .inertial import (
    ErrorStateKalmanFilter,
    KalmanPose,
    LatencyEstimator,
    WORLD_GRAVITY_G,
    _rotation_distance,
    wxyz_to_rotation_matrix,
)
from .pose import PoseEstimate, PoseEstimator
from .serial_pose import FusionSample, TimestampMapper


@dataclass
class _HistoryEntry:
    timestamp: float
    imu_rotation: np.ndarray
    acceleration_g: np.ndarray
    filter_state: dict[str, np.ndarray | bool]


@dataclass(frozen=True)
class FusionResult:
    state: str
    pose: KalmanPose | None
    camera_estimate: PoseEstimate | None
    latency_s: float | None
    latency_correlation: float | None
    visible_spheres: int
    calibration_progress: float
    calibration_detail: str


class FusionTracker:
    def __init__(self, model: ControllerModel, calibration: CameraCalibration, slot: str) -> None:
        self.model = model
        self.slot = slot
        self.pose_estimator = PoseEstimator(model, calibration)
        self.filter = ErrorStateKalmanFilter()
        self.timestamp_mapper = TimestampMapper()
        self.latency_estimator = LatencyEstimator()
        self.state = "CALIBRATING_STILL"
        self.latest_sample: FusionSample | None = None
        self.latest_imu_rotation: np.ndarray | None = None
        self.latest_imu_time: float | None = None
        self.latest_arrival_time: float | None = None
        self.previous_imu_rotation: np.ndarray | None = None
        self.previous_imu_time: float | None = None
        self.latest_angular_speed = 0.0
        self.stationary_since: float | None = None
        self.still_started: float | None = None
        self.bias_samples: list[np.ndarray] = []
        self.still_camera_poses: deque[tuple[float, np.ndarray, np.ndarray]] = deque()
        self.calibration_detail = "waiting for camera and IMU data"
        self.history: deque[_HistoryEntry] = deque()
        self.imu_motion: deque[tuple[float, float]] = deque(maxlen=1500)
        self.camera_motion: deque[tuple[float, float]] = deque(maxlen=300)
        self.previous_camera_rotation: np.ndarray | None = None
        self.previous_camera_time: float | None = None
        self.latency_s: float | None = None
        self.latency_correlation: float | None = None
        self.last_camera_update: float | None = None
        self.last_camera_estimate: PoseEstimate | None = None
        self.last_camera_read_time: float | None = None

    def reset(self) -> None:
        fresh = FusionTracker(self.model, self.pose_estimator.calibration, self.slot)
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
                self.imu_motion.append((timestamp, self.latest_angular_speed))
        self.latest_sample = sample
        self.latest_imu_rotation = rotation
        self.latest_imu_time = timestamp
        self.latest_arrival_time = arrival_time

        if self.filter.initialized and self.previous_imu_time is not None:
            dt = timestamp - self.previous_imu_time
            if 0.0 < dt < 0.2:
                self.filter.predict(rotation, sample.acceleration_g, dt)
                self._update_stationary(sample, timestamp)
                self.history.append(
                    _HistoryEntry(
                        timestamp,
                        rotation.copy(),
                        sample.acceleration_g.copy(),
                        self.filter.export_state(),
                    )
                )
                while self.history and timestamp - self.history[0].timestamp > 1.2:
                    self.history.popleft()
        self.previous_imu_rotation = rotation.copy()
        self.previous_imu_time = timestamp

    def _update_stationary(self, sample: FusionSample, timestamp: float) -> None:
        still = (
            abs(float(np.linalg.norm(sample.acceleration_g)) - 1.0) < 0.05
            and self.latest_angular_speed < math.radians(5.0)
        )
        if still:
            if self.stationary_since is None:
                self.stationary_since = timestamp
            elif timestamp - self.stationary_since >= 0.2:
                self.filter.update_zero_velocity()
        else:
            self.stationary_since = None

    def process_camera(
        self,
        detections: dict[str, SphereDetection],
        read_time: float,
    ) -> FusionResult:
        visible = len(detections)
        self.last_camera_read_time = read_time
        full = self.pose_estimator.estimate_camera_pose(
            detections,
            self.filter.rotation if self.filter.initialized and self.latency_s is not None else None,
            self.filter.position * 1000.0
            if self.filter.initialized and self.latency_s is not None
            else None,
        )
        if not self.filter.initialized:
            return self._calibrate_still(full, visible, read_time)

        if full is not None:
            camera_rotation = self._record_camera_motion(full, read_time)
        else:
            camera_rotation = None

        if self.latency_s is None:
            self._try_estimate_latency()
            if full is not None:
                self.filter.update_camera(
                    full.tvec.reshape(3) / 1000.0,
                    camera_rotation,
                    position_sigma_m=0.012,
                )
                self.last_camera_update = read_time
                self.last_camera_estimate = full
            self.state = "CALIBRATING_DELAY"
            return self.result(visible)

        imu_fresh = self.latest_arrival_time is not None and read_time - self.latest_arrival_time < 0.15
        measurement_time = read_time - self.latency_s
        camera_estimate: PoseEstimate | None = None
        if imu_fresh and visible >= 2:
            past_rotation, past_translation = self._predicted_pose_at(measurement_time)
            camera_estimate = self.pose_estimator.estimate_translation(
                detections,
                past_rotation,
                past_translation * 1000.0,
            )
            if camera_estimate is not None:
                orientation = full and self._rotation(full)
                self._delayed_camera_update(
                    measurement_time,
                    camera_estimate.tvec.reshape(3) / 1000.0,
                    orientation,
                    0.01 if visible == 3 else 0.025,
                )
                self.last_camera_update = read_time
                self.last_camera_estimate = camera_estimate
                self.state = "FULL" if visible == 3 else "DEGRADED_2"
        elif full is not None:
            self.filter.update_camera(
                full.tvec.reshape(3) / 1000.0,
                self._rotation(full),
                position_sigma_m=0.018,
                orientation_sigma_deg=5.0,
            )
            self.last_camera_update = read_time
            self.last_camera_estimate = full
            self.state = "CAMERA_ONLY"

        if camera_estimate is None and not (not imu_fresh and full is not None):
            age = float("inf") if self.last_camera_update is None else read_time - self.last_camera_update
            self.state = "IMU_ONLY" if imu_fresh and age <= 0.25 else "LOST"
            if self.state == "LOST":
                self.filter.velocity.fill(0.0)
        return self.result(visible)

    def _calibrate_still(
        self, full: PoseEstimate | None, visible: int, read_time: float
    ) -> FusionResult:
        if full is None:
            self._reset_still_window("need a stable pose from all three spheres")
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

        camera_position = full.tvec.reshape(3) / 1000.0
        camera_rotation = self._rotation(full)
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
            assert self.latest_imu_time is not None
            self.history.append(
                _HistoryEntry(
                    self.latest_imu_time,
                    self.latest_imu_rotation.copy(),
                    self.latest_sample.acceleration_g.copy(),
                    self.filter.export_state(),
                )
            )
            self.state = "CALIBRATING_DELAY"
            self.calibration_detail = "rotate smoothly to measure camera latency"
        self.last_camera_estimate = full
        return self.result(visible)

    def _reset_still_window(self, detail: str) -> None:
        self.still_started = None
        self.bias_samples.clear()
        self.still_camera_poses.clear()
        self.calibration_detail = detail

    def _record_camera_motion(self, estimate: PoseEstimate, timestamp: float) -> np.ndarray:
        rotation = self._rotation(estimate)
        if self.previous_camera_rotation is not None and self.previous_camera_time is not None:
            dt = timestamp - self.previous_camera_time
            if 0.0 < dt < 0.2:
                speed = _rotation_distance(rotation, self.previous_camera_rotation) / dt
                self.camera_motion.append((timestamp, speed))
        self.previous_camera_rotation = rotation.copy()
        self.previous_camera_time = timestamp
        return rotation

    def _try_estimate_latency(self) -> None:
        if len(self.imu_motion) < 20 or len(self.camera_motion) < 10:
            self.calibration_detail = (
                f"collecting motion: IMU {min(len(self.imu_motion), 20)}/20, "
                f"camera {min(len(self.camera_motion), 10)}/10"
            )
            return
        imu = np.asarray(self.imu_motion)
        camera = np.asarray(self.camera_motion)
        peak_speed = math.degrees(float(np.max(imu[:, 1])))
        if peak_speed < 20.0:
            self.calibration_detail = (
                f"rotate faster with varying speed; peak is {peak_speed:.0f} deg/s"
            )
            return
        estimate = self.latency_estimator.estimate(imu[:, 0], imu[:, 1], camera[:, 0], camera[:, 1])
        if estimate is None:
            self.calibration_detail = "keep rotating with all three spheres visible"
            return
        self.latency_correlation = estimate[1]
        if estimate[1] >= 0.65:
            self.latency_s = estimate[0]
            self.calibration_detail = (
                f"latency calibrated: {estimate[0] * 1000.0:.0f} ms, "
                f"correlation {estimate[1]:.2f}"
            )
        else:
            self.calibration_detail = (
                f"weak motion match {estimate[1]:.2f}/0.65; vary rotation speed"
            )

    def _predicted_pose_at(self, timestamp: float) -> tuple[np.ndarray, np.ndarray]:
        if not self.history:
            return self.filter.rotation.copy(), self.filter.position.copy()
        entry = min(self.history, key=lambda item: abs(item.timestamp - timestamp))
        state = entry.filter_state
        return np.asarray(state["rotation"]).copy(), np.asarray(state["position"]).copy()

    def _delayed_camera_update(
        self,
        timestamp: float,
        position_m: np.ndarray,
        rotation: np.ndarray | None,
        position_sigma_m: float,
    ) -> None:
        if not self.history:
            self.filter.update_camera(position_m, rotation, position_sigma_m)
            return
        history = list(self.history)
        index = min(range(len(history)), key=lambda i: abs(history[i].timestamp - timestamp))
        self.filter.restore_state(history[index].filter_state)
        self.filter.update_camera(position_m, rotation, position_sigma_m)
        history[index].filter_state = self.filter.export_state()
        previous_time = history[index].timestamp
        for later in history[index + 1 :]:
            self.filter.predict(later.imu_rotation, later.acceleration_g, later.timestamp - previous_time)
            later.filter_state = self.filter.export_state()
            previous_time = later.timestamp
        self.history = deque(history)

    @staticmethod
    def _rotation(estimate: PoseEstimate) -> np.ndarray:
        import cv2

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
            self.latency_s,
            self.latency_correlation,
            visible,
            progress,
            self.calibration_detail,
        )
