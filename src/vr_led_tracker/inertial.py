from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .filtering import quaternion_to_rotation_matrix, rotation_matrix_to_euler_zyx
from .serial_pose import FusionSample


GRAVITY_M_S2 = 9.80665
WORLD_GRAVITY_G = np.array([0.0, 0.0, 1.0], dtype=np.float64)


def _rotation_distance(first: np.ndarray, second: np.ndarray) -> float:
    relative = first @ second.T
    cosine = np.clip((float(np.trace(relative)) - 1.0) / 2.0, -1.0, 1.0)
    return math.acos(cosine)


def wxyz_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray | None:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all() or norm < 0.5:
        return None
    w, x, y, z = quaternion / norm
    return quaternion_to_rotation_matrix(np.array([x, y, z, w]))


@dataclass(frozen=True)
class InertialPose:
    state: str
    rotation_matrix: np.ndarray
    position_m: np.ndarray
    velocity_m_s: np.ndarray
    acceleration_m_s2: np.ndarray
    euler_zyx_deg: np.ndarray
    calibration_progress: float


class InertialPoseTracker:
    calibration_time_s = 1.0
    calibration_samples = 30
    stale_timeout_s = 0.25
    max_dt_s = 0.05
    acceleration_deadband_m_s2 = 0.15
    stationary_time_s = 0.2

    def __init__(self) -> None:
        self.state = "WAITING"
        self.position = np.zeros(3, dtype=np.float64)
        self.velocity = np.zeros(3, dtype=np.float64)
        self.linear_acceleration = np.zeros(3, dtype=np.float64)
        self.reference_rotation: np.ndarray | None = None
        self.body_accel_bias = np.zeros(3, dtype=np.float64)
        self.latest_rotation: np.ndarray | None = None
        self.latest_acceleration_g: np.ndarray | None = None
        self.previous_rotation: np.ndarray | None = None
        self.previous_time: float | None = None
        self.last_valid_time: float | None = None
        self.stationary_since: float | None = None
        self.calibration_started: float | None = None
        self.calibration_biases: list[np.ndarray] = []

    def _valid_sample(
        self, sample: FusionSample
    ) -> tuple[np.ndarray, np.ndarray] | None:
        rotation = wxyz_to_rotation_matrix(sample.quaternion_wxyz)
        acceleration = np.asarray(sample.acceleration_g, dtype=np.float64)
        if (
            rotation is None
            or acceleration.shape != (3,)
            or not np.isfinite(acceleration).all()
            or not 0.1 < float(np.linalg.norm(acceleration)) < 8.5
        ):
            return None
        return rotation, acceleration

    def _angular_speed_deg_s(self, rotation: np.ndarray, timestamp: float) -> float:
        if self.previous_rotation is None or self.previous_time is None:
            return 0.0
        dt = timestamp - self.previous_time
        if dt <= 1e-6:
            return 0.0
        return math.degrees(_rotation_distance(rotation, self.previous_rotation)) / dt

    def _calibrate(
        self,
        rotation: np.ndarray,
        acceleration_g: np.ndarray,
        timestamp: float,
        angular_speed: float,
    ) -> None:
        still = abs(float(np.linalg.norm(acceleration_g)) - 1.0) < 0.08 and angular_speed < 5.0
        if not still:
            self.calibration_started = None
            self.calibration_biases.clear()
            return
        if self.calibration_started is None:
            self.calibration_started = timestamp
        expected_body_gravity = rotation.T @ WORLD_GRAVITY_G
        self.calibration_biases.append(acceleration_g - expected_body_gravity)
        elapsed = timestamp - self.calibration_started
        if (
            elapsed >= self.calibration_time_s
            and len(self.calibration_biases) >= self.calibration_samples
        ):
            self.body_accel_bias = np.mean(self.calibration_biases, axis=0)
            self.reference_rotation = rotation.copy()
            self.position.fill(0.0)
            self.velocity.fill(0.0)
            self.linear_acceleration.fill(0.0)
            self.stationary_since = timestamp
            self.state = "LIVE"

    def update(self, sample: FusionSample, timestamp: float) -> InertialPose:
        valid = self._valid_sample(sample)
        if valid is None:
            if self.state == "WAITING":
                return self.snapshot()
            self.mark_stale(timestamp)
            return self.snapshot()
        rotation, acceleration_g = valid
        angular_speed = self._angular_speed_deg_s(rotation, timestamp)
        self.latest_rotation = rotation
        self.latest_acceleration_g = acceleration_g.copy()
        self.last_valid_time = timestamp

        if self.reference_rotation is None:
            self.state = "CALIBRATING"
            self._calibrate(rotation, acceleration_g, timestamp, angular_speed)
        else:
            if self.state == "STALE":
                self.previous_time = timestamp
            self.state = "LIVE"
            dt = 0.0 if self.previous_time is None else timestamp - self.previous_time
            dt = min(max(dt, 0.0), self.max_dt_s)
            corrected_body = acceleration_g - self.body_accel_bias
            world_specific_force = rotation @ corrected_body
            raw_linear = (
                self.reference_rotation.T @ (world_specific_force - WORLD_GRAVITY_G)
            ) * GRAVITY_M_S2
            if dt > 0.0:
                alpha = 1.0 - math.exp(-2.0 * math.pi * 8.0 * dt)
                self.linear_acceleration += alpha * (raw_linear - self.linear_acceleration)
            else:
                self.linear_acceleration = raw_linear
            if np.linalg.norm(self.linear_acceleration) < self.acceleration_deadband_m_s2:
                self.linear_acceleration.fill(0.0)

            stationary = (
                abs(float(np.linalg.norm(corrected_body)) - 1.0) < 0.05
                and angular_speed < 5.0
                and float(np.linalg.norm(raw_linear)) < 0.35
            )
            if stationary:
                if self.stationary_since is None:
                    self.stationary_since = timestamp
                if timestamp - self.stationary_since >= self.stationary_time_s:
                    self.velocity.fill(0.0)
                    self.linear_acceleration.fill(0.0)
            else:
                self.stationary_since = None
            if dt > 0.0:
                self.velocity += self.linear_acceleration * dt
                self.position += self.velocity * dt

        self.previous_rotation = rotation.copy()
        self.previous_time = timestamp
        return self.snapshot()

    def mark_stale(self, timestamp: float) -> None:
        if self.last_valid_time is None or timestamp - self.last_valid_time <= self.stale_timeout_s:
            return
        if self.state != "WAITING":
            self.state = "STALE"
            self.velocity.fill(0.0)
            self.linear_acceleration.fill(0.0)

    def recenter(self) -> None:
        if self.latest_rotation is not None:
            self.reference_rotation = self.latest_rotation.copy()
        self.position.fill(0.0)
        self.velocity.fill(0.0)
        self.linear_acceleration.fill(0.0)
        self.stationary_since = self.previous_time
        if self.reference_rotation is not None:
            self.state = "LIVE"

    def recalibrate(self) -> None:
        self.reference_rotation = None
        self.body_accel_bias.fill(0.0)
        self.position.fill(0.0)
        self.velocity.fill(0.0)
        self.linear_acceleration.fill(0.0)
        self.calibration_started = None
        self.calibration_biases.clear()
        self.stationary_since = None
        self.state = "CALIBRATING" if self.latest_rotation is not None else "WAITING"

    def snapshot(self) -> InertialPose:
        rotation = np.eye(3)
        if self.latest_rotation is not None:
            rotation = self.latest_rotation
            if self.reference_rotation is not None:
                rotation = self.reference_rotation.T @ rotation
        progress = 0.0
        if (
            self.state == "CALIBRATING"
            and self.calibration_started is not None
            and self.previous_time is not None
        ):
            progress = min(
                1.0,
                (self.previous_time - self.calibration_started) / self.calibration_time_s,
            )
        return InertialPose(
            self.state,
            rotation.copy(),
            self.position.copy(),
            self.velocity.copy(),
            self.linear_acceleration.copy(),
            rotation_matrix_to_euler_zyx(rotation),
            progress,
        )
