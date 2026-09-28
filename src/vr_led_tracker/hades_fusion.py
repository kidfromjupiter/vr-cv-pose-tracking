"""HadesVR-derived adaptive position fusion with an IMU orientation path.

The filter equations and default tuning are adapted from HadesVR's V3Kalman,
SimpleKalman, and IMU data-handler code at commit
0a6de3c19e22978dbd2a17e23fcb2e041b0cee48. See THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .errors import TrackerError
from .inertial import GRAVITY_M_S2, WORLD_GRAVITY_G


@dataclass(frozen=True)
class HadesFusionSettings:
    camera_measurement_uncertainty: float = 2.0
    camera_estimation_uncertainty: float = 1.0
    camera_process_noise: float = 7.5
    imu_measurement_uncertainty: float = 2.0
    imu_estimation_uncertainty: float = 5.0
    imu_process_noise: float = 22.55
    velocity_damping: float = 0.9
    yaw_drift_enabled: bool = False
    yaw_lower_velocity_m_s: float = 2.0
    yaw_upper_velocity_m_s: float = 5.0
    yaw_measurement_uncertainty: float = 5.0
    yaw_estimation_uncertainty: float = 1.25
    yaw_process_noise: float = 2.0
    identity_max_speed_px_s: float = 6000.0
    identity_reacquire_timeout_s: float = 0.25
    stereo_allow_mono_fallback: bool = True
    stereo_epipolar_error_limit_px: float = 3.0
    stereo_rigid_error_limit_mm: float = 10.0
    stereo_min_triangulation_angle_deg: float = 1.5
    stereo_quality_reference_area_px2: float = 256.0
    stereo_quality_reference_angle_deg: float = 8.0
    stereo_max_measurement_scale: float = 4.0
    stereo_mono_base_measurement_scale: float = 2.0
    stereo_max_mono_measurement_scale: float = 6.0

    def __post_init__(self) -> None:
        positive = (
            self.camera_measurement_uncertainty,
            self.camera_estimation_uncertainty,
            self.camera_process_noise,
            self.imu_measurement_uncertainty,
            self.imu_estimation_uncertainty,
            self.imu_process_noise,
            self.yaw_measurement_uncertainty,
            self.yaw_estimation_uncertainty,
            self.yaw_process_noise,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in positive):
            raise TrackerError("Hades fusion uncertainties and process noise must be positive")
        if not math.isfinite(self.velocity_damping) or not 0.0 <= self.velocity_damping <= 1.0:
            raise TrackerError("Hades velocity_damping must be between 0 and 1")
        if (
            not math.isfinite(self.yaw_lower_velocity_m_s)
            or not math.isfinite(self.yaw_upper_velocity_m_s)
            or self.yaw_lower_velocity_m_s < 0.0
            or self.yaw_upper_velocity_m_s <= self.yaw_lower_velocity_m_s
        ):
            raise TrackerError("Hades yaw velocity thresholds must be finite and increasing")
        if not math.isfinite(self.identity_max_speed_px_s) or self.identity_max_speed_px_s <= 0.0:
            raise TrackerError("identity max_speed_px_s must be positive")
        if (
            not math.isfinite(self.identity_reacquire_timeout_s)
            or self.identity_reacquire_timeout_s <= 0.0
        ):
            raise TrackerError("identity reacquire_timeout_s must be positive")
        stereo_positive = (
            self.stereo_epipolar_error_limit_px, self.stereo_rigid_error_limit_mm,
            self.stereo_min_triangulation_angle_deg, self.stereo_quality_reference_area_px2,
            self.stereo_quality_reference_angle_deg, self.stereo_max_measurement_scale,
            self.stereo_mono_base_measurement_scale, self.stereo_max_mono_measurement_scale,
        )
        if not all(math.isfinite(value) and value > 0 for value in stereo_positive):
            raise TrackerError("Stereo tracking limits and quality scales must be positive")
        if self.stereo_max_measurement_scale < 1.0 or self.stereo_max_mono_measurement_scale < self.stereo_mono_base_measurement_scale:
            raise TrackerError("Stereo measurement scale limits are invalid")

    @classmethod
    def load(cls, path: str | Path) -> "HadesFusionSettings":
        try:
            with Path(path).open(encoding="utf-8") as handle:
                raw = json.load(handle)
            camera = raw["camera"]
            imu = raw["imu"]
            yaw = raw["yaw_drift_correction"]
            identity = raw.get("identity_tracking", {})
            stereo = raw.get("stereo_tracking", {})
            return cls(
                camera_measurement_uncertainty=float(camera["measurement_uncertainty"]),
                camera_estimation_uncertainty=float(camera["estimation_uncertainty"]),
                camera_process_noise=float(camera["process_noise"]),
                imu_measurement_uncertainty=float(imu["measurement_uncertainty"]),
                imu_estimation_uncertainty=float(imu["estimation_uncertainty"]),
                imu_process_noise=float(imu["process_noise"]),
                velocity_damping=float(raw["velocity_damping"]),
                yaw_drift_enabled=bool(yaw["enabled"]),
                yaw_lower_velocity_m_s=float(yaw["lower_velocity_m_s"]),
                yaw_upper_velocity_m_s=float(yaw["upper_velocity_m_s"]),
                yaw_measurement_uncertainty=float(yaw["measurement_uncertainty"]),
                yaw_estimation_uncertainty=float(yaw["estimation_uncertainty"]),
                yaw_process_noise=float(yaw["process_noise"]),
                identity_max_speed_px_s=float(identity.get("max_speed_px_s", 6000.0)),
                identity_reacquire_timeout_s=float(
                    identity.get("reacquire_timeout_s", 0.25)
                ),
                stereo_allow_mono_fallback=bool(stereo.get("allow_mono_fallback", True)),
                stereo_epipolar_error_limit_px=float(stereo.get("epipolar_error_limit_px", 3.0)),
                stereo_rigid_error_limit_mm=float(stereo.get("rigid_error_limit_mm", 10.0)),
                stereo_min_triangulation_angle_deg=float(stereo.get("min_triangulation_angle_deg", 1.5)),
                stereo_quality_reference_area_px2=float(stereo.get("quality_reference_area_px2", 256.0)),
                stereo_quality_reference_angle_deg=float(stereo.get("quality_reference_angle_deg", 8.0)),
                stereo_max_measurement_scale=float(stereo.get("max_stereo_measurement_scale", 4.0)),
                stereo_mono_base_measurement_scale=float(stereo.get("mono_base_measurement_scale", 2.0)),
                stereo_max_mono_measurement_scale=float(stereo.get("max_mono_measurement_scale", 6.0)),
            )
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise TrackerError(f"Cannot load Hades fusion settings {path}: {exc}") from exc


class HadesVectorFilter:
    """Python adaptation of HadesVR's independent XYZ V3Kalman filters."""

    def __init__(self, settings: HadesFusionSettings) -> None:
        self.settings = settings
        self.reading = np.zeros(3)
        self.current_estimate = np.zeros(3)
        self.last_estimate = np.zeros(3)
        self.camera_estimation_error = np.full(
            3, settings.camera_estimation_uncertainty, dtype=np.float64
        )
        self.imu_estimation_error = np.full(
            3, settings.imu_estimation_uncertainty, dtype=np.float64
        )
        self.camera_gain = np.zeros(3)
        self.imu_gain = np.zeros(3)

    def initialize(self, position_m: np.ndarray) -> None:
        position = np.asarray(position_m, dtype=np.float64).reshape(3)
        self.reading = position.copy()
        self.current_estimate = position.copy()
        self.last_estimate = position.copy()

    def _update(self, measurement_error: float, process_noise: float, source: str) -> np.ndarray:
        estimation_error = (
            self.camera_estimation_error if source == "camera" else self.imu_estimation_error
        )
        gain = estimation_error / (estimation_error + measurement_error)
        current = self.last_estimate + gain * (self.reading - self.last_estimate)
        estimation_error[:] = (
            (1.0 - gain) * estimation_error
            + np.abs(self.last_estimate - current) * process_noise
        )
        self.current_estimate = current
        self.last_estimate = current.copy()
        if source == "camera":
            self.camera_gain = gain
        else:
            self.imu_gain = gain
        return current.copy()

    def update_camera(self, position_m: np.ndarray, measurement_scale: float = 1.0) -> np.ndarray:
        self.reading = np.asarray(position_m, dtype=np.float64).reshape(3).copy()
        return self._update(
            self.settings.camera_measurement_uncertainty * max(float(measurement_scale), 1.0),
            self.settings.camera_process_noise,
            "camera",
        )

    def update_imu(self, displacement_m: np.ndarray) -> np.ndarray:
        self.reading += np.asarray(displacement_m, dtype=np.float64).reshape(3)
        return self._update(
            self.settings.imu_measurement_uncertainty,
            self.settings.imu_process_noise,
            "imu",
        )


class HadesScalarFilter:
    def __init__(self, measurement_error: float, estimation_error: float, process_noise: float) -> None:
        self.measurement_error = float(measurement_error)
        self.estimation_error = float(estimation_error)
        self.process_noise = float(process_noise)
        self.current_estimate = 0.0
        self.last_estimate = 0.0
        self.gain = 0.0

    def update(self, measurement: float) -> float:
        self.gain = self.estimation_error / (
            self.estimation_error + self.measurement_error
        )
        current = self.last_estimate + self.gain * (measurement - self.last_estimate)
        self.estimation_error = (
            (1.0 - self.gain) * self.estimation_error
            + abs(self.last_estimate - current) * self.process_noise
        )
        self.current_estimate = current
        self.last_estimate = current
        return current


@dataclass(frozen=True)
class HadesPose:
    position_m: np.ndarray
    velocity_m_s: np.ndarray
    rotation_matrix: np.ndarray
    acceleration_bias_g: np.ndarray
    camera_gain: np.ndarray
    imu_gain: np.ndarray
    yaw_correction_rad: float


def _axis_angle_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    vector = np.asarray(axis, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12 or not math.isfinite(angle):
        return np.eye(3)
    matrix, _ = cv2.Rodrigues(vector / norm * angle)
    return matrix


class HadesMotionFilter:
    def __init__(self, settings: HadesFusionSettings) -> None:
        self.settings = settings
        self.position_filter = HadesVectorFilter(settings)
        self.yaw_filter = HadesScalarFilter(
            settings.yaw_measurement_uncertainty,
            settings.yaw_estimation_uncertainty,
            settings.yaw_process_noise,
        )
        self.initialized = False
        self.velocity = np.zeros(3)
        self.acceleration_bias = np.zeros(3)
        self.fixed_alignment = np.eye(3)
        self.rotation = np.eye(3)
        self.gravity_camera = WORLD_GRAVITY_G * GRAVITY_M_S2
        self.latest_imu_rotation: np.ndarray | None = None
        self.imu_velocity_window = np.zeros(3)
        self.camera_tracking = False
        self.last_camera_time: float | None = None
        self.last_camera_position: np.ndarray | None = None
        self.last_camera_estimate: np.ndarray | None = None
        self.yaw_correction_rad = 0.0

    def initialize(
        self,
        position_m: np.ndarray,
        camera_rotation: np.ndarray,
        imu_rotation: np.ndarray,
        acceleration_bias_g: np.ndarray,
    ) -> None:
        self.position_filter.initialize(position_m)
        self.velocity.fill(0.0)
        self.acceleration_bias = np.asarray(acceleration_bias_g, dtype=np.float64).reshape(3).copy()
        self.fixed_alignment = (
            np.asarray(camera_rotation, dtype=np.float64)
            @ np.asarray(imu_rotation, dtype=np.float64).T
        )
        self.latest_imu_rotation = np.asarray(imu_rotation, dtype=np.float64).copy()
        self.gravity_camera = self.fixed_alignment @ WORLD_GRAVITY_G * GRAVITY_M_S2
        self.rotation = np.asarray(camera_rotation, dtype=np.float64).copy()
        self.initialized = True

    def _update_rotation(self, imu_rotation: np.ndarray) -> None:
        self.latest_imu_rotation = np.asarray(imu_rotation, dtype=np.float64).copy()
        yaw = _axis_angle_rotation(self.gravity_camera, self.yaw_correction_rad)
        self.rotation = yaw @ self.fixed_alignment @ self.latest_imu_rotation

    def predict(
        self, imu_rotation: np.ndarray, acceleration_g: np.ndarray, dt: float
    ) -> HadesPose:
        if not self.initialized:
            raise RuntimeError("Hades motion filter is not initialized")
        dt = min(max(float(dt), 1e-4), 0.05)
        self._update_rotation(imu_rotation)
        specific_force = (
            np.asarray(acceleration_g, dtype=np.float64).reshape(3) - self.acceleration_bias
        )
        acceleration = self.rotation @ specific_force * GRAVITY_M_S2 - self.gravity_camera
        self.velocity += acceleration * dt
        self.imu_velocity_window += acceleration * dt
        self.velocity *= self.settings.velocity_damping
        self.position_filter.update_imu(self.velocity * dt)
        return self.snapshot()

    def _update_yaw_correction(self, camera_velocity: np.ndarray) -> None:
        if not self.settings.yaw_drift_enabled or self.latest_imu_rotation is None:
            self.imu_velocity_window.fill(0.0)
            return
        up = self.gravity_camera / max(float(np.linalg.norm(self.gravity_camera)), 1e-12)
        camera_horizontal = camera_velocity - up * float(np.dot(camera_velocity, up))
        imu_horizontal = self.imu_velocity_window - up * float(
            np.dot(self.imu_velocity_window, up)
        )
        camera_speed = float(np.linalg.norm(camera_horizontal))
        imu_speed = float(np.linalg.norm(imu_horizontal))
        if (
            self.settings.yaw_lower_velocity_m_s <= camera_speed
            <= self.settings.yaw_upper_velocity_m_s
            and imu_speed > 1e-4
        ):
            camera_direction = camera_horizontal / camera_speed
            imu_direction = imu_horizontal / imu_speed
            sine = float(np.dot(up, np.cross(imu_direction, camera_direction)))
            cosine = float(np.clip(np.dot(imu_direction, camera_direction), -1.0, 1.0))
            self.yaw_correction_rad = self.yaw_filter.update(math.atan2(sine, cosine))
            self._update_rotation(self.latest_imu_rotation)
        self.imu_velocity_window.fill(0.0)

    def update_camera(self, position_m: np.ndarray, timestamp: float, measurement_scale: float = 1.0) -> HadesPose:
        raw_position = np.asarray(position_m, dtype=np.float64).reshape(3)
        previous_estimate = self.position_filter.current_estimate.copy()
        filtered = self.position_filter.update_camera(raw_position, measurement_scale)
        if (
            self.camera_tracking
            and self.last_camera_time is not None
            and self.last_camera_position is not None
            and self.last_camera_estimate is not None
        ):
            dt = timestamp - self.last_camera_time
            if 1e-4 < dt < 1.0:
                self.velocity = (filtered - previous_estimate) / dt
                camera_velocity = (raw_position - self.last_camera_position) / dt
                self._update_yaw_correction(camera_velocity)
            else:
                self.velocity.fill(0.0)
                self.imu_velocity_window.fill(0.0)
        else:
            self.velocity.fill(0.0)
            self.imu_velocity_window.fill(0.0)
        self.camera_tracking = True
        self.last_camera_time = float(timestamp)
        self.last_camera_position = raw_position.copy()
        self.last_camera_estimate = filtered.copy()
        return self.snapshot()

    def mark_camera_missing(self) -> None:
        if self.camera_tracking:
            self.velocity.fill(0.0)
            self.imu_velocity_window.fill(0.0)
        self.camera_tracking = False
        self.last_camera_time = None
        self.last_camera_position = None
        self.last_camera_estimate = None

    def snapshot(self) -> HadesPose:
        return HadesPose(
            self.position_filter.current_estimate.copy(),
            self.velocity.copy(),
            self.rotation.copy(),
            self.acceleration_bias.copy(),
            self.position_filter.camera_gain.copy(),
            self.position_filter.imu_gain.copy(),
            self.yaw_correction_rad,
        )
