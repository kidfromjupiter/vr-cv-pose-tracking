from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import BallModel, CameraCalibration
from .detection import BallDetection
from .filtering import OneEuroVector
from .inertial import InertialPoseTracker
from .pose import PositionEstimate, PositionEstimator
from .serial_pose import FusionSample, TimestampMapper


@dataclass(frozen=True)
class TrackingResult:
    state: str
    position_m: np.ndarray | None
    rotation_matrix: np.ndarray | None
    velocity_m_s: np.ndarray
    position_age_s: float
    position_stale: bool
    camera_estimate: PositionEstimate | None
    calibration_progress: float


class BallTracker:
    camera_hold_s = 0.25
    imu_fresh_s = 0.15

    def __init__(
        self,
        model: BallModel,
        calibration: CameraCalibration,
        slot: str,
    ) -> None:
        self.model = model
        self.slot = slot
        self.position_estimator = PositionEstimator(model, calibration)
        self.position_filter = OneEuroVector()
        self.orientation_tracker = InertialPoseTracker()
        self.timestamp_mapper = TimestampMapper()
        self.latest_imu_arrival: float | None = None
        self.last_camera_update: float | None = None
        self.last_position_m: np.ndarray | None = None
        self.last_camera_estimate: PositionEstimate | None = None
        self.state = "CALIBRATING_IMU"

    def reset(self) -> None:
        fresh = BallTracker(self.model, self.position_estimator.calibration, self.slot)
        self.__dict__.update(fresh.__dict__)

    def add_imu(self, sample: FusionSample, arrival_time: float) -> None:
        if sample.slot != self.slot:
            return
        timestamp = self.timestamp_mapper.map(sample.timestamp_us, arrival_time)
        self.orientation_tracker.update(sample, timestamp)
        self.latest_imu_arrival = arrival_time

    def process_camera(
        self,
        detections: list[BallDetection],
        read_time: float,
    ) -> TrackingResult:
        estimate = self.position_estimator.estimate(detections)
        if estimate is not None:
            measured_m = estimate.position_mm / 1000.0
            self.last_position_m = self.position_filter.update(measured_m, read_time)
            self.last_camera_update = read_time
            self.last_camera_estimate = estimate

        camera_current = estimate is not None
        position_age = (
            float("inf")
            if self.last_camera_update is None
            else max(0.0, read_time - self.last_camera_update)
        )
        if position_age > self.camera_hold_s:
            position = None
            if self.last_position_m is not None:
                self.position_filter.reset()
                self.position_estimator.reset()
                self.last_position_m = None
        else:
            position = None if self.last_position_m is None else self.last_position_m.copy()

        imu_fresh = (
            self.latest_imu_arrival is not None
            and read_time - self.latest_imu_arrival < self.imu_fresh_s
        )
        orientation = self.orientation_tracker.snapshot()
        imu_ready = imu_fresh and orientation.state == "LIVE"
        imu_calibrating = self.latest_imu_arrival is None or (
            imu_fresh and orientation.state != "LIVE"
        )
        rotation = orientation.rotation_matrix.copy() if imu_ready else None

        if imu_calibrating:
            self.state = "CALIBRATING_IMU"
        elif camera_current and imu_ready:
            self.state = "FULL"
        elif camera_current:
            self.state = "CAMERA_ONLY"
        elif imu_ready:
            self.state = "IMU_ONLY"
        else:
            self.state = "LOST"

        derivative = self.position_filter.derivative
        velocity = np.zeros(3) if derivative is None else derivative.copy()
        return TrackingResult(
            self.state,
            position,
            rotation,
            velocity,
            position_age,
            not camera_current and position is not None,
            estimate,
            orientation.calibration_progress,
        )


FusionResult = TrackingResult
FusionTracker = BallTracker
