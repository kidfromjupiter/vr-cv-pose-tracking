from __future__ import annotations

import math

import numpy as np
import pytest

from conftest import make_detection
from vr_led_tracker.fusion import BallTracker
from vr_led_tracker.serial_pose import FusionSample


def imu_sample(
    timestamp_us: int,
    quaternion=(1.0, 0.0, 0.0, 0.0),
    acceleration=(0.0, 0.0, 1.0),
):
    return FusionSample(
        "right",
        timestamp_us // 10000,
        timestamp_us,
        np.asarray(quaternion, dtype=np.float64),
        np.asarray(acceleration, dtype=np.float64),
        0,
    )


def calibrate_imu(tracker: BallTracker) -> None:
    for index in range(70):
        now = index * 0.02
        tracker.add_imu(imu_sample(index * 20000), now)


def test_camera_xyz_and_imu_orientation_become_full(model, calibration):
    tracker = BallTracker(model, calibration, "right")
    calibrate_imu(tracker)
    detection = make_detection(model, calibration, np.array([20.0, -10.0, 750.0]))

    result = tracker.process_camera([detection], 1.4)

    assert result.state == "FULL"
    assert result.position_m is not None
    assert result.rotation_matrix is not None
    np.testing.assert_allclose(result.position_m, [0.02, -0.01, 0.75], atol=0.001)
    np.testing.assert_allclose(result.rotation_matrix, np.eye(3), atol=1e-8)


def test_orientation_is_relative_to_startup(model, calibration):
    tracker = BallTracker(model, calibration, "right")
    calibrate_imu(tracker)
    angle = math.radians(30.0)
    tracker.add_imu(
        imu_sample(1_400_000, (math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2))),
        1.4,
    )
    result = tracker.process_camera([], 1.4)
    assert result.rotation_matrix is not None
    yaw = math.degrees(math.atan2(result.rotation_matrix[1, 0], result.rotation_matrix[0, 0]))
    assert yaw == pytest.approx(30.0)


def test_ball_loss_freezes_then_invalidates_xyz(model, calibration):
    tracker = BallTracker(model, calibration, "right")
    calibrate_imu(tracker)
    detection = make_detection(model, calibration, np.array([0.0, 0.0, 800.0]))
    first = tracker.process_camera([detection], 1.4)
    tracker.add_imu(imu_sample(1_540_000), 1.54)
    frozen = tracker.process_camera([], 1.55)
    tracker.add_imu(imu_sample(1_650_000), 1.65)
    expired = tracker.process_camera([], 1.66)

    assert first.position_m is not None
    assert frozen.state == "IMU_ONLY"
    assert frozen.position_stale
    np.testing.assert_allclose(frozen.position_m, first.position_m)
    assert expired.position_m is None


def test_camera_only_when_imu_is_missing(model, calibration):
    tracker = BallTracker(model, calibration, "right")
    detection = make_detection(model, calibration, np.array([0.0, 0.0, 800.0]))
    result = tracker.process_camera([detection], 1.0)
    assert result.state == "CALIBRATING_IMU"
    assert result.position_m is not None
    assert result.rotation_matrix is None


def test_camera_only_after_calibrated_imu_becomes_stale(model, calibration):
    tracker = BallTracker(model, calibration, "right")
    calibrate_imu(tracker)
    detection = make_detection(model, calibration, np.array([0.0, 0.0, 800.0]))
    result = tracker.process_camera([detection], 2.0)
    assert result.state == "CAMERA_ONLY"
    assert result.position_m is not None
    assert result.rotation_matrix is None


def test_reset_returns_to_calibration(model, calibration):
    tracker = BallTracker(model, calibration, "right")
    calibrate_imu(tracker)
    tracker.reset()
    assert tracker.state == "CALIBRATING_IMU"
    assert tracker.last_position_m is None
