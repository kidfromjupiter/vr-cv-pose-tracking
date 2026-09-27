from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from conftest import make_detections
from vr_led_tracker.fusion import FusionTracker
from vr_led_tracker.inertial import ErrorStateKalmanFilter
from vr_led_tracker.serial_pose import FusionSample


def imu_sample(timestamp_us: int, quaternion=(1.0, 0.0, 0.0, 0.0), acceleration=(0.0, 0.0, 1.0)):
    return FusionSample(
        "right",
        timestamp_us // 10000,
        timestamp_us,
        np.asarray(quaternion, dtype=np.float64),
        np.asarray(acceleration, dtype=np.float64),
        0,
    )


def test_error_state_filter_cancels_gravity_and_accepts_camera_correction():
    filter_ = ErrorStateKalmanFilter()
    filter_.initialize(np.zeros(3), np.eye(3), np.eye(3), np.zeros(3))
    for _ in range(100):
        filter_.predict(np.eye(3), np.array([0.0, 0.0, 1.0]), 0.01)
    np.testing.assert_allclose(filter_.position, np.zeros(3), atol=1e-9)
    np.testing.assert_allclose(filter_.velocity, np.zeros(3), atol=1e-9)

    corrected = filter_.update_camera(np.array([0.1, -0.05, 0.8]), np.eye(3), 0.005)
    assert corrected.position_m[0] > 0.09
    assert corrected.position_m[2] > 0.7


def test_guided_stillness_initializes_fusion(model, calibration):
    tracker = FusionTracker(model, calibration, "right")
    rvec = np.array([[0.0], [0.0], [0.0]])
    tvec = np.array([[20.0], [-10.0], [750.0]])
    detections = make_detections(model, calibration, rvec, tvec)
    result = None
    for index in range(70):
        now = index * 0.02
        tracker.add_imu(imu_sample(index * 20000), now)
        result = tracker.process_camera(detections, now)
    assert result is not None
    assert result.state == "FULL"
    assert result.pose is not None
    assert result.camera_latency_s == 0.0
    np.testing.assert_allclose(result.pose.position_m, tvec.reshape(3) / 1000.0, atol=0.01)


def test_guided_stillness_tolerates_imu_yaw_drift(model, calibration):
    tracker = FusionTracker(model, calibration, "right")
    rvec = np.zeros((3, 1))
    tvec = np.array([[20.0], [-10.0], [750.0]])
    detections = make_detections(model, calibration, rvec, tvec)

    result = None
    for index in range(70):
        # Ten degrees/second exceeds the old angular-speed gate, but a fixed
        # camera pose confirms that the physical controller is stationary.
        angle = math.radians(10.0) * index * 0.02
        quaternion = (math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0))
        tracker.add_imu(imu_sample(index * 20000, quaternion), index * 0.02)
        result = tracker.process_camera(detections, index * 0.02)

    assert result is not None
    assert result.state == "FULL"
    assert result.pose is not None


def test_fixed_camera_latency_is_preserved_across_reset(model, calibration):
    tracker = FusionTracker(model, calibration, "right", camera_latency_s=0.025)
    tracker.reset()
    assert tracker.camera_latency_s == 0.025
    assert tracker.result(0).camera_latency_s == 0.025


def test_fixed_camera_latency_offsets_measurement_time(model, calibration):
    tracker = FusionTracker(model, calibration, "right", camera_latency_s=0.025)
    tracker.filter.initialize(np.zeros(3), np.eye(3), np.eye(3), np.zeros(3))
    tracker.latest_arrival_time = 10.0
    detections = make_detections(
        model,
        calibration,
        np.zeros((3, 1)),
        np.array([[0.0], [0.0], [750.0]]),
    )
    queried_times = []

    def predicted_pose_at(timestamp):
        queried_times.append(timestamp)
        return np.eye(3), np.array([0.0, 0.0, 0.75])

    tracker._predicted_pose_at = predicted_pose_at
    result = tracker.process_camera(detections, 10.0)

    assert result.state == "FULL"
    assert queried_times == [pytest.approx(9.975)]
