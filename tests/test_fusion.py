from __future__ import annotations

import math

import numpy as np

from conftest import make_detections
from vr_led_tracker.fusion import FusionTracker
from vr_led_tracker.hades_fusion import HadesFusionSettings
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


def initialize_tracker(tracker: FusionTracker, now: float = 0.0) -> None:
    tracker.filter.initialize(np.zeros(3), np.eye(3), np.eye(3), np.zeros(3))
    tracker.latest_arrival_time = now


def test_guided_stillness_initializes_hades_fusion(model, calibration):
    tracker = FusionTracker(model, calibration, "right")
    rvec = np.zeros((3, 1))
    tvec = np.array([[20.0], [-10.0], [750.0]])
    detections = list(reversed(list(make_detections(model, calibration, rvec, tvec).values())))
    result = None
    for index in range(70):
        now = index * 0.02
        tracker.add_imu(imu_sample(index * 20000), now)
        result = tracker.process_camera(detections, now)
    assert result is not None
    assert result.state == "FULL"
    assert result.pose is not None
    np.testing.assert_allclose(result.pose.position_m, tvec.reshape(3) / 1000.0, atol=0.01)


def test_two_white_candidates_fall_back_to_imu_only(model, calibration):
    tracker = FusionTracker(model, calibration, "right")
    initialize_tracker(tracker, 2.0)
    tracker.last_camera_update = 1.9
    detections = list(
        make_detections(
            model,
            calibration,
            np.zeros((3, 1)),
            np.array([[0.0], [0.0], [750.0]]),
        ).values()
    )[:2]

    result = tracker.process_camera(detections, 2.0)

    assert result.state == "IMU_ONLY"
    assert tracker.pose_estimator.last_assignment == {}


def test_guided_stillness_tolerates_imu_yaw_drift(model, calibration):
    tracker = FusionTracker(model, calibration, "right")
    rvec = np.zeros((3, 1))
    tvec = np.array([[20.0], [-10.0], [750.0]])
    detections = make_detections(model, calibration, rvec, tvec)

    result = None
    for index in range(70):
        angle = math.radians(10.0) * index * 0.02
        quaternion = (math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0))
        tracker.add_imu(imu_sample(index * 20000, quaternion), index * 0.02)
        result = tracker.process_camera(detections, index * 0.02)

    assert result is not None
    assert result.state == "FULL"
    assert result.pose is not None


def test_settings_are_preserved_across_reset(model, calibration):
    settings = HadesFusionSettings(velocity_damping=0.75)
    tracker = FusionTracker(model, calibration, "right", settings)
    tracker.reset()
    assert tracker.settings is settings
    assert tracker.filter.settings.velocity_damping == 0.75


def test_device_timestamp_drives_imu_prediction(model, calibration):
    tracker = FusionTracker(model, calibration, "right")
    initialize_tracker(tracker, 10.0)
    tracker.add_imu(imu_sample(1_000_000, acceleration=(1.0, 0.0, 1.0)), 10.0)
    tracker.add_imu(imu_sample(1_010_000, acceleration=(1.0, 0.0, 1.0)), 10.0)
    assert tracker.filter.position_filter.current_estimate[0] > 0.0


def test_live_imu_continues_after_long_camera_dropout(model, calibration):
    tracker = FusionTracker(model, calibration, "right")
    initialize_tracker(tracker, 10.0)
    tracker.last_camera_update = 1.0

    result = tracker.process_camera([], 10.0)

    assert result.state == "IMU_ONLY"
    assert result.pose is not None


def test_camera_measurement_uses_hades_adaptive_correction(model, calibration):
    tracker = FusionTracker(model, calibration, "right")
    initialize_tracker(tracker, 5.0)
    detections = make_detections(
        model,
        calibration,
        np.zeros((3, 1)),
        np.array([[80.0], [-40.0], [750.0]]),
    )

    result = tracker.process_camera(detections, 5.0)

    assert result.state == "FULL"
    assert result.pose is not None
    np.testing.assert_allclose(result.pose.camera_gain, np.full(3, 1.0 / 3.0))
    np.testing.assert_allclose(
        result.pose.position_m,
        np.array([0.08, -0.04, 0.75]) / 3.0,
        atol=0.002,
    )


def test_stale_imu_with_camera_is_camera_only(model, calibration):
    tracker = FusionTracker(model, calibration, "right")
    initialize_tracker(tracker, 1.0)
    detections = make_detections(
        model,
        calibration,
        np.zeros((3, 1)),
        np.array([[0.0], [0.0], [750.0]]),
    )
    result = tracker.process_camera(detections, 2.0)
    assert result.state == "CAMERA_ONLY"
