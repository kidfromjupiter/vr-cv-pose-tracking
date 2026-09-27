from __future__ import annotations

import math

import numpy as np

from vr_led_tracker.inertial import InertialPoseTracker, wxyz_to_rotation_matrix
from vr_led_tracker.serial_pose import FusionSample


def sample(quaternion=(1.0, 0.0, 0.0, 0.0), acceleration=(0.0, 0.0, 1.0)):
    return FusionSample(
        "right", 0, 0,
        np.asarray(quaternion, dtype=np.float64),
        np.asarray(acceleration, dtype=np.float64),
        0,
    )


def calibrate(tracker: InertialPoseTracker, value=None, start=0.0):
    value = sample() if value is None else value
    pose = None
    for index in range(61):
        pose = tracker.update(value, start + index * 0.02)
    assert pose is not None and pose.state == "LIVE"
    return pose


def test_wxyz_quaternion_rotation_and_invalid_input():
    half = math.sqrt(0.5)
    rotation = wxyz_to_rotation_matrix(np.array([half, half, 0.0, 0.0]))
    assert rotation is not None
    np.testing.assert_allclose(
        rotation @ np.array([0.0, 1.0, 0.0]), [0.0, 0.0, 1.0], atol=1e-7
    )
    assert wxyz_to_rotation_matrix(np.zeros(4)) is None


def test_calibration_removes_identity_gravity_and_bias():
    tracker = InertialPoseTracker()
    biased = sample(acceleration=(0.02, -0.01, 1.03))
    calibrate(tracker, biased)
    pose = tracker.update(biased, 1.22)
    np.testing.assert_allclose(tracker.body_accel_bias, [0.02, -0.01, 0.03], atol=1e-9)
    np.testing.assert_allclose(pose.acceleration_m_s2, np.zeros(3), atol=1e-9)


def test_calibration_removes_gravity_for_rotated_controller():
    half = math.sqrt(0.5)
    rotated = sample((half, half, 0.0, 0.0), (0.0, 1.0, 0.0))
    tracker = InertialPoseTracker()
    calibrate(tracker, rotated)
    pose = tracker.update(rotated, 1.22)
    np.testing.assert_allclose(pose.acceleration_m_s2, np.zeros(3), atol=1e-8)
    np.testing.assert_allclose(pose.rotation_matrix, np.eye(3), atol=1e-8)


def test_integrates_linear_acceleration_into_displacement():
    tracker = InertialPoseTracker()
    calibrate(tracker)
    moving = sample(acceleration=(0.2, 0.0, 1.0))
    pose = None
    for index in range(1, 51):
        pose = tracker.update(moving, 1.2 + index * 0.02)
    assert pose is not None
    assert pose.velocity_m_s[0] > 1.5
    assert pose.position_m[0] > 0.7
    assert abs(pose.position_m[1]) < 1e-8


def test_zero_velocity_update_and_recenter():
    tracker = InertialPoseTracker()
    calibrate(tracker)
    tracker.velocity[:] = [1.0, 0.0, 0.0]
    pose = None
    for index in range(1, 17):
        pose = tracker.update(sample(), 1.2 + index * 0.02)
    assert pose is not None
    np.testing.assert_allclose(pose.velocity_m_s, np.zeros(3))

    tracker.recenter()
    pose = tracker.snapshot()
    np.testing.assert_allclose(pose.position_m, np.zeros(3))
    np.testing.assert_allclose(pose.velocity_m_s, np.zeros(3))


def test_stale_stream_freezes_velocity_and_resumes_without_large_step():
    tracker = InertialPoseTracker()
    calibrate(tracker)
    tracker.velocity[:] = [1.0, 2.0, 3.0]
    tracker.mark_stale(1.6)
    assert tracker.snapshot().state == "STALE"
    np.testing.assert_allclose(tracker.snapshot().velocity_m_s, np.zeros(3))
    before = tracker.snapshot().position_m
    after = tracker.update(sample(acceleration=(0.2, 0.0, 1.0)), 2.0)
    np.testing.assert_allclose(after.position_m, before)
    assert after.state == "LIVE"
