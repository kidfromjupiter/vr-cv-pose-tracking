from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np
import pytest

from conftest import make_detections
from vr_led_tracker.pose import PoseEstimator, projected_sphere_radius_px


def rotation_error_degrees(first, second):
    a, _ = cv2.Rodrigues(first)
    b, _ = cv2.Rodrigues(second)
    relative = a @ b.T
    cosine = np.clip((np.trace(relative) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(cosine))


def test_recovers_full_pose(model, calibration):
    expected_rvec = np.array([[0.18], [-0.12], [0.08]])
    expected_tvec = np.array([[35.0], [-20.0], [760.0]])
    detections = make_detections(model, calibration, expected_rvec, expected_tvec)
    estimate = PoseEstimator(model, calibration).estimate_camera_pose(detections)
    assert estimate is not None
    assert estimate.state == "FULL"
    np.testing.assert_allclose(estimate.tvec, expected_tvec, atol=0.5)
    assert rotation_error_degrees(estimate.rvec, expected_rvec) < 0.1
    assert estimate.reprojection_error_px < 0.01


def test_two_spheres_use_imu_orientation(model, calibration):
    rvec = np.array([[0.12], [0.05], [-0.06]])
    tvec = np.array([[20.0], [12.0], [700.0]])
    detections = make_detections(model, calibration, rvec, tvec)
    estimator = PoseEstimator(model, calibration)
    detections.pop(model.labels[-1])
    rotation, _ = cv2.Rodrigues(rvec)
    degraded = estimator.estimate_translation(detections, rotation)
    assert degraded is not None
    assert degraded.state == "DEGRADED_2"
    np.testing.assert_allclose(degraded.tvec, tvec, atol=1.0)


def test_pose_survives_mask_radius_changes(model, calibration):
    rvec = np.array([[0.12], [0.05], [-0.06]])
    tvec = np.array([[20.0], [12.0], [700.0]])
    detections = make_detections(model, calibration, rvec, tvec)
    detections = {
        label: replace(detection, radius=detection.radius * 0.35, area=detection.area * 0.1225)
        for label, detection in detections.items()
    }
    estimator = PoseEstimator(model, calibration)
    full = estimator.estimate_camera_pose(detections)
    assert full is not None

    detections.pop(model.labels[-1])
    rotation, _ = cv2.Rodrigues(rvec)
    degraded = estimator.estimate_translation(detections, rotation)
    assert degraded is not None
    np.testing.assert_allclose(degraded.tvec, tvec, atol=1.0)


def test_one_sphere_has_no_camera_solution(model, calibration):
    detections = make_detections(
        model, calibration, np.zeros((3, 1)), np.array([[0.0], [0.0], [800.0]])
    )
    detections.pop(model.labels[-1])
    detections.pop(model.labels[-2])
    assert PoseEstimator(model, calibration).estimate_translation(detections, np.eye(3)) is None


def test_translation_rejects_nonfinite_prediction_without_crashing(model, calibration):
    rvec = np.array([[0.12], [0.05], [-0.06]])
    tvec = np.array([[20.0], [12.0], [700.0]])
    detections = make_detections(model, calibration, rvec, tvec)
    rotation, _ = cv2.Rodrigues(rvec)
    assert PoseEstimator(model, calibration).estimate_translation(
        detections, rotation, np.array([np.nan, 0.0, 700.0])
    ) is None


def test_translation_rejects_nonfinite_detection_without_crashing(model, calibration):
    rvec = np.zeros((3, 1))
    tvec = np.array([[0.0], [0.0], [700.0]])
    detections = make_detections(model, calibration, rvec, tvec)
    detections[model.labels[0]].center[0] = np.nan
    assert PoseEstimator(model, calibration).estimate_translation(
        detections, np.eye(3)
    ) is None


def test_projected_sphere_radius_tracks_depth(calibration):
    near = projected_sphere_radius_px(np.array([0.0, 0.0, 500.0]), 20.0, calibration)
    far = projected_sphere_radius_px(np.array([0.0, 0.0, 1000.0]), 20.0, calibration)
    assert near == pytest.approx(far * 2.0, rel=0.01)
