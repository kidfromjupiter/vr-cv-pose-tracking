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

    np.testing.assert_allclose(full.tvec, tvec, atol=1.0)


def test_two_spheres_have_no_camera_solution(model, calibration):
    detections = make_detections(
        model, calibration, np.zeros((3, 1)), np.array([[0.0], [0.0], [800.0]])
    )
    detections.pop(model.labels[-1])
    assert PoseEstimator(model, calibration).estimate_camera_pose(list(detections.values())) is None


def test_pose_rejects_nonfinite_detection_without_crashing(model, calibration):
    rvec = np.zeros((3, 1))
    tvec = np.array([[0.0], [0.0], [700.0]])
    detections = make_detections(model, calibration, rvec, tvec)
    detections[model.labels[0]].center[0] = np.nan
    assert PoseEstimator(model, calibration).estimate_camera_pose(list(detections.values())) is None


def test_unordered_white_spheres_recover_identity(model, calibration):
    rvec = np.array([[0.18], [-0.12], [0.08]])
    tvec = np.array([[35.0], [-20.0], [760.0]])
    detections = make_detections(model, calibration, rvec, tvec)
    unordered = list(reversed(list(detections.values())))
    estimator = PoseEstimator(model, calibration)

    estimate = estimator.estimate_camera_pose(unordered)

    assert estimate is not None
    assert set(estimator.last_assignment) == set(model.labels)
    np.testing.assert_allclose(estimate.tvec, tvec, atol=1.0)


def test_white_distractor_is_not_assigned_with_pose_prediction(model, calibration):
    rvec = np.array([[0.12], [0.04], [-0.03]])
    tvec = np.array([[25.0], [10.0], [720.0]])
    detections = make_detections(model, calibration, rvec, tvec)
    distractor = replace(
        next(iter(detections.values())),
        label="unassigned",
        center=np.array([100.0, 100.0]),
        score=1.2,
    )
    rotation, _ = cv2.Rodrigues(rvec)
    estimator = PoseEstimator(model, calibration)

    estimate = estimator.estimate_camera_pose(
        [distractor, *reversed(list(detections.values()))],
        rotation,
        tvec.reshape(3),
    )

    assert estimate is not None
    np.testing.assert_allclose(estimate.tvec, tvec, atol=1.0)


def test_identity_assignment_stays_stable_between_frames(model, calibration):
    first_rvec = np.array([[0.08], [-0.04], [0.03]])
    first_tvec = np.array([[10.0], [-5.0], [740.0]])
    second_rvec = np.array([[0.09], [-0.035], [0.035]])
    second_tvec = np.array([[14.0], [-3.0], [738.0]])
    first = make_detections(model, calibration, first_rvec, first_tvec)
    second = make_detections(model, calibration, second_rvec, second_tvec)
    second_rotation, _ = cv2.Rodrigues(second_rvec)
    estimator = PoseEstimator(model, calibration)

    assert estimator.estimate_camera_pose(list(reversed(list(first.values())))) is not None
    estimate = estimator.estimate_camera_pose(
        [second[model.labels[1]], second[model.labels[2]], second[model.labels[0]]],
        second_rotation,
        second_tvec.reshape(3),
    )

    assert estimate is not None
    for label in model.labels:
        np.testing.assert_allclose(
            estimator.last_assignment[label].center,
            second[label].center,
            atol=0.01,
        )


def test_projected_sphere_radius_tracks_depth(calibration):
    near = projected_sphere_radius_px(np.array([0.0, 0.0, 500.0]), 20.0, calibration)
    far = projected_sphere_radius_px(np.array([0.0, 0.0, 1000.0]), 20.0, calibration)
    assert near == pytest.approx(far * 2.0, rel=0.01)
