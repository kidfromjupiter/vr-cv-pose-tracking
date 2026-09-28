from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np
import pytest

from conftest import make_detections, make_stereo_detections
from vr_led_tracker.pose import PoseEstimator, StereoPoseEstimator, projected_sphere_radius_px


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


def test_stereo_recovers_pose_from_unordered_detections(model, stereo_calibration):
    rvec = np.array([[0.16], [-0.08], [0.05]])
    tvec = np.array([[30.0], [-15.0], [780.0]])
    left, right = make_stereo_detections(model, stereo_calibration, rvec, tvec)
    estimator = StereoPoseEstimator(model, stereo_calibration)

    estimate = estimator.estimate_camera_pose(
        list(reversed(list(left.values()))),
        [right[model.labels[1]], right[model.labels[2]], right[model.labels[0]]],
    )

    assert estimate is not None
    assert estimate.source == "STEREO"
    np.testing.assert_allclose(estimate.tvec, tvec, atol=0.5)
    assert rotation_error_degrees(estimate.rvec, rvec) < 0.1
    assert set(estimator.last_left_assignment) == set(model.labels)
    assert set(estimator.last_right_assignment) == set(model.labels)


def test_stereo_right_only_fallback_is_returned_in_left_frame(model, stereo_calibration):
    rvec = np.array([[0.1], [0.04], [-0.03]])
    tvec = np.array([[45.0], [12.0], [720.0]])
    _, right = make_stereo_detections(model, stereo_calibration, rvec, tvec)
    estimator = StereoPoseEstimator(model, stereo_calibration)

    estimate = estimator.estimate_camera_pose(None, list(reversed(list(right.values()))))

    assert estimate is not None
    assert estimate.source == "RIGHT_MONO"
    np.testing.assert_allclose(estimate.tvec, tvec, atol=1.0)
    assert rotation_error_degrees(estimate.rvec, rvec) < 0.1


def test_stereo_rejects_bad_epipolar_matching(model, stereo_calibration):
    rvec = np.zeros((3, 1))
    tvec = np.array([[20.0], [5.0], [760.0]])
    left, right = make_stereo_detections(model, stereo_calibration, rvec, tvec)
    shifted = {
        label: replace(detection, center=detection.center + np.array([0.0, 30.0]))
        for label, detection in right.items()
    }
    estimator = StereoPoseEstimator(model, stereo_calibration)

    estimate = estimator.estimate_camera_pose(list(left.values()), list(shifted.values()))

    assert estimate is not None
    assert estimate.source != "STEREO"


def test_stereo_tolerates_pixel_noise_and_white_distractors(model, stereo_calibration):
    rvec = np.array([[0.08], [-0.03], [0.06]])
    tvec = np.array([[25.0], [-8.0], [800.0]])
    left, right = make_stereo_detections(model, stereo_calibration, rvec, tvec)
    noise = [np.array([0.35, -0.25]), np.array([-0.4, 0.2]), np.array([0.2, 0.3])]
    noisy_left = [
        replace(left[label], center=left[label].center + noise[index])
        for index, label in enumerate(model.labels)
    ]
    noisy_right = [
        replace(right[label], center=right[label].center - noise[index])
        for index, label in enumerate(model.labels)
    ]
    distractor = replace(
        noisy_left[0],
        center=np.array([90.0, 100.0]),
        radius=6.0,
        area=36.0 * np.pi,
        score=1.0,
    )

    estimate = StereoPoseEstimator(model, stereo_calibration).estimate_camera_pose(
        [distractor, *reversed(noisy_left)],
        [*reversed(noisy_right)],
    )

    assert estimate is not None
    assert estimate.source == "STEREO"
    np.testing.assert_allclose(estimate.tvec, tvec, atol=8.0)
