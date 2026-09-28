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


def test_identity_gate_rejects_implausible_motion_then_reacquires(model, calibration):
    estimator = PoseEstimator(
        model,
        calibration,
        identity_max_speed_px_s=6000.0,
        identity_reacquire_timeout_s=0.25,
    )
    first = make_detections(
        model,
        calibration,
        np.zeros((3, 1)),
        np.array([[0.0], [0.0], [740.0]]),
    )
    jumped = make_detections(
        model,
        calibration,
        np.zeros((3, 1)),
        np.array([[100.0], [0.0], [740.0]]),
    )

    assert estimator.estimate_camera_pose(list(first.values()), frame_time=1.0) is not None
    assert estimator.estimate_camera_pose(list(jumped.values()), frame_time=1.01) is None
    assert estimator.identity_gate_rejected
    assert estimator.identity_gate_speed_px_s > 10_000.0
    assert "reacquire in" in estimator.identity_gate_detail
    assert len(estimator.assignment_history) == 1

    estimate = estimator.estimate_camera_pose(list(jumped.values()), frame_time=1.26)
    assert estimate is not None
    assert not estimator.identity_gate_rejected
    assert "geometric reacquisition" in estimator.identity_gate_detail
    assert len(estimator.assignment_history) == 1


def test_constant_velocity_prediction_preserves_identity_at_crossing(model, calibration):
    estimator = PoseEstimator(model, calibration)
    base = make_detections(
        model,
        calibration,
        np.zeros((3, 1)),
        np.array([[0.0], [0.0], [740.0]]),
    )

    def assignment(x0, x1, x2):
        return {
            model.labels[0]: replace(base[model.labels[0]], center=np.array([x0, 50.0])),
            model.labels[1]: replace(base[model.labels[1]], center=np.array([x1, 50.0])),
            model.labels[2]: replace(base[model.labels[2]], center=np.array([x2, 90.0])),
        }

    estimator._record_assignment(1.00, assignment(0.0, 30.0, 60.0))
    estimator._record_assignment(1.01, assignment(10.0, 20.0, 60.0))
    continuous = assignment(20.0, 10.0, 60.0)
    swapped = {
        model.labels[0]: continuous[model.labels[1]],
        model.labels[1]: continuous[model.labels[0]],
        model.labels[2]: continuous[model.labels[2]],
    }

    plausible, scores = estimator._temporal_assignment_scores(
        [swapped, continuous], 1.02
    )

    assert len(plausible) == 2
    continuous_key = tuple(id(continuous[label]) for label in model.labels)
    swapped_key = tuple(id(swapped[label]) for label in model.labels)
    assert scores[continuous_key] == pytest.approx(0.0)
    assert scores[continuous_key] < scores[swapped_key]


@pytest.mark.parametrize("frame_time", [None, float("nan"), 1.0])
def test_invalid_or_nonmonotonic_time_bypasses_identity_gate(model, calibration, frame_time):
    estimator = PoseEstimator(model, calibration)
    base = make_detections(
        model,
        calibration,
        np.zeros((3, 1)),
        np.array([[0.0], [0.0], [740.0]]),
    )
    estimator._record_assignment(1.0, base)
    assignments, scores = estimator._temporal_assignment_scores([base], frame_time)
    assert assignments == [base]
    assert scores == {}


def test_reset_clears_identity_motion_history(model, calibration):
    estimator = PoseEstimator(model, calibration)
    detections = make_detections(
        model,
        calibration,
        np.zeros((3, 1)),
        np.array([[0.0], [0.0], [740.0]]),
    )
    assert estimator.estimate_camera_pose(list(detections.values()), frame_time=1.0) is not None
    assert estimator.assignment_history
    estimator.reset()
    assert not estimator.assignment_history
    assert not estimator.identity_gate_rejected


def test_projected_sphere_radius_tracks_depth(calibration):
    near = projected_sphere_radius_px(np.array([0.0, 0.0, 500.0]), 20.0, calibration)
    far = projected_sphere_radius_px(np.array([0.0, 0.0, 1000.0]), 20.0, calibration)
    assert near == pytest.approx(far * 2.0, rel=0.01)


def test_stereo_recovers_unordered_pose(model, stereo_calibration):
    rvec = np.array([[0.16], [-0.08], [0.05]])
    tvec = np.array([[30.0], [-15.0], [780.0]])
    left, right = make_stereo_detections(model, stereo_calibration, rvec, tvec)
    estimator = StereoPoseEstimator(model, stereo_calibration)
    estimate = estimator.estimate_camera_pose(
        list(reversed(list(left.values()))),
        [right[model.labels[1]], right[model.labels[2]], right[model.labels[0]]],
        frame_time=1.0, frame_skew_s=.01,
    )
    assert estimate is not None and estimate.source == "STEREO"
    np.testing.assert_allclose(estimate.tvec, tvec, atol=.5)
    assert rotation_error_degrees(estimate.rvec, rvec) < .1
    assert 1.0 <= estimate.camera_measurement_scale <= 4.0
    assert estimate.triangulation_angle_deg > 1.5


def test_stereo_right_mono_fallback_is_in_left_frame(model, stereo_calibration):
    rvec = np.array([[0.1], [0.04], [-0.03]])
    tvec = np.array([[45.0], [12.0], [720.0]])
    _, right = make_stereo_detections(model, stereo_calibration, rvec, tvec)
    estimate = StereoPoseEstimator(model, stereo_calibration).estimate_camera_pose(
        None, list(reversed(list(right.values()))), frame_time=1.0
    )
    assert estimate is not None and estimate.source == "RIGHT_MONO"
    np.testing.assert_allclose(estimate.tvec, tvec, atol=1.0)
    assert 2.0 <= estimate.camera_measurement_scale <= 6.0


def test_stereo_bad_epipolar_pair_uses_mono_fallback(model, stereo_calibration):
    rvec, tvec = np.zeros((3, 1)), np.array([[20.0], [5.0], [760.0]])
    left, right = make_stereo_detections(model, stereo_calibration, rvec, tvec)
    shifted = {label: replace(d, center=d.center + [0, 30]) for label, d in right.items()}
    estimate = StereoPoseEstimator(model, stereo_calibration).estimate_camera_pose(
        list(left.values()), list(shifted.values()), frame_time=1.0
    )
    assert estimate is not None and estimate.source != "STEREO"


def test_stereo_identity_gate_rejects_impossible_motion(model, stereo_calibration):
    estimator = StereoPoseEstimator(model, stereo_calibration)
    first = make_stereo_detections(
        model, stereo_calibration, np.zeros((3, 1)), np.array([[0.0], [0.0], [740.0]])
    )
    jumped = make_stereo_detections(
        model, stereo_calibration, np.zeros((3, 1)), np.array([[100.0], [0.0], [740.0]])
    )
    assert estimator.estimate_camera_pose(
        list(first[0].values()), list(first[1].values()), frame_time=1.0
    ) is not None
    assert estimator.estimate_camera_pose(
        list(jumped[0].values()), list(jumped[1].values()), frame_time=1.01
    ) is None
    assert estimator.left_mono.identity_gate_rejected
    assert estimator.right_mono.identity_gate_rejected
