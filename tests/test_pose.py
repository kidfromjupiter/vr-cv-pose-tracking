from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from conftest import make_detection
from vr_led_tracker.pose import PositionEstimator, projected_sphere_radius_px


@pytest.mark.parametrize(
    "position",
    [
        np.array([0.0, 0.0, 500.0]),
        np.array([80.0, -45.0, 750.0]),
        np.array([-220.0, 110.0, 1500.0]),
    ],
)
def test_recovers_ball_xyz(model, calibration, position):
    detection = make_detection(model, calibration, position)
    estimate = PositionEstimator(model, calibration).estimate([detection])
    assert estimate is not None
    np.testing.assert_allclose(estimate.position_mm, position, atol=0.3)


def test_recovers_xyz_with_lens_distortion(model, calibration):
    distorted = replace(
        calibration,
        distortion=np.array([[0.16], [-0.45], [0.002], [-0.001], [0.3]]),
    )
    expected = np.array([130.0, -75.0, 900.0])
    estimate = PositionEstimator(model, distorted).estimate(
        [make_detection(model, distorted, expected)]
    )
    assert estimate is not None
    np.testing.assert_allclose(estimate.position_mm, expected, atol=0.5)


def test_apparent_radius_tracks_depth(calibration):
    near = projected_sphere_radius_px(np.array([0.0, 0.0, 500.0]), 20.0, calibration)
    far = projected_sphere_radius_px(np.array([0.0, 0.0, 1000.0]), 20.0, calibration)
    assert near == pytest.approx(far * 2.0, rel=0.01)


def test_rejects_invalid_detection(model, calibration):
    detection = make_detection(model, calibration, np.array([0.0, 0.0, 700.0]))
    invalid = replace(detection, radius=0.0)
    assert PositionEstimator(model, calibration).estimate([invalid]) is None


def test_temporal_continuity_prefers_nearby_candidate(model, calibration):
    estimator = PositionEstimator(model, calibration)
    first = make_detection(model, calibration, np.array([0.0, 0.0, 700.0]))
    nearby = make_detection(model, calibration, np.array([5.0, 0.0, 700.0]), score=0.8)
    distractor = make_detection(model, calibration, np.array([250.0, 100.0, 700.0]), score=1.0)
    assert estimator.estimate([first]) is not None
    estimate = estimator.estimate([distractor, nearby])
    assert estimate is not None
    np.testing.assert_allclose(estimate.position_mm, [5.0, 0.0, 700.0], atol=0.5)
