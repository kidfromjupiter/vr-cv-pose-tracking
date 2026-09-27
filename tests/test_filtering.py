from __future__ import annotations

import numpy as np
import pytest

from vr_led_tracker.filtering import OneEuroVector, PoseFilter


def test_one_euro_reduces_stationary_jitter():
    rng = np.random.default_rng(4)
    inputs = np.array([100.0, 50.0, 800.0]) + rng.normal(0, 3, size=(120, 3))
    filter_ = OneEuroVector(min_cutoff=1.0, beta=0.0)
    outputs = np.array([filter_.update(value, index / 60.0) for index, value in enumerate(inputs)])
    assert np.mean(np.std(outputs[30:], axis=0)) < np.mean(np.std(inputs[30:], axis=0)) * 0.6


def test_pose_filter_outputs_normalized_rotation():
    filter_ = PoseFilter()
    result = filter_.update(np.array([[0.1], [0.2], [-0.1]]), np.array([[1], [2], [3]]), 1.0)
    assert np.linalg.norm(result.quaternion) == pytest.approx(1.0)
    np.testing.assert_allclose(result.rotation_matrix @ result.rotation_matrix.T, np.eye(3), atol=1e-10)
