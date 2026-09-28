from __future__ import annotations

import json
import math

import cv2
import numpy as np
import pytest

from vr_led_tracker.errors import TrackerError
from vr_led_tracker.hades_fusion import (
    HadesFusionSettings,
    HadesMotionFilter,
    HadesScalarFilter,
    HadesVectorFilter,
)


def rotation_z(degrees: float) -> np.ndarray:
    matrix, _ = cv2.Rodrigues(np.array([0.0, 0.0, math.radians(degrees)]))
    return matrix


def test_vector_filter_matches_hades_camera_equations():
    filter_ = HadesVectorFilter(HadesFusionSettings())
    filter_.initialize(np.zeros(3))

    estimate = filter_.update_camera(np.array([3.0, -6.0, 9.0]))

    np.testing.assert_allclose(filter_.camera_gain, np.full(3, 1.0 / 3.0))
    np.testing.assert_allclose(estimate, [1.0, -2.0, 3.0])
    np.testing.assert_allclose(
        filter_.camera_estimation_error,
        [8.0 + 1.0 / 6.0, 15.5 + 1.0 / 6.0, 23.0 + 1.0 / 6.0],
    )


def test_camera_quality_scale_reduces_measurement_gain():
    normal = HadesVectorFilter(HadesFusionSettings())
    weak = HadesVectorFilter(HadesFusionSettings())
    normal.initialize(np.zeros(3)); weak.initialize(np.zeros(3))
    normal.update_camera(np.ones(3), measurement_scale=1.0)
    weak.update_camera(np.ones(3), measurement_scale=4.0)
    assert np.all(weak.camera_gain < normal.camera_gain)


def test_vector_filter_accumulates_imu_displacement():
    filter_ = HadesVectorFilter(HadesFusionSettings())
    filter_.initialize(np.array([1.0, 2.0, 3.0]))

    estimate = filter_.update_imu(np.array([0.7, 0.0, -0.7]))

    np.testing.assert_allclose(filter_.imu_gain, np.full(3, 5.0 / 7.0))
    np.testing.assert_allclose(estimate, [1.5, 2.0, 2.5])
    np.testing.assert_allclose(filter_.reading, [1.7, 2.0, 2.3])


def test_scalar_filter_matches_hades_equation():
    filter_ = HadesScalarFilter(5.0, 1.25, 2.0)
    assert filter_.update(math.pi / 2.0) == pytest.approx(math.pi / 10.0)
    assert filter_.gain == pytest.approx(0.2)


def test_stationary_prediction_cancels_gravity():
    filter_ = HadesMotionFilter(HadesFusionSettings())
    filter_.initialize(np.zeros(3), np.eye(3), np.eye(3), np.zeros(3))
    for _ in range(100):
        filter_.predict(np.eye(3), np.array([0.0, 0.0, 1.0]), 0.01)
    np.testing.assert_allclose(filter_.snapshot().position_m, np.zeros(3), atol=1e-12)
    np.testing.assert_allclose(filter_.snapshot().velocity_m_s, np.zeros(3), atol=1e-12)


def test_orientation_is_imu_based_after_static_optical_alignment():
    filter_ = HadesMotionFilter(HadesFusionSettings())
    filter_.initialize(np.zeros(3), rotation_z(30), rotation_z(10), np.zeros(3))
    filter_.predict(rotation_z(40), np.array([0.0, 0.0, 1.0]), 0.01)
    before_camera = filter_.snapshot().rotation_matrix

    filter_.update_camera(np.array([0.1, 0.0, 1.0]), 1.0)

    np.testing.assert_allclose(before_camera, rotation_z(60), atol=1e-10)
    np.testing.assert_allclose(filter_.snapshot().rotation_matrix, before_camera, atol=1e-12)


def test_camera_reacquisition_discards_pre_dropout_velocity():
    filter_ = HadesMotionFilter(HadesFusionSettings())
    filter_.initialize(np.zeros(3), np.eye(3), np.eye(3), np.zeros(3))
    filter_.camera_tracking = True
    filter_.last_camera_time = 0.0
    filter_.last_camera_position = np.zeros(3)
    filter_.last_camera_estimate = np.zeros(3)
    filter_.velocity[:] = 1.0
    filter_.mark_camera_missing()
    filter_.predict(np.eye(3), np.array([0.2, 0.0, 1.0]), 0.01)

    pose = filter_.update_camera(np.array([0.1, 0.0, 0.7]), 2.0)

    np.testing.assert_allclose(pose.velocity_m_s, np.zeros(3))


def test_optional_camera_velocity_yaw_correction():
    settings = HadesFusionSettings(
        yaw_drift_enabled=True,
        yaw_lower_velocity_m_s=0.1,
        yaw_upper_velocity_m_s=10.0,
    )
    filter_ = HadesMotionFilter(settings)
    filter_.initialize(np.zeros(3), np.eye(3), np.eye(3), np.zeros(3))
    filter_.imu_velocity_window[:] = [1.0, 0.0, 0.0]

    filter_._update_yaw_correction(np.array([0.0, 1.0, 0.0]))

    assert filter_.yaw_correction_rad == pytest.approx(math.pi / 10.0)


def test_settings_load_and_validation(tmp_path):
    path = tmp_path / "fusion.json"
    path.write_text(
        json.dumps(
            {
                "camera": {
                    "measurement_uncertainty": 3,
                    "estimation_uncertainty": 1,
                    "process_noise": 4,
                },
                "imu": {
                    "measurement_uncertainty": 2,
                    "estimation_uncertainty": 5,
                    "process_noise": 6,
                },
                "velocity_damping": 0.8,
                "yaw_drift_correction": {
                    "enabled": False,
                    "lower_velocity_m_s": 1,
                    "upper_velocity_m_s": 2,
                    "measurement_uncertainty": 5,
                    "estimation_uncertainty": 1.25,
                    "process_noise": 2,
                },
            }
        ),
        encoding="utf-8",
    )
    settings = HadesFusionSettings.load(path)
    assert settings.camera_measurement_uncertainty == 3.0
    assert settings.velocity_damping == 0.8
    assert settings.identity_max_speed_px_s == 6000.0
    assert settings.identity_reacquire_timeout_s == 0.25

    with pytest.raises(TrackerError):
        HadesFusionSettings(velocity_damping=1.1)
    with pytest.raises(TrackerError):
        HadesFusionSettings(identity_max_speed_px_s=0.0)
    with pytest.raises(TrackerError):
        HadesFusionSettings(identity_reacquire_timeout_s=0.0)


def test_settings_load_custom_identity_tracking(tmp_path):
    source = {
        "camera": {
            "measurement_uncertainty": 2,
            "estimation_uncertainty": 1,
            "process_noise": 7.5,
        },
        "imu": {
            "measurement_uncertainty": 2,
            "estimation_uncertainty": 5,
            "process_noise": 22.55,
        },
        "velocity_damping": 0.9,
        "yaw_drift_correction": {
            "enabled": False,
            "lower_velocity_m_s": 2,
            "upper_velocity_m_s": 5,
            "measurement_uncertainty": 5,
            "estimation_uncertainty": 1.25,
            "process_noise": 2,
        },
        "identity_tracking": {
            "max_speed_px_s": 8000,
            "reacquire_timeout_s": 0.4,
        },
    }
    path = tmp_path / "fusion.json"
    path.write_text(json.dumps(source), encoding="utf-8")

    settings = HadesFusionSettings.load(path)

    assert settings.identity_max_speed_px_s == 8000.0
    assert settings.identity_reacquire_timeout_s == 0.4
