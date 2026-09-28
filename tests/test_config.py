from __future__ import annotations

import json

import numpy as np
import pytest

from vr_led_tracker.config import CameraCalibration, ControllerModel, StereoCalibration
from vr_led_tracker.errors import TrackerError


def test_load_valid_model(model):
    assert model.labels == ("sphere_0", "sphere_1", "sphere_2")
    assert model.object_points.shape == (3, 3)
    np.testing.assert_allclose(model.diameters_mm, [20, 20, 20])


def test_rejects_collinear_model(tmp_path):
    path = tmp_path / "line.json"
    path.write_text(
        json.dumps(
            {
                "spheres": [
                    {"label": "sphere_0", "center_mm": [0, 0, 0], "diameter_mm": 20},
                    {"label": "sphere_1", "center_mm": [10, 0, 0], "diameter_mm": 20},
                    {"label": "sphere_2", "center_mm": [20, 0, 0], "diameter_mm": 20},
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(TrackerError, match="collinear"):
        ControllerModel.load(path)


def test_rejects_legacy_led_model(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({"leds": []}), encoding="utf-8")
    with pytest.raises(TrackerError, match="Legacy"):
        ControllerModel.load(path)


def test_calibration_scales_same_aspect(calibration):
    scaled = calibration.for_image_size((640, 360))
    assert scaled.camera_matrix[0, 0] == pytest.approx(450)
    assert scaled.camera_matrix[1, 2] == pytest.approx(180)
    np.testing.assert_allclose(scaled.distortion, calibration.distortion)


def test_calibration_rejects_aspect_change(calibration):
    with pytest.raises(TrackerError, match="aspect ratio"):
        calibration.for_image_size((640, 480))


def test_rejects_symmetric_white_sphere_geometry(tmp_path):
    path = tmp_path / "symmetric.json"
    path.write_text(
        json.dumps(
            {
                "spheres": [
                    {"label": "sphere_0", "center_mm": [-50, 0, 0], "diameter_mm": 20},
                    {"label": "sphere_1", "center_mm": [50, 0, 0], "diameter_mm": 20},
                    {"label": "sphere_2", "center_mm": [0, 80, 0], "diameter_mm": 20},
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(TrackerError, match="too symmetric"):
        ControllerModel.load(path)


def test_stereo_calibration_round_trip_and_scaling(tmp_path, stereo_calibration):
    path = tmp_path / "stereo.json"
    stereo_calibration.save(path)

    loaded = StereoCalibration.load(path)
    np.testing.assert_allclose(loaded.right_from_left_rotation, np.eye(3))
    np.testing.assert_allclose(loaded.right_from_left_translation_mm, [-120.0, 0.0, 0.0])
    scaled = loaded.for_image_sizes((640, 360), (640, 360))
    assert scaled.left.camera_matrix[0, 0] == pytest.approx(450.0)
    assert scaled.right.camera_matrix[1, 2] == pytest.approx(180.0)


def test_stereo_calibration_rejects_invalid_transform(calibration):
    with pytest.raises(TrackerError, match="proper orthonormal"):
        StereoCalibration(
            calibration,
            calibration,
            np.diag([1.0, 1.0, 2.0]),
            np.array([-120.0, 0.0, 0.0]),
            0.2,
            {},
        )
    with pytest.raises(TrackerError, match="baseline"):
        StereoCalibration(calibration, calibration, np.eye(3), np.zeros(3), 0.2, {})
