from __future__ import annotations

import json

import numpy as np
import pytest

from vr_led_tracker.config import CameraCalibration, ControllerModel, load_color_profiles
from vr_led_tracker.errors import TrackerError


def test_load_valid_model(model):
    assert model.labels == ("red", "blue", "white")
    assert model.object_points.shape == (3, 3)
    np.testing.assert_allclose(model.diameters_mm, [24, 30, 20])


def test_rejects_collinear_model(tmp_path):
    path = tmp_path / "line.json"
    path.write_text(
        json.dumps(
            {
                "spheres": [
                    {"label": "red", "center_mm": [0, 0, 0], "diameter_mm": 20},
                    {"label": "blue", "center_mm": [10, 0, 0], "diameter_mm": 20},
                    {"label": "white", "center_mm": [20, 0, 0], "diameter_mm": 20},
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


def test_rejects_out_of_range_hsv(model, tmp_path):
    path = tmp_path / "colors.json"
    path.write_text(
        json.dumps(
            {
                "profiles": [
                    {
                        "label": label,
                        "hsv_ranges": [{"lower": [-1, 100, 100], "upper": [20, 255, 255]}],
                    }
                    for label in model.labels
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(TrackerError, match="Invalid color profile"):
        load_color_profiles(path, model.labels)
