from __future__ import annotations

import json

import numpy as np
import pytest

from vr_led_tracker.config import BallModel, CameraCalibration, load_color_profile
from vr_led_tracker.errors import TrackerError


def test_load_valid_model(model):
    assert model.label == "blue"
    assert model.diameter_mm == 20.0


@pytest.mark.parametrize(
    "value, message",
    [
        ({"ball": {"label": "red", "diameter_mm": 20}}, "label"),
        ({"ball": {"label": "blue", "diameter_mm": 0}}, "positive"),
        ({"spheres": []}, "Legacy"),
    ],
)
def test_rejects_invalid_model(tmp_path, value, message):
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(TrackerError, match=message):
        BallModel.load(path)


def test_calibration_scales_same_aspect(calibration):
    scaled = calibration.for_image_size((640, 360))
    assert scaled.camera_matrix[0, 0] == pytest.approx(450)
    assert scaled.camera_matrix[1, 2] == pytest.approx(180)
    np.testing.assert_allclose(scaled.distortion, calibration.distortion)


def test_calibration_rejects_aspect_change(calibration):
    with pytest.raises(TrackerError, match="aspect ratio"):
        calibration.for_image_size((640, 480))


def test_load_blue_color_profile(tmp_path):
    path = tmp_path / "color.json"
    path.write_text(
        json.dumps(
            {
                "profiles": [
                    {
                        "label": "blue",
                        "hsv_ranges": [{"lower": [100, 80, 60], "upper": [130, 255, 255]}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    profile = load_color_profile(path)
    assert profile.label == "blue"


def test_rejects_out_of_range_hsv(tmp_path):
    path = tmp_path / "color.json"
    path.write_text(
        json.dumps(
            {
                "profiles": [
                    {
                        "label": "blue",
                        "hsv_ranges": [{"lower": [-1, 80, 60], "upper": [130, 255, 255]}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(TrackerError, match="Invalid blue"):
        load_color_profile(path)
