from __future__ import annotations

import numpy as np
import pytest

from vr_led_tracker.config import BallModel, CameraCalibration, ColorProfile
from vr_led_tracker.detection import BallDetection
from vr_led_tracker.pose import projected_sphere_observation


@pytest.fixture
def model_file(tmp_path):
    path = tmp_path / "controller.json"
    path.write_text(
        '{"ball": {"label": "blue", "diameter_mm": 20.0}}',
        encoding="utf-8",
    )
    return path


@pytest.fixture
def model(model_file) -> BallModel:
    return BallModel.load(model_file)


@pytest.fixture
def calibration() -> CameraCalibration:
    return CameraCalibration(
        (1280, 720),
        np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]]),
        np.zeros((5, 1)),
        0.2,
        {},
    )


@pytest.fixture
def blue_profile() -> ColorProfile:
    return ColorProfile(
        "blue",
        ((np.array([105, 120, 80], np.uint8), np.array([125, 255, 255], np.uint8)),),
    )


def make_detection(model, calibration, position_mm, score=1.0) -> BallDetection:
    observation = projected_sphere_observation(position_mm, model.diameter_mm, calibration)
    assert observation is not None
    center, radius = observation
    contour = np.array([[[0, 0]], [[1, 0]], [[1, 1]], [[0, 1]]], np.int32)
    return BallDetection(center, radius, np.pi * radius**2, 0.95, score, contour)
