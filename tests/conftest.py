from __future__ import annotations

import cv2
import numpy as np
import pytest

from vr_led_tracker.config import CameraCalibration, ControllerModel


@pytest.fixture
def model_file(tmp_path):
    path = tmp_path / "controller.json"
    path.write_text(
        """
        {
          "spheres": [
            {"label": "red", "center_mm": [-70, 0, 0], "diameter_mm": 24},
            {"label": "blue", "center_mm": [70, 0, 0], "diameter_mm": 30},
            {"label": "white", "center_mm": [0, 45, 30], "diameter_mm": 20}
          ]
        }
        """,
        encoding="utf-8",
    )
    return path


@pytest.fixture
def model(model_file) -> ControllerModel:
    return ControllerModel.load(model_file)


@pytest.fixture
def calibration() -> CameraCalibration:
    return CameraCalibration(
        (1280, 720),
        np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]]),
        np.zeros((5, 1)),
        0.2,
        {},
    )


def make_detections(model, calibration, rvec, tvec):
    from vr_led_tracker.detection import SphereDetection
    from vr_led_tracker.pose import projected_sphere_radius_px

    projected, _ = cv2.projectPoints(
        model.object_points, rvec, tvec, calibration.camera_matrix, calibration.distortion
    )
    contour = np.array([[[0, 0]], [[1, 0]], [[1, 1]], [[0, 1]]], np.int32)
    rotation, _ = cv2.Rodrigues(rvec)
    values = {}
    for sphere, point in zip(model.spheres, projected.reshape(-1, 2)):
        center_camera = rotation @ sphere.center_mm + np.asarray(tvec).reshape(3)
        radius = projected_sphere_radius_px(center_camera, sphere.diameter_mm, calibration)
        values[sphere.label] = SphereDetection(
            sphere.label, point, radius, np.pi * radius**2, 0.9, 1.0, contour
        )
    return values
