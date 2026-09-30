from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .config import BallModel, CameraCalibration
from .detection import BallDetection


@dataclass(frozen=True)
class PositionEstimate:
    position_mm: np.ndarray
    fit_error_px: float
    detection: BallDetection


def projected_sphere_observation(
    center_camera_mm: np.ndarray,
    diameter_mm: float,
    calibration: CameraCalibration,
    samples: int = 64,
) -> tuple[np.ndarray, float] | None:
    center = np.asarray(center_camera_mm, dtype=np.float64).reshape(3)
    sphere_radius = float(diameter_mm) * 0.5
    distance = float(np.linalg.norm(center))
    if (
        not np.isfinite(center).all()
        or not math.isfinite(sphere_radius)
        or sphere_radius <= 0
        or distance <= sphere_radius
        or center[2] <= 0
    ):
        return None
    normal = center / distance
    helper = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(helper, normal))) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    first = np.cross(normal, helper)
    first_norm = float(np.linalg.norm(first))
    if first_norm <= 1e-12:
        return None
    first /= first_norm
    second = np.cross(normal, first)
    silhouette_center = center * (1.0 - sphere_radius**2 / distance**2)
    silhouette_radius = sphere_radius * math.sqrt(1.0 - sphere_radius**2 / distance**2)
    angles = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
    circle = silhouette_center + silhouette_radius * (
        np.cos(angles)[:, None] * first + np.sin(angles)[:, None] * second
    )
    try:
        projected, _ = cv2.projectPoints(
            circle,
            np.zeros((3, 1)),
            np.zeros((3, 1)),
            calibration.camera_matrix,
            calibration.distortion,
        )
    except cv2.error:
        return None
    contour = projected.reshape(-1, 2).astype(np.float32)
    if not np.isfinite(contour).all():
        return None
    area = abs(float(cv2.contourArea(contour)))
    moments = cv2.moments(contour)
    if area <= 0 or abs(float(moments["m00"])) <= 1e-12:
        return None
    pixel_center = np.array(
        [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]],
        dtype=np.float64,
    )
    return pixel_center, math.sqrt(area / np.pi)


def projected_sphere_radius_px(
    center_camera_mm: np.ndarray,
    diameter_mm: float,
    calibration: CameraCalibration,
    samples: int = 64,
) -> float:
    observation = projected_sphere_observation(
        center_camera_mm, diameter_mm, calibration, samples
    )
    return float("nan") if observation is None else observation[1]


class PositionEstimator:
    def __init__(
        self,
        model: BallModel,
        calibration: CameraCalibration,
        minimum_depth_mm: float = 100.0,
        maximum_depth_mm: float = 5000.0,
    ) -> None:
        self.model = model
        self.calibration = calibration
        self.minimum_depth_mm = minimum_depth_mm
        self.maximum_depth_mm = maximum_depth_mm
        self.last_estimate: PositionEstimate | None = None

    def reset(self) -> None:
        self.last_estimate = None

    def _initial_position(self, detection: BallDetection) -> np.ndarray | None:
        if detection.radius <= 0 or not np.isfinite(detection.center).all():
            return None
        try:
            point = cv2.undistortPoints(
                detection.center.reshape(1, 1, 2),
                self.calibration.camera_matrix,
                self.calibration.distortion,
            ).reshape(2)
        except cv2.error:
            return None
        focal = math.sqrt(
            float(self.calibration.camera_matrix[0, 0])
            * float(self.calibration.camera_matrix[1, 1])
        )
        depth = focal * self.model.diameter_mm / (2.0 * detection.radius)
        result = np.array([point[0] * depth, point[1] * depth, depth], dtype=np.float64)
        return result if np.isfinite(result).all() else None

    def _fit(self, detection: BallDetection) -> PositionEstimate | None:
        position = self._initial_position(detection)
        if position is None:
            return None
        observed = np.array(
            [detection.center[0], detection.center[1], detection.radius], dtype=np.float64
        )
        for _ in range(15):
            prediction = projected_sphere_observation(
                position, self.model.diameter_mm, self.calibration
            )
            if prediction is None:
                return None
            predicted = np.array([prediction[0][0], prediction[0][1], prediction[1]])
            residual = predicted - observed
            if float(np.linalg.norm(residual)) < 1e-5:
                break
            jacobian = np.empty((3, 3), dtype=np.float64)
            for axis in range(3):
                step = max(0.05, abs(float(position[axis])) * 1e-4)
                perturbed = position.copy()
                perturbed[axis] += step
                shifted = projected_sphere_observation(
                    perturbed, self.model.diameter_mm, self.calibration
                )
                if shifted is None:
                    return None
                shifted_vector = np.array([shifted[0][0], shifted[0][1], shifted[1]])
                jacobian[:, axis] = (shifted_vector - predicted) / step
            try:
                delta = np.linalg.lstsq(jacobian, -residual, rcond=None)[0]
            except np.linalg.LinAlgError:
                return None
            delta_norm = float(np.linalg.norm(delta))
            if not np.isfinite(delta).all():
                return None
            if delta_norm > 250.0:
                delta *= 250.0 / delta_norm
            position += delta
            if position[2] <= self.model.diameter_mm * 0.5:
                return None
        final = projected_sphere_observation(position, self.model.diameter_mm, self.calibration)
        if final is None:
            return None
        error = float(
            np.linalg.norm(
                np.array([final[0][0], final[0][1], final[1]], dtype=np.float64) - observed
            )
        )
        if (
            not np.isfinite(position).all()
            or not math.isfinite(error)
            or not self.minimum_depth_mm <= position[2] <= self.maximum_depth_mm
            or error > 1.0
        ):
            return None
        return PositionEstimate(position.copy(), error, detection)

    def estimate(self, detections: list[BallDetection]) -> PositionEstimate | None:
        candidates: list[tuple[float, PositionEstimate]] = []
        for detection in detections:
            estimate = self._fit(detection)
            if estimate is None:
                continue
            score = -2.0 * detection.score + estimate.fit_error_px
            if self.last_estimate is not None:
                pixel_distance = float(
                    np.linalg.norm(detection.center - self.last_estimate.detection.center)
                )
                radius_change = abs(detection.radius - self.last_estimate.detection.radius)
                score += 0.02 * pixel_distance + 0.04 * radius_change
            candidates.append((score, estimate))
        if not candidates:
            return None
        estimate = min(candidates, key=lambda item: item[0])[1]
        self.last_estimate = estimate
        return estimate


PoseEstimate = PositionEstimate
PoseEstimator = PositionEstimator
