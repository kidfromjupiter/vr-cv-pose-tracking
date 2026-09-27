from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .config import CameraCalibration, ControllerModel
from .detection import SphereDetection


@dataclass(frozen=True)
class PoseEstimate:
    rvec: np.ndarray
    tvec: np.ndarray
    reprojection_error_px: float
    radius_error_fraction: float
    state: str
    visible_spheres: int


def _rotation_distance(first: np.ndarray, second: np.ndarray) -> float:
    relative = first @ second.T
    cosine = np.clip((float(np.trace(relative)) - 1.0) / 2.0, -1.0, 1.0)
    return math.acos(cosine)


def projected_sphere_radius_px(
    center_camera_mm: np.ndarray,
    diameter_mm: float,
    calibration: CameraCalibration,
    samples: int = 32,
) -> float:
    center = np.asarray(center_camera_mm, dtype=np.float64).reshape(3)
    radius = diameter_mm * 0.5
    distance = float(np.linalg.norm(center))
    if (
        not np.isfinite(center).all()
        or not math.isfinite(radius)
        or radius <= 0.0
        or not math.isfinite(distance)
        or distance <= radius
        or center[2] <= 0
    ):
        return float("nan")
    normal = center / distance
    helper = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(helper, normal))) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    first = np.cross(normal, helper)
    first_norm = float(np.linalg.norm(first))
    if not math.isfinite(first_norm) or first_norm <= 1e-12:
        return float("nan")
    first /= first_norm
    second = np.cross(normal, first)
    silhouette_center = center * (1.0 - (radius * radius) / (distance * distance))
    silhouette_radius = radius * math.sqrt(1.0 - (radius * radius) / (distance * distance))
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
        return float("nan")
    if not np.isfinite(projected).all():
        return float("nan")
    area = abs(float(cv2.contourArea(projected.reshape(-1, 2).astype(np.float32))))
    return math.sqrt(area / np.pi) if area > 0 else float("nan")


class PoseEstimator:
    def __init__(
        self,
        model: ControllerModel,
        calibration: CameraCalibration,
        center_error_limit_px: float = 5.0,
        radius_error_limit: float = 0.35,
    ) -> None:
        self.model = model
        self.calibration = calibration
        self.center_error_limit_px = center_error_limit_px
        self.radius_error_limit = radius_error_limit
        self.last_pose: PoseEstimate | None = None

    def reset(self) -> None:
        self.last_pose = None

    def _metrics(
        self,
        indices: list[int],
        detections: dict[str, SphereDetection],
        rotation: np.ndarray,
        translation: np.ndarray,
    ) -> tuple[float, float]:
        object_points = self.model.object_points[indices]
        rvec, _ = cv2.Rodrigues(rotation)
        projected, _ = cv2.projectPoints(
            object_points,
            rvec,
            translation.reshape(3, 1),
            self.calibration.camera_matrix,
            self.calibration.distortion,
        )
        observed = np.asarray(
            [detections[self.model.spheres[index].label].center for index in indices]
        )
        center_error = float(
            np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - observed) ** 2, axis=1)))
        )
        radius_errors = []
        for index in indices:
            center_camera = rotation @ self.model.object_points[index] + translation
            predicted = projected_sphere_radius_px(
                center_camera, self.model.diameters_mm[index], self.calibration
            )
            measured = detections[self.model.spheres[index].label].radius
            if not np.isfinite(predicted) or measured <= 0:
                return center_error, float("inf")
            radius_errors.append(abs(predicted - measured) / measured)
        return center_error, float(np.mean(radius_errors))

    def estimate_camera_pose(
        self,
        detections: dict[str, SphereDetection],
        predicted_rotation: np.ndarray | None = None,
        predicted_translation: np.ndarray | None = None,
    ) -> PoseEstimate | None:
        if not all(label in detections for label in self.model.labels):
            return None
        image_points = np.asarray(
            [detections[label].center for label in self.model.labels], dtype=np.float64
        )
        try:
            count, rvecs, tvecs = cv2.solveP3P(
                np.ascontiguousarray(self.model.object_points, dtype=np.float64),
                np.ascontiguousarray(image_points, dtype=np.float64),
                self.calibration.camera_matrix,
                self.calibration.distortion,
                flags=cv2.SOLVEPNP_P3P,
            )
        except cv2.error:
            return None
        candidates: list[tuple[float, PoseEstimate]] = []
        for rvec, tvec in zip(rvecs[:count], tvecs[:count]):
            rotation, _ = cv2.Rodrigues(rvec)
            translation = np.asarray(tvec, dtype=np.float64).reshape(3)
            camera_points = (rotation @ self.model.object_points.T).T + translation
            if np.any(camera_points[:, 2] <= 1.0):
                continue
            center_error, radius_error = self._metrics(
                [0, 1, 2], detections, rotation, translation
            )
            score = center_error + 12.0 * radius_error
            if predicted_rotation is not None:
                score += 2.0 * math.degrees(_rotation_distance(rotation, predicted_rotation))
            elif self.last_pose is not None:
                last_rotation, _ = cv2.Rodrigues(self.last_pose.rvec)
                score += 0.5 * math.degrees(_rotation_distance(rotation, last_rotation))
            if predicted_translation is not None:
                score += 0.01 * float(np.linalg.norm(translation - predicted_translation))
            elif self.last_pose is not None:
                score += 0.005 * float(
                    np.linalg.norm(translation - self.last_pose.tvec.reshape(3))
                )
            candidates.append(
                (
                    score,
                    PoseEstimate(
                        np.asarray(rvec).reshape(3, 1),
                        translation.reshape(3, 1),
                        center_error,
                        radius_error,
                        "FULL",
                        3,
                    ),
                )
            )
        if not candidates:
            return None
        estimate = min(candidates, key=lambda item: item[0])[1]
        if (
            estimate.reprojection_error_px > self.center_error_limit_px
            or estimate.radius_error_fraction > self.radius_error_limit
        ):
            return None
        self.last_pose = estimate
        return estimate

    def estimate_translation(
        self,
        detections: dict[str, SphereDetection],
        rotation: np.ndarray,
        initial_translation: np.ndarray | None = None,
    ) -> PoseEstimate | None:
        indices = [index for index, label in enumerate(self.model.labels) if label in detections]
        if len(indices) < 2:
            return None
        rotation = np.asarray(rotation, dtype=np.float64)
        image_points = np.asarray(
            [detections[self.model.labels[i]].center for i in indices], dtype=np.float64
        )
        if rotation.shape != (3, 3) or not np.isfinite(rotation).all() or not np.isfinite(image_points).all():
            return None
        if any(
            not math.isfinite(detections[self.model.labels[i]].radius)
            or detections[self.model.labels[i]].radius <= 0.0
            for i in indices
        ):
            return None
        try:
            normalized = cv2.undistortPoints(
                image_points.reshape(-1, 1, 2),
                self.calibration.camera_matrix,
                self.calibration.distortion,
            ).reshape(-1, 2)
        except cv2.error:
            return None
        if not np.isfinite(normalized).all():
            return None
        bearings = np.column_stack([normalized, np.ones(len(indices))])
        bearing_norms = np.linalg.norm(bearings, axis=1, keepdims=True)
        if not np.isfinite(bearing_norms).all() or np.any(bearing_norms <= 1e-12):
            return None
        bearings /= bearing_norms
        blocks = []
        targets = []
        for index, bearing in zip(indices, bearings):
            projection = np.eye(3) - np.outer(bearing, bearing)
            blocks.append(projection)
            targets.append(-projection @ (rotation @ self.model.object_points[index]))
        design = np.vstack(blocks)
        target = np.concatenate(targets)
        translation = self._damped_solve(design, target, 1e-9)
        if translation is None:
            return None
        if initial_translation is not None:
            initial = np.asarray(initial_translation, dtype=np.float64).reshape(3)
            if not np.isfinite(initial).all():
                return None
            translation = 0.75 * translation + 0.25 * initial

        for _ in range(8):
            residual = self._translation_residual(indices, detections, rotation, translation)
            if not np.isfinite(residual).all():
                return None
            jacobian = np.empty((len(residual), 3), dtype=np.float64)
            for axis in range(3):
                shifted = translation.copy()
                shifted[axis] += 0.1
                shifted_residual = self._translation_residual(
                    indices, detections, rotation, shifted
                )
                if not np.isfinite(shifted_residual).all():
                    return None
                jacobian[:, axis] = (shifted_residual - residual) / 0.1
            step = self._damped_solve(jacobian, -residual, 1e-3)
            if step is None:
                return None
            translation += np.clip(step, -50.0, 50.0)
            if not np.isfinite(translation).all():
                return None
            if np.linalg.norm(step) < 1e-3:
                break
        camera_points = (rotation @ self.model.object_points[indices].T).T + translation
        if np.any(camera_points[:, 2] <= 1.0):
            return None
        center_error, radius_error = self._metrics(indices, detections, rotation, translation)
        limit = self.center_error_limit_px if len(indices) == 3 else 3.5
        if center_error > limit or radius_error > self.radius_error_limit:
            return None
        rvec, _ = cv2.Rodrigues(rotation)
        estimate = PoseEstimate(
            rvec.reshape(3, 1),
            translation.reshape(3, 1),
            center_error,
            radius_error,
            "FULL" if len(indices) == 3 else "DEGRADED_2",
            len(indices),
        )
        self.last_pose = estimate
        return estimate

    @staticmethod
    def _damped_solve(
        matrix: np.ndarray,
        target: np.ndarray,
        damping: float,
    ) -> np.ndarray | None:
        matrix = np.asarray(matrix, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        if (
            matrix.ndim != 2
            or target.shape != (matrix.shape[0],)
            or matrix.shape[1] != 3
            or not np.isfinite(matrix).all()
            or not np.isfinite(target).all()
        ):
            return None
        normal = matrix.T @ matrix + np.eye(3) * damping
        right_hand_side = matrix.T @ target
        if not np.isfinite(normal).all() or not np.isfinite(right_hand_side).all():
            return None
        try:
            solution = np.linalg.solve(normal, right_hand_side)
        except np.linalg.LinAlgError:
            return None
        return solution if np.isfinite(solution).all() else None

    def _translation_residual(
        self,
        indices: list[int],
        detections: dict[str, SphereDetection],
        rotation: np.ndarray,
        translation: np.ndarray,
    ) -> np.ndarray:
        rvec, _ = cv2.Rodrigues(rotation)
        projected, _ = cv2.projectPoints(
            self.model.object_points[indices],
            rvec,
            translation.reshape(3, 1),
            self.calibration.camera_matrix,
            self.calibration.distortion,
        )
        observed = np.asarray([detections[self.model.labels[i]].center for i in indices])
        residuals = list((projected.reshape(-1, 2) - observed).reshape(-1))
        for index in indices:
            predicted = projected_sphere_radius_px(
                rotation @ self.model.object_points[index] + translation,
                self.model.diameters_mm[index],
                self.calibration,
            )
            observed_radius = detections[self.model.labels[index]].radius
            residuals.append(3.0 * (predicted - observed_radius) / max(observed_radius, 1.0))
        return np.asarray(residuals, dtype=np.float64)
