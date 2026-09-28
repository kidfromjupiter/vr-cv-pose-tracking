from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations, permutations

import cv2
import numpy as np

from .config import CameraCalibration, ControllerModel, StereoCalibration
from .detection import SphereDetection


@dataclass(frozen=True)
class PoseEstimate:
    rvec: np.ndarray
    tvec: np.ndarray
    reprojection_error_px: float
    radius_error_fraction: float
    state: str
    visible_spheres: int
    source: str = "MONOCULAR"


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
        self.last_assignment: dict[str, SphereDetection] = {}

    def reset(self) -> None:
        self.last_pose = None
        self.last_assignment.clear()

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
        detections: dict[str, SphereDetection] | list[SphereDetection],
        predicted_rotation: np.ndarray | None = None,
        predicted_translation: np.ndarray | None = None,
    ) -> PoseEstimate | None:
        previous_assignment = dict(self.last_assignment)
        self.last_assignment = {}
        projected_prediction: np.ndarray | None = None
        if isinstance(detections, dict):
            assignments = [detections]
        else:
            if len(detections) < 3:
                return None
            assignments = [
                dict(zip(self.model.labels, ordered))
                for group in combinations(detections[:8], 3)
                for ordered in permutations(group)
            ]
            if predicted_rotation is not None and predicted_translation is not None:
                try:
                    predicted_rvec, _ = cv2.Rodrigues(
                        np.asarray(predicted_rotation, dtype=np.float64)
                    )
                    projected, _ = cv2.projectPoints(
                        self.model.object_points,
                        predicted_rvec,
                        np.asarray(predicted_translation, dtype=np.float64).reshape(3, 1),
                        self.calibration.camera_matrix,
                        self.calibration.distortion,
                    )
                    projected = projected.reshape(-1, 2)
                except (ValueError, cv2.error):
                    projected = np.empty((0, 2))
                if projected.shape == (3, 2) and np.isfinite(projected).all():
                    projected_prediction = projected
                    ranked = sorted(
                        assignments,
                        key=lambda assignment: sum(
                            float(np.linalg.norm(assignment[label].center - point))
                            for label, point in zip(self.model.labels, projected)
                        ),
                    )
                    gated = [
                        assignment
                        for assignment in ranked
                        if max(
                            float(np.linalg.norm(assignment[label].center - point))
                            for label, point in zip(self.model.labels, projected)
                        ) <= 200.0
                    ]
                    assignments = (gated or ranked)[:12]

        best: tuple[float, PoseEstimate, dict[str, SphereDetection]] | None = None
        for assignment in assignments:
            if not all(label in assignment for label in self.model.labels):
                continue
            for score, estimate in self._labeled_pose_candidates(
                assignment,
                predicted_rotation,
                predicted_translation,
            ):
                detection_quality = sum(item.score for item in assignment.values())
                total = score - 2.0 * detection_quality
                if projected_prediction is not None:
                    total += 0.12 * sum(
                        float(np.linalg.norm(assignment[label].center - point))
                        for label, point in zip(self.model.labels, projected_prediction)
                    )
                if all(label in previous_assignment for label in self.model.labels):
                    total += 0.08 * sum(
                        float(
                            np.linalg.norm(
                                assignment[label].center
                                - previous_assignment[label].center
                            )
                        )
                        for label in self.model.labels
                    )
                if best is None or total < best[0]:
                    best = (total, estimate, assignment)
        if best is None:
            return None
        estimate = best[1]
        if estimate.reprojection_error_px > self.center_error_limit_px:
            return None
        self.last_pose = estimate
        self.last_assignment = dict(best[2])
        return estimate

    def _labeled_pose_candidates(
        self,
        detections: dict[str, SphereDetection],
        predicted_rotation: np.ndarray | None,
        predicted_translation: np.ndarray | None,
    ) -> list[tuple[float, PoseEstimate]]:
        image_points = np.asarray(
            [detections[label].center for label in self.model.labels], dtype=np.float64
        )
        if not np.isfinite(image_points).all():
            return []
        try:
            count, rvecs, tvecs = cv2.solveP3P(
                np.ascontiguousarray(self.model.object_points, dtype=np.float64),
                np.ascontiguousarray(image_points, dtype=np.float64),
                self.calibration.camera_matrix,
                self.calibration.distortion,
                flags=cv2.SOLVEPNP_P3P,
            )
        except cv2.error:
            return []
        candidates: list[tuple[float, PoseEstimate]] = []
        for rvec, tvec in zip(rvecs[:count], tvecs[:count]):
            rotation, _ = cv2.Rodrigues(rvec)
            translation = np.asarray(tvec, dtype=np.float64).reshape(3)
            camera_points = (rotation @ self.model.object_points.T).T + translation
            if (
                not np.isfinite(rotation).all()
                or not np.isfinite(translation).all()
                or np.any(camera_points[:, 2] <= 100.0)
                or np.any(camera_points[:, 2] >= 5000.0)
            ):
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
        return candidates


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float64).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _rigid_transform(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_center = np.mean(source, axis=0)
    target_center = np.mean(target, axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    left, _, right = np.linalg.svd(covariance)
    rotation = right.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right[-1, :] *= -1.0
        rotation = right.T @ left.T
    translation = target_center - rotation @ source_center
    return rotation, translation


class StereoPoseEstimator:
    def __init__(
        self,
        model: ControllerModel,
        calibration: StereoCalibration,
        center_error_limit_px: float = 5.0,
        epipolar_error_limit_px: float = 3.0,
        rigid_error_limit_mm: float = 10.0,
    ) -> None:
        self.model = model
        self.calibration = calibration
        self.center_error_limit_px = center_error_limit_px
        self.epipolar_error_limit_px = epipolar_error_limit_px
        self.rigid_error_limit_mm = rigid_error_limit_mm
        self.left_mono = PoseEstimator(model, calibration.left, center_error_limit_px)
        self.right_mono = PoseEstimator(model, calibration.right, center_error_limit_px)
        self.last_pose: PoseEstimate | None = None
        self.last_left_assignment: dict[str, SphereDetection] = {}
        self.last_right_assignment: dict[str, SphereDetection] = {}
        essential = _skew(calibration.right_from_left_translation_mm) @ calibration.right_from_left_rotation
        self.fundamental = (
            np.linalg.inv(calibration.right.camera_matrix).T
            @ essential
            @ np.linalg.inv(calibration.left.camera_matrix)
        )

    def reset(self) -> None:
        self.left_mono.reset()
        self.right_mono.reset()
        self.last_pose = None
        self.last_left_assignment.clear()
        self.last_right_assignment.clear()

    @staticmethod
    def _valid_detections(detections: list[SphereDetection] | None) -> list[SphereDetection]:
        if detections is None:
            return []
        return [
            item
            for item in detections[:8]
            if np.asarray(item.center).shape == (2,) and np.isfinite(item.center).all()
        ]

    def _undistorted_pixels(
        self, detections: list[SphereDetection], calibration: CameraCalibration
    ) -> np.ndarray:
        points = np.asarray([item.center for item in detections], dtype=np.float64).reshape(-1, 1, 2)
        return cv2.undistortPoints(
            points, calibration.camera_matrix, calibration.distortion, P=calibration.camera_matrix
        ).reshape(-1, 2)

    def _epipolar_edges(
        self, left: list[SphereDetection], right: list[SphereDetection]
    ) -> list[tuple[float, int, int]]:
        left_points = self._undistorted_pixels(left, self.calibration.left)
        right_points = self._undistorted_pixels(right, self.calibration.right)
        edges: list[tuple[float, int, int]] = []
        for left_index, left_point in enumerate(left_points):
            homogeneous_left = np.array([left_point[0], left_point[1], 1.0])
            right_line = self.fundamental @ homogeneous_left
            for right_index, right_point in enumerate(right_points):
                homogeneous_right = np.array([right_point[0], right_point[1], 1.0])
                left_line = self.fundamental.T @ homogeneous_right
                right_denominator = float(np.linalg.norm(right_line[:2]))
                left_denominator = float(np.linalg.norm(left_line[:2]))
                if min(right_denominator, left_denominator) <= 1e-12:
                    continue
                numerator = abs(float(homogeneous_right @ right_line))
                error = 0.5 * (
                    numerator / right_denominator + numerator / left_denominator
                )
                if error <= self.epipolar_error_limit_px:
                    edges.append((error, left_index, right_index))
        return edges

    def _triangulate(
        self,
        left: list[SphereDetection],
        right: list[SphereDetection],
        matching: tuple[tuple[float, int, int], ...],
    ) -> np.ndarray | None:
        left_points = np.asarray([left[item[1]].center for item in matching], dtype=np.float64)
        right_points = np.asarray([right[item[2]].center for item in matching], dtype=np.float64)
        left_normal = cv2.undistortPoints(
            left_points.reshape(-1, 1, 2),
            self.calibration.left.camera_matrix,
            self.calibration.left.distortion,
        ).reshape(-1, 2)
        right_normal = cv2.undistortPoints(
            right_points.reshape(-1, 1, 2),
            self.calibration.right.camera_matrix,
            self.calibration.right.distortion,
        ).reshape(-1, 2)
        left_projection = np.hstack([np.eye(3), np.zeros((3, 1))])
        right_projection = np.hstack(
            [
                self.calibration.right_from_left_rotation,
                self.calibration.right_from_left_translation_mm.reshape(3, 1),
            ]
        )
        homogeneous = cv2.triangulatePoints(
            left_projection, right_projection, left_normal.T, right_normal.T
        )
        scale = homogeneous[3]
        if np.any(np.abs(scale) <= 1e-9):
            return None
        points = (homogeneous[:3] / scale).T
        right_camera = (
            self.calibration.right_from_left_rotation @ points.T
        ).T + self.calibration.right_from_left_translation_mm
        if (
            not np.isfinite(points).all()
            or np.any(points[:, 2] <= 100.0)
            or np.any(points[:, 2] >= 5000.0)
            or np.any(right_camera[:, 2] <= 100.0)
            or np.any(right_camera[:, 2] >= 5000.0)
        ):
            return None
        return points

    def _stereo_metrics(
        self,
        rotation: np.ndarray,
        translation: np.ndarray,
        left_assignment: dict[str, SphereDetection],
        right_assignment: dict[str, SphereDetection],
    ) -> tuple[float, float]:
        left_rvec, _ = cv2.Rodrigues(rotation)
        right_rotation = self.calibration.right_from_left_rotation @ rotation
        right_translation = (
            self.calibration.right_from_left_rotation @ translation
            + self.calibration.right_from_left_translation_mm
        )
        right_rvec, _ = cv2.Rodrigues(right_rotation)
        left_projected, _ = cv2.projectPoints(
            self.model.object_points,
            left_rvec,
            translation.reshape(3, 1),
            self.calibration.left.camera_matrix,
            self.calibration.left.distortion,
        )
        right_projected, _ = cv2.projectPoints(
            self.model.object_points,
            right_rvec,
            right_translation.reshape(3, 1),
            self.calibration.right.camera_matrix,
            self.calibration.right.distortion,
        )
        observed_left = np.asarray([left_assignment[label].center for label in self.model.labels])
        observed_right = np.asarray([right_assignment[label].center for label in self.model.labels])
        residuals = np.vstack(
            [left_projected.reshape(-1, 2) - observed_left, right_projected.reshape(-1, 2) - observed_right]
        )
        center_error = float(np.sqrt(np.mean(np.sum(residuals**2, axis=1))))
        radius_errors = []
        for index, sphere in enumerate(self.model.spheres):
            left_center = rotation @ sphere.center_mm + translation
            right_center = right_rotation @ sphere.center_mm + right_translation
            for center, detection, calibration in (
                (left_center, left_assignment[sphere.label], self.calibration.left),
                (right_center, right_assignment[sphere.label], self.calibration.right),
            ):
                predicted = projected_sphere_radius_px(center, sphere.diameter_mm, calibration)
                if not np.isfinite(predicted) or detection.radius <= 0.0:
                    return center_error, float("inf")
                radius_errors.append(abs(predicted - detection.radius) / detection.radius)
        return center_error, float(np.mean(radius_errors))

    def _estimate_stereo(
        self,
        left: list[SphereDetection],
        right: list[SphereDetection],
        predicted_rotation: np.ndarray | None,
        predicted_translation: np.ndarray | None,
    ) -> PoseEstimate | None:
        edges = self._epipolar_edges(left, right)
        matchings = []
        for matching in combinations(edges, 3):
            if len({item[1] for item in matching}) == 3 and len({item[2] for item in matching}) == 3:
                matchings.append(matching)
        matchings.sort(key=lambda value: sum(item[0] for item in value))
        best: tuple[
            float,
            PoseEstimate,
            dict[str, SphereDetection],
            dict[str, SphereDetection],
        ] | None = None
        for matching in matchings[:256]:
            points = self._triangulate(left, right, matching)
            if points is None:
                continue
            for model_order in permutations(range(3)):
                source = self.model.object_points[list(model_order)]
                try:
                    rotation, translation = _rigid_transform(source, points)
                except np.linalg.LinAlgError:
                    continue
                fitted = (rotation @ source.T).T + translation
                rigid_error = float(np.sqrt(np.mean(np.sum((fitted - points) ** 2, axis=1))))
                if not np.isfinite(rigid_error) or rigid_error > self.rigid_error_limit_mm:
                    continue
                left_assignment = {
                    self.model.labels[model_index]: left[matching[point_index][1]]
                    for point_index, model_index in enumerate(model_order)
                }
                right_assignment = {
                    self.model.labels[model_index]: right[matching[point_index][2]]
                    for point_index, model_index in enumerate(model_order)
                }
                center_error, radius_error = self._stereo_metrics(
                    rotation, translation, left_assignment, right_assignment
                )
                if (
                    not np.isfinite(center_error)
                    or center_error > self.center_error_limit_px
                    or radius_error > 0.35
                ):
                    continue
                score = center_error + rigid_error + 12.0 * radius_error
                score -= sum(item.score for item in left_assignment.values())
                score -= sum(item.score for item in right_assignment.values())
                if all(label in self.last_left_assignment for label in self.model.labels):
                    score += 0.04 * sum(
                        float(
                            np.linalg.norm(
                                left_assignment[label].center
                                - self.last_left_assignment[label].center
                            )
                        )
                        for label in self.model.labels
                    )
                if all(label in self.last_right_assignment for label in self.model.labels):
                    score += 0.04 * sum(
                        float(
                            np.linalg.norm(
                                right_assignment[label].center
                                - self.last_right_assignment[label].center
                            )
                        )
                        for label in self.model.labels
                    )
                if predicted_rotation is not None:
                    score += 2.0 * math.degrees(_rotation_distance(rotation, predicted_rotation))
                elif self.last_pose is not None:
                    previous_rotation, _ = cv2.Rodrigues(self.last_pose.rvec)
                    score += 0.5 * math.degrees(_rotation_distance(rotation, previous_rotation))
                if predicted_translation is not None:
                    score += 0.01 * float(np.linalg.norm(translation - predicted_translation))
                elif self.last_pose is not None:
                    score += 0.005 * float(
                        np.linalg.norm(translation - self.last_pose.tvec.reshape(3))
                    )
                rvec, _ = cv2.Rodrigues(rotation)
                estimate = PoseEstimate(
                    rvec,
                    translation.reshape(3, 1),
                    center_error,
                    radius_error,
                    "FULL",
                    3,
                    "STEREO",
                )
                if best is None or score < best[0]:
                    best = (score, estimate, left_assignment, right_assignment)
        if best is None:
            return None
        self.last_pose = best[1]
        self.last_left_assignment = best[2]
        self.last_right_assignment = best[3]
        self.left_mono.last_pose = best[1]
        self.left_mono.last_assignment = dict(best[2])
        best_rotation, _ = cv2.Rodrigues(best[1].rvec)
        right_rotation = self.calibration.right_from_left_rotation @ best_rotation
        right_translation = (
            self.calibration.right_from_left_rotation @ best[1].tvec.reshape(3)
            + self.calibration.right_from_left_translation_mm
        )
        right_rvec, _ = cv2.Rodrigues(right_rotation)
        self.right_mono.last_pose = PoseEstimate(
            right_rvec,
            right_translation.reshape(3, 1),
            best[1].reprojection_error_px,
            best[1].radius_error_fraction,
            "FULL",
            3,
        )
        self.right_mono.last_assignment = dict(best[3])
        return best[1]

    def _right_prediction(
        self,
        rotation: np.ndarray | None,
        translation: np.ndarray | None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        right_rotation = None
        right_translation = None
        if rotation is not None:
            right_rotation = self.calibration.right_from_left_rotation @ rotation
        if translation is not None:
            right_translation = (
                self.calibration.right_from_left_rotation @ np.asarray(translation).reshape(3)
                + self.calibration.right_from_left_translation_mm
            )
        return right_rotation, right_translation

    def _right_to_left(self, estimate: PoseEstimate) -> PoseEstimate:
        right_rotation, _ = cv2.Rodrigues(estimate.rvec)
        left_rotation = self.calibration.right_from_left_rotation.T @ right_rotation
        left_translation = self.calibration.right_from_left_rotation.T @ (
            estimate.tvec.reshape(3) - self.calibration.right_from_left_translation_mm
        )
        left_rvec, _ = cv2.Rodrigues(left_rotation)
        return PoseEstimate(
            left_rvec,
            left_translation.reshape(3, 1),
            estimate.reprojection_error_px,
            estimate.radius_error_fraction,
            estimate.state,
            estimate.visible_spheres,
            "RIGHT_MONO",
        )

    def estimate_camera_pose(
        self,
        left_detections: list[SphereDetection] | None,
        right_detections: list[SphereDetection] | None,
        predicted_rotation: np.ndarray | None = None,
        predicted_translation: np.ndarray | None = None,
    ) -> PoseEstimate | None:
        left = self._valid_detections(left_detections)
        right = self._valid_detections(right_detections)
        if len(left) >= 3 and len(right) >= 3:
            stereo = self._estimate_stereo(left, right, predicted_rotation, predicted_translation)
            if stereo is not None:
                return stereo

        candidates: list[tuple[float, PoseEstimate, str]] = []
        if len(left) >= 3:
            estimate = self.left_mono.estimate_camera_pose(
                left, predicted_rotation, predicted_translation
            )
            if estimate is not None:
                estimate = PoseEstimate(
                    estimate.rvec,
                    estimate.tvec,
                    estimate.reprojection_error_px,
                    estimate.radius_error_fraction,
                    estimate.state,
                    estimate.visible_spheres,
                    "LEFT_MONO",
                )
                candidates.append(
                    (estimate.reprojection_error_px + 12.0 * estimate.radius_error_fraction, estimate, "left")
                )
        if len(right) >= 3:
            right_rotation, right_translation = self._right_prediction(
                predicted_rotation, predicted_translation
            )
            estimate = self.right_mono.estimate_camera_pose(
                right, right_rotation, right_translation
            )
            if estimate is not None:
                converted = self._right_to_left(estimate)
                candidates.append(
                    (
                        converted.reprojection_error_px + 12.0 * converted.radius_error_fraction + 1e-9,
                        converted,
                        "right",
                    )
                )
        if not candidates:
            self.last_left_assignment = {}
            self.last_right_assignment = {}
            return None
        _, selected, _ = min(candidates, key=lambda item: item[0])
        self.last_left_assignment = dict(self.left_mono.last_assignment) if len(left) >= 3 else {}
        self.last_right_assignment = dict(self.right_mono.last_assignment) if len(right) >= 3 else {}
        self.last_pose = selected
        return selected
