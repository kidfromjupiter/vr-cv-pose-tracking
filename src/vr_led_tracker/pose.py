from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from itertools import combinations, permutations

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
        identity_max_speed_px_s: float = 6000.0,
        identity_reacquire_timeout_s: float = 0.25,
    ) -> None:
        self.model = model
        self.calibration = calibration
        self.center_error_limit_px = center_error_limit_px
        self.radius_error_limit = radius_error_limit
        self.identity_max_speed_px_s = float(identity_max_speed_px_s)
        self.identity_reacquire_timeout_s = float(identity_reacquire_timeout_s)
        self.last_pose: PoseEstimate | None = None
        self.last_assignment: dict[str, SphereDetection] = {}
        self.assignment_history: deque[tuple[float, dict[str, np.ndarray]]] = deque(maxlen=2)
        self.identity_gate_rejected = False
        self.identity_gate_detail = ""
        self.identity_gate_speed_px_s = 0.0

    def reset(self) -> None:
        self.last_pose = None
        self.last_assignment.clear()
        self.assignment_history.clear()
        self.identity_gate_rejected = False
        self.identity_gate_detail = ""
        self.identity_gate_speed_px_s = 0.0

    def _temporal_assignment_scores(
        self,
        assignments: list[dict[str, SphereDetection]],
        frame_time: float | None,
    ) -> tuple[list[dict[str, SphereDetection]], dict[tuple[int, ...], float]]:
        if (
            frame_time is None
            or not math.isfinite(frame_time)
            or not self.assignment_history
        ):
            return assignments, {}
        last_time, last_centers = self.assignment_history[-1]
        dt = frame_time - last_time
        if dt <= 1e-6:
            return assignments, {}
        if dt >= self.identity_reacquire_timeout_s:
            self.assignment_history.clear()
            self.last_pose = None
            self.identity_gate_detail = "identity history expired; using geometric reacquisition"
            return assignments, {}

        predicted = {label: center.copy() for label, center in last_centers.items()}
        if len(self.assignment_history) == 2:
            previous_time, previous_centers = self.assignment_history[0]
            history_dt = last_time - previous_time
            if history_dt > 1e-6:
                for label in self.model.labels:
                    velocity = (last_centers[label] - previous_centers[label]) / history_dt
                    predicted[label] = last_centers[label] + velocity * dt

        plausible: list[dict[str, SphereDetection]] = []
        prediction_scores: dict[tuple[int, ...], float] = {}
        minimum_max_speed = float("inf")
        for assignment in assignments:
            speeds = [
                float(np.linalg.norm(assignment[label].center - last_centers[label])) / dt
                for label in self.model.labels
            ]
            max_speed = max(speeds)
            minimum_max_speed = min(minimum_max_speed, max_speed)
            if max_speed > self.identity_max_speed_px_s:
                continue
            key = tuple(id(assignment[label]) for label in self.model.labels)
            prediction_scores[key] = sum(
                float(np.linalg.norm(assignment[label].center - predicted[label]))
                for label in self.model.labels
            )
            plausible.append(assignment)

        if not plausible:
            self.identity_gate_rejected = True
            self.identity_gate_speed_px_s = minimum_max_speed
            remaining_ms = max(0.0, self.identity_reacquire_timeout_s - dt) * 1000.0
            self.identity_gate_detail = (
                f"identity motion rejected: {minimum_max_speed:.0f} px/s; "
                f"reacquire in {remaining_ms:.0f} ms"
            )
        return plausible, prediction_scores

    def _record_assignment(
        self,
        frame_time: float | None,
        assignment: dict[str, SphereDetection],
    ) -> None:
        if frame_time is None or not math.isfinite(frame_time):
            return
        if self.assignment_history and frame_time <= self.assignment_history[-1][0]:
            return
        centers = {
            label: np.asarray(assignment[label].center, dtype=np.float64).copy()
            for label in self.model.labels
        }
        self.assignment_history.append((frame_time, centers))

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
        frame_time: float | None = None,
    ) -> PoseEstimate | None:
        previous_assignment = dict(self.last_assignment)
        self.last_assignment = {}
        self.identity_gate_rejected = False
        self.identity_gate_detail = ""
        self.identity_gate_speed_px_s = 0.0
        projected_prediction: np.ndarray | None = None
        prediction_scores: dict[tuple[int, ...], float] = {}
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
            assignments, prediction_scores = self._temporal_assignment_scores(
                assignments, frame_time
            )
            if not assignments:
                return None
            if self.identity_gate_detail:
                previous_assignment = {}
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
                temporal_key = tuple(id(assignment[label]) for label in self.model.labels)
                if temporal_key in prediction_scores:
                    total += 0.25 * prediction_scores[temporal_key]
                elif all(label in previous_assignment for label in self.model.labels):
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
        self._record_assignment(frame_time, best[2])
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
