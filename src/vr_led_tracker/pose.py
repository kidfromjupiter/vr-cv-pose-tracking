from __future__ import annotations

import math
from collections import deque
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
    camera_measurement_scale: float = 1.0
    triangulation_angle_deg: float | None = None


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


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float64).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _rigid_transform(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_center, target_center = np.mean(source, axis=0), np.mean(target, axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    left, _, right = np.linalg.svd(covariance)
    rotation = right.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right[-1] *= -1.0
        rotation = right.T @ left.T
    return rotation, target_center - rotation @ source_center


class StereoPoseEstimator:
    """Triangulate three sphere centers and express the pose in the left-camera frame."""

    def __init__(
        self, model: ControllerModel, calibration: StereoCalibration,
        center_error_limit_px: float = 5.0, epipolar_error_limit_px: float = 3.0,
        rigid_error_limit_mm: float = 10.0, min_triangulation_angle_deg: float = 1.5,
        quality_reference_area_px2: float = 256.0, quality_reference_angle_deg: float = 8.0,
        max_stereo_measurement_scale: float = 4.0, mono_base_measurement_scale: float = 2.0,
        max_mono_measurement_scale: float = 6.0, identity_max_speed_px_s: float = 6000.0,
        identity_reacquire_timeout_s: float = 0.25, allow_mono_fallback: bool = True,
    ) -> None:
        self.model, self.calibration = model, calibration
        self.center_error_limit_px = float(center_error_limit_px)
        self.epipolar_error_limit_px = float(epipolar_error_limit_px)
        self.rigid_error_limit_mm = float(rigid_error_limit_mm)
        self.min_triangulation_angle_deg = float(min_triangulation_angle_deg)
        self.quality_reference_area_px2 = float(quality_reference_area_px2)
        self.quality_reference_angle_deg = float(quality_reference_angle_deg)
        self.max_stereo_measurement_scale = float(max_stereo_measurement_scale)
        self.mono_base_measurement_scale = float(mono_base_measurement_scale)
        self.max_mono_measurement_scale = float(max_mono_measurement_scale)
        self.allow_mono_fallback = bool(allow_mono_fallback)
        mono_args = (identity_max_speed_px_s, identity_reacquire_timeout_s)
        self.left_mono = PoseEstimator(model, calibration.left, center_error_limit_px, 0.35, *mono_args)
        self.right_mono = PoseEstimator(model, calibration.right, center_error_limit_px, 0.35, *mono_args)
        self.last_pose: PoseEstimate | None = None
        self.last_left_assignment: dict[str, SphereDetection] = {}
        self.last_right_assignment: dict[str, SphereDetection] = {}
        essential = _skew(calibration.right_from_left_translation_mm) @ calibration.right_from_left_rotation
        self.fundamental = np.linalg.inv(calibration.right.camera_matrix).T @ essential @ np.linalg.inv(calibration.left.camera_matrix)

    def reset(self) -> None:
        self.left_mono.reset(); self.right_mono.reset()
        self.last_pose = None; self.last_left_assignment.clear(); self.last_right_assignment.clear()

    @staticmethod
    def _valid(detections: list[SphereDetection] | None) -> list[SphereDetection]:
        return [] if detections is None else [d for d in detections[:8] if np.asarray(d.center).shape == (2,) and np.isfinite(d.center).all()]

    def _undistorted(self, detections, calibration, pixels=True):
        points = np.asarray([d.center for d in detections], dtype=np.float64).reshape(-1, 1, 2)
        return cv2.undistortPoints(points, calibration.camera_matrix, calibration.distortion,
                                   P=calibration.camera_matrix if pixels else None).reshape(-1, 2)

    def _edges(self, left, right):
        lp, rp = self._undistorted(left, self.calibration.left), self._undistorted(right, self.calibration.right)
        edges = []
        for li, point_l in enumerate(lp):
            line_r = self.fundamental @ np.r_[point_l, 1.0]
            for ri, point_r in enumerate(rp):
                hr = np.r_[point_r, 1.0]; line_l = self.fundamental.T @ hr
                numerator = abs(float(hr @ line_r))
                denominators = (np.linalg.norm(line_r[:2]), np.linalg.norm(line_l[:2]))
                if min(denominators) <= 1e-12: continue
                error = 0.5 * numerator * (1.0 / denominators[0] + 1.0 / denominators[1])
                if error <= self.epipolar_error_limit_px: edges.append((error, li, ri))
        return edges

    def _triangulate(self, left, right, matching):
        ln = self._undistorted([left[e[1]] for e in matching], self.calibration.left, False)
        rn = self._undistorted([right[e[2]] for e in matching], self.calibration.right, False)
        p1 = np.hstack([np.eye(3), np.zeros((3, 1))])
        p2 = np.hstack([self.calibration.right_from_left_rotation,
                        self.calibration.right_from_left_translation_mm.reshape(3, 1)])
        homogeneous = cv2.triangulatePoints(p1, p2, ln.T, rn.T)
        if np.any(np.abs(homogeneous[3]) <= 1e-9): return None
        points = (homogeneous[:3] / homogeneous[3]).T
        right_points = (self.calibration.right_from_left_rotation @ points.T).T + self.calibration.right_from_left_translation_mm
        if not np.isfinite(points).all() or np.any(points[:, 2] <= 100) or np.any(points[:, 2] >= 5000) or np.any(right_points[:, 2] <= 100) or np.any(right_points[:, 2] >= 5000):
            return None
        camera_right = -self.calibration.right_from_left_rotation.T @ self.calibration.right_from_left_translation_mm
        angles = []
        for point in points:
            ray_l = point / np.linalg.norm(point); ray_r = (point - camera_right) / np.linalg.norm(point - camera_right)
            angles.append(math.degrees(math.acos(np.clip(float(ray_l @ ray_r), -1, 1))))
        return points, min(angles)

    def _metrics(self, rotation, translation, la, ra):
        rr = self.calibration.right_from_left_rotation @ rotation
        rt = self.calibration.right_from_left_rotation @ translation + self.calibration.right_from_left_translation_mm
        lrvec = cv2.Rodrigues(rotation)[0]; rrvec = cv2.Rodrigues(rr)[0]
        lp = cv2.projectPoints(self.model.object_points, lrvec, translation, self.calibration.left.camera_matrix, self.calibration.left.distortion)[0].reshape(-1, 2)
        rp = cv2.projectPoints(self.model.object_points, rrvec, rt, self.calibration.right.camera_matrix, self.calibration.right.distortion)[0].reshape(-1, 2)
        observed_l = np.asarray([la[label].center for label in self.model.labels])
        observed_r = np.asarray([ra[label].center for label in self.model.labels])
        center_error = float(np.sqrt(np.mean(np.sum(np.vstack([lp-observed_l, rp-observed_r]) ** 2, axis=1))))
        radius_errors = []
        for i, sphere in enumerate(self.model.spheres):
            for center, detection, cal in ((rotation @ sphere.center_mm + translation, la[sphere.label], self.calibration.left), (rr @ sphere.center_mm + rt, ra[sphere.label], self.calibration.right)):
                radius = projected_sphere_radius_px(center, sphere.diameter_mm, cal)
                if not np.isfinite(radius) or detection.radius <= 0: return center_error, float("inf")
                radius_errors.append(abs(radius-detection.radius)/detection.radius)
        return center_error, float(np.mean(radius_errors))

    def _quality(self, reprojection, angle, assignments, frame_skew_s, max_frame_skew_s, mono=False):
        reproj = 1.0 + (reprojection / max(self.center_error_limit_px, 1e-6)) ** 2
        mean_area = max(float(np.mean([d.area for a in assignments for d in a.values()])), 1e-6)
        area = max(1.0, math.sqrt(self.quality_reference_area_px2 / mean_area))
        if mono: return min(self.max_mono_measurement_scale, self.mono_base_measurement_scale * reproj * area)
        angle_factor = max(1.0, math.sin(math.radians(self.quality_reference_angle_deg)) / max(math.sin(math.radians(angle)), 1e-6))
        skew = 1.0 + max(frame_skew_s, 0.0) / max(max_frame_skew_s, 1e-6)
        return min(self.max_stereo_measurement_scale, reproj * area * angle_factor * skew)

    def _estimate_stereo(self, left, right, predicted_rotation, predicted_translation, frame_time, frame_skew_s, max_frame_skew_s):
        matchings = [m for m in combinations(self._edges(left, right), 3) if len({x[1] for x in m}) == 3 and len({x[2] for x in m}) == 3]
        matchings.sort(key=lambda m: sum(x[0] for x in m))
        candidates = []
        for matching in matchings[:256]:
            result = self._triangulate(left, right, matching)
            if result is None: continue
            points, angle = result
            if angle < self.min_triangulation_angle_deg: continue
            for order in permutations(range(3)):
                source = self.model.object_points[list(order)]
                try: rotation, translation = _rigid_transform(source, points)
                except np.linalg.LinAlgError: continue
                rigid = float(np.sqrt(np.mean(np.sum(((rotation @ source.T).T + translation - points) ** 2, axis=1))))
                if not np.isfinite(rigid) or rigid > self.rigid_error_limit_mm: continue
                la = {self.model.labels[mi]: left[matching[pi][1]] for pi, mi in enumerate(order)}
                ra = {self.model.labels[mi]: right[matching[pi][2]] for pi, mi in enumerate(order)}
                reproj, radius = self._metrics(rotation, translation, la, ra)
                if not np.isfinite(reproj) or reproj > self.center_error_limit_px or radius > 0.35: continue
                score = reproj + rigid + 12*radius - sum(d.score for d in la.values()) - sum(d.score for d in ra.values())
                if predicted_rotation is not None: score += 2*math.degrees(_rotation_distance(rotation, predicted_rotation))
                if predicted_translation is not None: score += .01*float(np.linalg.norm(translation-predicted_translation))
                scale = self._quality(reproj, angle, (la, ra), frame_skew_s, max_frame_skew_s)
                estimate = PoseEstimate(cv2.Rodrigues(rotation)[0], translation.reshape(3,1), reproj, radius, "FULL", 3, "STEREO", scale, angle)
                candidates.append((score, estimate, la, ra))
        if not candidates: return None
        left_allowed, left_scores = self.left_mono._temporal_assignment_scores([x[2] for x in candidates], frame_time)
        right_allowed, right_scores = self.right_mono._temporal_assignment_scores([x[3] for x in candidates], frame_time)
        left_keys = {tuple(id(a[l]) for l in self.model.labels) for a in left_allowed}
        right_keys = {tuple(id(a[l]) for l in self.model.labels) for a in right_allowed}
        plausible = []
        for item in candidates:
            lk = tuple(id(item[2][l]) for l in self.model.labels); rk = tuple(id(item[3][l]) for l in self.model.labels)
            if lk in left_keys and rk in right_keys: plausible.append((item[0] + .25*left_scores.get(lk,0) + .25*right_scores.get(rk,0), *item[1:]))
        if not plausible: return None
        _, estimate, la, ra = min(plausible, key=lambda x: x[0])
        self.last_pose = estimate; self.last_left_assignment = dict(la); self.last_right_assignment = dict(ra)
        self.left_mono.last_pose = estimate; self.left_mono.last_assignment = dict(la); self.left_mono._record_assignment(frame_time, la)
        right_rotation = self.calibration.right_from_left_rotation @ cv2.Rodrigues(estimate.rvec)[0]
        right_translation = self.calibration.right_from_left_rotation @ estimate.tvec.reshape(3) + self.calibration.right_from_left_translation_mm
        self.right_mono.last_pose = PoseEstimate(cv2.Rodrigues(right_rotation)[0], right_translation.reshape(3,1), estimate.reprojection_error_px, estimate.radius_error_fraction, "FULL", 3)
        self.right_mono.last_assignment = dict(ra); self.right_mono._record_assignment(frame_time, ra)
        return estimate

    def estimate_camera_pose(self, left_detections, right_detections, predicted_rotation=None, predicted_translation=None, frame_time=None, frame_skew_s=0.0, max_frame_skew_s=0.02):
        left, right = self._valid(left_detections), self._valid(right_detections)
        if len(left) >= 3 and len(right) >= 3:
            estimate = self._estimate_stereo(left, right, predicted_rotation, predicted_translation, frame_time, frame_skew_s, max_frame_skew_s)
            if estimate is not None: return estimate
        if not self.allow_mono_fallback: return None
        choices = []
        if len(left) >= 3:
            e = self.left_mono.estimate_camera_pose(left, predicted_rotation, predicted_translation, frame_time)
            if e is not None:
                scale = self._quality(e.reprojection_error_px, 0, (self.left_mono.last_assignment,), 0, 1, True)
                choices.append((e.reprojection_error_px+12*e.radius_error_fraction, PoseEstimate(e.rvec,e.tvec,e.reprojection_error_px,e.radius_error_fraction,e.state,e.visible_spheres,"LEFT_MONO",scale)))
        if len(right) >= 3:
            rp = None if predicted_rotation is None else self.calibration.right_from_left_rotation @ predicted_rotation
            tp = None if predicted_translation is None else self.calibration.right_from_left_rotation @ predicted_translation + self.calibration.right_from_left_translation_mm
            e = self.right_mono.estimate_camera_pose(right, rp, tp, frame_time)
            if e is not None:
                rr = cv2.Rodrigues(e.rvec)[0]; lr = self.calibration.right_from_left_rotation.T @ rr
                lt = self.calibration.right_from_left_rotation.T @ (e.tvec.reshape(3)-self.calibration.right_from_left_translation_mm)
                scale = self._quality(e.reprojection_error_px, 0, (self.right_mono.last_assignment,), 0, 1, True)
                choices.append((e.reprojection_error_px+12*e.radius_error_fraction+1e-9, PoseEstimate(cv2.Rodrigues(lr)[0],lt.reshape(3,1),e.reprojection_error_px,e.radius_error_fraction,e.state,e.visible_spheres,"RIGHT_MONO",scale)))
        if not choices:
            self.last_left_assignment.clear(); self.last_right_assignment.clear(); return None
        estimate = min(choices, key=lambda x:x[0])[1]
        self.last_left_assignment = dict(self.left_mono.last_assignment) if len(left) >= 3 else {}
        self.last_right_assignment = dict(self.right_mono.last_assignment) if len(right) >= 3 else {}
        self.last_pose = estimate
        return estimate
