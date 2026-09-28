from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .errors import TrackerError


def _read_json(path: str | Path) -> dict[str, Any]:
    try:
        with Path(path).open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise TrackerError(f"Cannot read JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TrackerError(f"{path} must contain a JSON object")
    return value


def _write_json(path: str | Path, value: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2)
            handle.write("\n")
    except OSError as exc:
        raise TrackerError(f"Cannot write {path}: {exc}") from exc


@dataclass(frozen=True)
class SphereDefinition:
    label: str
    center_mm: np.ndarray
    diameter_mm: float


@dataclass(frozen=True)
class ControllerModel:
    spheres: tuple[SphereDefinition, ...]

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(sphere.label for sphere in self.spheres)

    @property
    def object_points(self) -> np.ndarray:
        return np.asarray([sphere.center_mm for sphere in self.spheres], dtype=np.float64)

    @property
    def diameters_mm(self) -> np.ndarray:
        return np.asarray([sphere.diameter_mm for sphere in self.spheres], dtype=np.float64)

    @classmethod
    def load(cls, path: str | Path) -> "ControllerModel":
        raw = _read_json(path)
        if "leds" in raw:
            raise TrackerError("Legacy LED models are unsupported; define three white spheres")
        raw_spheres = raw.get("spheres")
        if not isinstance(raw_spheres, list) or len(raw_spheres) != 3:
            raise TrackerError("Controller model must define exactly three spheres")
        spheres: list[SphereDefinition] = []
        for index, item in enumerate(raw_spheres):
            if not isinstance(item, dict):
                raise TrackerError(f"Sphere {index} must be an object")
            label = item.get("label")
            center = item.get("center_mm")
            if not isinstance(label, str) or not label.strip():
                raise TrackerError(f"Sphere {index} needs a non-empty label")
            try:
                point = np.asarray(center, dtype=np.float64)
                diameter = float(item["diameter_mm"])
            except (TypeError, ValueError) as exc:
                raise TrackerError(f"Sphere {label} has invalid geometry") from exc
            except KeyError as exc:
                raise TrackerError(f"Sphere {label} needs diameter_mm") from exc
            if point.shape != (3,) or not np.isfinite(point).all():
                raise TrackerError(f"Sphere {label} center_mm must contain three finite numbers")
            if not np.isfinite(diameter) or diameter <= 0:
                raise TrackerError(f"Sphere {label} diameter_mm must be positive")
            spheres.append(SphereDefinition(label.strip().lower(), point, diameter))

        labels = [sphere.label for sphere in spheres]
        required_labels = {"sphere_0", "sphere_1", "sphere_2"}
        if set(labels) != required_labels or len(set(labels)) != 3:
            raise TrackerError("Sphere labels must be exactly sphere_0, sphere_1, and sphere_2")
        by_label = {sphere.label: sphere for sphere in spheres}
        spheres = [by_label[label] for label in ("sphere_0", "sphere_1", "sphere_2")]
        points = np.asarray([sphere.center_mm for sphere in spheres])
        distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
        if np.any((distances + np.eye(3)) < 1e-6):
            raise TrackerError("Sphere centers must be distinct")
        extent = float(np.max(np.ptp(points, axis=0)))
        area2 = float(np.linalg.norm(np.cross(points[1] - points[0], points[2] - points[0])))
        if extent <= 0 or area2 / (extent**2) < 0.01:
            raise TrackerError("Sphere centers are collinear or nearly collinear")
        pair_distances = np.sort(distances[np.triu_indices(3, 1)])
        relative_separation = np.diff(pair_distances) / pair_distances[-1]
        if np.min(relative_separation) < 0.05:
            raise TrackerError(
                "White-sphere geometry is too symmetric; pair distances must differ by at least 5%"
            )
        return cls(tuple(spheres))


@dataclass(frozen=True)
class CameraCalibration:
    image_size: tuple[int, int]
    camera_matrix: np.ndarray
    distortion: np.ndarray
    rms_error: float
    board: dict[str, float | int]

    def save(self, path: str | Path) -> None:
        _write_json(
            path,
            {
                "image_size": list(self.image_size),
                "camera_matrix": self.camera_matrix.tolist(),
                "distortion": self.distortion.reshape(-1).tolist(),
                "rms_error": self.rms_error,
                "board": self.board,
            },
        )

    @classmethod
    def load(cls, path: str | Path) -> "CameraCalibration":
        raw = _read_json(path)
        try:
            size = tuple(int(v) for v in raw["image_size"])
            matrix = np.asarray(raw["camera_matrix"], dtype=np.float64)
            distortion = np.asarray(raw["distortion"], dtype=np.float64).reshape(-1, 1)
            rms = float(raw["rms_error"])
            board = dict(raw.get("board", {}))
        except (KeyError, TypeError, ValueError) as exc:
            raise TrackerError(f"Invalid camera calibration file {path}") from exc
        if len(size) != 2 or min(size) <= 0 or matrix.shape != (3, 3):
            raise TrackerError(f"Invalid camera calibration dimensions in {path}")
        if distortion.size < 4 or not np.isfinite(matrix).all() or not np.isfinite(distortion).all():
            raise TrackerError(f"Invalid camera calibration values in {path}")
        return cls((size[0], size[1]), matrix, distortion, rms, board)

    def for_image_size(self, size: tuple[int, int]) -> "CameraCalibration":
        if size == self.image_size:
            return self
        old_aspect = self.image_size[0] / self.image_size[1]
        new_aspect = size[0] / size[1]
        if abs(old_aspect - new_aspect) > 0.01:
            raise TrackerError(
                f"Camera was calibrated at {self.image_size[0]}x{self.image_size[1]}, "
                f"but the camera is {size[0]}x{size[1]} with a different aspect ratio"
            )
        sx = size[0] / self.image_size[0]
        sy = size[1] / self.image_size[1]
        matrix = self.camera_matrix.copy()
        matrix[0, :] *= sx
        matrix[1, :] *= sy
        matrix[2, 2] = 1.0
        return CameraCalibration(size, matrix, self.distortion.copy(), self.rms_error, self.board)
