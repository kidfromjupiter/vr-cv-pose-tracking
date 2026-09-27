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
            raise TrackerError(
                "Legacy four-LED models are unsupported; define red, blue, and white spheres"
            )
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
        if set(labels) != {"red", "blue", "white"} or len(set(labels)) != 3:
            raise TrackerError("Sphere labels must be exactly red, blue, and white")
        by_label = {sphere.label: sphere for sphere in spheres}
        spheres = [by_label[label] for label in ("red", "blue", "white")]
        points = np.asarray([sphere.center_mm for sphere in spheres])
        distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
        if np.any((distances + np.eye(3)) < 1e-6):
            raise TrackerError("Sphere centers must be distinct")
        extent = float(np.max(np.ptp(points, axis=0)))
        area2 = float(np.linalg.norm(np.cross(points[1] - points[0], points[2] - points[0])))
        if extent <= 0 or area2 / (extent**2) < 0.01:
            raise TrackerError("Sphere centers are collinear or nearly collinear")
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
                f"but DroidCam is {size[0]}x{size[1]} with a different aspect ratio"
            )
        sx = size[0] / self.image_size[0]
        sy = size[1] / self.image_size[1]
        matrix = self.camera_matrix.copy()
        matrix[0, :] *= sx
        matrix[1, :] *= sy
        matrix[2, 2] = 1.0
        return CameraCalibration(size, matrix, self.distortion.copy(), self.rms_error, self.board)


@dataclass(frozen=True)
class ColorProfile:
    label: str
    hsv_ranges: tuple[tuple[np.ndarray, np.ndarray], ...]
    min_area_px: float = 6.0
    max_area_px: float = 20000.0
    min_circularity: float = 0.25

    def to_json(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "hsv_ranges": [
                {"lower": lower.tolist(), "upper": upper.tolist()}
                for lower, upper in self.hsv_ranges
            ],
            "min_area_px": self.min_area_px,
            "max_area_px": self.max_area_px,
            "min_circularity": self.min_circularity,
        }


def save_color_profiles(path: str | Path, profiles: list[ColorProfile]) -> None:
    _write_json(path, {"profiles": [profile.to_json() for profile in profiles]})


def load_color_profiles(path: str | Path, expected_labels: tuple[str, ...]) -> tuple[ColorProfile, ...]:
    raw = _read_json(path)
    items = raw.get("profiles")
    if not isinstance(items, list):
        raise TrackerError("Color calibration must contain a profiles list")
    profiles: list[ColorProfile] = []
    for item in items:
        try:
            label = str(item["label"])
            ranges = []
            for hsv_range in item["hsv_ranges"]:
                lower_values = np.asarray(hsv_range["lower"], dtype=np.float64)
                upper_values = np.asarray(hsv_range["upper"], dtype=np.float64)
                if (
                    lower_values.shape != (3,)
                    or upper_values.shape != (3,)
                    or not np.isfinite(lower_values).all()
                    or not np.isfinite(upper_values).all()
                    or np.any(lower_values < 0)
                    or np.any(upper_values > [179, 255, 255])
                    or np.any(lower_values > upper_values)
                ):
                    raise ValueError
                lower = lower_values.astype(np.uint8)
                upper = upper_values.astype(np.uint8)
                ranges.append((lower, upper))
            min_area = float(item.get("min_area_px", 6.0))
            max_area = float(item.get("max_area_px", 20000.0))
            min_circularity = float(item.get("min_circularity", 0.25))
            if min_area <= 0 or max_area <= min_area or not 0 <= min_circularity <= 1:
                raise ValueError
            profile = ColorProfile(
                label,
                tuple(ranges),
                min_area,
                max_area,
                min_circularity,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise TrackerError("Invalid color profile") from exc
        if not profile.hsv_ranges:
            raise TrackerError(f"Color profile {label} has no HSV ranges")
        profiles.append(profile)
    labels = tuple(profile.label for profile in profiles)
    if len(set(labels)) != len(labels) or set(labels) != set(expected_labels):
        raise TrackerError("Color profile labels must exactly match the controller model")
    by_label = {profile.label: profile for profile in profiles}
    return tuple(by_label[label] for label in expected_labels)
