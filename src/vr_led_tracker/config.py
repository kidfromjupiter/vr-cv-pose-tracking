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
class BallModel:
    label: str
    diameter_mm: float

    @classmethod
    def load(cls, path: str | Path) -> "BallModel":
        raw = _read_json(path)
        if "spheres" in raw or "leds" in raw:
            raise TrackerError("Legacy multi-marker models are unsupported; define one blue ball")
        item = raw.get("ball")
        if not isinstance(item, dict):
            raise TrackerError("Tracker model must define a ball object")
        label = item.get("label")
        if not isinstance(label, str) or label.strip().lower() != "blue":
            raise TrackerError("Ball label must be blue")
        try:
            diameter = float(item["diameter_mm"])
        except (KeyError, TypeError, ValueError) as exc:
            raise TrackerError("Ball needs a valid diameter_mm") from exc
        if not np.isfinite(diameter) or diameter <= 0:
            raise TrackerError("Ball diameter_mm must be positive")
        return cls("blue", diameter)


@dataclass(frozen=True)
class ColorProfile:
    label: str
    hsv_ranges: tuple[tuple[np.ndarray, np.ndarray], ...]
    min_area_px: float = 8.0
    max_area_px: float = 20000.0
    min_circularity: float = 0.35

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


def save_color_profile(path: str | Path, profile: ColorProfile) -> None:
    _write_json(path, {"profiles": [profile.to_json()]})


def load_color_profile(path: str | Path) -> ColorProfile:
    raw = _read_json(path)
    items = raw.get("profiles")
    if not isinstance(items, list) or len(items) != 1:
        raise TrackerError("Color calibration must contain exactly one profile")
    item = items[0]
    try:
        label = str(item["label"]).strip().lower()
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
            ranges.append((lower_values.astype(np.uint8), upper_values.astype(np.uint8)))
        min_area = float(item.get("min_area_px", 8.0))
        max_area = float(item.get("max_area_px", 20000.0))
        min_circularity = float(item.get("min_circularity", 0.35))
        if (
            label != "blue"
            or not ranges
            or not np.isfinite([min_area, max_area, min_circularity]).all()
            or min_area <= 0
            or max_area <= min_area
            or not 0 <= min_circularity <= 1
        ):
            raise ValueError
    except (KeyError, TypeError, ValueError) as exc:
        raise TrackerError("Invalid blue color profile") from exc
    return ColorProfile(label, tuple(ranges), min_area, max_area, min_circularity)


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
