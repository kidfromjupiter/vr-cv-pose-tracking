from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class SphereDetection:
    label: str
    center: np.ndarray
    radius: float
    area: float
    circularity: float
    score: float
    contour: np.ndarray


class SphereDetector:
    """Detect an unordered set of bright, low-saturation sphere candidates."""

    def __init__(self, max_candidates: int = 8) -> None:
        self.max_candidates = max_candidates
        self.open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    @staticmethod
    def make_mask(hsv: np.ndarray) -> tuple[np.ndarray, int]:
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        low_saturation = saturation <= 110
        sample = value[low_saturation]
        if sample.size >= 100:
            threshold, _ = cv2.threshold(
                sample.reshape(-1, 1),
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )
            value_threshold = int(np.clip(threshold, 140, 235))
        else:
            value_threshold = 190
        mask = np.where(low_saturation & (value >= value_threshold), 255, 0).astype(np.uint8)
        return mask, value_threshold

    def detect(self, frame: np.ndarray) -> tuple[list[SphereDetection], np.ndarray, int]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask, value_threshold = self.make_mask(hsv)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.close_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.open_kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[SphereDetection] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < 8.0 or area > 20000.0:
                continue
            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 0.0:
                continue
            circularity = float(4.0 * np.pi * area / (perimeter * perimeter))
            if circularity < 0.35:
                continue
            moments = cv2.moments(contour)
            if moments["m00"] <= 0.0:
                continue
            center = np.asarray(
                [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]],
                dtype=np.float64,
            )
            radius = float(np.sqrt(area / np.pi))
            contour_mask = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(contour_mask, [contour], -1, 255, -1)
            brightness = cv2.mean(hsv[:, :, 2], mask=contour_mask)[0] / 255.0
            whiteness = 1.0 - cv2.mean(hsv[:, :, 1], mask=contour_mask)[0] / 255.0
            area_score = min(1.0, area / 32.0)
            score = 0.40 * min(circularity, 1.0) + 0.30 * brightness + 0.20 * whiteness + 0.10 * area_score
            candidates.append(
                SphereDetection(
                    "unassigned",
                    center,
                    radius,
                    area,
                    circularity,
                    float(score),
                    contour,
                )
            )
        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates[: self.max_candidates], mask, value_threshold


LedDetection = SphereDetection
LedDetector = SphereDetector
