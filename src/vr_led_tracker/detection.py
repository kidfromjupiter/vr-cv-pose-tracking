from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import ColorProfile


@dataclass(frozen=True)
class BallDetection:
    center: np.ndarray
    radius: float
    area: float
    circularity: float
    score: float
    contour: np.ndarray


class BallDetector:
    def __init__(self, profile: ColorProfile, max_candidates: int = 8) -> None:
        self.profile = profile
        self.max_candidates = max_candidates
        self.open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def make_mask(self, hsv: np.ndarray) -> np.ndarray:
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in self.profile.hsv_ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
        return mask

    def detect(self, frame: np.ndarray) -> tuple[list[BallDetection], np.ndarray]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = self.make_mask(hsv)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.close_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.open_kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[BallDetection] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.profile.min_area_px or area > self.profile.max_area_px:
                continue
            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 0:
                continue
            circularity = float(4.0 * np.pi * area / (perimeter * perimeter))
            if circularity < self.profile.min_circularity:
                continue
            moments = cv2.moments(contour)
            if moments["m00"] <= 0:
                continue
            center = np.asarray(
                [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]],
                dtype=np.float64,
            )
            radius = float(np.sqrt(area / np.pi))
            contour_mask = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(contour_mask, [contour], -1, 255, -1)
            saturation = cv2.mean(hsv[:, :, 1], mask=contour_mask)[0] / 255.0
            brightness = cv2.mean(hsv[:, :, 2], mask=contour_mask)[0] / 255.0
            area_score = min(1.0, area / max(self.profile.min_area_px * 4.0, 1.0))
            score = (
                0.40 * min(circularity, 1.0)
                + 0.25 * saturation
                + 0.20 * brightness
                + 0.15 * area_score
            )
            candidates.append(
                BallDetection(center, radius, area, circularity, float(score), contour)
            )
        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates[: self.max_candidates], mask


SphereDetection = BallDetection
SphereDetector = BallDetector
