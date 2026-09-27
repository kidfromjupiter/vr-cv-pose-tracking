from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import ColorProfile


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
    def __init__(self, profiles: tuple[ColorProfile, ...]) -> None:
        self.profiles = profiles
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    @staticmethod
    def make_mask(hsv: np.ndarray, profile: ColorProfile) -> np.ndarray:
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in profile.hsv_ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
        return mask

    def detect(
        self,
        frame: np.ndarray,
        previous_pixels: dict[str, np.ndarray] | None = None,
    ) -> tuple[dict[str, SphereDetection], dict[str, np.ndarray]]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        previous_pixels = previous_pixels or {}
        all_candidates: list[SphereDetection] = []
        masks: dict[str, np.ndarray] = {}

        for profile in self.profiles:
            mask = self.make_mask(hsv, profile)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
            masks[profile.label] = mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = float(cv2.contourArea(contour))
                if area < profile.min_area_px or area > profile.max_area_px:
                    continue
                perimeter = float(cv2.arcLength(contour, True))
                if perimeter <= 0:
                    continue
                circularity = float(4.0 * np.pi * area / (perimeter * perimeter))
                if circularity < profile.min_circularity:
                    continue
                moments = cv2.moments(contour)
                if moments["m00"] <= 0:
                    continue
                center = np.asarray(
                    [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]],
                    dtype=np.float64,
                )
                radius = float(np.sqrt(area / np.pi))
                brightness_mask = np.zeros(mask.shape, dtype=np.uint8)
                cv2.drawContours(brightness_mask, [contour], -1, 255, -1)
                brightness = cv2.mean(hsv[:, :, 2], mask=brightness_mask)[0] / 255.0
                area_score = min(1.0, area / max(profile.min_area_px * 4.0, 1.0))
                score = 0.45 * min(circularity, 1.0) + 0.35 * brightness + 0.20 * area_score
                previous = previous_pixels.get(profile.label)
                if previous is not None:
                    distance = float(np.linalg.norm(center - previous))
                    score += 0.25 * np.exp(-distance / 80.0)
                all_candidates.append(
                    SphereDetection(
                        profile.label,
                        center,
                        radius,
                        area,
                        circularity,
                        float(score),
                        contour,
                    )
                )

        # Resolve overlapping color masks globally so one physical blob cannot be
        # assigned to two labels. Highest-quality candidates win.
        selected: dict[str, SphereDetection] = {}
        used_centers: list[np.ndarray] = []
        for candidate in sorted(all_candidates, key=lambda item: item.score, reverse=True):
            if candidate.label in selected:
                continue
            if any(np.linalg.norm(candidate.center - used) < max(5.0, candidate.radius) for used in used_centers):
                continue
            selected[candidate.label] = candidate
            used_centers.append(candidate.center)
        return selected, masks


# Retained as aliases for callers outside the package while the model format is
# intentionally migrated to spheres.
LedDetection = SphereDetection
LedDetector = SphereDetector
