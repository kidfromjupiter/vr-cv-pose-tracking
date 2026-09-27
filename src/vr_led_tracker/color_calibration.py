from __future__ import annotations

import cv2
import numpy as np

from .camera import open_camera, read_frame
from .config import ColorProfile, ControllerModel, save_color_profiles
from .errors import TrackerError


def profile_from_samples(label: str, samples: np.ndarray) -> ColorProfile:
    samples = np.asarray(samples, dtype=np.float64).reshape(-1, 3)
    if len(samples) < 10:
        raise TrackerError(f"Not enough color samples for {label}")
    if label.lower() == "white":
        saturation_high = int(np.clip(np.percentile(samples[:, 1], 90) + 25, 20, 150))
        value_low = int(np.clip(np.percentile(samples[:, 2], 10) - 35, 80, 190))
        return ColorProfile(
            label,
            ((np.array([0, 0, value_low], np.uint8),
              np.array([179, saturation_high, 255], np.uint8)),),
            min_circularity=0.45,
        )
    hues = samples[:, 0]
    angles = hues * (2.0 * np.pi / 180.0)
    center = (np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles))) * 180.0 / (2 * np.pi)) % 180
    hue_delta = ((hues - center + 90.0) % 180.0) - 90.0
    margin = float(np.clip(np.percentile(np.abs(hue_delta), 90) + 6.0, 4.0, 30.0))
    saturation_low = int(np.clip(np.percentile(samples[:, 1], 10) - 35, 0, 180))
    value_low = int(np.clip(np.percentile(samples[:, 2], 10) - 45, 0, 160))
    lower_hue = center - margin
    upper_hue = center + margin
    ranges: list[tuple[np.ndarray, np.ndarray]] = []
    if lower_hue < 0:
        ranges.append((np.array([0, saturation_low, value_low], np.uint8), np.array([upper_hue, 255, 255], np.uint8)))
        ranges.append((np.array([180 + lower_hue, saturation_low, value_low], np.uint8), np.array([179, 255, 255], np.uint8)))
    elif upper_hue > 179:
        ranges.append((np.array([lower_hue, saturation_low, value_low], np.uint8), np.array([179, 255, 255], np.uint8)))
        ranges.append((np.array([0, saturation_low, value_low], np.uint8), np.array([upper_hue - 180, 255, 255], np.uint8)))
    else:
        ranges.append((np.array([lower_hue, saturation_low, value_low], np.uint8), np.array([upper_hue, 255, 255], np.uint8)))
    return ColorProfile(label, tuple(ranges))


class _Sampler:
    def __init__(self, labels: tuple[str, ...], clicks_per_label: int = 10) -> None:
        self.labels = labels
        self.clicks_per_label = clicks_per_label
        self.index = 0
        self.current_hsv: np.ndarray | None = None
        self.samples: dict[str, list[np.ndarray]] = {label: [] for label in labels}
        self.clicks: dict[str, list[tuple[int, int]]] = {label: [] for label in labels}

    @property
    def complete(self) -> bool:
        return self.index >= len(self.labels)

    @property
    def label(self) -> str:
        return self.labels[min(self.index, len(self.labels) - 1)]

    def reset_current(self) -> None:
        if self.complete:
            self.index = len(self.labels) - 1
        self.samples[self.label].clear()
        self.clicks[self.label].clear()

    def callback(self, event: int, x: int, y: int, _flags: int, _param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN or self.complete or self.current_hsv is None:
            return
        height, width = self.current_hsv.shape[:2]
        x0, x1 = max(0, x - 5), min(width, x + 6)
        y0, y1 = max(0, y - 5), min(height, y + 6)
        patch = self.current_hsv[y0:y1, x0:x1].reshape(-1, 3)
        if patch.size == 0:
            return
        bright = patch[patch[:, 2] >= np.percentile(patch[:, 2], 55)]
        self.samples[self.label].append(bright.copy())
        self.clicks[self.label].append((x, y))
        if len(self.samples[self.label]) >= self.clicks_per_label:
            self.index += 1


def run_color_calibration(
    device: str,
    model: ControllerModel,
    output: str,
    samples_per_color: int = 10,
) -> list[ColorProfile]:
    if samples_per_color <= 0:
        raise TrackerError("Color samples per sphere must be positive")
    capture = open_camera(device)
    sampler = _Sampler(model.labels, samples_per_color)
    window = "Sphere color calibration"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, sampler.callback)
    palette = [(20, 20, 240), (240, 100, 20), (245, 245, 245)]
    try:
        while not sampler.complete:
            frame = read_frame(capture)
            sampler.current_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            display = frame.copy()
            for index, label in enumerate(model.labels):
                for point in sampler.clicks[label]:
                    cv2.circle(display, point, 8, palette[index], 2, cv2.LINE_AA)
            count = len(sampler.samples[sampler.label])
            pixel_count = sum(len(patch) for patch in sampler.samples[sampler.label])
            lines = [
                f"Sample: {sampler.label} ({count}/{sampler.clicks_per_label})",
                f"Collected HSV pixels: {pixel_count}",
                "Click inside the sphere from different frames and angles",
                "R redo current color | Q cancel",
            ]
            y = 28
            for line in lines:
                cv2.putText(display, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 240, 30), 2, cv2.LINE_AA)
                y += 27
            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                raise TrackerError("Color calibration cancelled")
            if key == ord("r"):
                sampler.reset_current()
    finally:
        capture.release()
        cv2.destroyWindow(window)

    profiles = [
        profile_from_samples(label, np.concatenate(sampler.samples[label], axis=0))
        for label in model.labels
    ]
    save_color_profiles(output, profiles)
    return profiles
