from __future__ import annotations

import cv2
import numpy as np

from .camera import open_camera, read_frame
from .config import ColorProfile, save_color_profile
from .errors import TrackerError


def profile_from_samples(samples: np.ndarray) -> ColorProfile:
    samples = np.asarray(samples, dtype=np.float64).reshape(-1, 3)
    if len(samples) < 10 or not np.isfinite(samples).all():
        raise TrackerError("Not enough valid blue color samples")
    hues = samples[:, 0]
    angles = hues * (2.0 * np.pi / 180.0)
    center = (
        np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
        * 180.0
        / (2.0 * np.pi)
    ) % 180.0
    hue_delta = ((hues - center + 90.0) % 180.0) - 90.0
    margin = float(np.clip(np.percentile(np.abs(hue_delta), 90) + 6.0, 4.0, 30.0))
    saturation_low = int(np.clip(np.percentile(samples[:, 1], 10) - 35, 20, 220))
    value_low = int(np.clip(np.percentile(samples[:, 2], 10) - 45, 20, 220))
    lower_hue = center - margin
    upper_hue = center + margin
    ranges: list[tuple[np.ndarray, np.ndarray]] = []
    if lower_hue < 0:
        ranges.append(
            (
                np.array([0, saturation_low, value_low], np.uint8),
                np.array([upper_hue, 255, 255], np.uint8),
            )
        )
        ranges.append(
            (
                np.array([180 + lower_hue, saturation_low, value_low], np.uint8),
                np.array([179, 255, 255], np.uint8),
            )
        )
    elif upper_hue > 179:
        ranges.append(
            (
                np.array([lower_hue, saturation_low, value_low], np.uint8),
                np.array([179, 255, 255], np.uint8),
            )
        )
        ranges.append(
            (
                np.array([0, saturation_low, value_low], np.uint8),
                np.array([upper_hue - 180, 255, 255], np.uint8),
            )
        )
    else:
        ranges.append(
            (
                np.array([lower_hue, saturation_low, value_low], np.uint8),
                np.array([upper_hue, 255, 255], np.uint8),
            )
        )
    return ColorProfile("blue", tuple(ranges))


class _Sampler:
    def __init__(self, required_clicks: int) -> None:
        self.required_clicks = required_clicks
        self.current_hsv: np.ndarray | None = None
        self.samples: list[np.ndarray] = []
        self.clicks: list[tuple[int, int]] = []

    def reset(self) -> None:
        self.samples.clear()
        self.clicks.clear()

    def callback(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if (
            event != cv2.EVENT_LBUTTONDOWN
            or self.current_hsv is None
            or len(self.samples) >= self.required_clicks
        ):
            return
        height, width = self.current_hsv.shape[:2]
        x0, x1 = max(0, x - 5), min(width, x + 6)
        y0, y1 = max(0, y - 5), min(height, y + 6)
        patch = self.current_hsv[y0:y1, x0:x1].reshape(-1, 3)
        if patch.size == 0:
            return
        bright = patch[patch[:, 2] >= np.percentile(patch[:, 2], 55)]
        if bright.size:
            self.samples.append(bright.copy())
            self.clicks.append((x, y))


def run_color_calibration(
    device: str,
    output: str,
    required_clicks: int = 10,
) -> ColorProfile:
    if required_clicks <= 0:
        raise TrackerError("Color sample count must be positive")
    capture = open_camera(device)
    sampler = _Sampler(required_clicks)
    window = "Blue ball color calibration"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, sampler.callback)
    try:
        while len(sampler.samples) < required_clicks:
            frame = read_frame(capture)
            sampler.current_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            display = frame.copy()
            for point in sampler.clicks:
                cv2.circle(display, point, 8, (255, 120, 20), 2, cv2.LINE_AA)
            lines = [
                f"Sample blue ball ({len(sampler.samples)}/{required_clicks})",
                "Click inside the ball in different frames and positions",
                "R restart | Q cancel",
            ]
            for index, line in enumerate(lines):
                cv2.putText(
                    display,
                    line,
                    (12, 28 + index * 27),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (30, 240, 30),
                    2,
                    cv2.LINE_AA,
                )
            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                raise TrackerError("Color calibration cancelled")
            if key == ord("r"):
                sampler.reset()
    finally:
        capture.release()
        cv2.destroyWindow(window)
    profile = profile_from_samples(np.concatenate(sampler.samples, axis=0))
    save_color_profile(output, profile)
    return profile
