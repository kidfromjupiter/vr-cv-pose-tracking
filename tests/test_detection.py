from __future__ import annotations

import cv2
import numpy as np

from vr_led_tracker.color_calibration import profile_from_samples
from vr_led_tracker.config import ColorProfile
from vr_led_tracker.detection import SphereDetector


def test_color_profile_handles_hue_wraparound():
    samples = np.array([[178, 240, 250], [179, 245, 255], [0, 250, 250], [2, 235, 245]] * 4)
    profile = profile_from_samples("red", samples)
    assert len(profile.hsv_ranges) == 2


def test_white_profile_uses_low_saturation_and_high_value():
    samples = np.array([[20, 12, 235], [90, 18, 245], [150, 8, 250]] * 5)
    profile = profile_from_samples("white", samples)
    lower, upper = profile.hsv_ranges[0]
    assert lower[0] == 0 and upper[0] == 179
    assert upper[1] <= 150
    assert lower[2] >= 80


def test_detects_distinct_colored_blobs():
    profiles = (
        ColorProfile("first", ((np.array([15, 150, 150], np.uint8), np.array([25, 255, 255], np.uint8)),)),
        ColorProfile("second", ((np.array([105, 150, 150], np.uint8), np.array([115, 255, 255], np.uint8)),)),
    )
    hsv = np.zeros((160, 220, 3), np.uint8)
    cv2.circle(hsv, (60, 80), 9, (20, 255, 255), -1)
    cv2.circle(hsv, (160, 80), 10, (110, 255, 255), -1)
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    detections, masks = SphereDetector(profiles).detect(frame)
    assert set(detections) == {"first", "second"}
    np.testing.assert_allclose(detections["first"].center, [60, 80], atol=1)
    assert masks["second"][80, 160] == 255
