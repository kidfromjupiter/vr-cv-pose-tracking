from __future__ import annotations

import cv2
import numpy as np

from vr_led_tracker.color_calibration import profile_from_samples
from vr_led_tracker.detection import BallDetector


def test_blue_profile_uses_sampled_hue_and_minimum_saturation():
    samples = np.array([[113, 220, 230], [115, 240, 250], [117, 210, 245]] * 5)
    profile = profile_from_samples(samples)
    assert profile.label == "blue"
    assert len(profile.hsv_ranges) == 1
    lower, upper = profile.hsv_ranges[0]
    assert lower[0] < 115 < upper[0]
    assert lower[1] >= 20


def test_profile_handles_hue_wraparound():
    samples = np.array([[178, 240, 250], [179, 245, 255], [0, 250, 250], [2, 235, 245]] * 4)
    assert len(profile_from_samples(samples).hsv_ranges) == 2


def test_detector_finds_blue_ball_and_ignores_other_colors(blue_profile):
    hsv = np.zeros((180, 260, 3), np.uint8)
    cv2.circle(hsv, (70, 90), 12, (115, 240, 245), -1)
    cv2.circle(hsv, (190, 90), 14, (5, 240, 245), -1)
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    detections, mask = BallDetector(blue_profile).detect(frame)

    assert len(detections) == 1
    np.testing.assert_allclose(detections[0].center, [70, 90], atol=1)
    assert mask[90, 70] == 255
    assert mask[90, 190] == 0


def test_detector_limits_candidate_count(blue_profile):
    hsv = np.zeros((240, 320, 3), np.uint8)
    for index in range(12):
        cv2.circle(hsv, (20 + (index % 6) * 50, 50 + (index // 6) * 120), 8, (115, 240, 245), -1)
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    detections, _ = BallDetector(blue_profile, max_candidates=8).detect(frame)
    assert len(detections) == 8
