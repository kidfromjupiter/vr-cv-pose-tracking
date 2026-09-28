from __future__ import annotations

import cv2
import numpy as np
import pytest

from vr_led_tracker.detection import SphereDetector


@pytest.mark.parametrize("brightness", [165, 205, 245])
def test_adaptive_detector_finds_white_spheres_across_brightness(brightness):
    frame = np.full((180, 260, 3), 25, np.uint8)
    for center, radius in [((45, 80), 9), ((130, 60), 11), ((210, 115), 10)]:
        cv2.circle(frame, center, radius, (brightness, brightness, brightness), -1)
    cv2.circle(frame, (90, 140), 12, (20, 20, 230), -1)

    detections, mask, threshold = SphereDetector().detect(frame)

    assert len(detections) == 3
    assert 140 <= threshold <= 235
    assert mask[80, 45] == 255
    assert mask[140, 90] == 0


def test_adaptive_detector_limits_candidate_count():
    frame = np.full((240, 320, 3), 20, np.uint8)
    for index in range(12):
        cv2.circle(frame, (20 + (index % 6) * 50, 50 + (index // 6) * 120), 8, (240, 240, 240), -1)
    detections, _mask, _threshold = SphereDetector(max_candidates=8).detect(frame)
    assert len(detections) == 8
