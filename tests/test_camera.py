from __future__ import annotations

import numpy as np
import pytest

from vr_led_tracker.camera import FrameSynchronizer, TimestampedFrame


def frame(timestamp: float) -> TimestampedFrame:
    return TimestampedFrame(np.zeros((2, 2, 3), dtype=np.uint8), timestamp)


def test_frame_synchronizer_pairs_frames_within_skew():
    synchronizer = FrameSynchronizer(0.02)
    synchronizer.add("left", frame(1.000))
    synchronizer.add("right", frame(1.012))

    observation = synchronizer.pop_ready(1.012)

    assert observation is not None
    assert observation.left is not None and observation.right is not None
    assert observation.skew_s == pytest.approx(0.012)
    assert observation.timestamp == pytest.approx(1.006)


def test_frame_synchronizer_releases_unmatched_frame_for_fallback():
    synchronizer = FrameSynchronizer(0.02)
    synchronizer.add("left", frame(2.0))

    assert synchronizer.pop_ready(2.01) is None
    observation = synchronizer.pop_ready(2.021)

    assert observation is not None
    assert observation.left is not None
    assert observation.right is None


def test_frame_synchronizer_drops_old_side_when_pair_is_too_far_apart():
    synchronizer = FrameSynchronizer(0.02)
    synchronizer.add("left", frame(3.0))
    synchronizer.add("right", frame(3.05))

    observation = synchronizer.pop_ready(3.05)

    assert observation is not None
    assert observation.left is not None
    assert observation.right is None
