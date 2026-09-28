from __future__ import annotations

import numpy as np
import pytest

from vr_led_tracker.camera import FrameSynchronizer, TimestampedFrame


def frame(timestamp: float) -> TimestampedFrame:
    return TimestampedFrame(np.zeros((2, 2, 3), dtype=np.uint8), timestamp)


def test_frame_synchronizer_pairs_frames_within_skew():
    sync = FrameSynchronizer(0.02)
    sync.add("left", frame(1.0)); sync.add("right", frame(1.012))
    observation = sync.pop_ready(1.012)
    assert observation.left is not None and observation.right is not None
    assert observation.skew_s == pytest.approx(0.012)
    assert observation.timestamp == pytest.approx(1.006)


def test_frame_synchronizer_releases_old_frame_for_mono_fallback():
    sync = FrameSynchronizer(0.02)
    sync.add("left", frame(2.0))
    assert sync.pop_ready(2.01) is None
    observation = sync.pop_ready(2.021)
    assert observation.left is not None and observation.right is None


def test_frame_synchronizer_rejects_unknown_side():
    sync = FrameSynchronizer(0.02)
    with pytest.raises(ValueError, match="Unknown"):
        sync.add("middle", frame(1.0))
