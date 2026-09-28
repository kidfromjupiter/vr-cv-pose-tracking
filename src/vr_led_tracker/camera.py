from __future__ import annotations

from pathlib import Path
from collections import deque
from dataclasses import dataclass
import queue
import threading
import time

import cv2
import numpy as np

from .errors import TrackerError


def normalize_device(device: str) -> int | str:
    if device.isdecimal():
        return int(device)
    return device


def open_camera(device: str) -> cv2.VideoCapture:
    source = normalize_device(device)
    if isinstance(source, str) and source.startswith("/dev/") and not Path(source).exists():
        raise TrackerError(
            f"Camera device {source} does not exist. Start the camera stream and verify the V4L2 device first."
        )
    capture = cv2.VideoCapture(source, cv2.CAP_V4L2)
    if not capture.isOpened():
        capture.release()
        raise TrackerError(f"Could not open camera {device}")
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def read_frame(capture: cv2.VideoCapture):
    ok, frame = capture.read()
    if not ok or frame is None:
        raise TrackerError("Camera stopped providing frames")
    return frame


@dataclass(frozen=True)
class TimestampedFrame:
    frame: np.ndarray
    timestamp: float


@dataclass(frozen=True)
class CameraObservation:
    left: TimestampedFrame | None
    right: TimestampedFrame | None

    @property
    def timestamp(self) -> float:
        values = [item.timestamp for item in (self.left, self.right) if item is not None]
        return float(sum(values) / len(values))

    @property
    def skew_s(self) -> float | None:
        if self.left is None or self.right is None:
            return None
        return abs(self.left.timestamp - self.right.timestamp)


class FrameSynchronizer:
    """Pairs timestamped frames and releases old unmatched frames for mono fallback."""

    def __init__(self, max_skew_s: float) -> None:
        if max_skew_s <= 0.0:
            raise ValueError("max_skew_s must be positive")
        self.max_skew_s = float(max_skew_s)
        self.frames: dict[str, deque[TimestampedFrame]] = {
            "left": deque(maxlen=3),
            "right": deque(maxlen=3),
        }

    def add(self, side: str, frame: TimestampedFrame) -> None:
        if side not in self.frames:
            raise ValueError(f"Unknown camera side {side}")
        self.frames[side].append(frame)

    def pop_ready(self, now: float) -> CameraObservation | None:
        left = self.frames["left"]
        right = self.frames["right"]
        if left and right:
            left_frame = left[0]
            right_frame = right[0]
            skew = left_frame.timestamp - right_frame.timestamp
            if abs(skew) <= self.max_skew_s:
                return CameraObservation(left.popleft(), right.popleft())
            older_side = left if skew < 0.0 else right
            if now - older_side[0].timestamp >= self.max_skew_s:
                old = older_side.popleft()
                return CameraObservation(old, None) if older_side is left else CameraObservation(None, old)
            return None
        for side, other in ((left, right), (right, left)):
            if side and not other and now - side[0].timestamp >= self.max_skew_s:
                old = side.popleft()
                return CameraObservation(old, None) if side is left else CameraObservation(None, old)
        return None


class StereoCameraCapture:
    def __init__(self, left_device: str, right_device: str, max_skew_s: float) -> None:
        left_capture = open_camera(left_device)
        try:
            right_capture = open_camera(right_device)
        except Exception:
            left_capture.release()
            raise
        self.captures = {"left": left_capture, "right": right_capture}
        self.events: queue.Queue[tuple[str, TimestampedFrame]] = queue.Queue(maxsize=8)
        self.synchronizer = FrameSynchronizer(max_skew_s)
        self.stop_event = threading.Event()
        self.error: Exception | None = None
        self.threads = [
            threading.Thread(target=self._capture, args=(side,), name=f"camera-{side}", daemon=True)
            for side in ("left", "right")
        ]
        for thread in self.threads:
            thread.start()

    def _capture(self, side: str) -> None:
        capture = self.captures[side]
        try:
            while not self.stop_event.is_set():
                packet = TimestampedFrame(read_frame(capture), time.monotonic())
                try:
                    self.events.put_nowait((side, packet))
                except queue.Full:
                    try:
                        self.events.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        self.events.put_nowait((side, packet))
                    except queue.Full:
                        pass
        except Exception as exc:
            if not self.stop_event.is_set():
                self.error = exc
                self.stop_event.set()

    def next_observation(self, timeout_s: float = 0.1) -> CameraObservation:
        deadline = time.monotonic() + timeout_s
        while True:
            if self.error is not None:
                raise TrackerError(f"Stereo camera capture failed: {self.error}")
            now = time.monotonic()
            ready = self.synchronizer.pop_ready(now)
            if ready is not None:
                return ready
            remaining = deadline - now
            if remaining <= 0.0:
                raise TrackerError("Timed out waiting for camera frames")
            try:
                side, packet = self.events.get(timeout=min(remaining, self.synchronizer.max_skew_s))
                self.synchronizer.add(side, packet)
                while True:
                    try:
                        queued_side, queued_packet = self.events.get_nowait()
                    except queue.Empty:
                        break
                    self.synchronizer.add(queued_side, queued_packet)
            except queue.Empty:
                continue

    def close(self) -> None:
        self.stop_event.set()
        for capture in self.captures.values():
            capture.release()
        for thread in self.threads:
            thread.join(timeout=0.3)
