from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np


FUSION_MAGIC = b"\xa5\x5a"
FUSION_VERSION = 1
FUSION_PAYLOAD_SIZE = 24
FUSION_FRAME_SIZE = 31


def crc16_ccitt(data: bytes | bytearray) -> int:
    crc = 0xFFFF
    for value in data:
        crc ^= value << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


@dataclass(frozen=True)
class FusionSample:
    slot: str
    sequence: int
    timestamp_us: int
    quaternion_wxyz: np.ndarray
    acceleration_g: np.ndarray
    status: int
    arrival_time: float | None = None


def decode_tracker_payload(payload: bytes, slot: int) -> FusionSample:
    if len(payload) != FUSION_PAYLOAD_SIZE or payload[0:2] != b"\x01\x01":
        raise ValueError("invalid tracker payload")
    sequence, timestamp_us = struct.unpack_from("<HI", payload, 2)
    values = struct.unpack_from("<4h3hH", payload, 8)
    quaternion = np.asarray(values[:4], dtype=np.float64) / 32767.0
    acceleration = np.asarray(values[4:7], dtype=np.float64) / 2048.0
    return FusionSample(
        "right" if slot == 0 else "left",
        sequence,
        timestamp_us,
        quaternion,
        acceleration,
        values[7],
    )


class FusionFrameParser:
    """Incrementally decode CRC-protected timestamped receiver frames."""

    def __init__(self) -> None:
        self.buffer = bytearray()
        self.invalid_bytes = 0
        self.crc_errors = 0
        self.frames_decoded = 0
        self.dropped_frames = {"right": 0, "left": 0}
        self.last_sequence: dict[str, int] = {}

    def feed(self, data: bytes, arrival_time: float | None = None) -> list[FusionSample]:
        self.buffer.extend(data)
        samples: list[FusionSample] = []
        while True:
            magic = self.buffer.find(FUSION_MAGIC)
            if magic < 0:
                keep = 1 if self.buffer[-1:] == FUSION_MAGIC[:1] else 0
                discarded = len(self.buffer) - keep
                self.invalid_bytes += discarded
                if discarded:
                    del self.buffer[:discarded]
                break
            if magic:
                self.invalid_bytes += magic
                del self.buffer[:magic]
            if len(self.buffer) < FUSION_FRAME_SIZE:
                break
            candidate = self.buffer[:FUSION_FRAME_SIZE]
            if (
                candidate[2] != FUSION_VERSION
                or candidate[3] not in (0, 1)
                or candidate[4] != FUSION_PAYLOAD_SIZE
            ):
                self.invalid_bytes += 1
                del self.buffer[0]
                continue
            expected_crc = struct.unpack_from("<H", candidate, 29)[0]
            if crc16_ccitt(candidate[2:29]) != expected_crc:
                self.crc_errors += 1
                self.invalid_bytes += 1
                del self.buffer[0]
                continue
            try:
                sample = decode_tracker_payload(bytes(candidate[5:29]), candidate[3])
            except ValueError:
                self.invalid_bytes += 1
                del self.buffer[0]
                continue
            previous = self.last_sequence.get(sample.slot)
            if previous is not None:
                gap = (sample.sequence - previous) & 0xFFFF
                if 1 < gap < 0x8000:
                    self.dropped_frames[sample.slot] += gap - 1
            self.last_sequence[sample.slot] = sample.sequence
            samples.append(
                FusionSample(
                    sample.slot,
                    sample.sequence,
                    sample.timestamp_us,
                    sample.quaternion_wxyz,
                    sample.acceleration_g,
                    sample.status,
                    arrival_time,
                )
            )
            self.frames_decoded += 1
            del self.buffer[:FUSION_FRAME_SIZE]
        return samples


class TimestampMapper:
    """Unwrap timestamps while preserving controller-measured sample intervals."""

    def __init__(self) -> None:
        self.wraps = 0
        self.last_raw: int | None = None
        self.host_offset: float | None = None

    def map(self, timestamp_us: int, arrival_time: float) -> float:
        if self.last_raw is not None and timestamp_us < self.last_raw - 0x80000000:
            self.wraps += 1
        self.last_raw = timestamp_us
        device_time = (timestamp_us + self.wraps * (1 << 32)) * 1e-6
        if self.host_offset is None:
            self.host_offset = arrival_time - device_time
        return device_time + self.host_offset
