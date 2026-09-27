from __future__ import annotations

import struct

import numpy as np

from vr_led_tracker.serial_pose import FusionFrameParser, TimestampMapper, crc16_ccitt


def make_frame(sequence=0, slot=0, timestamp=0x89ABCDEF):
    payload = bytearray(24)
    payload[0:2] = b"\x01\x01"
    struct.pack_into("<HI4h3hH", payload, 2, sequence, timestamp, 32767, -100, 200, -300, 1024, -512, 2048, 0x13)
    frame = bytearray(b"\xa5\x5a" + bytes((1, slot, 24)))
    frame.extend(payload)
    frame.extend(struct.pack("<H", crc16_ccitt(frame[2:])))
    return bytes(frame)


def test_crc16_ccitt_known_vector():
    assert crc16_ccitt(b"123456789") == 0x29B1


def test_decodes_timestamped_sample_and_scales_values():
    sample = FusionFrameParser().feed(make_frame(19, 1), 12.5)[0]
    assert sample.slot == "left"
    assert sample.sequence == 19
    assert sample.timestamp_us == 0x89ABCDEF
    assert sample.arrival_time == 12.5
    np.testing.assert_allclose(sample.acceleration_g, [0.5, -0.25, 1.0])
    assert sample.status == 0x13


def test_fragmentation_resync_crc_and_per_slot_drops():
    parser = FusionFrameParser()
    bad = bytearray(make_frame(1))
    bad[-1] ^= 1
    stream = b"garbage" + bytes(bad) + make_frame(4, 0) + make_frame(6, 0) + make_frame(9, 1)
    assert parser.feed(stream[:20]) == []
    samples = parser.feed(stream[20:])
    assert [(sample.slot, sample.sequence) for sample in samples] == [
        ("right", 4), ("right", 6), ("left", 9)
    ]
    assert parser.crc_errors == 1
    assert parser.dropped_frames["right"] == 1


def test_timestamp_mapper_unwraps_32_bit_counter():
    mapper = TimestampMapper()
    first = mapper.map(0xFFFFFF00, 100.0)
    second = mapper.map(0x00000100, 100.001)
    assert second > first
    assert second - first == pytest.approx(512e-6)


import pytest
