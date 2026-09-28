from __future__ import annotations

import pytest

from vr_led_tracker.cli import build_parser, main


def test_inertial_preview_defaults():
    args = build_parser().parse_args(["inertial-preview"])
    assert args.device == "/dev/ttyACM0"
    assert args.baud == 230400


def test_color_calibration_command_is_removed():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["calibrate-colors"])


def test_tracking_uses_zero_fixed_camera_latency_by_default():
    args = build_parser().parse_args(["track"])
    assert args.camera_latency_ms == 0.0


def test_tracking_rejects_out_of_range_camera_latency(capsys):
    assert main(["track", "--camera-latency-ms", "-1"]) == 2
    assert "--camera-latency-ms must be between 0 and 500" in capsys.readouterr().err
    assert main(["track", "--camera-latency-ms", "501"]) == 2


def test_inertial_preview_rejects_nonpositive_baud(capsys):
    assert main(["inertial-preview", "--baud", "0"]) == 2
    assert "--baud must be positive" in capsys.readouterr().err
