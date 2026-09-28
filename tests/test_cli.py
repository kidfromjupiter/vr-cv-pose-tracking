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


def test_tracking_uses_hades_settings_by_default():
    args = build_parser().parse_args(["track"])
    assert args.fusion_settings == "config/hades_fusion.json"


def test_stereo_tracking_has_separate_command_and_defaults():
    args = build_parser().parse_args(["track-stereo"])
    assert args.left_device == "/dev/video0"
    assert args.right_device == "/dev/video2"
    assert args.max_frame_skew_ms == 20.0


def test_tracking_camera_latency_option_is_removed():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["track", "--camera-latency-ms", "25"])


def test_inertial_preview_rejects_nonpositive_baud(capsys):
    assert main(["inertial-preview", "--baud", "0"]) == 2
    assert "--baud must be positive" in capsys.readouterr().err
