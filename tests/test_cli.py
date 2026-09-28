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
    assert args.left_device == "/dev/video0"
    assert args.right_device == "/dev/video2"
    assert args.max_frame_skew_ms == 20.0


def test_tracking_rejects_out_of_range_camera_latency(capsys):
    assert main(["track", "--camera-latency-ms", "-1"]) == 2
    assert "--camera-latency-ms must be between 0 and 500" in capsys.readouterr().err
    assert main(["track", "--camera-latency-ms", "501"]) == 2


def test_inertial_preview_rejects_nonpositive_baud(capsys):
    assert main(["inertial-preview", "--baud", "0"]) == 2
    assert "--baud must be positive" in capsys.readouterr().err


def test_tracking_rejects_same_camera_and_invalid_skew(capsys):
    assert main(["track", "--left-device", "2", "--right-device", "2"]) == 2
    assert "must be different" in capsys.readouterr().err
    assert main(["track", "--max-frame-skew-ms", "0"]) == 2
    assert "--max-frame-skew-ms must be between 1 and 100" in capsys.readouterr().err


def test_stereo_calibration_defaults():
    args = build_parser().parse_args(["calibrate-stereo", "--square-mm", "25"])
    assert args.frames == 20
    assert args.output == "config/stereo_camera.json"
