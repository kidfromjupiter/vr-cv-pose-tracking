from __future__ import annotations

import pytest

from vr_led_tracker.cli import build_parser, main


def test_inertial_preview_defaults():
    args = build_parser().parse_args(["inertial-preview"])
    assert args.device == "/dev/ttyACM0"
    assert args.baud == 230400


def test_color_calibration_defaults():
    args = build_parser().parse_args(["calibrate-color"])
    assert args.output == "config/color.json"
    assert args.samples == 10


def test_tracking_defaults_to_ball_model_and_color_profile():
    args = build_parser().parse_args(["track"])
    assert args.model == "config/controller.json"
    assert args.color == "config/color.json"
    assert not hasattr(args, "camera_latency_ms")


def test_removed_plural_color_calibration_command():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["calibrate-colors"])


def test_tracking_rejects_nonpositive_baud(capsys):
    assert main(["track", "--baud", "0"]) == 2
    assert "--baud must be positive" in capsys.readouterr().err


def test_inertial_preview_rejects_nonpositive_baud(capsys):
    assert main(["inertial-preview", "--baud", "0"]) == 2
    assert "--baud must be positive" in capsys.readouterr().err
