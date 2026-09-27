from __future__ import annotations

from vr_led_tracker.cli import build_parser, main


def test_inertial_preview_defaults():
    args = build_parser().parse_args(["inertial-preview"])
    assert args.device == "/dev/ttyACM0"
    assert args.baud == 230400


def test_color_calibration_collects_more_frames_by_default():
    args = build_parser().parse_args(["calibrate-colors"])
    assert args.samples_per_color == 10


def test_inertial_preview_rejects_nonpositive_baud(capsys):
    assert main(["inertial-preview", "--baud", "0"]) == 2
    assert "--baud must be positive" in capsys.readouterr().err
