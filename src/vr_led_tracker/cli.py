from __future__ import annotations

import argparse
import sys

from .calibration import run_camera_calibration
from .color_calibration import run_color_calibration
from .errors import TrackerError
from .inertial_preview import run_inertial_preview
from .preview import run_preview


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Track a blue ball's XYZ with a camera and orientation with an IMU"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    camera = subparsers.add_parser("calibrate-camera", help="Calibrate the camera with a checkerboard")
    camera.add_argument("--device", default="/dev/video0")
    camera.add_argument("--output", default="config/camera.json")
    camera.add_argument("--columns", type=int, default=9, help="checkerboard inner corners across")
    camera.add_argument("--rows", type=int, default=6, help="checkerboard inner corners down")
    camera.add_argument("--square-mm", type=float, required=True, help="measured checker square size")
    camera.add_argument("--frames", type=int, default=15)

    color = subparsers.add_parser("calibrate-color", help="Sample the blue ball color")
    color.add_argument("--device", default="/dev/video0")
    color.add_argument("--output", default="config/color.json")
    color.add_argument(
        "--samples",
        type=int,
        default=10,
        help="clicks from separate frames (default: 10)",
    )

    track = subparsers.add_parser("track", help="Open the camera position and IMU orientation preview")
    track.add_argument("--device", default="/dev/video0")
    track.add_argument("--serial-device", default="/dev/ttyACM0")
    track.add_argument("--baud", type=int, default=230400)
    track.add_argument("--imu-slot", choices=("right", "left"), default="right")
    track.add_argument("--model", default="config/controller.json")
    track.add_argument("--camera", default="config/camera.json")
    track.add_argument("--color", default="config/color.json")

    inertial = subparsers.add_parser(
        "inertial-preview", help="Animate controllers from the timestamped fusion stream"
    )
    inertial.add_argument("--device", default="/dev/ttyACM0")
    inertial.add_argument("--baud", type=int, default=230400)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "calibrate-camera":
            calibration = run_camera_calibration(
                args.device, args.output, args.columns, args.rows, args.square_mm, args.frames
            )
            print(f"Saved {args.output}; RMS reprojection error {calibration.rms_error:.4f} px")
        elif args.command == "calibrate-color":
            run_color_calibration(args.device, args.output, args.samples)
            print(f"Saved blue ball color profile to {args.output}")
        elif args.command == "track":
            if args.baud <= 0:
                raise TrackerError("--baud must be positive")
            run_preview(
                args.device,
                args.serial_device,
                args.baud,
                args.imu_slot,
                args.model,
                args.camera,
                args.color,
            )
        elif args.command == "inertial-preview":
            if args.baud <= 0:
                raise TrackerError("--baud must be positive")
            run_inertial_preview(args.device, args.baud)
    except TrackerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
    return 0
