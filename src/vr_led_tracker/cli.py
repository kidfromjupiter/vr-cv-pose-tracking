from __future__ import annotations

import argparse
import sys

from .calibration import run_camera_calibration
from .color_calibration import run_color_calibration
from .config import ControllerModel
from .errors import TrackerError
from .inertial_preview import run_inertial_preview
from .preview import run_preview


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Track a three-sphere VR controller with a camera and IMU")
    subparsers = parser.add_subparsers(dest="command", required=True)

    camera = subparsers.add_parser("calibrate-camera", help="Calibrate the camera with a checkerboard")
    camera.add_argument("--device", default="/dev/video0")
    camera.add_argument("--output", default="config/camera.json")
    camera.add_argument("--columns", type=int, default=9, help="checkerboard inner corners across")
    camera.add_argument("--rows", type=int, default=6, help="checkerboard inner corners down")
    camera.add_argument("--square-mm", type=float, required=True, help="measured checker square size")
    camera.add_argument("--frames", type=int, default=15)

    colors = subparsers.add_parser("calibrate-colors", help="Sample red, blue, and white spheres")
    colors.add_argument("--device", default="/dev/video0")
    colors.add_argument("--model", default="config/controller.json")
    colors.add_argument("--output", default="config/colors.json")
    colors.add_argument(
        "--samples-per-color",
        type=int,
        default=10,
        help="clicks from separate frames for each sphere (default: 10)",
    )

    track = subparsers.add_parser("track", help="Open the fused camera/IMU pose preview")
    track.add_argument("--device", default="/dev/video0")
    track.add_argument("--serial-device", default="/dev/ttyACM0")
    track.add_argument("--baud", type=int, default=230400)
    track.add_argument("--imu-slot", choices=("right", "left"), default="right")
    track.add_argument("--model", default="config/controller.json")
    track.add_argument("--camera", default="config/camera.json")
    track.add_argument("--colors", default="config/colors.json")
    track.add_argument("--max-reprojection-px", type=float, default=5.0)
    track.add_argument(
        "--camera-latency-ms",
        type=float,
        default=0.0,
        help="fixed camera delay in milliseconds, from 0 to 500 (default: 0)",
    )

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
        elif args.command == "calibrate-colors":
            model = ControllerModel.load(args.model)
            run_color_calibration(
                args.device,
                model,
                args.output,
                args.samples_per_color,
            )
            print(f"Saved three sphere color profiles to {args.output}")
        elif args.command == "track":
            if args.max_reprojection_px <= 0 or args.baud <= 0:
                raise TrackerError("--max-reprojection-px and --baud must be positive")
            if not 0.0 <= args.camera_latency_ms <= 500.0:
                raise TrackerError("--camera-latency-ms must be between 0 and 500")
            run_preview(
                args.device,
                args.serial_device,
                args.baud,
                args.imu_slot,
                args.model,
                args.camera,
                args.colors,
                args.max_reprojection_px,
                args.camera_latency_ms,
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
