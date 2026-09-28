from __future__ import annotations

import argparse
import sys

from .calibration import run_camera_calibration, run_stereo_calibration
from .config import ControllerModel
from .errors import TrackerError
from .inertial_preview import run_inertial_preview
from .preview import run_preview
from .stereo_preview import run_stereo_preview


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Track a three-white-sphere VR controller with a camera and IMU"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    camera = subparsers.add_parser("calibrate-camera", help="Calibrate the camera with a checkerboard")
    camera.add_argument("--device", default="/dev/video0")
    camera.add_argument("--output", default="config/camera.json")
    camera.add_argument("--columns", type=int, default=9, help="checkerboard inner corners across")
    camera.add_argument("--rows", type=int, default=6, help="checkerboard inner corners down")
    camera.add_argument("--square-mm", type=float, required=True, help="measured checker square size")
    camera.add_argument("--frames", type=int, default=15)

    stereo = subparsers.add_parser("calibrate-stereo", help="Calibrate two cameras from synchronized checkerboard views")
    stereo.add_argument("--left-device", default="/dev/video0")
    stereo.add_argument("--right-device", default="/dev/video2")
    stereo.add_argument("--output", default="config/stereo_camera.json")
    stereo.add_argument("--columns", type=int, default=9)
    stereo.add_argument("--rows", type=int, default=6)
    stereo.add_argument("--square-mm", type=float, required=True)
    stereo.add_argument("--frames", type=int, default=20)
    stereo.add_argument("--max-frame-skew-ms", type=float, default=20.0)

    track = subparsers.add_parser("track", help="Open the fused camera/IMU pose preview")
    track.add_argument("--device", default="/dev/video0")
    track.add_argument("--serial-device", default="/dev/ttyACM0")
    track.add_argument("--baud", type=int, default=230400)
    track.add_argument("--imu-slot", choices=("right", "left"), default="right")
    track.add_argument("--model", default="config/controller.json")
    track.add_argument("--camera", default="config/camera.json")
    track.add_argument("--fusion-settings", default="config/hades_fusion.json")
    track.add_argument("--max-reprojection-px", type=float, default=5.0)

    track_stereo = subparsers.add_parser("track-stereo", help="Open the stereo fused camera/IMU pose preview")
    track_stereo.add_argument("--left-device", default="/dev/video0")
    track_stereo.add_argument("--right-device", default="/dev/video2")
    track_stereo.add_argument("--serial-device", default="/dev/ttyACM0")
    track_stereo.add_argument("--baud", type=int, default=230400)
    track_stereo.add_argument("--imu-slot", choices=("right", "left"), default="right")
    track_stereo.add_argument("--model", default="config/controller.json")
    track_stereo.add_argument("--stereo-camera", default="config/stereo_camera.json")
    track_stereo.add_argument("--fusion-settings", default="config/hades_fusion.json")
    track_stereo.add_argument("--max-reprojection-px", type=float, default=5.0)
    track_stereo.add_argument("--max-frame-skew-ms", type=float, default=20.0)

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
        elif args.command == "calibrate-stereo":
            if args.left_device == args.right_device:
                raise TrackerError("--left-device and --right-device must be different")
            if not 1.0 <= args.max_frame_skew_ms <= 100.0:
                raise TrackerError("--max-frame-skew-ms must be between 1 and 100")
            calibration = run_stereo_calibration(
                args.left_device, args.right_device, args.output, args.columns, args.rows,
                args.square_mm, args.frames, args.max_frame_skew_ms,
            )
            print(f"Saved {args.output}; stereo RMS reprojection error {calibration.rms_error:.4f} px")
        elif args.command == "track":
            if args.max_reprojection_px <= 0 or args.baud <= 0:
                raise TrackerError("--max-reprojection-px and --baud must be positive")
            run_preview(
                args.device,
                args.serial_device,
                args.baud,
                args.imu_slot,
                args.model,
                args.camera,
                args.fusion_settings,
                args.max_reprojection_px,
            )
        elif args.command == "track-stereo":
            if args.max_reprojection_px <= 0 or args.baud <= 0:
                raise TrackerError("--max-reprojection-px and --baud must be positive")
            if args.left_device == args.right_device:
                raise TrackerError("--left-device and --right-device must be different")
            if not 1.0 <= args.max_frame_skew_ms <= 100.0:
                raise TrackerError("--max-frame-skew-ms must be between 1 and 100")
            run_stereo_preview(
                args.left_device, args.right_device, args.serial_device, args.baud, args.imu_slot,
                args.model, args.stereo_camera, args.fusion_settings,
                args.max_reprojection_px, args.max_frame_skew_ms,
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
