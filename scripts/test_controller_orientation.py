"""Run after installing the project: python scripts/test_controller_orientation.py."""

from __future__ import annotations

import argparse
import sys

from vr_led_tracker.errors import TrackerError
from vr_led_tracker.orientation_preview import run_orientation_preview


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Draw left/right orientations from receiver quaternions")
    parser.add_argument("--device", default="/dev/ttyACM0", help="receiver serial device")
    parser.add_argument("--baud", type=int, default=230400)
    args = parser.parse_args(argv)
    try:
        run_orientation_preview(args.device, args.baud)
    except TrackerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
