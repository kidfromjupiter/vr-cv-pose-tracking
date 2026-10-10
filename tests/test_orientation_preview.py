from __future__ import annotations

import math
import runpy
import struct
from pathlib import Path

import cv2
import numpy as np
import pytest
import serial

from vr_led_tracker import orientation_preview as viewer
from vr_led_tracker.errors import TrackerError
from vr_led_tracker.serial_pose import FusionFrameParser, FusionSample, crc16_ccitt


def sample(quaternion=(1.0, 0.0, 0.0, 0.0), slot="right"):
    return FusionSample(slot, 0, 0, np.array(quaternion), np.zeros(3), 0)


def frame(slot, quaternion, sequence=0):
    values = np.rint(np.array(quaternion) * 32767).astype(int)
    payload = b"\x01\x01" + struct.pack("<HI4h3hH", sequence, 0, *values, 0, 0, 0, 0)
    body = bytes((1, slot, 24)) + payload
    return b"\xa5\x5a" + body + struct.pack("<H", crc16_ccitt(body))


@pytest.mark.parametrize("axis", range(3))
def test_received_quaternion_rotates_each_axis_without_calibration(axis):
    half = math.sqrt(0.5)
    quaternion = np.zeros(4)
    quaternion[0] = quaternion[axis + 1] = half
    controller = viewer.ControllerOrientation()
    assert controller.update(sample(quaternion), 0.0)
    assert controller.state(0.0) == "LIVE"
    basis = np.eye(3)
    rotation = controller.displayed_rotation()
    np.testing.assert_allclose(rotation @ basis[:, axis], basis[:, axis], atol=1e-7)
    np.testing.assert_allclose(rotation @ basis[:, (axis + 1) % 3], basis[:, (axis + 2) % 3], atol=1e-7)
    controller.update(sample(-quaternion * 0.9), 0.1)
    np.testing.assert_allclose(controller.displayed_rotation(), rotation, atol=1e-7)


@pytest.mark.parametrize("quaternion", [(0, 0, 0, 0), (float("nan"), 0, 0, 0), (float("inf"), 0, 0, 0)])
def test_invalid_quaternion_preserves_last_pose_and_does_not_refresh_age(quaternion):
    controller = viewer.ControllerOrientation()
    assert not controller.update(sample(quaternion), 0.0)
    assert controller.state(0.0) == "WAITING"
    controller.update(sample(), 1.0)
    assert not controller.update(sample(quaternion), 1.3)
    assert controller.state(1.3) == "STALE"
    assert controller.last_valid_time == 1.0
    np.testing.assert_allclose(controller.displayed_rotation(), np.eye(3))


def test_interleaved_fragmented_frames_keep_slots_and_freshness_independent():
    parser = FusionFrameParser()
    controllers = {slot: viewer.ControllerOrientation() for slot in ("left", "right")}
    half = math.sqrt(0.5)
    data = frame(0, (1, 0, 0, 0)) + frame(1, (half, 0, 0, half))
    assert parser.feed(data[:17], 0.0) == []
    for value in parser.feed(data[17:], 0.0):
        controllers[value.slot].update(value, value.arrival_time)
    for value in parser.feed(frame(0, (1, 0, 0, 0), 1), 0.3):
        controllers[value.slot].update(value, value.arrival_time)
    assert controllers["right"].state(0.3) == "LIVE"
    assert controllers["left"].state(0.3) == "STALE"
    np.testing.assert_allclose(controllers["left"].displayed_rotation() @ [1, 0, 0], [0, 1, 0], atol=1e-7)
    for value in parser.feed(frame(1, (1, 0, 0, 0), 1), 0.4):
        controllers[value.slot].update(value, value.arrival_time)
    assert controllers["left"].state(0.4) == "LIVE"


def test_recenter_skips_waiting_and_stale_and_preserves_received_values():
    controller = viewer.ControllerOrientation()
    controller.recenter(0.0)
    assert controller.reference is None
    half = math.sqrt(0.5)
    value = sample((half, half, 0, 0))
    controller.update(value, 0.0)
    controller.recenter(0.3)
    assert controller.reference is None
    controller.update(value, 0.4)
    controller.recenter(0.4)
    np.testing.assert_allclose(controller.displayed_rotation(), np.eye(3), atol=1e-7)
    np.testing.assert_array_equal(controller.sample.quaternion_wxyz, value.quaternion_wxyz)
    controller.update(sample(), 0.5)
    np.testing.assert_allclose(controller.displayed_rotation() @ [0, 1, 0], [0, 0, -1], atol=1e-7)
    controller.reference = None
    np.testing.assert_allclose(controller.displayed_rotation(), np.eye(3))


@pytest.fixture
def fake_viewer(monkeypatch):
    class Port:
        closed = False
        data = frame(0, (1, 0, 0, 0)) + frame(1, (1, 0, 0, 0))
        error = None

        @property
        def in_waiting(self):
            if self.error:
                raise self.error
            return len(self.data)

        def read(self, size):
            result, self.data = self.data[:size], self.data[size:]
            return result

        def close(self):
            self.closed = True

    port = Port()
    opened = []
    destroyed = []
    displays = []

    def open_port(*args, **kwargs):
        opened.append((args, kwargs))
        return port

    monkeypatch.setattr(serial, "Serial", open_port)
    monkeypatch.setattr(cv2, "namedWindow", lambda *args: None)
    monkeypatch.setattr(cv2, "resizeWindow", lambda *args: None)
    monkeypatch.setattr(cv2, "imshow", lambda name, image: displays.append(image.copy()))
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *args: 1)
    monkeypatch.setattr(cv2, "waitKey", lambda delay: ord("q"))
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: destroyed.append(True))
    return port, opened, destroyed, displays


@pytest.mark.parametrize("exit_mode", ["q", "escape", "window", "interrupt"])
def test_preview_cleanup_for_all_exit_modes(fake_viewer, monkeypatch, exit_mode):
    port, opened, destroyed, displays = fake_viewer
    if exit_mode == "escape":
        monkeypatch.setattr(cv2, "waitKey", lambda delay: 27)
    elif exit_mode == "window":
        monkeypatch.setattr(cv2, "waitKey", lambda delay: -1)
        monkeypatch.setattr(cv2, "getWindowProperty", lambda *args: 0)
    elif exit_mode == "interrupt":
        def interrupt(delay):
            raise KeyboardInterrupt
        monkeypatch.setattr(cv2, "waitKey", interrupt)
    if exit_mode == "interrupt":
        with pytest.raises(KeyboardInterrupt):
            viewer.run_orientation_preview("test-port", 115200)
    else:
        viewer.run_orientation_preview("test-port", 115200)
    assert port.closed and destroyed == [True]
    assert opened == [(("test-port",), {"baudrate": 115200, "timeout": 0})]
    assert displays[0].shape == (720, 1280, 3)


def test_recenter_and_clear_keys_apply_to_both_live_controllers(fake_viewer, monkeypatch):
    recorded = []
    draw = viewer.draw_orientation_panel

    def record(label, controller, now, size):
        recorded.append((label, controller.reference is not None))
        return draw(label, controller, now, size)

    keys = iter([ord("r"), ord("c"), ord("q")])
    monkeypatch.setattr(viewer, "draw_orientation_panel", record)
    monkeypatch.setattr(viewer.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(cv2, "waitKey", lambda delay: next(keys))
    viewer.run_orientation_preview()
    assert recorded == [("LEFT", False), ("RIGHT", False), ("LEFT", True), ("RIGHT", True), ("LEFT", False), ("RIGHT", False)]


def test_serial_read_failure_closes_resources(fake_viewer):
    port, _, destroyed, displays = fake_viewer
    port.error = serial.SerialException("receiver disconnected")
    with pytest.raises(TrackerError, match="Cannot read serial device.*receiver disconnected"):
        viewer.run_orientation_preview()
    assert port.closed and destroyed == [True] and not displays


def test_serial_open_failure_is_actionable(monkeypatch):
    def fail(*args, **kwargs):
        raise serial.SerialException("port busy")
    monkeypatch.setattr(serial, "Serial", fail)
    with pytest.raises(TrackerError, match="Cannot open serial device.*port busy"):
        viewer.run_orientation_preview()


def test_window_setup_failure_closes_port(fake_viewer, monkeypatch):
    port, _, destroyed, _ = fake_viewer
    def fail(*args):
        raise cv2.error("display unavailable")
    monkeypatch.setattr(cv2, "namedWindow", fail)
    with pytest.raises(TrackerError, match="Cannot display orientation viewer"):
        viewer.run_orientation_preview()
    assert port.closed and destroyed == [True]


def test_standalone_script_defaults_and_invalid_baud(monkeypatch, capsys):
    script = Path(__file__).resolve().parents[1] / "scripts/test_controller_orientation.py"
    namespace = runpy.run_path(str(script))
    calls = []
    monkeypatch.setitem(namespace["main"].__globals__, "run_orientation_preview", lambda *args: calls.append(args))
    assert namespace["main"]([]) == 0
    assert calls == [("/dev/ttyACM0", 230400)]
    monkeypatch.setitem(namespace["main"].__globals__, "run_orientation_preview", viewer.run_orientation_preview)
    assert namespace["main"](["--baud", "0"]) == 2
    assert "--baud must be positive" in capsys.readouterr().err
