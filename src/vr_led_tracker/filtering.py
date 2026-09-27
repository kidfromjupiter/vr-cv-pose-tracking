from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


def rotation_matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        quat = np.array([(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s,
                         (m[1, 0] - m[0, 1]) / s, 0.25 * s])
    else:
        index = int(np.argmax(np.diag(m)))
        if index == 0:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            quat = np.array([0.25 * s, (m[0, 1] + m[1, 0]) / s,
                             (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s])
        elif index == 1:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            quat = np.array([(m[0, 1] + m[1, 0]) / s, 0.25 * s,
                             (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s])
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            quat = np.array([(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s,
                             0.25 * s, (m[1, 0] - m[0, 1]) / s])
    return quat / np.linalg.norm(quat)


def quaternion_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion, dtype=np.float64) / np.linalg.norm(quaternion)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotation_matrix_to_euler_zyx(matrix: np.ndarray) -> np.ndarray:
    sy = math.hypot(float(matrix[0, 0]), float(matrix[1, 0]))
    if sy > 1e-8:
        roll = math.atan2(float(matrix[2, 1]), float(matrix[2, 2]))
        pitch = math.atan2(float(-matrix[2, 0]), sy)
        yaw = math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))
    else:
        roll = math.atan2(float(-matrix[1, 2]), float(matrix[1, 1]))
        pitch = math.atan2(float(-matrix[2, 0]), sy)
        yaw = 0.0
    return np.degrees([yaw, pitch, roll])


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroVector:
    def __init__(self, min_cutoff: float = 1.3, beta: float = 0.015, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.value: np.ndarray | None = None
        self.derivative: np.ndarray | None = None
        self.timestamp: float | None = None

    def reset(self) -> None:
        self.value = None
        self.derivative = None
        self.timestamp = None

    def update(self, value: np.ndarray, timestamp: float) -> np.ndarray:
        value = np.asarray(value, dtype=np.float64)
        if self.value is None or self.timestamp is None or timestamp <= self.timestamp:
            self.value = value.copy()
            self.derivative = np.zeros_like(value)
            self.timestamp = timestamp
            return self.value.copy()
        dt = min(max(timestamp - self.timestamp, 1e-4), 0.2)
        raw_derivative = (value - self.value) / dt
        derivative_alpha = _alpha(self.d_cutoff, dt)
        self.derivative = derivative_alpha * raw_derivative + (1.0 - derivative_alpha) * self.derivative
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(self.derivative))
        value_alpha = _alpha(cutoff, dt)
        self.value = value_alpha * value + (1.0 - value_alpha) * self.value
        self.timestamp = timestamp
        return self.value.copy()


def _slerp(start: np.ndarray, end: np.ndarray, amount: float) -> np.ndarray:
    dot = float(np.dot(start, end))
    if dot < 0:
        end = -end
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        result = start + amount * (end - start)
        return result / np.linalg.norm(result)
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    return (math.sin((1 - amount) * theta) / sin_theta) * start + (
        math.sin(amount * theta) / sin_theta
    ) * end


class AdaptiveQuaternionFilter:
    def __init__(self, min_cutoff: float = 1.5, beta: float = 0.08, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.value: np.ndarray | None = None
        self.speed = 0.0
        self.timestamp: float | None = None

    def reset(self) -> None:
        self.value = None
        self.speed = 0.0
        self.timestamp = None

    def update(self, value: np.ndarray, timestamp: float) -> np.ndarray:
        value = np.asarray(value, dtype=np.float64)
        value /= np.linalg.norm(value)
        if self.value is None or self.timestamp is None or timestamp <= self.timestamp:
            self.value = value.copy()
            self.timestamp = timestamp
            return self.value.copy()
        if np.dot(self.value, value) < 0:
            value = -value
        dt = min(max(timestamp - self.timestamp, 1e-4), 0.2)
        angle = 2.0 * math.acos(min(1.0, abs(float(np.dot(self.value, value)))))
        raw_speed = angle / dt
        da = _alpha(self.d_cutoff, dt)
        self.speed = da * raw_speed + (1.0 - da) * self.speed
        cutoff = self.min_cutoff + self.beta * self.speed
        self.value = _slerp(self.value, value, _alpha(cutoff, dt))
        self.value /= np.linalg.norm(self.value)
        self.timestamp = timestamp
        return self.value.copy()


@dataclass(frozen=True)
class FilteredTransform:
    rvec: np.ndarray
    tvec: np.ndarray
    rotation_matrix: np.ndarray
    quaternion: np.ndarray
    euler_zyx_deg: np.ndarray


class PoseFilter:
    def __init__(self) -> None:
        self.translation = OneEuroVector()
        self.rotation = AdaptiveQuaternionFilter()

    def reset(self) -> None:
        self.translation.reset()
        self.rotation.reset()

    def update(self, rvec: np.ndarray, tvec: np.ndarray, timestamp: float) -> FilteredTransform:
        matrix, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
        quaternion = rotation_matrix_to_quaternion(matrix)
        filtered_t = self.translation.update(np.asarray(tvec).reshape(3), timestamp)
        filtered_q = self.rotation.update(quaternion, timestamp)
        filtered_matrix = quaternion_to_rotation_matrix(filtered_q)
        filtered_rvec, _ = cv2.Rodrigues(filtered_matrix)
        return FilteredTransform(
            filtered_rvec.reshape(3, 1),
            filtered_t.reshape(3, 1),
            filtered_matrix,
            filtered_q,
            rotation_matrix_to_euler_zyx(filtered_matrix),
        )
