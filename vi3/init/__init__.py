"""The IMU init block: the GT-free visual-inertial initialization of a window
(gravity, gyro bias, metric scale, initial velocity) and its IMU target
trajectory, which every grounding method (``vi3.grounding_methods``) builds on."""

from vi3.init.imu_init import IMUInit
from vi3.init.presets import PRESETS, resolve_preset

__all__ = ["IMUInit", "PRESETS", "resolve_preset"]
