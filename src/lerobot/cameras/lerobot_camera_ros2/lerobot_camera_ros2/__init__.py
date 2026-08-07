"""
LeRobot ROS2 Camera Plugin

This package provides ROS 2 camera integration for the LeRobot framework,
following LeRobot's plugin naming conventions.
"""

from .config import LeRobotCameraROS2Config, ROS2CameraConfig

__version__ = "0.1.0"


def __getattr__(name: str):
    if name == "ROS2Camera":
        from .camera import ROS2Camera

        return ROS2Camera
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "LeRobotCameraROS2Config",
    "ROS2CameraConfig",
    "ROS2Camera",
]
