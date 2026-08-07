"""ROS 2 camera plugin package for LeRobot."""

from .lerobot_camera_ros2.config import LeRobotCameraROS2Config, ROS2CameraConfig


def __getattr__(name: str):
    if name == "ROS2Camera":
        from .lerobot_camera_ros2.camera import ROS2Camera

        return ROS2Camera
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["LeRobotCameraROS2Config", "ROS2CameraConfig", "ROS2Camera"]
