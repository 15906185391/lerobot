"""Configuration classes for ROS 2 cameras in LeRobot."""

from dataclasses import dataclass

from lerobot.cameras.configs import CameraConfig


@CameraConfig.register_subclass("ros2")
@dataclass
class ROS2CameraConfig(CameraConfig):
    """Configuration class for ROS 2 cameras.

    This configuration allows LeRobot to receive image data from ROS 2 topics,
    enabling integration with ROS 2-based camera systems.

    Example:
        ```python
        config = ROS2CameraConfig(
            topic_name="/camera/color/image_raw",
            node_name="wrist_camera_node",
            namespace="robot",
            fps=30,
            width=640,
            height=480
        )
        ```

    Args:
        topic_name: ROS 2 image topic name (e.g., "/camera/color/image_raw")
        node_name: ROS 2 node name for this camera (e.g., "wrist_camera_node")
        namespace: ROS 2 namespace (optional, defaults to empty string)
        timeout_ms: Timeout for receiving images in milliseconds
        queue_size: ROS 2 subscription queue size
        encoding: Expected image encoding (e.g., "bgr8", "rgb8", "mono8")
        warmup_s: Time waiting for a first frame before returning from connect
    """

    topic_name: str = "/camera/color/image_raw"
    node_name: str = "lerobot_ros2_camera"
    namespace: str = ""
    timeout_ms: float = 1000.0
    queue_size: int = 10
    encoding: str = "bgr8"
    warmup_s: float = 1.0
    depth_topic_name: str | None = None
    depth_encoding: str = "32FC1"
    # LeRobot required parameters with default values
    width: int = 1280
    height: int = 720
    fps: int = 30


@CameraConfig.register_subclass("lerobot_camera_ros2")
@dataclass
class LeRobotCameraROS2Config(ROS2CameraConfig):
    """Backward-compatible alias for older configs using `type: lerobot_camera_ros2`."""
