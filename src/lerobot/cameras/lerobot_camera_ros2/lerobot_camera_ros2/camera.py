"""ROS 2 camera implementation for LeRobot."""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from lerobot.cameras.camera import Camera
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .config import ROS2CameraConfig

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import rclpy
    from cv_bridge import CvBridge
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
    from sensor_msgs.msg import CompressedImage, Image
else:
    rclpy = None
    CvBridge = Any
    SingleThreadedExecutor = Any
    Node = Any
    QoSHistoryPolicy = Any
    QoSProfile = Any
    QoSReliabilityPolicy = Any
    Image = Any
    CompressedImage = Any


def _require_ros2_dependencies() -> None:
    global rclpy, CvBridge, SingleThreadedExecutor, Node
    global QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy, Image, CompressedImage

    if rclpy is not None:
        return

    try:
        import rclpy as _rclpy
        from rclpy.executors import SingleThreadedExecutor as _SingleThreadedExecutor
        from rclpy.node import Node as _Node
        from rclpy.qos import (
            QoSHistoryPolicy as _QoSHistoryPolicy,
            QoSProfile as _QoSProfile,
            QoSReliabilityPolicy as _QoSReliabilityPolicy,
        )
        from sensor_msgs.msg import CompressedImage as _CompressedImage, Image as _Image
    except AttributeError as exc:
        if "_ARRAY_API" in str(exc):
            raise ImportError(
                "ROS 2 `cv_bridge` was compiled against NumPy 1.x, but this Python "
                "environment is using NumPy 2.x. Use a NumPy 1.x environment for ROS "
                "Humble cv_bridge, for example `pip install 'numpy<2'`, or rebuild "
                "cv_bridge against NumPy 2.x."
            ) from exc
        raise
    except ImportError as exc:
        raise ImportError(
            "ROS 2 camera support requires `rclpy` and `sensor_msgs` to be "
            "available in the current Python environment."
        ) from exc

    _cv_bridge_cls = None
    numpy_major = int(np.__version__.split(".", maxsplit=1)[0])
    should_try_cv_bridge = os.getenv("LEROBOT_ROS2_DISABLE_CV_BRIDGE", "0") != "1" and (
        numpy_major < 2 or os.getenv("LEROBOT_ROS2_FORCE_CV_BRIDGE", "0") == "1"
    )
    if should_try_cv_bridge:
        try:
            import cv_bridge as _cv_bridge

            _cv_bridge_cls = _cv_bridge.CvBridge
        except AttributeError as exc:
            if "_ARRAY_API" not in str(exc):
                raise
            logger.warning(
                "ROS 2 `cv_bridge` is not compatible with the current NumPy; "
                "falling back to manual RGB image conversion."
            )
        except ImportError:
            logger.warning("`cv_bridge` is unavailable; falling back to manual RGB image conversion.")
    elif numpy_major >= 2:
        logger.warning(
            "Skipping ROS 2 `cv_bridge` because NumPy %s is installed; "
            "falling back to manual RGB image conversion.",
            np.__version__,
        )

    rclpy = _rclpy
    CvBridge = _cv_bridge_cls
    SingleThreadedExecutor = _SingleThreadedExecutor
    Node = _Node
    QoSHistoryPolicy = _QoSHistoryPolicy
    QoSProfile = _QoSProfile
    QoSReliabilityPolicy = _QoSReliabilityPolicy
    Image = _Image
    CompressedImage = _CompressedImage


class ROS2Camera(Camera):
    """ROS 2 camera implementation for LeRobot.

    This class provides a LeRobot camera interface that receives image data
    from ROS 2 topics. It supports both synchronous and asynchronous frame reading.

    Example:
        ```python
        from lerobot_ros2_devices.cameras import ROS2Camera, ROS2CameraConfig

        # Create configuration
        config = ROS2CameraConfig(topic_name="/camera/color/image_raw", fps=30, width=640, height=480)

        # Create and connect camera
        camera = ROS2Camera(config)
        camera.connect()

        # Read frames
        image = camera.read()
        async_image = camera.async_read()

        # Disconnect when done
        camera.disconnect()
        ```
    """

    def __init__(self, config: ROS2CameraConfig):
        """Initialize the ROS 2 camera.

        Args:
            config: ROS 2 camera configuration
        """
        _require_ros2_dependencies()
        super().__init__(config)
        self.config = config
        self.topic_name = config.topic_name
        self.node_name = config.node_name
        self.namespace = config.namespace
        self.timeout_ms = config.timeout_ms
        self.queue_size = config.queue_size
        self.qos_profile = (config.qos_profile or "sensor_data").lower().strip()
        if self.qos_profile not in {"sensor_data", "default"}:
            raise ValueError(
                f"ROS2Camera qos_profile must be 'sensor_data' or 'default', got {config.qos_profile!r}"
            )
        self.encoding = config.encoding
        self.image_transport = (config.image_transport or "raw").lower().strip()
        if self.image_transport not in {"raw", "compressed"}:
            raise ValueError(
                f"ROS2Camera image_transport must be 'raw' or 'compressed', got {config.image_transport!r}"
            )
        self.warmup_s = config.warmup_s
        self.depth_topic_name = config.depth_topic_name
        self.depth_encoding = config.depth_encoding

        # ROS 2 components
        self.ros_node: Node | None = None
        self.executor: SingleThreadedExecutor | None = None
        self.executor_thread: threading.Thread | None = None
        self._disconnecting = False
        self.image_subscription = None
        self.depth_subscription = None

        # Image processing
        self.latest_image: np.ndarray | None = None
        self.latest_depth: np.ndarray | None = None
        self.latest_image_timestamp: float | None = None
        self.latest_depth_timestamp: float | None = None
        self.image_lock = threading.Lock()
        self.depth_lock = threading.Lock()
        self.image_received_event = threading.Event()
        self.depth_received_event = threading.Event()
        self._dimensions_initialized = False
        self._cv_bridge_enabled = (
            os.getenv("LEROBOT_ROS2_DISABLE_CV_BRIDGE", "0") != "1" and CvBridge is not None
        )
        if self.depth_topic_name and not self._cv_bridge_enabled:
            raise ImportError(
                "ROS 2 depth camera support requires `cv_bridge`. Fix the cv_bridge/NumPy "
                "environment or disable `depth_topic_name`."
            )
        self.bridge = CvBridge() if self._cv_bridge_enabled else None
        self._last_cv_bridge_error_log_ts = 0.0
        self._cv_bridge_error_log_interval_s = 2.0

        # Connection state
        self._connected = False

    def __str__(self) -> str:
        return f"ROS2Camera(topic={self.topic_name}, transport={self.image_transport}, node={self.node_name})"

    @property
    def is_connected(self) -> bool:
        """Check if the camera is currently connected.

        Returns:
            bool: True if connected and ready to capture frames, False otherwise.
        """
        return self._connected and self.ros_node is not None

    def get_actual_image_dimensions(self) -> tuple[int, int] | None:
        """Get the actual image dimensions from the received images.

        Returns:
            tuple[int, int] | None: (width, height) if images have been received, None otherwise.
        """
        with self.image_lock:
            if self.latest_image is not None:
                # (width, height)
                return self.latest_image.shape[1], self.latest_image.shape[0]
            return None

    @staticmethod
    def find_cameras(discovery_timeout_s: float = 3.0) -> list[dict[str, Any]]:
        """Discover available ROS 2 camera topics.

        This method scans the ROS 2 system for available raw and compressed image topics.

        Returns:
            List[Dict[str, Any]]: List of dictionaries containing camera information.
        """
        cameras: list[dict[str, Any]] = []

        try:
            _require_ros2_dependencies()

            # Initialize ROS 2 if not already done
            if not rclpy.ok():
                rclpy.init()

            temp_node = Node("ros2_camera_discovery")
            try:
                deadline_s = time.monotonic() + max(0.0, discovery_timeout_s)
                while True:
                    cameras = ROS2Camera._cameras_from_topic_names_and_types(
                        temp_node.get_topic_names_and_types()
                    )
                    if cameras or time.monotonic() >= deadline_s:
                        break
                    rclpy.spin_once(temp_node, timeout_sec=0.1)
            finally:
                temp_node.destroy_node()

        except Exception as e:
            logger.warning(f"Failed to discover ROS 2 cameras: {e}")

        return cameras

    @staticmethod
    def _cameras_from_topic_names_and_types(
        topic_names_and_types: list[tuple[str, list[str]]],
    ) -> list[dict[str, Any]]:
        cameras: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()

        for topic_name, topic_types in topic_names_and_types:
            if "sensor_msgs/msg/Image" in topic_types:
                key = (topic_name, "raw")
                if key not in seen:
                    topic_parts = topic_name.split("/")
                    cameras.append(
                        {
                            "id": topic_name,
                            "topic_name": topic_name,
                            "image_transport": "raw",
                            "type": "ROS2",
                            "description": f"ROS 2 Image topic: {topic_name}",
                            "namespace": topic_parts[1] if len(topic_parts) > 2 else "",
                        }
                    )
                    seen.add(key)

            if "sensor_msgs/msg/CompressedImage" in topic_types:
                base_topic_name = topic_name.removesuffix("/compressed")
                key = (base_topic_name, "compressed")
                if key not in seen:
                    topic_parts = base_topic_name.split("/")
                    cameras.append(
                        {
                            "id": topic_name,
                            "topic_name": base_topic_name,
                            "image_transport": "compressed",
                            "type": "ROS2",
                            "description": f"ROS 2 CompressedImage topic: {topic_name}",
                            "namespace": topic_parts[1] if len(topic_parts) > 2 else "",
                        }
                    )
                    seen.add(key)

        return cameras

    def connect(self, warmup: bool = True) -> None:
        """Connect to the ROS 2 camera topic.

        Args:
            warmup: If True, wait for the first image before returning.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        try:
            _require_ros2_dependencies()

            # Initialize ROS 2 if not already done
            if not rclpy.ok():
                rclpy.init()

            # Create ROS 2 node with unique name
            self.ros_node = Node(self.node_name, namespace=self.namespace)

            # Create image subscription
            image_topic_name = self._image_subscription_topic_name()
            image_msg_type = CompressedImage if self.image_transport == "compressed" else Image
            image_qos = self._subscription_qos()
            image_callback = (
                self._compressed_image_callback
                if self.image_transport == "compressed"
                else self._image_callback
            )
            self.image_subscription = self.ros_node.create_subscription(
                image_msg_type,
                image_topic_name,
                image_callback,
                image_qos,
            )
            if self.depth_topic_name:
                self.depth_subscription = self.ros_node.create_subscription(
                    Image,
                    self.depth_topic_name,
                    self._depth_callback,
                    image_qos,
                )

            # Start executor in separate thread
            self._disconnecting = False
            self.executor = SingleThreadedExecutor()
            self.executor.add_node(self.ros_node)
            self.executor_thread = threading.Thread(target=self._spin_executor, daemon=True)
            self.executor_thread.start()
            self._connected = True

            # Wait for connection to establish
            if self.warmup_s > 0:
                time.sleep(self.warmup_s)

            # Warmup: wait for first image
            if warmup:
                logger.info(f"Warming up {self}...")
                if not self.image_received_event.wait(timeout=max(self.warmup_s, self.timeout_ms / 1000.0)):
                    logger.warning(f"No image received from {self.topic_name} during warmup")

            logger.info("Connected to ROS 2 camera: %s", image_topic_name)

        except Exception as e:
            logger.error(f"Failed to connect to ROS 2 camera: {e}")
            self.disconnect()
            raise

    def _subscription_qos(self) -> QoSProfile | int:
        if self.qos_profile == "default":
            return self.queue_size
        return QoSProfile(
            depth=self.queue_size,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
        )

    def _image_subscription_topic_name(self) -> str:
        if self.image_transport != "compressed":
            return self.topic_name
        if self.topic_name.endswith("/compressed"):
            return self.topic_name
        return f"{self.topic_name}/compressed"

    def _update_image_dimensions(self, actual_width: int, actual_height: int) -> None:
        if self._dimensions_initialized:
            return

        # Set camera dimensions from first received image
        if self.width is None or self.height is None:
            self.width = actual_width
            self.height = actual_height
            logger.info("Auto-detected image dimensions: %sx%s", actual_width, actual_height)
        elif self.width != actual_width or self.height != actual_height:
            logger.warning(
                "Image dimensions mismatch: configured %sx%s, actual %sx%s. Updating to actual dimensions.",
                self.width,
                self.height,
                actual_width,
                actual_height,
            )
            self.width = actual_width
            self.height = actual_height

        self._dimensions_initialized = True

    def _image_callback(self, msg: Image) -> None:
        """ROS 2 image callback function.

        Args:
            msg: ROS 2 Image message
        """
        try:
            self._update_image_dimensions(int(msg.width), int(msg.height))

            # Convert ROS image message to OpenCV format
            cv_image = None
            if self._cv_bridge_enabled:
                try:
                    cv_image = self.bridge.imgmsg_to_cv2(msg, self.encoding)
                except Exception as e:
                    now = time.monotonic()
                    if (now - self._last_cv_bridge_error_log_ts) >= self._cv_bridge_error_log_interval_s:
                        logger.error(f"Failed to convert ROS image message: {e}")
                        logger.warning(
                            "Disabling cv_bridge for camera '%s' and falling back to manual converter.",
                            self.topic_name,
                        )
                        self._last_cv_bridge_error_log_ts = now
                    self._cv_bridge_enabled = False

            if cv_image is None:
                cv_image = self._manual_convert_image_msg(msg, self.encoding)
                if cv_image is None:
                    return

            # Convert to RGB format for LeRobot
            try:
                if self.encoding in ["bgr8", "bgra8"]:
                    rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                elif self.encoding in ["rgb8", "rgba8"]:
                    rgb_image = cv_image
                elif self.encoding == "mono8":
                    rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)
                else:
                    # Default to BGR to RGB conversion
                    rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            except Exception as e:
                logger.error(f"Failed to convert image color format: {e}")
                # Fallback: use the image as-is
                rgb_image = cv_image

            with self.image_lock:
                self.latest_image = rgb_image.copy()
                self.latest_image_timestamp = time.perf_counter()
                self.image_received_event.set()

        except Exception as e:
            logger.error(f"Error processing image: {e}")

    def _compressed_image_callback(self, msg: CompressedImage) -> None:
        """ROS 2 compressed image callback function."""
        try:
            encoded = np.frombuffer(msg.data, dtype=np.uint8)
            bgr_image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if bgr_image is None:
                logger.error("Failed to decode compressed ROS image from %s", self.topic_name)
                return

            actual_height, actual_width = bgr_image.shape[:2]
            self._update_image_dimensions(int(actual_width), int(actual_height))

            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

            with self.image_lock:
                self.latest_image = rgb_image.copy()
                self.latest_image_timestamp = time.perf_counter()
                self.image_received_event.set()

        except Exception as e:
            logger.error(f"Error processing compressed image: {e}")

    def _manual_convert_image_msg(self, msg: Image, encoding: str) -> np.ndarray | None:
        """Fallback conversion path that avoids cv_bridge."""
        try:
            src = (msg.encoding or "").lower()
            dst = (encoding or src or "bgr8").lower()
            h, w = int(msg.height), int(msg.width)
            if h <= 0 or w <= 0:
                logger.error("Manual conversion got invalid image size: %sx%s", w, h)
                return None

            raw = np.frombuffer(msg.data, dtype=np.uint8)
            if src in ("rgb8", "bgr8"):
                expected = h * w * 3
                if raw.size < expected:
                    logger.error(
                        "Manual conversion failed: RGB payload too small (%s < %s)",
                        raw.size,
                        expected,
                    )
                    return None
                img = raw[:expected].reshape(h, w, 3)
                if src != dst and dst in ("rgb8", "bgr8"):
                    return img[..., ::-1].copy()
                return img.copy()

            if src in ("rgba8", "bgra8"):
                expected = h * w * 4
                if raw.size < expected:
                    logger.error(
                        "Manual conversion failed: RGBA payload too small (%s < %s)",
                        raw.size,
                        expected,
                    )
                    return None
                img = raw[:expected].reshape(h, w, 4)
                rgb = img[..., :3] if src == "rgba8" else img[..., [2, 1, 0]]
                if dst == "bgr8":
                    return rgb[..., ::-1].copy()
                return rgb.copy()

            if src == "mono8":
                expected = h * w
                if raw.size < expected:
                    logger.error(
                        "Manual conversion failed: mono payload too small (%s < %s)",
                        raw.size,
                        expected,
                    )
                    return None
                gray = raw[:expected].reshape(h, w)
                if dst in ("rgb8", "bgr8", "rgba8", "bgra8"):
                    return np.repeat(gray[:, :, None], 3, axis=2)
                return gray.copy()

            logger.error("Manual conversion unsupported encoding: src=%s dst=%s", src, dst)
            return None
        except Exception as exc:
            logger.error("Manual conversion exception: %s", exc)
            return None

    def _depth_callback(self, msg: Image) -> None:
        if not self.depth_topic_name:
            return
        try:
            cv_depth = self.bridge.imgmsg_to_cv2(msg, self.depth_encoding)
            depth_float = np.array(cv_depth, dtype=np.float32, copy=False)
            depth_clean = np.nan_to_num(depth_float, nan=0.0, posinf=0.0, neginf=0.0)

            valid_mask = np.isfinite(depth_clean)
            if np.any(valid_mask):
                min_val = float(depth_clean[valid_mask].min())
                max_val = float(depth_clean[valid_mask].max())
            else:
                min_val = 0.0
                max_val = 0.0

            if max_val > min_val:
                depth_norm = (depth_clean - min_val) / (max_val - min_val)
            else:
                depth_norm = np.zeros_like(depth_clean, dtype=np.float32)

            depth_vis = np.clip(depth_norm * 255.0, 0.0, 255.0).astype(np.uint8)
            if depth_vis.ndim == 2:
                depth_vis = np.repeat(depth_vis[:, :, None], 3, axis=2)
            elif depth_vis.shape[-1] == 1:
                depth_vis = np.repeat(depth_vis, 3, axis=2)

            with self.depth_lock:
                self.latest_depth = depth_vis.copy()
                self.latest_depth_timestamp = time.perf_counter()
                self.depth_received_event.set()
        except Exception as exc:
            logger.error(f"Failed to process depth image: {exc}")

    def _spin_executor(self) -> None:
        executor = self.executor
        if executor is None:
            return
        try:
            executor.spin()
        except Exception as exc:  # noqa: BLE001 - rclpy raises its own shutdown sentinel from this thread.
            if self._disconnecting or exc.__class__.__name__ == "ExternalShutdownException":
                logger.debug("ROS 2 camera executor stopped during shutdown: %s", exc)
                return
            logger.exception("ROS 2 camera executor stopped unexpectedly")

    def read(self) -> Any:
        return self.async_read(timeout_ms=self.timeout_ms)

    def async_read(self, timeout_ms: float | None = None) -> Any:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

        timeout_ms = timeout_ms or self.timeout_ms

        if not self.image_received_event.wait(timeout=timeout_ms / 1000.0):
            raise TimeoutError(f"No image received within {timeout_ms}ms")

        rgb_image = None
        with self.image_lock:
            if self.latest_image is not None:
                rgb_image = self.latest_image.copy()

        if self.depth_topic_name:
            depth_image = None
            if self.depth_received_event.wait(timeout=timeout_ms / 1000.0):
                with self.depth_lock:
                    if self.latest_depth is not None:
                        depth_image = self.latest_depth.copy()
            return {"rgb": rgb_image, "depth": depth_image}

        if rgb_image is None:
            raise RuntimeError("No image available")
        return rgb_image

    def read_latest(self, max_age_ms: int = 500) -> Any:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

        with self.image_lock:
            rgb_image = self.latest_image.copy() if self.latest_image is not None else None
            timestamp = self.latest_image_timestamp

        if rgb_image is None or timestamp is None:
            raise RuntimeError(f"{self} has not received any images yet")

        age_ms = (time.perf_counter() - timestamp) * 1e3
        if age_ms > max_age_ms:
            raise TimeoutError(
                f"{self} latest image is too old: {age_ms:.1f} ms (max allowed: {max_age_ms} ms)."
            )

        return rgb_image

    def read_latest_depth(self, max_age_ms: int = 500) -> Any:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        if not self.depth_topic_name:
            raise RuntimeError(f"{self} is not configured with a depth topic")

        with self.depth_lock:
            depth_image = self.latest_depth.copy() if self.latest_depth is not None else None
            timestamp = self.latest_depth_timestamp

        if depth_image is None or timestamp is None:
            raise RuntimeError(f"{self} has not received any depth images yet")

        age_ms = (time.perf_counter() - timestamp) * 1e3
        if age_ms > max_age_ms:
            raise TimeoutError(
                f"{self} latest depth image is too old: {age_ms:.1f} ms (max allowed: {max_age_ms} ms)."
            )

        return depth_image

    def disconnect(self) -> None:
        """Disconnect from the camera and release resources."""
        self._connected = False
        self._disconnecting = True

        # Stop executor
        if self.executor:
            self.executor.shutdown()
            self.executor = None

        # Wait for thread to finish
        if self.executor_thread:
            self.executor_thread.join(timeout=2.0)
            self.executor_thread = None

        # Destroy node
        if self.ros_node:
            self.ros_node.destroy_node()
            self.ros_node = None
        self.image_subscription = None
        self.depth_subscription = None

        # Clean up resources
        with self.image_lock:
            self.latest_image = None
            self.latest_image_timestamp = None
            self.image_received_event.clear()
            self._dimensions_initialized = False
        with self.depth_lock:
            self.latest_depth = None
            self.latest_depth_timestamp = None
            self.depth_received_event.clear()

        self._disconnecting = False
        logger.info(f"Disconnected from ROS 2 camera: {self.topic_name}")
