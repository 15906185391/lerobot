# LeRobot ROS 2 Camera

ROS 2 camera integration for LeRobot. This package subscribes to `sensor_msgs/msg/Image` topics and exposes frames through the LeRobot camera interface.

Before using it, make sure your ROS 2 environment is sourced so `rclpy`, `cv_bridge`, and `sensor_msgs` are importable from Python.

Example:

```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.cameras="{ front: {type: ros2, topic_name: /camera/color/image_raw, width: 640, height: 480, fps: 30} }" \
    --dataset.repo_id=${HF_USER}/my-dataset \
    --dataset.num_episodes=5
```
