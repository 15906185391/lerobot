#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from lerobot.lerobot_types import RobotAction


def ordered_action_items(
    action: RobotAction, action_order: Iterable[str] | None = None
) -> list[tuple[str, float]]:
    """Return action items in robot feature order, with unknown keys appended deterministically."""
    used_keys = set()
    items: list[tuple[str, float]] = []

    if action_order is not None:
        for key in action_order:
            if key in action:
                items.append((key, float(action[key])))
                used_keys.add(key)

    for key in sorted(set(action) - used_keys):
        items.append((key, float(action[key])))

    return items


def action_key_to_joint_name(key: str) -> str:
    return key.removesuffix(".pos")


class ROS2ActionPublisher:
    """Publish LeRobot actions as `sensor_msgs/msg/JointState` for live jitter inspection."""

    def __init__(
        self,
        *,
        topic: str = "/lerobot/action",
        node_name: str = "lerobot_action_publisher",
        frame_id: str = "",
        queue_size: int = 10,
    ) -> None:
        try:
            import rclpy
            from rclpy.node import Node
            from sensor_msgs.msg import JointState
        except ImportError as exc:
            raise ImportError(
                "Publishing actions to ROS 2 requires `rclpy` and `sensor_msgs`. "
                "Please source your ROS 2 environment before running LeRobot."
            ) from exc

        if not rclpy.ok():
            rclpy.init()

        self._rclpy = rclpy
        self._JointState: Any = JointState
        self._node: Node | None = Node(node_name)
        self._publisher = self._node.create_publisher(JointState, topic, queue_size)
        self._frame_id = frame_id
        self.topic = topic

    def publish(self, action: RobotAction, action_order: Iterable[str] | None = None) -> None:
        if self._node is None:
            return

        items = ordered_action_items(action, action_order)
        msg = self._JointState()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id
        msg.name = [action_key_to_joint_name(key) for key, _value in items]
        msg.position = [value for _key, value in items]
        self._publisher.publish(msg)

    def close(self) -> None:
        if self._node is None:
            return
        self._node.destroy_node()
        self._node = None
