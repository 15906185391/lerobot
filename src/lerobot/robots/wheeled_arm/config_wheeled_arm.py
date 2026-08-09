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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.cameras.lerobot_camera_ros2 import LeRobotCameraROS2Config

from ..config import RobotConfig

WHEELED_ARM_ARM_JOINT_NAMES = [
    *(f"left_arm_{idx}" for idx in range(7)),
    *(f"right_arm_{idx}" for idx in range(7)),
]
WHEELED_ARM_GRIPPER_NAMES = [
    "left_gripper",
    "right_gripper",
]
WHEELED_ARM_JOINT_NAMES = [
    *WHEELED_ARM_ARM_JOINT_NAMES,
    *WHEELED_ARM_GRIPPER_NAMES,
]

WHEELED_ARM_PARTS = (
    "left_arm",
    "right_arm",
    "left_gripper",
    "right_gripper",
)


def wheeled_arm_cameras_config() -> dict[str, CameraConfig]:
    return {
        "front": LeRobotCameraROS2Config(
            topic_name="/camera/color/image_raw",
            node_name="wheeled_arm_front_camera",
            width=1280,
            height=720,
            fps=30,
        )
    }


@RobotConfig.register_subclass("wheeled_arm")
@dataclass
class WheeledArmConfig(RobotConfig):
    cameras: dict[str, CameraConfig] = field(
        default_factory=wheeled_arm_cameras_config)

    joint_names: list[str] = field(
        default_factory=lambda: WHEELED_ARM_JOINT_NAMES.copy())

    # Only parts listed here will be commanded through LCM.
    controlled_parts: list[str] = field(
        default_factory=lambda: list(WHEELED_ARM_PARTS))

    # Caps target jumps relative to the latest observed joint position. None disables clipping.
    max_relative_target: float | dict[str, float] | None = None

    # Wait after creating the LCM handler so subscriptions can receive the first robot state.
    connect_timeout_s: float = 1.0

    # Robot joint feedback older than this is ignored by closed-loop teleoperators.
    # Set to None to accept the latest received feedback regardless of age.
    state_feedback_timeout_s: float | None = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()

        if len(self.joint_names) != 16:
            raise ValueError(
                f"`joint_names` must contain exactly 16 arm/gripper names, got {len(self.joint_names)}."
            )

        duplicate_names = {
            name for name in self.joint_names if self.joint_names.count(name) > 1}
        if duplicate_names:
            raise ValueError(
                f"`joint_names` must be unique, got duplicates: {sorted(duplicate_names)}.")

        unknown_parts = set(self.controlled_parts) - set(WHEELED_ARM_PARTS)
        if unknown_parts:
            raise ValueError(
                f"`controlled_parts` contains unknown parts {sorted(unknown_parts)}. "
                f"Known parts are {list(WHEELED_ARM_PARTS)}."
            )
