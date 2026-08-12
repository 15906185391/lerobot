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

WHEELED_ARM_ACTIVE_ARMS_ACTION_KEY = "__wheeled_arm_active_arms"


def wheeled_arm_cameras_config() -> dict[str, CameraConfig]:
    return {
        "front": LeRobotCameraROS2Config(
            topic_name="/camera/color/image_raw",
            node_name="wheeled_arm_front_camera",
            width=640,
            height=480,
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

    # Robot-side command loop. When enabled, send_action() updates the latest target
    # and a background loop republishes it at control_dt, decoupling hardware command
    # timing from camera/Rerun/dataset loop timing.
    use_control_loop: bool = True
    control_dt: float = 1.0 / 250.0
    # Interpolate between incoming low-rate actions inside the robot-side control loop.
    # With the default 30Hz record/IK loop, this fills the 250Hz command stream with
    # intermediate joint targets instead of repeating a stair-stepped position.
    interpolate_control_loop_actions: bool = True
    action_interpolation_duration_s: float = 1.0 / 30.0

    # Wait after creating the LCM handler so subscriptions can receive the first robot state.
    connect_timeout_s: float = 1.0

    # Manipulation LCM URL used to communicate with the robot controller.
    lcm_url: str = "udpm://239.255.76.67:8880?ttl=1"

    # Robot joint feedback older than this is ignored by closed-loop teleoperators.
    # Set to None to accept the latest received feedback regardless of age.
    state_feedback_timeout_s: float | None = 1.0

    # Safety gate for physical operation: require fresh left/right arm LCM feedback before
    # starting teleoperation and before computing reset movej start positions.
    require_fresh_feedback: bool = True

    # Maximum time to wait for fresh left/right arm state feedback at safety gates.
    feedback_wait_timeout_s: float = 5.0

    # Software-only robot body mode. When enabled, wheeled_arm does not create an LCM
    # handler and joint observations come from the last commanded action. Cameras are
    # still real by default so recording can be tested with a connected camera but no robot.
    mock: bool = False

    # Generate synthetic camera images in mock mode. Leave this false when a real camera is
    # connected and only the robot body/LCM feedback should be simulated.
    mock_cameras: bool = False

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

        if self.feedback_wait_timeout_s < 0:
            raise ValueError("`feedback_wait_timeout_s` must be non-negative.")
        if self.control_dt <= 0:
            raise ValueError("`control_dt` must be positive.")
        if self.action_interpolation_duration_s <= 0:
            raise ValueError("`action_interpolation_duration_s` must be positive.")
