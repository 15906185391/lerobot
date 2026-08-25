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
from typing import Literal

from lerobot.cameras import CameraConfig
from lerobot.cameras.lerobot_camera_ros2 import ROS2CameraConfig

from ..config import RobotConfig

WheeledArmEndEffector = Literal["gripper", "suction"]
WHEELED_ARM_END_EFFECTOR_TYPES: tuple[WheeledArmEndEffector, ...] = ("gripper", "suction")

WHEELED_ARM_ARM_JOINT_NAMES = [
    *(f"left_arm_{idx}" for idx in range(7)),
    *(f"right_arm_{idx}" for idx in range(7)),
]
WHEELED_ARM_GRIPPER_NAMES = [
    "left_gripper",
    "right_gripper",
]
WHEELED_ARM_SUCTION_NAMES = [
    "left_suction",
    "right_suction",
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


WHEELED_ARM_DEFAULT_ROS2_CAMERA_TOPIC = "/camera/color/image_raw"


def wheeled_arm_end_effector_names(end_effector: WheeledArmEndEffector) -> list[str]:
    if end_effector == "gripper":
        return WHEELED_ARM_GRIPPER_NAMES.copy()
    if end_effector == "suction":
        return WHEELED_ARM_SUCTION_NAMES.copy()
    raise ValueError(
        f"Unknown wheeled_arm end effector {end_effector!r}. "
        f"Known end effectors are {list(WHEELED_ARM_END_EFFECTOR_TYPES)}."
    )


def wheeled_arm_joint_names(end_effector: WheeledArmEndEffector) -> list[str]:
    return [*WHEELED_ARM_ARM_JOINT_NAMES, *wheeled_arm_end_effector_names(end_effector)]


def wheeled_arm_parts(end_effector: WheeledArmEndEffector) -> tuple[str, ...]:
    return ("left_arm", "right_arm", *wheeled_arm_end_effector_names(end_effector))


def wheeled_arm_suction_operation_mode(
    side: str,
    active: bool,
    *,
    left_suction_operation_mode: int,
    right_suction_operation_mode: int,
    left_release_operation_mode: int,
    right_release_operation_mode: int,
) -> int:
    if side == "left":
        return left_suction_operation_mode if active else left_release_operation_mode
    if side == "right":
        return right_suction_operation_mode if active else right_release_operation_mode
    raise ValueError(f"Unknown suction side: {side!r}")


def wheeled_arm_cameras_config() -> dict[str, CameraConfig]:
    return {
        "front": ROS2CameraConfig(
            topic_name=WHEELED_ARM_DEFAULT_ROS2_CAMERA_TOPIC,
            image_transport="compressed",
            width=640,
            height=480,
            fps=30,
            queue_size=1,
            warmup_s=3,
        )
    }


@RobotConfig.register_subclass("wheeled_arm")
@dataclass
class WheeledArmConfig(RobotConfig):
    cameras: dict[str, CameraConfig] = field(default_factory=wheeled_arm_cameras_config)

    # Physical end effector mounted on the left/right wrists. This controls the
    # final two data features and the Manipulation LCM command topics.
    end_effector: WheeledArmEndEffector = "gripper"

    joint_names: list[str] = field(default_factory=lambda: WHEELED_ARM_JOINT_NAMES.copy())

    # Only parts listed here will be commanded through LCM.
    controlled_parts: list[str] = field(default_factory=lambda: list(WHEELED_ARM_PARTS))

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
    # Fallback interpolation duration used before the second action arrives or when adaptive duration is off.
    action_interpolation_duration_s: float = 1.0 / 30.0
    # Adapt the interpolation duration to the measured interval between incoming actions. This keeps
    # the 250Hz command stream aligned with the real record/teleop loop when cameras or visualization
    # make the outer loop run slower or jitter around the nominal fps.
    adaptive_action_interpolation_duration: bool = True
    action_interpolation_min_duration_s: float = 0.02
    action_interpolation_max_duration_s: float = 0.06
    # Stop publishing moving commands if no fresh action arrives for this long.
    # This protects the real robot when the outer record/teleop loop stalls.
    # Set to None to disable the watchdog.
    action_watchdog_timeout_s: float | None = 0.1

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

    # Suction command mapping aligned with `suction.lcm`.
    # Legacy single-mode defaults are kept for backward compatibility.
    suction_on_threshold: float = 0.5
    suction_off_operation_mode: int = 0
    suction_on_operation_mode: int = 1
    suction_activation_threshold: float = 0.05
    suction_left_suction_operation_mode: int = 11
    suction_right_suction_operation_mode: int = 12
    suction_left_release_operation_mode: int = 13
    suction_right_release_operation_mode: int = 14
    suction_max_vacuum_pct: float = 70.0
    suction_detect_vacuum_pct: float = 60.0
    suction_grip_timeout_100ms: float = 20.0

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.end_effector not in WHEELED_ARM_END_EFFECTOR_TYPES:
            raise ValueError(
                f"`end_effector` must be one of {list(WHEELED_ARM_END_EFFECTOR_TYPES)}, "
                f"got {self.end_effector!r}."
            )

        if self.joint_names == WHEELED_ARM_JOINT_NAMES:
            self.joint_names = wheeled_arm_joint_names(self.end_effector)
        if self.controlled_parts == list(WHEELED_ARM_PARTS):
            self.controlled_parts = list(wheeled_arm_parts(self.end_effector))

        if len(self.joint_names) != 16:
            raise ValueError(
                f"`joint_names` must contain exactly 16 arm/gripper names, got {len(self.joint_names)}."
            )

        duplicate_names = {name for name in self.joint_names if self.joint_names.count(name) > 1}
        if duplicate_names:
            raise ValueError(f"`joint_names` must be unique, got duplicates: {sorted(duplicate_names)}.")

        known_parts = set(wheeled_arm_parts(self.end_effector))
        unknown_parts = set(self.controlled_parts) - known_parts
        if unknown_parts:
            raise ValueError(
                f"`controlled_parts` contains unknown parts {sorted(unknown_parts)}. "
                f"Known parts are {list(wheeled_arm_parts(self.end_effector))}."
            )

        if self.feedback_wait_timeout_s < 0:
            raise ValueError("`feedback_wait_timeout_s` must be non-negative.")
        if self.control_dt <= 0:
            raise ValueError("`control_dt` must be positive.")
        if self.action_interpolation_duration_s <= 0:
            raise ValueError("`action_interpolation_duration_s` must be positive.")
        if self.action_interpolation_min_duration_s <= 0:
            raise ValueError("`action_interpolation_min_duration_s` must be positive.")
        if self.action_interpolation_max_duration_s <= 0:
            raise ValueError("`action_interpolation_max_duration_s` must be positive.")
        if self.action_interpolation_min_duration_s > self.action_interpolation_max_duration_s:
            raise ValueError(
                "`action_interpolation_min_duration_s` must be less than or equal to "
                "`action_interpolation_max_duration_s`."
            )
        if self.action_watchdog_timeout_s is not None and self.action_watchdog_timeout_s <= 0:
            raise ValueError("`action_watchdog_timeout_s` must be positive or None.")
        if not 0.0 <= self.suction_on_threshold <= 1.0:
            raise ValueError("`suction_on_threshold` must be in [0, 1].")
        if not 0.0 <= self.suction_activation_threshold <= 1.0:
            raise ValueError("`suction_activation_threshold` must be in [0, 1].")
        if not 0.0 <= self.suction_max_vacuum_pct <= 70.0:
            raise ValueError("`suction_max_vacuum_pct` must be in [0, 70].")
        if not -1.0 <= self.suction_detect_vacuum_pct <= 70.0:
            raise ValueError("`suction_detect_vacuum_pct` must be -1 or in [0, 70].")
        if self.suction_grip_timeout_100ms < -1.0:
            raise ValueError("`suction_grip_timeout_100ms` must be >= -1.")
