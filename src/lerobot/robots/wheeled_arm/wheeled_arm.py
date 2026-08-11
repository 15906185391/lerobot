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

import logging
import time
from collections.abc import Callable
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np

from lerobot.cameras import make_cameras_from_configs
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_wheeled_arm import WheeledArmConfig

if TYPE_CHECKING:
    from .hardware_interface.lcm_handler import LCMHandler

logger = logging.getLogger(__name__)


def _make_lcm_handler(lcm_url: str) -> LCMHandler:
    try:
        from .hardware_interface.lcm_handler import LCMHandler
    except ModuleNotFoundError as exc:
        if exc.name == "lcm":
            raise ImportError(
                "'lcm' is required to control wheeled_arm. Install the robot SDK dependencies "
                "that provide the Python `lcm` module before connecting the robot. "
                "For the `xr` conda environment, run: "
                "`/home/kuanli/miniconda3/envs/xr/bin/python -m pip install lcm`."
            ) from exc
        raise

    try:
        return LCMHandler(lcm_url=lcm_url)
    except RuntimeError as exc:
        if "Couldn't create LCM" in str(exc):
            raise RuntimeError(
                f"Could not create wheeled_arm LCM connection for URL '{lcm_url}'. "
                "Ensure Linux has a multicast route for LCM. For loopback-only testing, "
                "run: `sudo ip link set lo multicast on` and "
                "`sudo ip route add 224.0.0.0/4 dev lo`."
            ) from exc
        raise


_PART_SLICES: dict[str, slice] = {
    "left_arm": slice(0, 7),
    "right_arm": slice(7, 14),
    "left_gripper": slice(14, 15),
    "right_gripper": slice(15, 16),
}

_PART_MOVING_FLAGS: dict[str, str] = {
    "left_arm": "left_arm_moving",
    "right_arm": "right_arm_moving",
    "left_gripper": "left_gripper_moving",
    "right_gripper": "right_gripper_moving",
    "head": "head_moving",
    "waist": "waist_moving",
    "leg": "leg_moving",
}

_RESET_LEFT_ARM_RAD = np.deg2rad([20.0, 70.0, -75.0, 100.0, -25.0, 0.0, 0.0]).astype(
    np.float32
)
_RESET_RIGHT_ARM_RAD = np.deg2rad([-20.0, 70.0, 75.0, 100.0, 25.0, 0.0, 0.0]).astype(
    np.float32
)


class WheeledArm(Robot):
    """LeRobot wrapper for the LCM-controlled wheeled arm upper body."""

    config_class = WheeledArmConfig
    name = "wheeled_arm"

    def __init__(self, config: WheeledArmConfig):
        super().__init__(config)
        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras)
        self._handler: LCMHandler | None = None
        self._mock_connected = False
        self._mock_joint_pos: np.ndarray | None = None
        self._mock_frame_index = 0

        self._joint_index_by_action_key = {
            f"{joint_name}.pos": idx for idx, joint_name in enumerate(self.config.joint_names)
        }

    @property
    def _joints_ft(self) -> dict[str, type]:
        return {f"{joint_name}.pos": float for joint_name in self.config.joint_names}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        features: dict[str, tuple] = {}
        for cam_name, cam in self.cameras.items():
            if getattr(cam, "use_rgb", True):
                features[cam_name] = (cam.height, cam.width, 3)
            if getattr(cam, "use_depth", False):
                features[f"{cam_name}_depth"] = (cam.height, cam.width, 1)
        return features

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._joints_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._joints_ft

    @property
    def is_connected(self) -> bool:
        if self.config.mock:
            cameras_connected = self.config.mock_cameras or all(
                cam.is_connected for cam in self.cameras.values()
            )
            return self._mock_connected and cameras_connected
        return self._handler is not None and all(cam.is_connected for cam in self.cameras.values())

    @property
    def is_calibrated(self) -> bool:
        return True

    @property
    def has_valid_feedback(self) -> bool:
        if self.config.mock:
            return self.is_connected
        if self._handler is None:
            return False
        if hasattr(self._handler, "has_arm_state_feedback"):
            return bool(self._handler.has_arm_state_feedback(self.config.state_feedback_timeout_s))
        return bool(
            getattr(self._handler, "left_arm_state_updated", None)
            and self._handler.left_arm_state_updated.is_set()
            and getattr(self._handler, "right_arm_state_updated", None)
            and self._handler.right_arm_state_updated.is_set()
        )

    def wait_for_valid_feedback(
        self, timeout_s: float | None = None, min_time_s: float | None = None
    ) -> bool:
        """Wait until fresh left/right arm LCM feedback is available."""
        if self.config.mock:
            return self.has_valid_feedback
        if self._handler is None:
            return False
        timeout_s = self.config.feedback_wait_timeout_s if timeout_s is None else timeout_s
        deadline = time.monotonic() + timeout_s
        while True:
            if self._has_valid_feedback(min_time_s=min_time_s):
                return True
            if timeout_s <= 0 or time.monotonic() >= deadline:
                return False
            time.sleep(0.02)

    def require_valid_feedback(self, context: str, min_time_s: float | None = None) -> None:
        if self.wait_for_valid_feedback(min_time_s=min_time_s):
            return
        raise RuntimeError(
            f"wheeled_arm did not receive fresh left/right arm LCM feedback before {context}. "
            "为避免实物机器人从零位或旧状态跳变，已停止继续执行。"
            "请检查 MANIP_LEFT_ARM_STATE / MANIP_RIGHT_ARM_STATE、LCM URL、组播路由和控制器状态发布。"
        )

    def _has_valid_feedback(self, min_time_s: float | None = None) -> bool:
        assert self._handler is not None
        if hasattr(self._handler, "has_arm_state_feedback"):
            try:
                return bool(
                    self._handler.has_arm_state_feedback(
                        self.config.state_feedback_timeout_s, min_time_s=min_time_s
                    )
                )
            except TypeError:
                if min_time_s is not None:
                    return False
                return bool(self._handler.has_arm_state_feedback(self.config.state_feedback_timeout_s))
        if min_time_s is not None:
            return False
        return self.has_valid_feedback

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        if self.config.mock:
            self._mock_joint_pos = self._default_mock_joint_pos()
            self._mock_frame_index = 0
            if not self.config.mock_cameras:
                for cam in self.cameras.values():
                    cam.connect()
            self._mock_connected = True
            logger.warning(
                "%s connected in mock robot-body mode. LCM is not used and joint feedback "
                "will follow commanded actions. Cameras are %s.",
                self,
                "synthetic" if self.config.mock_cameras else "real",
            )
            return

        self._handler = _make_lcm_handler(self.config.lcm_url)
        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        wait_timeout_s = max(self.config.connect_timeout_s, self.config.feedback_wait_timeout_s)
        if not self.wait_for_valid_feedback(wait_timeout_s):
            if self.config.require_fresh_feedback:
                raise RuntimeError(
                    "%s did not receive fresh left/right arm LCM state feedback while connecting. "
                    "为避免实物遥操作初始状态跳变，连接已中止。"
                    "请检查 LCM URL、组播路由和控制器状态发布程序。"
                    % self
                )
            logger.warning(
                "%s has not received fresh left/right arm state feedback. "
                "机器人没有读到新鲜的左右臂 LCM 状态；请检查 LCM URL、组播路由和控制器状态发布程序。"
                " Closed-loop teleoperator feedback will be skipped until LCM state arrives.",
                self,
            )

        logger.info(f"{self} connected.")

    def calibrate(self) -> None:
        logger.info("%s does not require LeRobot-side calibration.", self)

    def configure(self) -> None:
        if self._handler is None:
            return

        self._set_moving_flags(set())

    @check_if_not_connected
    def reset_to_rest_pose(
        self,
        stop_requested: Callable[[], bool] | None = None,
        progress_callback: Callable[[RobotObservation], None] | None = None,
    ) -> bool:
        """Move both arms to the PICO data-collection reset pose using MOVEJ."""
        if self.config.mock:
            return self._reset_mock_rest_pose(progress_callback=progress_callback)

        assert self._handler is not None

        from .hardware_interface.dynamics_related_functions.collision_detection import (
            Collision_Detection,
        )
        from .hardware_interface.trajectory_plan.moveJ import MOVEJ

        reset_start_t = time.monotonic()
        if self.config.require_fresh_feedback:
            self.require_valid_feedback("reset_to_rest_pose")

        with self._handler.joint_current_pos_lock:
            current_joint_position = np.asarray(
                self._handler.joint_current_pos, dtype=np.float32
            ).copy()

        target_joint_position = current_joint_position.copy()
        target_joint_position[_PART_SLICES["left_arm"]] = _RESET_LEFT_ARM_RAD
        target_joint_position[_PART_SLICES["right_arm"]] = _RESET_RIGHT_ARM_RAD

        self._set_movej_reset_flags()
        movej = MOVEJ(
            self._handler,
            Collision_Detection(self._handler),
            stop_requested=stop_requested,
            progress_callback=self._make_reset_progress_callback(progress_callback),
        )
        completed = bool(movej.moveJ2target(current_joint_position, target_joint_position))
        if not completed:
            self._stop_all_moving_flags()
            logger.warning("wheeled_arm movej reset interrupted by emergency stop request.")
            return False

        if self.config.require_fresh_feedback:
            self.require_valid_feedback("post-reset LCM feedback", min_time_s=reset_start_t)
        return True

    def joint_observation_from_package(self, joint_positions: np.ndarray) -> RobotObservation:
        joint_positions = np.asarray(joint_positions, dtype=np.float32)
        return {
            f"{joint_name}.pos": float(joint_positions[idx])
            for idx, joint_name in enumerate(self.config.joint_names)
        }

    def _make_reset_progress_callback(
        self, progress_callback: Callable[[RobotObservation], None] | None
    ) -> Callable[[np.ndarray], None] | None:
        if progress_callback is None:
            return None

        def callback(joint_positions: np.ndarray) -> None:
            progress_callback(self.joint_observation_from_package(joint_positions))

        return callback

    def _set_movej_reset_flags(self) -> None:
        assert self._handler is not None

        self._handler.left_arm_moving = True
        self._handler.right_arm_moving = True
        self._handler.left_gripper_moving = False
        self._handler.right_gripper_moving = False
        self._handler.head_moving = False
        self._handler.waist_moving = False
        self._handler.leg_moving = False

    def _stop_all_moving_flags(self) -> None:
        assert self._handler is not None

        self._handler.left_arm_moving = False
        self._handler.right_arm_moving = False
        self._handler.left_gripper_moving = False
        self._handler.right_gripper_moving = False
        self._handler.head_moving = False
        self._handler.waist_moving = False
        self._handler.leg_moving = False

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        if self.config.mock:
            return self._get_mock_observation()

        assert self._handler is not None

        start = time.perf_counter()
        with self._handler.joint_current_pos_lock:
            joint_positions = np.asarray(self._handler.joint_current_pos, dtype=np.float32).copy()

        obs_dict: RobotObservation = {
            f"{joint_name}.pos": float(joint_positions[idx])
            for idx, joint_name in enumerate(self.config.joint_names)
        }
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        return self._read_camera_observations(obs_dict)

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        if self.config.mock:
            return self._send_mock_action(action)

        assert self._handler is not None

        unknown_keys = set(action) - set(self.action_features)
        if unknown_keys:
            raise ValueError(f"Unknown wheeled_arm action keys: {sorted(unknown_keys)}")

        with self._handler.joint_current_pos_lock:
            package = np.asarray(self._handler.joint_current_pos, dtype=np.float32).copy()

        goal_pos = {key: float(value) for key, value in action.items() if key.endswith(".pos")}

        if self.config.max_relative_target is not None:
            goal_present_pos = {
                key: (target, float(package[self._joint_index_by_action_key[key]]))
                for key, target in goal_pos.items()
            }
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        for key, value in goal_pos.items():
            package[self._joint_index_by_action_key[key]] = value

        self._set_moving_flags(set(goal_pos))
        self._handler.upper_body_data_publisher(package)

        return goal_pos

    def _set_moving_flags(self, action_keys: set[str]) -> None:
        assert self._handler is not None

        controlled_parts = set(self.config.controlled_parts)
        for part, flag_name in _PART_MOVING_FLAGS.items():
            part_slice = _PART_SLICES.get(part)
            part_action_keys = (
                {f"{joint_name}.pos" for joint_name in self.config.joint_names[part_slice]}
                if part_slice is not None
                else set()
            )
            should_move = part in controlled_parts and bool(action_keys & part_action_keys)
            setattr(self._handler, flag_name, should_move)

    @check_if_not_connected
    def disconnect(self) -> None:
        if self.config.mock:
            if not self.config.mock_cameras:
                for cam in self.cameras.values():
                    if cam.is_connected:
                        cam.disconnect()
            self._mock_connected = False
            self._mock_joint_pos = None
            logger.info(f"{self} disconnected.")
            return

        assert self._handler is not None

        if hasattr(self._handler, "stop"):
            self._handler.stop()
        self._handler = None

        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")

    def _default_mock_joint_pos(self) -> np.ndarray:
        joint_pos = np.zeros(len(self.config.joint_names), dtype=np.float32)
        joint_pos[_PART_SLICES["left_arm"]] = _RESET_LEFT_ARM_RAD
        joint_pos[_PART_SLICES["right_arm"]] = _RESET_RIGHT_ARM_RAD
        return joint_pos

    def _reset_mock_rest_pose(
        self, progress_callback: Callable[[RobotObservation], None] | None = None
    ) -> bool:
        self._mock_joint_pos = self._default_mock_joint_pos()
        if progress_callback is not None:
            progress_callback(self.joint_observation_from_package(self._mock_joint_pos))
        return True

    def _get_mock_observation(self) -> RobotObservation:
        assert self._mock_joint_pos is not None
        obs_dict = self.joint_observation_from_package(self._mock_joint_pos)
        if self.config.mock_cameras:
            for cam_key, cam in self.cameras.items():
                if getattr(cam, "use_rgb", True):
                    obs_dict[cam_key] = self._mock_rgb_image(
                        cam_key, int(cam.height), int(cam.width)
                    )
                if getattr(cam, "use_depth", False):
                    obs_dict[f"{cam_key}_depth"] = self._mock_depth_image(
                        int(cam.height), int(cam.width)
                    )
            self._mock_frame_index += 1
            return obs_dict
        return self._read_camera_observations(obs_dict)

    def _send_mock_action(self, action: RobotAction) -> RobotAction:
        assert self._mock_joint_pos is not None

        unknown_keys = set(action) - set(self.action_features)
        if unknown_keys:
            raise ValueError(f"Unknown wheeled_arm action keys: {sorted(unknown_keys)}")

        goal_pos = {key: float(value) for key, value in action.items() if key.endswith(".pos")}

        if self.config.max_relative_target is not None:
            goal_present_pos = {
                key: (target, float(self._mock_joint_pos[self._joint_index_by_action_key[key]]))
                for key, target in goal_pos.items()
            }
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        for key, value in goal_pos.items():
            self._mock_joint_pos[self._joint_index_by_action_key[key]] = value

        return goal_pos

    def _read_camera_observations(self, obs_dict: RobotObservation) -> RobotObservation:
        for cam_key, cam in self.cameras.items():
            if getattr(cam, "use_rgb", True):
                start = time.perf_counter()
                try:
                    obs_dict[cam_key] = cam.read_latest()
                except (RuntimeError, TimeoutError) as exc:
                    topic = getattr(cam, "topic_name", None) or getattr(cam, "camera_index", None) or cam_key
                    raise RuntimeError(
                        f"没有相机输入：读取相机 '{cam_key}' 失败（{topic}）。"
                        "请确认相机节点已启动、topic/设备索引正确，并先用 GUI 的“常用命令 > 查找相机”验证图像流。"
                    ) from exc
                dt_ms = (time.perf_counter() - start) * 1e3
                logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

            if getattr(cam, "use_depth", False):
                start = time.perf_counter()
                try:
                    obs_dict[f"{cam_key}_depth"] = cam.read_latest_depth()
                except (RuntimeError, TimeoutError) as exc:
                    topic = getattr(cam, "depth_topic_name", None) or getattr(cam, "topic_name", None) or cam_key
                    raise RuntimeError(
                        f"没有深度相机输入：读取相机 '{cam_key}' 深度图失败（{topic}）。"
                        "请确认深度 topic 已发布、相机驱动正常，并先用 GUI 的“常用命令 > 查找相机”验证图像流。"
                    ) from exc
                dt_ms = (time.perf_counter() - start) * 1e3
                logger.debug(f"{self} read {cam_key} depth: {dt_ms:.1f}ms")

        return obs_dict

    def _mock_rgb_image(self, camera_name: str, height: int, width: int) -> np.ndarray:
        x = np.linspace(0, 255, width, dtype=np.uint32)[None, :]
        y = np.linspace(0, 255, height, dtype=np.uint32)[:, None]
        frame = self._mock_frame_index
        name_offset = sum(camera_name.encode("utf-8")) % 255
        image = np.empty((height, width, 3), dtype=np.uint8)
        image[..., 0] = ((x + frame * 3) % 255).astype(np.uint8)
        image[..., 1] = ((y + frame * 5) % 255).astype(np.uint8)
        image[..., 2] = (name_offset + frame * 7) % 255
        return image

    def _mock_depth_image(self, height: int, width: int) -> np.ndarray:
        depth = np.linspace(0.2, 1.2, height, dtype=np.float32)[:, None]
        return np.repeat(depth, width, axis=1)[..., None]
