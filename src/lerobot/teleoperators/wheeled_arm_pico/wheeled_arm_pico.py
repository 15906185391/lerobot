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
from functools import cached_property

import numpy as np

from lerobot.lerobot_types import RobotAction
from lerobot.robots.wheeled_arm.config_wheeled_arm import (
    WHEELED_ARM_ARM_JOINT_NAMES,
    WHEELED_ARM_GRIPPER_NAMES,
    WHEELED_ARM_JOINT_NAMES,
)
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from .config_wheeled_arm_pico import WheeledArmPicoConfig
from .ik_utils import (
    LEFT_TCP,
    RIGHT_TCP,
    MockXrClient,
    RelativeTeleopTarget,
    arm_action_from_q,
    arm_joint_indices,
    arm_q_from_feedback,
    build_primitive_collision_model,
    configure_self_collision,
    default_urdf_path,
    import_runtime_dependencies,
    initial_configuration,
    locked_joint_indices,
    make_locked_joints_task_class,
    package_dirs_for_urdf,
    select_solver,
    xr_pose_to_world_se3,
)

logger = logging.getLogger(__name__)


class WheeledArmPico(Teleoperator):
    """PICO controller teleoperator that outputs wheeled_arm left/right arm joint targets."""

    config_class = WheeledArmPicoConfig
    name = "wheeled_arm_pico"

    def __init__(self, config: WheeledArmPicoConfig):
        super().__init__(config)
        self.config = config
        self.arm_joint_names = WHEELED_ARM_ARM_JOINT_NAMES.copy()
        self.gripper_names = WHEELED_ARM_GRIPPER_NAMES.copy()
        self.joint_names = WHEELED_ARM_JOINT_NAMES.copy()

        self._connected = False
        self._deps = None
        self._xr_client = None
        self._robot = None
        self._configuration = None
        self._tasks = None
        self._barriers = None
        self._constraints = None
        self._collision_barrier = None
        self._arm_q_indices = None
        self._locked_q_indices = None
        self._q_ref = None
        self._solver = None
        self._dt = 0.0
        self._visualizer = None
        self._left_mapper = RelativeTeleopTarget()
        self._right_mapper = RelativeTeleopTarget()
        self._last_reset_button = False
        self._gripper_positions = {
            f"{name}.pos": self.config.gripper_open_pos for name in self.gripper_names
        }
        self._last_action = dict.fromkeys(self.action_features, 0.0)

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {f"{joint_name}.pos": float for joint_name in self.joint_names}

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return self.action_features

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        if self.config.solve_frequency_hz <= 0:
            raise ValueError("`solve_frequency_hz` must be positive.")
        if self.config.visualization_update_hz < 0:
            raise ValueError("`visualization_update_hz` must be non-negative.")
        self._dt = 1.0 / self.config.solve_frequency_hz

        self._deps = import_runtime_dependencies()
        urdf_path = self.config.urdf_path or default_urdf_path()
        if not urdf_path.is_file():
            raise FileNotFoundError(f"Cannot find wheeled_arm URDF: {urdf_path}")

        pin = self._deps.pin
        self._robot = pin.RobotWrapper.BuildFromURDF(
            filename=str(urdf_path),
            package_dirs=package_dirs_for_urdf(urdf_path),
            root_joint=None,
        )
        if self.config.use_self_collision:
            self._robot.collision_model = build_primitive_collision_model(
                self._robot.model, pin, self._deps.fcl
            )

        self._q_ref = initial_configuration(self._robot.model, pin)
        self._arm_q_indices, _ = arm_joint_indices(self._robot.model)
        self._locked_q_indices, locked_v_indices = locked_joint_indices(self._robot.model)

        if self.config.use_self_collision:
            initial_ignore_distance = (
                self.config.initial_ignore_distance
                if self.config.initial_ignore_distance is not None
                else self.config.d_min
            )
            self._robot.collision_data = configure_self_collision(
                self._robot, self._q_ref, initial_ignore_distance, pin
            )
            self._collision_barrier = self._deps.SelfCollisionBarrier(
                n_collision_pairs=len(self._robot.collision_model.collisionPairs),
                gain=10.0,
                safe_displacement_gain=5.0,
                d_min=self.config.d_min,
            )
            self._barriers = [self._collision_barrier]
        else:
            self._barriers = []

        LockedJointsTask = make_locked_joints_task_class(self._deps.Task)
        self._constraints = [LockedJointsTask(self._locked_q_indices, locked_v_indices, self._q_ref)]

        self._configuration = self._deps.pink.Configuration(
            self._robot.model,
            self._robot.data,
            self._q_ref,
            collision_model=self._robot.collision_model if self.config.use_self_collision else None,
            collision_data=self._robot.collision_data if self.config.use_self_collision else None,
        )

        left_task = self._deps.FrameTask(
            LEFT_TCP,
            position_cost=self.config.position_cost,
            orientation_cost=self.config.orientation_cost,
            gain=self.config.task_gain,
        )
        right_task = self._deps.FrameTask(
            RIGHT_TCP,
            position_cost=self.config.position_cost,
            orientation_cost=self.config.orientation_cost,
            gain=self.config.task_gain,
        )
        posture_task = self._deps.PostureTask(cost=self.config.posture_cost)
        self._tasks = [left_task, right_task, posture_task]
        for task in self._tasks:
            task.set_target_from_configuration(self._configuration)

        self._solver = select_solver(self._deps.qpsolvers, self.config.solver)
        self._xr_client = MockXrClient() if self.config.mock_xr else self._make_xr_client()
        self._last_action = self._make_action(self._configuration.q)
        if self.config.visualize:
            from .visualization import WheeledArmPicoVisualizer

            self._visualizer = WheeledArmPicoVisualizer(
                self.config,
                self._deps,
                self._robot,
                self._configuration,
                self._arm_q_indices,
                urdf_path,
            )
        self._connected = True
        logger.info("%s connected using IK solver '%s'.", self, self._solver)

    def _make_xr_client(self):
        from .ik_utils import ensure_vendor_path

        ensure_vendor_path()
        try:
            from xrobotoolkit_teleop.common.xr_client import XrClient
        except ImportError as exc:
            raise ImportError(
                "wheeled_arm_pico requires `xrobotoolkit_sdk` and the bundled "
                "`xrobotoolkit_teleop` client to read PICO controller poses."
            ) from exc
        return XrClient()

    def calibrate(self) -> None:
        self.reset_baseline()

    def configure(self) -> None:
        pass

    def reset_baseline(self) -> None:
        self._left_mapper.reset()
        self._right_mapper.reset()

    @check_if_not_connected
    def send_feedback(self, feedback: dict) -> None:
        assert self._configuration is not None
        assert self._arm_q_indices is not None
        assert self._locked_q_indices is not None
        assert self._q_ref is not None

        for gripper_name in self.gripper_names:
            key = f"{gripper_name}.pos"
            if key in feedback:
                self._gripper_positions[key] = float(feedback[key])

        arm_q = arm_q_from_feedback(feedback, self.arm_joint_names)
        if arm_q is None:
            return

        q = self._configuration.q.copy()
        q[self._arm_q_indices] = arm_q
        q[self._locked_q_indices] = self._q_ref[self._locked_q_indices]
        self._configuration.update(q)

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        assert self._configuration is not None
        assert self._tasks is not None
        assert self._xr_client is not None
        assert self._deps is not None
        assert self._arm_q_indices is not None
        assert self._locked_q_indices is not None
        assert self._q_ref is not None

        pin = self._deps.pin
        left_task, right_task, _posture_task = self._tasks
        current_left = self._configuration.get_transform_frame_to_world(LEFT_TCP).copy()
        current_right = self._configuration.get_transform_frame_to_world(RIGHT_TCP).copy()
        left_target_pose = current_left
        right_target_pose = current_right
        xr_ok = False
        left_active = False
        right_active = False
        collision_status = "unknown"
        min_barrier = None
        left_gripper_value = None
        right_gripper_value = None

        try:
            reset_pressed = bool(self._xr_client.get_button_state_by_name(self.config.reset_button))
            if reset_pressed and not self._last_reset_button:
                self.reset_baseline()
            self._last_reset_button = reset_pressed

            left_grip = float(self._xr_client.get_key_value_by_name(self.config.left_grip_name))
            right_grip = float(self._xr_client.get_key_value_by_name(self.config.right_grip_name))
            left_active = left_grip >= self.config.activation_threshold
            right_active = right_grip >= self.config.activation_threshold
            left_gripper_value = float(
                self._xr_client.get_key_value_by_name(self.config.left_gripper_input_name)
            )
            right_gripper_value = float(
                self._xr_client.get_key_value_by_name(self.config.right_gripper_input_name)
            )

            left_controller_pose = xr_pose_to_world_se3(
                self._xr_client.get_pose_by_name(self.config.left_controller_name), pin
            )
            right_controller_pose = xr_pose_to_world_se3(
                self._xr_client.get_pose_by_name(self.config.right_controller_name), pin
            )
            xr_ok = True
        except Exception as exc:
            logger.warning("Waiting for valid PICO XR data: %s", exc)
            self.reset_baseline()
            self._update_visualization(
                left_target_pose,
                right_target_pose,
                xr_ok,
                left_active,
                right_active,
                collision_status,
                min_barrier,
            )
            return self._last_action.copy()

        if left_active:
            left_target_pose = self._left_mapper.update(
                left_controller_pose,
                current_left,
                self.config.scale,
                orientation_enabled=not self.config.position_only,
            )
        else:
            self._left_mapper.reset()
            left_target_pose = current_left

        if right_active:
            right_target_pose = self._right_mapper.update(
                right_controller_pose,
                current_right,
                self.config.scale,
                orientation_enabled=not self.config.position_only,
            )
        else:
            self._right_mapper.reset()
            right_target_pose = current_right

        left_task.transform_target_to_world = left_target_pose
        right_task.transform_target_to_world = right_target_pose
        self._update_gripper_positions(left_gripper_value, right_gripper_value)

        if self._collision_barrier is not None:
            min_barrier = float(np.min(self._collision_barrier.compute_barrier(self._configuration)))
            if min_barrier <= 0.0:
                collision_status = "collision"
                logger.warning("PICO IK target rejected by self-collision barrier: %.4f", min_barrier)
                self._last_action = self._make_action(self._configuration.q)
                self._update_visualization(
                    left_target_pose,
                    right_target_pose,
                    xr_ok,
                    left_active,
                    right_active,
                    collision_status,
                    min_barrier,
                )
                return self._last_action.copy()
            collision_status = "warning" if min_barrier < 0.01 else "safe"
        else:
            collision_status = "disabled"

        try:
            velocity = self._deps.solve_ik(
                self._configuration,
                self._tasks,
                self._dt,
                solver=self._solver,
                barriers=self._barriers,
                constraints=self._constraints,
                safety_break=False,
            )
        except self._deps.NoSolutionFound as exc:
            logger.warning("PICO IK solver failed: %s", exc)
            velocity = np.zeros(self._robot.model.nv)

        self._configuration.integrate_inplace(velocity, self._dt)
        q_locked = self._configuration.q.copy()
        q_locked[self._locked_q_indices] = self._q_ref[self._locked_q_indices]
        self._configuration.update(q_locked)

        self._last_action = self._make_action(self._configuration.q)
        self._update_visualization(
            left_target_pose,
            right_target_pose,
            xr_ok,
            left_active,
            right_active,
            collision_status,
            min_barrier,
        )
        return self._last_action.copy()

    def _make_action(self, q: np.ndarray) -> RobotAction:
        action = arm_action_from_q(q, self._arm_q_indices, self.arm_joint_names)
        action.update(self._gripper_positions)
        return action

    def _update_gripper_positions(
        self, left_value: float | None, right_value: float | None
    ) -> None:
        values = {
            "left_gripper.pos": left_value,
            "right_gripper.pos": right_value,
        }
        for key, raw_value in values.items():
            if raw_value is None:
                continue
            ratio = float(np.clip(raw_value, 0.0, 1.0))
            self._gripper_positions[key] = (
                self.config.gripper_open_pos
                + ratio * (self.config.gripper_closed_pos - self.config.gripper_open_pos)
            )

    def _update_visualization(
        self,
        left_target_pose,
        right_target_pose,
        xr_ok: bool,
        left_active: bool,
        right_active: bool,
        collision_status: str,
        min_barrier: float | None,
    ) -> None:
        if self._visualizer is None:
            return
        self._visualizer.update(
            left_target_pose=left_target_pose,
            right_target_pose=right_target_pose,
            xr_ok=xr_ok,
            left_active=left_active,
            right_active=right_active,
            collision_status=collision_status,
            min_barrier=min_barrier,
        )

    @check_if_not_connected
    def disconnect(self) -> None:
        if self._xr_client is not None:
            self._xr_client.close()
        if self._visualizer is not None:
            self._visualizer.close()
        self._xr_client = None
        self._visualizer = None
        self._connected = False
        logger.info("%s disconnected.", self)
