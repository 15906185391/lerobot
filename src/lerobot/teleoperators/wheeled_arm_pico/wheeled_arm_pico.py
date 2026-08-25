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
from functools import cached_property

import numpy as np

from lerobot.lerobot_types import RobotAction
from lerobot.robots.wheeled_arm.config_wheeled_arm import (
    WHEELED_ARM_ACTIVE_ARMS_ACTION_KEY,
    WHEELED_ARM_ARM_JOINT_NAMES,
    wheeled_arm_end_effector_names,
    wheeled_arm_joint_names,
)
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from .config_wheeled_arm_pico import WheeledArmPicoConfig
from .ik_utils import (
    LEFT_TCP,
    RIGHT_TCP,
    MockXrClient,
    RelativeTeleopTarget,
    XrPoseFilter,
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
    smooth_joint_positions,
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
        self.end_effector_names = wheeled_arm_end_effector_names(
            self.config.left_end_effector, self.config.right_end_effector
        )
        self.gripper_names = self.end_effector_names
        self.joint_names = wheeled_arm_joint_names(
            self.config.left_end_effector, self.config.right_end_effector
        )
        self._end_effector_types = {
            "left": self.config.left_end_effector,
            "right": self.config.right_end_effector,
        }

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
        self._urdf_path = None
        self._rerun_robot_last_update_t = 0.0
        self._rerun_robot_failed = False
        self._last_visualization_state = None
        self._filtered_arm_q = None
        self._previous_arm_step = None
        self._left_pose_filter = XrPoseFilter(
            position_alpha=self.config.pico_position_smoothing_alpha,
            orientation_alpha=self.config.pico_orientation_smoothing_alpha,
            position_deadband_m=self.config.pico_position_deadband_m,
            orientation_deadband_rad=self.config.pico_orientation_deadband_rad,
        )
        self._right_pose_filter = XrPoseFilter(
            position_alpha=self.config.pico_position_smoothing_alpha,
            orientation_alpha=self.config.pico_orientation_smoothing_alpha,
            position_deadband_m=self.config.pico_position_deadband_m,
            orientation_deadband_rad=self.config.pico_orientation_deadband_rad,
        )
        self._left_mapper = RelativeTeleopTarget()
        self._right_mapper = RelativeTeleopTarget()
        self._left_active = False
        self._right_active = False
        self._left_release_until_t = 0.0
        self._right_release_until_t = 0.0
        self._last_reset_button = False
        self._last_recording_control_buttons: dict[str, bool] = {}
        self._gripper_positions = {
            f"{name}.pos": self._end_effector_initial_pos(self._end_effector_types[side])
            for side, name in zip(("left", "right"), self.gripper_names, strict=True)
        }
        self._filtered_gripper_positions = self._gripper_positions.copy()
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

        self._deps = import_runtime_dependencies(require_collision_backend=self.config.use_self_collision)
        urdf_path = self.config.urdf_path or default_urdf_path()
        if not urdf_path.is_file():
            raise FileNotFoundError(
                f"Cannot find wheeled_arm URDF: {urdf_path}. "
                "Provide the robot model with `--teleop.urdf_path=/path/to/real_robot.urdf`, "
                "or place the URDF at the default bundled path."
            )
        self._urdf_path = urdf_path

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
                gain=self.config.self_collision_gain,
                safe_displacement_gain=self.config.self_collision_safe_displacement_gain,
                d_min=self.config.d_min,
            )
            self._barriers = [self._collision_barrier]
        else:
            self._barriers = []

        locked_joints_task_cls = make_locked_joints_task_class(self._deps.Task)
        self._constraints = [
            locked_joints_task_cls(
                self._locked_q_indices,
                locked_v_indices,
                self._q_ref,
                gain=self.config.locked_joints_gain,
                lm_damping=self.config.locked_joints_lm_damping,
            )
        ]

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
            lm_damping=self.config.frame_lm_damping,
            gain=self.config.task_gain,
        )
        right_task = self._deps.FrameTask(
            RIGHT_TCP,
            position_cost=self.config.position_cost,
            orientation_cost=self.config.orientation_cost,
            lm_damping=self.config.frame_lm_damping,
            gain=self.config.task_gain,
        )
        damping_task = self._deps.DampingTask(cost=self.config.damping_task_cost)
        posture_task = self._deps.PostureTask(
            cost=self.config.posture_cost,
            lm_damping=self.config.posture_lm_damping,
            gain=self.config.posture_gain,
        )
        self._tasks = [left_task, right_task, damping_task, posture_task]
        for task in (left_task, right_task, posture_task):
            task.set_target_from_configuration(self._configuration)

        wheeled_arm_pico_visualizer_cls = None
        if self.config.visualize:
            try:
                from .visualization import WheeledArmPicoVisualizer, require_visualization_dependencies

                wheeled_arm_pico_visualizer_cls = WheeledArmPicoVisualizer
                require_visualization_dependencies()
            except Exception as exc:
                logger.warning("Disabling wheeled_arm_pico Viser visualization: %s", exc)
                self.config.visualize = False

        self._solver = select_solver(self._deps.qpsolvers, self.config.solver)
        self._xr_client = MockXrClient() if self.config.mock_xr else self._make_xr_client()
        self._reset_action_filter_from_q(self._configuration.q)
        self._last_action = self._make_action(self._configuration.q, set())
        if self.config.visualize and wheeled_arm_pico_visualizer_cls is not None:
            try:
                self._visualizer = wheeled_arm_pico_visualizer_cls(
                    self.config,
                    self._deps,
                    self._robot,
                    self._configuration,
                    self._arm_q_indices,
                    urdf_path,
                )
            except Exception as exc:
                logger.warning("Disabling wheeled_arm_pico Viser visualization: %s", exc)
                self._visualizer = None
                self.config.visualize = False
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
        self._left_pose_filter.reset()
        self._right_pose_filter.reset()

    @check_if_not_connected
    def get_recording_control(self) -> str | None:
        """Return a keyboard-compatible recording control triggered by a PICO button.

        The return values match ``lerobot.utils.keyboard_input.apply_recording_control``:
        ``"right"`` advances the current recording phase, ``"left"`` requests a re-record,
        and ``"esc"`` stops the whole recording session.
        """
        if not self.config.recording_control:
            return None

        assert self._xr_client is not None

        controls = (
            (self.config.recording_advance_button, "right"),
            (self.config.recording_rerecord_button, "left"),
            (self.config.recording_stop_button, "esc"),
        )
        for button_name, control in controls:
            if not button_name:
                continue
            try:
                pressed = bool(self._xr_client.get_button_state_by_name(button_name))
            except Exception as exc:
                logger.warning(
                    "Could not read PICO recording control button '%s': %s",
                    button_name,
                    exc,
                )
                continue

            was_pressed = self._last_recording_control_buttons.get(button_name, False)
            self._last_recording_control_buttons[button_name] = pressed
            if pressed and not was_pressed:
                logger.info(
                    "PICO recording control '%s' triggered by button '%s'.", control, button_name
                )
                return control

        return None

    def emergency_stop_requested(self) -> bool:
        """Return True while the configured PICO emergency stop button is pressed."""
        button_name = self.config.emergency_stop_button
        if not button_name or self._xr_client is None:
            return False

        try:
            pressed = bool(self._xr_client.get_button_state_by_name(button_name))
        except Exception as exc:
            logger.warning("Could not read PICO emergency stop button '%s': %s", button_name, exc)
            return False

        if pressed:
            logger.warning("PICO emergency stop requested by button '%s'.", button_name)
        return pressed

    @check_if_not_connected
    def send_feedback(self, feedback: dict) -> None:
        assert self._configuration is not None
        assert self._arm_q_indices is not None
        assert self._locked_q_indices is not None
        assert self._q_ref is not None

        for gripper_name in self.gripper_names:
            key = f"{gripper_name}.pos"
            if key in feedback:
                value = float(feedback[key])
                self._gripper_positions[key] = value
                self._filtered_gripper_positions[key] = value

        arm_q = arm_q_from_feedback(feedback, self.arm_joint_names)
        if arm_q is None:
            return

        q = self._configuration.q.copy()
        q[self._arm_q_indices] = arm_q
        q[self._locked_q_indices] = self._q_ref[self._locked_q_indices]
        self._configuration.update(q)
        self._filtered_arm_q = arm_q.copy()

    @check_if_not_connected
    def refresh_visualization_from_feedback(
        self,
        feedback: dict,
        *,
        collision_status: str = "sync",
        min_barrier: float | None = None,
    ) -> None:
        """Force the visualizers to match robot feedback outside the normal IK loop."""
        self.send_feedback(feedback)

        assert self._configuration is not None

        left_pose = self._configuration.get_transform_frame_to_world(LEFT_TCP).copy()
        right_pose = self._configuration.get_transform_frame_to_world(RIGHT_TCP).copy()
        self._update_visualization(
            left_pose,
            right_pose,
            xr_ok=False,
            left_active=False,
            right_active=False,
            collision_status=collision_status,
            min_barrier=min_barrier,
            force=True,
        )

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
        left_task, right_task, damping_task, posture_task = self._tasks
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
            left_active = self._update_activation_state("left", left_grip)
            right_active = self._update_activation_state("right", right_grip)
            left_gripper_value = float(
                self._xr_client.get_key_value_by_name(self.config.left_gripper_input_name)
            )
            right_gripper_value = float(
                self._xr_client.get_key_value_by_name(self.config.right_gripper_input_name)
            )

            left_controller_pose = xr_pose_to_world_se3(
                self._left_pose_filter.update(
                    self._xr_client.get_pose_by_name(self.config.left_controller_name)
                ),
                pin,
            )
            right_controller_pose = xr_pose_to_world_se3(
                self._right_pose_filter.update(
                    self._xr_client.get_pose_by_name(self.config.right_controller_name)
                ),
                pin,
            )
            xr_ok = True
        except Exception as exc:
            logger.warning("Waiting for valid PICO XR data: %s", exc)
            self.reset_baseline()
            self._clear_release_deceleration()
            self._last_action[WHEELED_ARM_ACTIVE_ARMS_ACTION_KEY] = ()
            self._update_visualization(
                left_target_pose,
                right_target_pose,
                xr_ok,
                left_active,
                right_active,
                collision_status,
                min_barrier,
                force=True,
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
        hold_arm_q = np.asarray(self._configuration.q[self._arm_q_indices], dtype=float).copy()
        ik_arm_mask = self._active_arm_mask(left_active, right_active)
        release_arm_mask = self._release_deceleration_arm_mask(time.monotonic())
        command_arm_mask = ik_arm_mask | release_arm_mask

        if not np.any(command_arm_mask):
            self._reset_action_filter_from_q(self._configuration.q)
            self._last_action = self._make_action(self._configuration.q, set())
            self._update_visualization(
                left_target_pose,
                right_target_pose,
                xr_ok,
                left_active,
                right_active,
                "idle",
                min_barrier,
            )
            return self._last_action.copy()

        if not np.any(ik_arm_mask):
            self._apply_action_smoothing_to_configuration(command_arm_mask)
            self._last_action = self._make_action(
                self._configuration.q,
                self._arm_names_from_mask(command_arm_mask),
            )
            self._update_visualization(
                left_target_pose,
                right_target_pose,
                xr_ok,
                left_active,
                right_active,
                "decelerating",
                min_barrier,
            )
            return self._last_action.copy()

        if self._collision_barrier is not None:
            min_barrier = float(np.min(self._collision_barrier.compute_barrier(self._configuration)))
            if min_barrier <= 0.0:
                collision_status = "collision"
                logger.warning("PICO IK target rejected by self-collision barrier: %.4f", min_barrier)
                self._reset_action_filter_from_q(self._configuration.q)
                self._last_action = self._make_action(
                    self._configuration.q,
                    self._arm_names_from_mask(command_arm_mask),
                )
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
            collision_status = (
                "warning" if min_barrier < self.config.collision_warning_distance else "safe"
            )
        else:
            collision_status = "disabled"

        active_tasks = []
        if left_active:
            active_tasks.append(left_task)
        if right_active:
            active_tasks.append(right_task)
        active_tasks.append(damping_task)
        active_tasks.append(posture_task)

        try:
            velocity = self._deps.solve_ik(
                self._configuration,
                active_tasks,
                self._dt,
                solver=self._solver,
                damping=self.config.ik_damping,
                limits=None if self.config.enforce_limits else [],
                barriers=self._barriers,
                constraints=self._constraints,
                safety_break=self.config.ik_safety_break,
                **self.config.solver_kwargs,
            )
        except self._deps.NoSolutionFound as exc:
            logger.warning("PICO IK solver failed: %s", exc)
            velocity = np.zeros(self._robot.model.nv)

        self._configuration.integrate_inplace(velocity, self._dt)
        q_locked = self._configuration.q.copy()
        q_locked[self._locked_q_indices] = self._q_ref[self._locked_q_indices]
        q_locked[self._arm_q_indices[~ik_arm_mask]] = hold_arm_q[~ik_arm_mask]
        self._configuration.update(q_locked)
        self._apply_action_smoothing_to_configuration(command_arm_mask)

        self._last_action = self._make_action(
            self._configuration.q,
            self._arm_names_from_mask(command_arm_mask),
        )
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

    def _update_activation_state(self, side: str, grip: float) -> bool:
        if side == "left":
            was_active = self._left_active
        elif side == "right":
            was_active = self._right_active
        else:
            raise ValueError(f"Unknown arm side: {side}")

        release_threshold = self.config.activation_threshold - self.config.activation_hysteresis
        is_active = grip >= (release_threshold if was_active else self.config.activation_threshold)

        if was_active and not is_active:
            release_until_t = time.monotonic() + self.config.grip_release_deceleration_s
        else:
            release_until_t = 0.0

        if side == "left":
            self._left_active = is_active
            if is_active or release_until_t > 0.0:
                self._left_release_until_t = release_until_t
        else:
            self._right_active = is_active
            if is_active or release_until_t > 0.0:
                self._right_release_until_t = release_until_t
        return is_active

    def _make_action(self, q: np.ndarray, active_arms: set[str] | None = None) -> RobotAction:
        action = arm_action_from_q(q, self._arm_q_indices, self.arm_joint_names)
        action.update(self._gripper_positions)
        action[WHEELED_ARM_ACTIVE_ARMS_ACTION_KEY] = tuple(sorted(active_arms or set()))
        return action

    def _end_effector_initial_pos(self, end_effector: str) -> float:
        if end_effector == "suction":
            return self.config.suction_off_pos
        return self.config.gripper_open_pos

    def _end_effector_target_pos(self, end_effector: str, ratio: float) -> float:
        ratio = float(np.clip(ratio, 0.0, 1.0))
        if end_effector == "suction":
            return (
                self.config.suction_on_pos
                if ratio >= self.config.suction_trigger_threshold
                else self.config.suction_off_pos
            )
        return self.config.gripper_open_pos + ratio * (
            self.config.gripper_closed_pos - self.config.gripper_open_pos
        )

    def _reset_action_filter_from_q(self, q: np.ndarray) -> None:
        assert self._arm_q_indices is not None
        self._filtered_arm_q = np.asarray(q[self._arm_q_indices], dtype=float).copy()
        self._previous_arm_step = np.zeros_like(self._filtered_arm_q)

    def _active_arm_mask(self, left_active: bool, right_active: bool) -> np.ndarray:
        return np.array([left_active] * 7 + [right_active] * 7, dtype=bool)

    def _active_arm_names(self, left_active: bool, right_active: bool) -> set[str]:
        active_arms = set()
        if left_active:
            active_arms.add("left_arm")
        if right_active:
            active_arms.add("right_arm")
        return active_arms

    def _arm_names_from_mask(self, active_arm_mask: np.ndarray) -> set[str]:
        active_arm_mask = np.asarray(active_arm_mask, dtype=bool)
        active_arms = set()
        if bool(np.any(active_arm_mask[:7])):
            active_arms.add("left_arm")
        if bool(np.any(active_arm_mask[7:14])):
            active_arms.add("right_arm")
        return active_arms

    def _clear_release_deceleration(self) -> None:
        self._left_release_until_t = 0.0
        self._right_release_until_t = 0.0

    def _release_deceleration_arm_mask(self, now: float) -> np.ndarray:
        return self._active_arm_mask(
            self._arm_is_release_decelerating("left", now),
            self._arm_is_release_decelerating("right", now),
        )

    def _arm_is_release_decelerating(self, side: str, now: float) -> bool:
        if side == "left":
            if self._left_active or now > self._left_release_until_t:
                return False
            arm_slice = slice(0, 7)
        elif side == "right":
            if self._right_active or now > self._right_release_until_t:
                return False
            arm_slice = slice(7, 14)
        else:
            raise ValueError(f"Unknown arm side: {side}")

        if self._previous_arm_step is None:
            return False
        previous_step = np.asarray(self._previous_arm_step, dtype=float)[arm_slice]
        return bool(np.any(np.abs(previous_step) > self.config.grip_release_stop_step_rad))

    def _apply_action_smoothing_to_configuration(self, active_arm_mask: np.ndarray) -> None:
        assert self._configuration is not None
        assert self._arm_q_indices is not None
        assert self._locked_q_indices is not None
        assert self._q_ref is not None

        target_arm_q = np.asarray(self._configuration.q[self._arm_q_indices], dtype=float)
        active_arm_mask = np.asarray(active_arm_mask, dtype=bool)
        if active_arm_mask.shape != target_arm_q.shape:
            raise ValueError(
                f"`active_arm_mask` shape {active_arm_mask.shape} does not match "
                f"arm q shape {target_arm_q.shape}."
            )
        current_arm_q = (
            target_arm_q
            if self._filtered_arm_q is None
            else np.asarray(self._filtered_arm_q, dtype=float)
        )
        previous_arm_step = (
            np.zeros_like(target_arm_q)
            if self._previous_arm_step is None
            else np.asarray(self._previous_arm_step, dtype=float)
        )
        filtered_arm_q = target_arm_q.copy()
        arm_step = np.zeros_like(target_arm_q)

        if np.any(active_arm_mask):
            smoothed_q, smoothed_step = smooth_joint_positions(
                target_arm_q[active_arm_mask],
                current_arm_q[active_arm_mask],
                previous_arm_step[active_arm_mask],
                self._dt,
                alpha=self.config.arm_action_smoothing_alpha,
                max_velocity_rad_s=self.config.max_joint_velocity_rad_s,
                max_acceleration_rad_s2=self.config.max_joint_acceleration_rad_s2,
            )
            filtered_arm_q[active_arm_mask] = smoothed_q
            arm_step[active_arm_mask] = smoothed_step

        self._filtered_arm_q = filtered_arm_q
        self._previous_arm_step = arm_step

        q_filtered = self._configuration.q.copy()
        q_filtered[self._arm_q_indices] = filtered_arm_q
        q_filtered[self._locked_q_indices] = self._q_ref[self._locked_q_indices]
        self._configuration.update(q_filtered)

    def _update_gripper_positions(
        self, left_value: float | None, right_value: float | None
    ) -> None:
        values = (
            ("left", left_value),
            ("right", right_value),
        )
        for side, raw_value in values:
            if raw_value is None:
                continue
            end_effector = self._end_effector_types[side]
            key = f"{side}_{end_effector}.pos"
            target_pos = self._end_effector_target_pos(end_effector, float(raw_value))
            current_pos = self._filtered_gripper_positions[key]
            if end_effector == "suction":
                filtered_pos = target_pos
            elif abs(target_pos - current_pos) <= self.config.gripper_input_deadband:
                filtered_pos = current_pos
            else:
                filtered_pos = current_pos + self.config.gripper_position_smoothing_alpha * (
                    target_pos - current_pos
                )
            self._filtered_gripper_positions[key] = filtered_pos
            self._gripper_positions[key] = filtered_pos

    def _update_visualization(
        self,
        left_target_pose,
        right_target_pose,
        xr_ok: bool,
        left_active: bool,
        right_active: bool,
        collision_status: str,
        min_barrier: float | None,
        force: bool = False,
    ) -> None:
        self._last_visualization_state = {
            "left_target_pose": left_target_pose.copy(),
            "right_target_pose": right_target_pose.copy(),
            "xr_ok": xr_ok,
            "left_active": left_active,
            "right_active": right_active,
            "collision_status": collision_status,
            "min_barrier": min_barrier,
        }
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
            force=force,
        )

    def log_rerun_robot_visualization(
        self,
        frame_index: int | None = None,
        timestamp: float | None = None,
        force: bool = False,
    ) -> None:
        if (
            not self.config.rerun_visualize_robot
            or self._rerun_robot_failed
            or self._last_visualization_state is None
        ):
            return
        if not force and self.config.rerun_robot_update_hz > 0:
            now = time.monotonic()
            if now - self._rerun_robot_last_update_t < 1.0 / self.config.rerun_robot_update_hz:
                return
            self._rerun_robot_last_update_t = now

        assert self._deps is not None
        assert self._robot is not None
        assert self._configuration is not None

        try:
            from .rerun_robot_visualization import log_rerun_robot_visualization

            log_rerun_robot_visualization(
                self.config,
                self._deps,
                self._robot,
                self._configuration,
                frame_index=frame_index,
                timestamp=timestamp,
                urdf_path=self._urdf_path,
                **self._last_visualization_state,
            )
        except Exception as exc:
            self._rerun_robot_failed = True
            logger.warning("Disabling wheeled_arm_pico Rerun robot visualization: %s", exc)

    @check_if_not_connected
    def disconnect(self) -> None:
        if self._xr_client is not None:
            self._xr_client.close()
        if self._visualizer is not None:
            self._visualizer.close()
        self._xr_client = None
        self._visualizer = None
        self._last_visualization_state = None
        self._connected = False
        logger.info("%s disconnected.", self)
