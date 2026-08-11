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

import numpy as np

from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.teleoperators.utils import make_teleoperator_from_config
from lerobot.teleoperators.wheeled_arm_pico import WheeledArmPico, WheeledArmPicoConfig
from lerobot.teleoperators.wheeled_arm_pico.ik_utils import (
    XrPoseFilter,
    arm_q_from_feedback,
    make_locked_joints_task_class,
    smooth_joint_positions,
)


def test_wheeled_arm_pico_config_is_registered():
    assert "wheeled_arm_pico" in TeleoperatorConfig.get_known_choices()
    assert isinstance(make_teleoperator_from_config(WheeledArmPicoConfig()), WheeledArmPico)


def test_wheeled_arm_pico_config_exposes_ik_parameters():
    cfg = WheeledArmPicoConfig(
        position_cost=[5.0, 4.0, 3.0],
        orientation_cost=[1.0, 0.5, 0.25],
        pico_position_smoothing_alpha=0.35,
        pico_orientation_smoothing_alpha=0.45,
        pico_position_deadband_m=0.002,
        pico_orientation_deadband_rad=0.02,
        frame_lm_damping=1e-3,
        task_gain=0.75,
        posture_gain=0.5,
        posture_lm_damping=1e-4,
        locked_joints_gain=0.8,
        locked_joints_lm_damping=1e-5,
        ik_damping=1e-8,
        ik_safety_break=True,
        enforce_limits=False,
        arm_action_smoothing_alpha=0.4,
        max_joint_velocity_rad_s=1.2,
        max_joint_acceleration_rad_s2=6.0,
        self_collision_gain=12.0,
        self_collision_safe_displacement_gain=6.0,
        collision_warning_distance=0.02,
        solver_kwargs={"verbose": False},
        rerun_visualize_robot=False,
        rerun_robot_update_hz=5.0,
        rerun_robot_prefix="ik_robot",
        rerun_robot_axis_length=0.2,
    )

    assert cfg.position_cost == [5.0, 4.0, 3.0]
    assert cfg.orientation_cost == [1.0, 0.5, 0.25]
    assert cfg.pico_position_smoothing_alpha == 0.35
    assert cfg.pico_orientation_smoothing_alpha == 0.45
    assert cfg.pico_position_deadband_m == 0.002
    assert cfg.pico_orientation_deadband_rad == 0.02
    assert cfg.frame_lm_damping == 1e-3
    assert cfg.task_gain == 0.75
    assert cfg.posture_gain == 0.5
    assert cfg.posture_lm_damping == 1e-4
    assert cfg.locked_joints_gain == 0.8
    assert cfg.locked_joints_lm_damping == 1e-5
    assert cfg.ik_damping == 1e-8
    assert cfg.ik_safety_break is True
    assert cfg.enforce_limits is False
    assert cfg.arm_action_smoothing_alpha == 0.4
    assert cfg.max_joint_velocity_rad_s == 1.2
    assert cfg.max_joint_acceleration_rad_s2 == 6.0
    assert cfg.self_collision_gain == 12.0
    assert cfg.self_collision_safe_displacement_gain == 6.0
    assert cfg.collision_warning_distance == 0.02
    assert cfg.solver_kwargs == {"verbose": False}
    assert cfg.rerun_visualize_robot is False
    assert cfg.rerun_robot_update_hz == 5.0
    assert cfg.rerun_robot_prefix == "ik_robot"
    assert cfg.rerun_robot_axis_length == 0.2


def test_action_features_match_wheeled_arm_arm_and_gripper_joints():
    teleop = WheeledArmPico(WheeledArmPicoConfig())

    assert list(teleop.action_features) == [
        *(f"left_arm_{idx}.pos" for idx in range(7)),
        *(f"right_arm_{idx}.pos" for idx in range(7)),
        "left_gripper.pos",
        "right_gripper.pos",
    ]


def test_pico_recording_controls_are_edge_triggered():
    class ButtonXrClient:
        def __init__(self):
            self.buttons = {"A": False, "B": False, "X": False}

        def get_button_state_by_name(self, name: str) -> bool:
            return self.buttons[name]

    xr_client = ButtonXrClient()
    teleop = WheeledArmPico(WheeledArmPicoConfig())
    teleop._xr_client = xr_client
    teleop._connected = True

    assert teleop.get_recording_control() is None

    xr_client.buttons["A"] = True
    assert teleop.get_recording_control() == "right"
    assert teleop.get_recording_control() is None

    xr_client.buttons["A"] = False
    assert teleop.get_recording_control() is None

    xr_client.buttons["B"] = True
    assert teleop.get_recording_control() == "left"

    xr_client.buttons["B"] = False
    xr_client.buttons["X"] = True
    assert teleop.get_recording_control() == "esc"


def test_pico_emergency_stop_button_is_level_triggered():
    class ButtonXrClient:
        def __init__(self):
            self.buttons = {"X": False}

        def get_button_state_by_name(self, name: str) -> bool:
            return self.buttons[name]

    xr_client = ButtonXrClient()
    teleop = WheeledArmPico(WheeledArmPicoConfig())
    teleop._xr_client = xr_client

    assert teleop.emergency_stop_requested() is False

    xr_client.buttons["X"] = True
    assert teleop.emergency_stop_requested() is True
    assert teleop.emergency_stop_requested() is True

    xr_client.buttons["X"] = False
    assert teleop.emergency_stop_requested() is False


def test_arm_q_from_feedback_requires_complete_left_and_right_arm_state():
    joint_names = [
        *(f"left_arm_{idx}" for idx in range(7)),
        *(f"right_arm_{idx}" for idx in range(7)),
    ]
    feedback = {f"{name}.pos": float(idx) for idx, name in enumerate(joint_names)}

    arm_q = arm_q_from_feedback(feedback, joint_names)

    np.testing.assert_allclose(arm_q, np.arange(14, dtype=float))
    assert arm_q_from_feedback({"left_arm_0.pos": 0.0}, joint_names) is None


def test_smooth_joint_positions_applies_ema_velocity_and_acceleration_limits():
    target_q = np.array([10.0, -10.0])
    current_q = np.zeros(2)
    previous_step = np.array([0.01, -0.01])

    smoothed_q, step = smooth_joint_positions(
        target_q,
        current_q,
        previous_step,
        0.1,
        alpha=0.5,
        max_velocity_rad_s=1.0,
        max_acceleration_rad_s2=2.0,
    )

    np.testing.assert_allclose(step, [0.015, -0.015])
    np.testing.assert_allclose(smoothed_q, [0.015, -0.015])


def test_smooth_joint_positions_can_be_disabled_with_passthrough_settings():
    target_q = np.array([0.3, -0.2])
    current_q = np.zeros(2)

    smoothed_q, step = smooth_joint_positions(
        target_q,
        current_q,
        None,
        0.1,
        alpha=1.0,
        max_velocity_rad_s=None,
        max_acceleration_rad_s2=None,
    )

    np.testing.assert_allclose(smoothed_q, target_q)
    np.testing.assert_allclose(step, target_q)


def test_xr_pose_filter_applies_position_deadband_and_smoothing():
    pose_filter = XrPoseFilter(
        position_alpha=0.5,
        orientation_alpha=1.0,
        position_deadband_m=0.01,
        orientation_deadband_rad=0.0,
    )
    initial = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    jitter = np.array([0.005, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    moved = np.array([0.105, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    np.testing.assert_allclose(pose_filter.update(initial), initial)
    np.testing.assert_allclose(pose_filter.update(jitter), initial)
    np.testing.assert_allclose(pose_filter.update(moved)[:3], [0.0525, 0.0, 0.0])


def test_xr_pose_filter_slerps_orientation_and_resets_state():
    pose_filter = XrPoseFilter(
        position_alpha=1.0,
        orientation_alpha=0.5,
        position_deadband_m=0.0,
        orientation_deadband_rad=0.0,
    )
    initial = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    yaw_90 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)])

    pose_filter.update(initial)
    filtered = pose_filter.update(yaw_90)

    np.testing.assert_allclose(filtered[3:], [0.0, 0.0, np.sin(np.pi / 8), np.cos(np.pi / 8)])

    pose_filter.reset()
    np.testing.assert_allclose(pose_filter.update(yaw_90), yaw_90)


def test_inactive_arm_smoothing_holds_feedback_and_clears_previous_step():
    class FakeConfiguration:
        def __init__(self):
            self.q = np.arange(18, dtype=float)

        def update(self, q):
            self.q = np.asarray(q, dtype=float).copy()

    teleop = WheeledArmPico(WheeledArmPicoConfig())
    teleop._configuration = FakeConfiguration()
    teleop._arm_q_indices = np.arange(14)
    teleop._locked_q_indices = np.array([14, 15, 16, 17])
    teleop._q_ref = np.zeros(18)
    teleop._dt = 0.1
    teleop._filtered_arm_q = np.arange(14, dtype=float) + 1.0
    teleop._previous_arm_step = np.ones(14, dtype=float) * 0.2

    left_active_mask = teleop._active_arm_mask(left_active=True, right_active=False)
    target_q = teleop._configuration.q.copy()
    target_q[:7] += 1.0
    target_q[7:14] = np.arange(7, 14, dtype=float)
    teleop._configuration.update(target_q)

    teleop._apply_action_smoothing_to_configuration(left_active_mask)

    assert np.all(teleop._previous_arm_step[:7] != 0.0)
    np.testing.assert_allclose(teleop._previous_arm_step[7:], np.zeros(7))
    np.testing.assert_allclose(teleop._configuration.q[7:14], np.arange(7, 14, dtype=float))


def test_locked_joints_task_uses_configured_gain_and_lm_damping():
    class BaseTask:
        def __init__(self, cost=None, gain=1.0, lm_damping=0.0):
            self.cost = cost
            self.gain = gain
            self.lm_damping = lm_damping

    LockedJointsTask = make_locked_joints_task_class(BaseTask)

    task = LockedJointsTask(
        q_indices=np.array([0, 1]),
        v_indices=np.array([0, 1]),
        target_q=np.array([1.0, 2.0]),
        gain=0.6,
        lm_damping=1e-4,
    )

    assert task.gain == 0.6
    assert task.lm_damping == 1e-4
