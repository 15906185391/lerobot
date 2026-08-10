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

import threading
from unittest.mock import patch

import numpy as np

from lerobot.robots import make_robot_from_config
from lerobot.robots.config import RobotConfig
from lerobot.robots.wheeled_arm import WheeledArm, WheeledArmConfig
from lerobot.robots.wheeled_arm.config_wheeled_arm import wheeled_arm_cameras_config


class FakeLCMHandler:
    def __init__(self):
        self.joint_current_pos = np.arange(23, dtype=np.float32)
        self.joint_current_pos_lock = threading.Lock()
        self.last_package = None
        self.stopped = False
        self.has_feedback = True

        for flag in (
            "left_arm_moving",
            "right_arm_moving",
            "left_gripper_moving",
            "right_gripper_moving",
            "head_moving",
            "waist_moving",
            "leg_moving",
        ):
            setattr(self, flag, True)

    def upper_body_data_publisher(self, package):
        self.last_package = np.asarray(package).copy()

    def has_arm_state_feedback(self, max_age_s=None):
        return self.has_feedback

    def stop(self):
        self.stopped = True


def _make_robot(**overrides):
    handler = FakeLCMHandler()
    with patch("lerobot.robots.wheeled_arm.wheeled_arm._make_lcm_handler", return_value=handler):
        robot = WheeledArm(WheeledArmConfig(
            cameras={}, connect_timeout_s=0, **overrides))
        robot.connect()
    return robot, handler


def test_wheeled_arm_config_is_registered():
    assert "wheeled_arm" in RobotConfig.get_known_choices()
    assert isinstance(make_robot_from_config(
        WheeledArmConfig(cameras={}, connect_timeout_s=0)), WheeledArm)


def test_default_camera_config_uses_ros2_camera():
    cameras = wheeled_arm_cameras_config()

    assert list(cameras) == ["front"]
    assert cameras["front"].type == "lerobot_camera_ros2"
    assert cameras["front"].topic_name == "/camera/color/image_raw"
    assert cameras["front"].width == 640
    assert cameras["front"].height == 480


def test_get_observation_reads_left_and_right_arm_joint_positions():
    robot, _handler = _make_robot()

    obs = robot.get_observation()

    assert len(obs) == 16
    assert obs["left_arm_0.pos"] == 0.0
    assert obs["right_arm_0.pos"] == 7.0
    assert obs["right_arm_6.pos"] == 13.0
    assert obs["left_gripper.pos"] == 14.0
    assert obs["right_gripper.pos"] == 15.0


def test_has_valid_feedback_reflects_lcm_arm_state_availability():
    robot, handler = _make_robot()

    assert robot.has_valid_feedback is True

    handler.has_feedback = False

    assert robot.has_valid_feedback is False


def test_send_action_publishes_23_dim_package_for_arm_joints_only():
    robot, handler = _make_robot()

    returned = robot.send_action(
        {
            "left_arm_0.pos": 1.5,
            "right_arm_6.pos": 2.5,
            "left_gripper.pos": 0.25,
        }
    )

    assert returned == {
        "left_arm_0.pos": 1.5,
        "right_arm_6.pos": 2.5,
        "left_gripper.pos": 0.25,
    }
    assert handler.last_package[0] == 1.5
    assert handler.last_package[13] == 2.5
    assert handler.last_package[14] == 0.25
    assert handler.last_package[1] == 1.0
    assert handler.last_package[15] == 15.0

    assert handler.left_arm_moving is True
    assert handler.right_arm_moving is True
    assert handler.left_gripper_moving is True
    assert handler.right_gripper_moving is False
    assert handler.head_moving is False
    assert handler.waist_moving is False
    assert handler.leg_moving is False


def test_send_action_respects_disabled_parts_and_relative_limit():
    robot, handler = _make_robot(
        controlled_parts=["left_arm", "left_gripper"], max_relative_target=2.0)

    returned = robot.send_action(
        {"left_arm_0.pos": 100.0, "right_arm_0.pos": 100.0, "left_gripper.pos": 100.0})

    assert returned == {"left_arm_0.pos": 2.0, "right_arm_0.pos": 9.0, "left_gripper.pos": 16.0}
    assert handler.last_package[0] == 2.0
    assert handler.last_package[7] == 9.0
    assert handler.last_package[14] == 16.0
    assert handler.left_arm_moving is True
    assert handler.right_arm_moving is False
    assert handler.left_gripper_moving is True


def test_reset_to_rest_pose_uses_movej_with_arm_targets_in_radians():
    robot, handler = _make_robot()
    calls = []

    class FakeMOVEJ:
        def __init__(self, lcm_handler, collision_detection, stop_requested=None):
            self.lcm_handler = lcm_handler
            self.collision_detection = collision_detection
            self.stop_requested = stop_requested

        def moveJ2target(self, current_position, target_position):
            calls.append(
                (np.asarray(current_position).copy(), np.asarray(target_position).copy())
            )
            return True

    with patch("lerobot.robots.wheeled_arm.hardware_interface.trajectory_plan.moveJ.MOVEJ", FakeMOVEJ):
        robot.reset_to_rest_pose()

    assert len(calls) == 1
    current_position, target_position = calls[0]
    np.testing.assert_allclose(current_position, np.arange(23, dtype=np.float32))
    np.testing.assert_allclose(
        target_position[:7],
        np.deg2rad([20.0, 70.0, -75.0, 100.0, -25.0, 0.0, 0.0]),
    )
    np.testing.assert_allclose(
        target_position[7:14],
        np.deg2rad([-20.0, 70.0, 75.0, 100.0, 25.0, 0.0, 0.0]),
    )
    np.testing.assert_allclose(target_position[14:], np.arange(14, 23, dtype=np.float32))
    np.testing.assert_allclose(handler.joint_current_pos, target_position)
    assert handler.left_arm_moving is True
    assert handler.right_arm_moving is True
    assert handler.left_gripper_moving is False
    assert handler.right_gripper_moving is False
    assert handler.head_moving is False
    assert handler.waist_moving is False
    assert handler.leg_moving is False


def test_reset_to_rest_pose_stops_when_movej_is_interrupted():
    robot, handler = _make_robot()

    class InterruptedMOVEJ:
        def __init__(self, lcm_handler, collision_detection, stop_requested=None):
            self.stop_requested = stop_requested

        def moveJ2target(self, current_position, target_position):
            assert self.stop_requested is not None
            assert self.stop_requested() is True
            return False

    with patch("lerobot.robots.wheeled_arm.hardware_interface.trajectory_plan.moveJ.MOVEJ", InterruptedMOVEJ):
        completed = robot.reset_to_rest_pose(stop_requested=lambda: True)

    assert completed is False
    np.testing.assert_allclose(handler.joint_current_pos, np.arange(23, dtype=np.float32))
    assert handler.left_arm_moving is False
    assert handler.right_arm_moving is False
    assert handler.left_gripper_moving is False
    assert handler.right_gripper_moving is False
    assert handler.head_moving is False
    assert handler.waist_moving is False
    assert handler.leg_moving is False


def test_disconnect_stops_handler():
    robot, handler = _make_robot()

    robot.disconnect()

    assert handler.stopped is True
    assert robot.is_connected is False


def test_unknown_action_key_raises():
    robot, _handler = _make_robot()

    try:
        robot.send_action({"not_a_joint.pos": 1.0})
    except ValueError as exc:
        assert "not_a_joint.pos" in str(exc)
    else:
        raise AssertionError(
            "Expected ValueError for an unknown wheeled_arm action key.")
