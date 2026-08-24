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
import time
from unittest.mock import patch

import numpy as np

from lerobot.robots import make_robot_from_config
from lerobot.robots.config import RobotConfig
from lerobot.robots.wheeled_arm import WheeledArm, WheeledArmConfig, WheeledArmWithHipYawConfig
from lerobot.robots.wheeled_arm.config_wheeled_arm import (
    WHEELED_ARM_ACTIVE_ARMS_ACTION_KEY,
    WHEELED_ARM_DEFAULT_ROS2_CAMERA_TOPIC,
    wheeled_arm_cameras_config,
)


class FakeLCMHandler:
    def __init__(self):
        self.joint_current_pos = np.arange(23, dtype=np.float32)
        self.joint_current_pos_lock = threading.Lock()
        self.last_package = None
        self.published_packages = []
        self.publish_started_event = threading.Event()
        self.release_publish_event = threading.Event()
        self.block_publish = False
        self.stopped = False
        self.has_feedback = True
        self.arm_state_time_s = time.monotonic()

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
        if self.block_publish:
            self.publish_started_event.set()
            self.release_publish_event.wait(timeout=0.2)
        self.last_package = np.asarray(package).copy()
        self.published_packages.append(self.last_package)

    def has_arm_state_feedback(self, max_age_s=None, min_time_s=None):
        if not self.has_feedback:
            return False
        return min_time_s is None or self.arm_state_time_s >= min_time_s

    def simulate_arm_state_feedback(self, joint_position):
        with self.joint_current_pos_lock:
            self.joint_current_pos[: len(joint_position)] = joint_position
            self.arm_state_time_s = time.monotonic()

    def stop(self):
        self.stopped = True


class FakeCamera:
    def __init__(self):
        self.height = 480
        self.width = 640
        self.use_rgb = True
        self.use_depth = False
        self.is_connected = False
        self.connect_calls = 0
        self.disconnect_calls = 0

    def connect(self):
        self.is_connected = True
        self.connect_calls += 1

    def disconnect(self):
        self.is_connected = False
        self.disconnect_calls += 1

    def read_latest(self):
        return np.full((self.height, self.width, 3), 127, dtype=np.uint8)


def _make_robot(**overrides):
    handler = FakeLCMHandler()
    overrides.setdefault("use_control_loop", False)
    with patch("lerobot.robots.wheeled_arm.wheeled_arm._make_lcm_handler", return_value=handler):
        robot = WheeledArm(WheeledArmConfig(
            cameras={}, connect_timeout_s=0, **overrides))
        robot.connect()
    return robot, handler


def test_wheeled_arm_config_is_registered():
    assert "wheeled_arm" in RobotConfig.get_known_choices()
    assert isinstance(make_robot_from_config(
        WheeledArmConfig(cameras={}, connect_timeout_s=0)), WheeledArm)


def test_wheeled_arm_with_hip_yaw_config_is_registered():
    assert "wheeled_arm_with_hip_yaw" in RobotConfig.get_known_choices()
    robot = make_robot_from_config(WheeledArmWithHipYawConfig(cameras={}, connect_timeout_s=0))

    assert isinstance(robot, WheeledArm)
    assert "hip_yaw.pos" in robot.action_features


def test_default_camera_config_uses_ros2_camera():
    cameras = wheeled_arm_cameras_config()

    assert list(cameras) == ["front"]
    assert cameras["front"].type == "ros2"
    assert cameras["front"].topic_name == WHEELED_ARM_DEFAULT_ROS2_CAMERA_TOPIC
    assert cameras["front"].image_transport == "compressed"
    assert cameras["front"].queue_size == 1
    assert cameras["front"].width == 640
    assert cameras["front"].height == 480
    assert cameras["front"].fps == 30


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


def test_robot_side_control_loop_publishes_latest_target_at_control_dt():
    robot, handler = _make_robot(
        use_control_loop=True,
        control_dt=0.01,
        interpolate_control_loop_actions=False,
    )

    returned = robot.send_action({"left_arm_0.pos": 1.5})

    deadline = time.monotonic() + 0.2
    while len(handler.published_packages) < 2 and time.monotonic() < deadline:
        time.sleep(0.005)

    assert returned == {"left_arm_0.pos": 1.5}
    assert len(handler.published_packages) >= 2
    assert handler.last_package[0] == 1.5
    assert handler.left_arm_moving is True
    assert handler.right_arm_moving is False

    robot.disconnect()
    assert robot._control_loop_thread is None


def test_control_loop_interpolates_between_last_command_and_new_target():
    robot, _handler = _make_robot()
    target = np.arange(23, dtype=np.float32)
    target[0] = 2.0

    robot._commanded_package = np.zeros(23, dtype=np.float32)
    robot._target_package = target
    robot._target_action_keys = {"left_arm_0.pos"}
    robot._target_indices = np.array([0], dtype=int)
    robot._interpolation_start_package = np.zeros(23, dtype=np.float32)
    robot._interpolation_start_t = 10.0
    robot._interpolation_duration_s = 1.0

    package = robot._interpolated_control_loop_package(now=10.5)

    assert package[0] == 1.0
    np.testing.assert_allclose(package[1:], target[1:])
    robot.disconnect()


def test_control_loop_interpolation_can_be_disabled():
    robot, _handler = _make_robot(
        interpolate_control_loop_actions=False,
    )
    target = np.arange(23, dtype=np.float32)
    target[0] = 2.0

    robot._target_package = target
    robot._target_action_keys = {"left_arm_0.pos"}
    robot._target_indices = np.array([0], dtype=int)
    robot._interpolation_start_package = np.zeros(23, dtype=np.float32)
    robot._interpolation_start_t = 10.0
    robot._interpolation_duration_s = 1.0

    package = robot._interpolated_control_loop_package(now=10.5)

    np.testing.assert_allclose(package, target)
    robot.disconnect()


def test_control_loop_adapts_interpolation_duration_to_action_interval():
    robot, _handler = _make_robot(
        action_interpolation_duration_s=1.0 / 30.0,
        action_interpolation_min_duration_s=0.02,
        action_interpolation_max_duration_s=0.06,
    )
    package = np.arange(23, dtype=np.float32)

    with patch("lerobot.robots.wheeled_arm.wheeled_arm.time.perf_counter", return_value=10.0):
        robot._set_control_loop_target(package, {"left_arm_0.pos"})
    assert robot._interpolation_duration_s == 1.0 / 30.0

    with patch("lerobot.robots.wheeled_arm.wheeled_arm.time.perf_counter", return_value=10.05):
        robot._set_control_loop_target(package, {"left_arm_0.pos"})
    np.testing.assert_allclose(robot._interpolation_duration_s, 0.05)

    with patch("lerobot.robots.wheeled_arm.wheeled_arm.time.perf_counter", return_value=10.20):
        robot._set_control_loop_target(package, {"left_arm_0.pos"})
    np.testing.assert_allclose(robot._interpolation_duration_s, 0.06)

    with patch("lerobot.robots.wheeled_arm.wheeled_arm.time.perf_counter", return_value=10.205):
        robot._set_control_loop_target(package, {"left_arm_0.pos"})
    np.testing.assert_allclose(robot._interpolation_duration_s, 0.02)

    robot.disconnect()


def test_control_loop_can_use_fixed_interpolation_duration():
    robot, _handler = _make_robot(
        adaptive_action_interpolation_duration=False,
        action_interpolation_duration_s=0.04,
    )
    package = np.arange(23, dtype=np.float32)

    with patch("lerobot.robots.wheeled_arm.wheeled_arm.time.perf_counter", return_value=10.0):
        robot._set_control_loop_target(package, {"left_arm_0.pos"})
    with patch("lerobot.robots.wheeled_arm.wheeled_arm.time.perf_counter", return_value=10.20):
        robot._set_control_loop_target(package, {"left_arm_0.pos"})

    assert robot._interpolation_duration_s == 0.04
    robot.disconnect()


def test_robot_side_control_loop_publishes_interpolated_targets():
    robot, handler = _make_robot(
        use_control_loop=True,
        control_dt=0.01,
        action_interpolation_duration_s=1.0,
    )

    robot.send_action({"left_arm_0.pos": 2.0})

    deadline = time.monotonic() + 0.2
    while not handler.published_packages and time.monotonic() < deadline:
        time.sleep(0.005)

    assert handler.published_packages
    assert 0.0 <= handler.published_packages[0][0] < 2.0
    robot.disconnect()


def test_control_loop_publish_does_not_block_target_updates():
    robot, handler = _make_robot(
        use_control_loop=True,
        control_dt=0.01,
        interpolate_control_loop_actions=False,
    )
    handler.block_publish = True

    robot.send_action({"left_arm_0.pos": 1.0})
    assert handler.publish_started_event.wait(timeout=0.2)

    start = time.perf_counter()
    robot.send_action({"left_arm_0.pos": 2.0})
    elapsed_s = time.perf_counter() - start

    handler.release_publish_event.set()
    robot.disconnect()

    assert elapsed_s < 0.05


def test_control_loop_watchdog_stops_moving_flags_after_stale_action():
    robot, handler = _make_robot(
        use_control_loop=True,
        control_dt=0.005,
        action_watchdog_timeout_s=0.02,
        interpolate_control_loop_actions=False,
    )

    robot.send_action({"left_arm_0.pos": 1.0})
    deadline = time.monotonic() + 0.2
    while not handler.left_arm_moving and time.monotonic() < deadline:
        time.sleep(0.005)
    assert handler.left_arm_moving is True

    deadline = time.monotonic() + 0.2
    while handler.left_arm_moving and time.monotonic() < deadline:
        time.sleep(0.005)

    assert handler.left_arm_moving is False
    assert robot._target_package is None
    robot.disconnect()


def test_empty_control_loop_action_stops_previous_moving_flags_immediately():
    robot, handler = _make_robot(
        use_control_loop=True,
        control_dt=0.01,
        interpolate_control_loop_actions=False,
    )

    robot.send_action({"left_arm_0.pos": 1.0})
    deadline = time.monotonic() + 0.2
    while not handler.left_arm_moving and time.monotonic() < deadline:
        time.sleep(0.005)

    robot.send_action(
        {
            "left_arm_0.pos": 1.0,
            WHEELED_ARM_ACTIVE_ARMS_ACTION_KEY: (),
        }
    )

    assert handler.left_arm_moving is False
    robot.disconnect()


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


def test_send_action_respects_pico_active_arm_metadata():
    robot, handler = _make_robot()

    returned = robot.send_action(
        {
            "left_arm_0.pos": 1.5,
            "right_arm_6.pos": 2.5,
            "left_gripper.pos": 0.25,
            WHEELED_ARM_ACTIVE_ARMS_ACTION_KEY: ("left_arm",),
        }
    )

    assert returned == {
        "left_arm_0.pos": 1.5,
        "right_arm_6.pos": 2.5,
        "left_gripper.pos": 0.25,
    }
    assert handler.last_package[0] == 1.5
    assert handler.last_package[13] == 2.5
    assert handler.left_arm_moving is True
    assert handler.right_arm_moving is False
    assert handler.left_gripper_moving is True


def test_send_action_can_command_hip_yaw_when_waist_is_configured():
    robot, handler = _make_robot(
        joint_names=[
            *(f"left_arm_{idx}" for idx in range(7)),
            *(f"right_arm_{idx}" for idx in range(7)),
            "left_gripper",
            "right_gripper",
            "hip_yaw",
        ],
        controlled_parts=["left_arm", "right_arm", "left_gripper", "right_gripper", "waist"],
    )

    returned = robot.send_action(
        {
            "left_arm_0.pos": 1.5,
            "hip_yaw.pos": -0.25,
            WHEELED_ARM_ACTIVE_ARMS_ACTION_KEY: ("left_arm", "waist"),
        }
    )

    assert returned == {"left_arm_0.pos": 1.5, "hip_yaw.pos": -0.25}
    assert handler.last_package[0] == 1.5
    assert handler.last_package[20] == -0.25
    assert handler.left_arm_moving is True
    assert handler.waist_moving is True
    assert handler.right_arm_moving is False


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
        def __init__(
            self,
            lcm_handler,
            collision_detection,
            stop_requested=None,
            progress_callback=None,
        ):
            self.lcm_handler = lcm_handler
            self.collision_detection = collision_detection
            self.stop_requested = stop_requested
            self.progress_callback = progress_callback

        def moveJ2target(self, current_position, target_position):  # noqa: N802
            calls.append(
                (np.asarray(current_position).copy(), np.asarray(target_position).copy())
            )
            self.lcm_handler.simulate_arm_state_feedback(target_position)
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
        def __init__(
            self,
            lcm_handler,
            collision_detection,
            stop_requested=None,
            progress_callback=None,
        ):
            self.stop_requested = stop_requested
            self.progress_callback = progress_callback

        def moveJ2target(self, current_position, target_position):  # noqa: N802
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


def test_mock_wheeled_arm_uses_software_joints_without_lcm_and_keeps_real_cameras():
    camera = FakeCamera()

    with patch("lerobot.robots.wheeled_arm.wheeled_arm._make_lcm_handler") as make_handler:
        robot = WheeledArm(WheeledArmConfig(cameras={}, mock=True, connect_timeout_s=0))
        robot.cameras = {"front": camera}
        robot.connect()

    make_handler.assert_not_called()
    assert robot.is_connected is True
    assert robot.has_valid_feedback is True
    assert camera.connect_calls == 1

    returned = robot.send_action({"right_gripper.pos": 0.6, "left_arm_0.pos": 0.2})
    assert returned == {"right_gripper.pos": 0.6, "left_arm_0.pos": 0.2}

    obs = robot.get_observation()
    np.testing.assert_allclose(obs["right_gripper.pos"], 0.6)
    np.testing.assert_allclose(obs["left_arm_0.pos"], 0.2)
    assert obs["front"].shape == (480, 640, 3)
    assert obs["front"].dtype == np.uint8

    robot.disconnect()
    assert camera.disconnect_calls == 1
    assert robot.is_connected is False


def test_mock_wheeled_arm_can_generate_synthetic_camera_images():
    camera = FakeCamera()

    with patch("lerobot.robots.wheeled_arm.wheeled_arm._make_lcm_handler") as make_handler:
        robot = WheeledArm(
            WheeledArmConfig(cameras={}, mock=True, mock_cameras=True, connect_timeout_s=0)
        )
        robot.cameras = {"front": camera}
        robot.connect()

    make_handler.assert_not_called()
    assert camera.connect_calls == 0
    obs = robot.get_observation()
    assert obs["front"].shape == (480, 640, 3)
    assert obs["front"].dtype == np.uint8

    robot.disconnect()
    assert camera.disconnect_calls == 0


def test_real_wheeled_arm_can_use_synthetic_camera_images_without_connecting_camera():
    camera = FakeCamera()
    handler = FakeLCMHandler()

    with patch(
        "lerobot.robots.wheeled_arm.wheeled_arm._make_lcm_handler", return_value=handler
    ) as make_handler:
        robot = WheeledArm(
            WheeledArmConfig(
                cameras={}, mock=False, mock_cameras=True, connect_timeout_s=0, use_control_loop=False
            )
        )
        robot.cameras = {"front": camera}
        robot.connect()

    make_handler.assert_called_once()
    assert robot.is_connected is True
    assert camera.connect_calls == 0

    obs = robot.get_observation()
    np.testing.assert_allclose(obs["left_arm_0.pos"], 0.0)
    assert obs["front"].shape == (480, 640, 3)
    assert obs["front"].dtype == np.uint8

    robot.disconnect()
    assert camera.disconnect_calls == 0
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
