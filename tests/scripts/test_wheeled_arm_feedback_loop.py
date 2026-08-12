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

from types import SimpleNamespace

from lerobot.scripts.lerobot_record import record_loop
from lerobot.scripts.lerobot_teleoperate import teleop_loop
from lerobot.teleoperators.teleoperator import Teleoperator


class IdentityProcessor:
    def __call__(self, data):
        if isinstance(data, tuple):
            return data[0]
        return data


class FakeWheeledArmRobot:
    name = "wheeled_arm"
    action_features = {"left_arm_0.pos": float}

    def __init__(self):
        self.config = SimpleNamespace(require_fresh_feedback=False)
        self.has_valid_feedback = True
        self.sent_actions = []

    def get_observation(self):
        return {"left_arm_0.pos": 0.0}

    def send_action(self, action):
        self.sent_actions.append(action.copy())
        return action


class FakeTeleop(Teleoperator):
    name = "fake"

    def __init__(self, use_continuous_robot_feedback=False):
        self.id = None
        self.config = SimpleNamespace(use_continuous_robot_feedback=use_continuous_robot_feedback)
        self.feedback_count = 0

    @property
    def action_features(self):
        return {"left_arm_0.pos": float}

    @property
    def feedback_features(self):
        return {"left_arm_0.pos": float}

    @property
    def is_connected(self):
        return True

    @property
    def is_calibrated(self):
        return True

    def connect(self, calibrate=True):
        return None

    def calibrate(self):
        return None

    def configure(self):
        return None

    def send_feedback(self, feedback):
        self.feedback_count += 1

    def get_action(self):
        return {"left_arm_0.pos": 0.0}

    def disconnect(self):
        return None


def test_teleoperate_loop_skips_continuous_wheeled_arm_feedback_by_default():
    robot = FakeWheeledArmRobot()
    teleop = FakeTeleop(use_continuous_robot_feedback=False)
    processor = IdentityProcessor()

    teleop_loop(
        teleop=teleop,
        robot=robot,
        fps=100,
        teleop_action_processor=processor,
        robot_action_processor=processor,
        robot_observation_processor=processor,
        duration=0.02,
    )

    assert teleop.feedback_count == 0
    assert robot.sent_actions


def test_teleoperate_loop_can_use_continuous_wheeled_arm_feedback():
    robot = FakeWheeledArmRobot()
    teleop = FakeTeleop(use_continuous_robot_feedback=True)
    processor = IdentityProcessor()

    teleop_loop(
        teleop=teleop,
        robot=robot,
        fps=100,
        teleop_action_processor=processor,
        robot_action_processor=processor,
        robot_observation_processor=processor,
        duration=0.02,
    )

    assert teleop.feedback_count > 0


def test_record_loop_skips_continuous_wheeled_arm_feedback_by_default():
    robot = FakeWheeledArmRobot()
    teleop = FakeTeleop(use_continuous_robot_feedback=False)
    processor = IdentityProcessor()
    events = {"exit_early": False}

    record_loop(
        robot=robot,
        events=events,
        fps=100,
        teleop_action_processor=processor,
        robot_action_processor=processor,
        robot_observation_processor=processor,
        teleop=teleop,
        control_time_s=0.02,
    )

    assert teleop.feedback_count == 0
    assert robot.sent_actions


def test_record_loop_can_use_continuous_wheeled_arm_feedback():
    robot = FakeWheeledArmRobot()
    teleop = FakeTeleop(use_continuous_robot_feedback=True)
    processor = IdentityProcessor()
    events = {"exit_early": False}

    record_loop(
        robot=robot,
        events=events,
        fps=100,
        teleop_action_processor=processor,
        robot_action_processor=processor,
        robot_observation_processor=processor,
        teleop=teleop,
        control_time_s=0.02,
    )

    assert teleop.feedback_count > 0
