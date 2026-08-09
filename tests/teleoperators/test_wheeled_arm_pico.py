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
from lerobot.teleoperators.wheeled_arm_pico.ik_utils import arm_q_from_feedback


def test_wheeled_arm_pico_config_is_registered():
    assert "wheeled_arm_pico" in TeleoperatorConfig.get_known_choices()
    assert isinstance(make_teleoperator_from_config(WheeledArmPicoConfig()), WheeledArmPico)


def test_action_features_match_wheeled_arm_arm_and_gripper_joints():
    teleop = WheeledArmPico(WheeledArmPicoConfig())

    assert list(teleop.action_features) == [
        *(f"left_arm_{idx}.pos" for idx in range(7)),
        *(f"right_arm_{idx}.pos" for idx in range(7)),
        "left_gripper.pos",
        "right_gripper.pos",
    ]


def test_arm_q_from_feedback_requires_complete_left_and_right_arm_state():
    joint_names = [
        *(f"left_arm_{idx}" for idx in range(7)),
        *(f"right_arm_{idx}" for idx in range(7)),
    ]
    feedback = {f"{name}.pos": float(idx) for idx, name in enumerate(joint_names)}

    arm_q = arm_q_from_feedback(feedback, joint_names)

    np.testing.assert_allclose(arm_q, np.arange(14, dtype=float))
    assert arm_q_from_feedback({"left_arm_0.pos": 0.0}, joint_names) is None
