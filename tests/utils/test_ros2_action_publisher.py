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

from lerobot.utils.ros2_action_publisher import action_key_to_joint_name, ordered_action_items


def test_ordered_action_items_uses_robot_feature_order_and_appends_unknown_keys():
    action = {
        "right_arm_0.pos": 2.0,
        "left_arm_0.pos": 1.0,
        "extra.pos": 3.0,
    }
    order = ["left_arm_0.pos", "right_arm_0.pos"]

    assert ordered_action_items(action, order) == [
        ("left_arm_0.pos", 1.0),
        ("right_arm_0.pos", 2.0),
        ("extra.pos", 3.0),
    ]


def test_action_key_to_joint_name_removes_position_suffix_only():
    assert action_key_to_joint_name("left_arm_0.pos") == "left_arm_0"
    assert action_key_to_joint_name("base.x") == "base.x"
