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

from dataclasses import dataclass
from pathlib import Path

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("wheeled_arm_pico")
@dataclass
class WheeledArmPicoConfig(TeleoperatorConfig):
    urdf_path: Path | None = None

    scale: float = 1.0
    activation_threshold: float = 0.9
    position_only: bool = False

    solve_frequency_hz: float = 30.0
    solver: str | None = None

    use_self_collision: bool = True
    d_min: float = 0.03
    initial_ignore_distance: float | None = None

    mock_xr: bool = False
    reset_button: str = "Y"
    left_controller_name: str = "left_controller"
    right_controller_name: str = "right_controller"
    left_grip_name: str = "left_grip"
    right_grip_name: str = "right_grip"
    left_gripper_input_name: str = "left_trigger"
    right_gripper_input_name: str = "right_trigger"
    gripper_open_pos: float = 0.0
    gripper_closed_pos: float = 1.0

    position_cost: float = 5.0
    orientation_cost: float = 1.0
    task_gain: float = 0.5
    posture_cost: float = 1e-4

    visualize: bool = False
    visualization_host: str = "0.0.0.0"
    visualization_port: int = 8082
    visualization_open_browser: bool = True
    visualization_update_hz: float = 10.0
