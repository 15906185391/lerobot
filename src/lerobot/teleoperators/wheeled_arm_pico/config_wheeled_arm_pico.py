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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from ..config import TeleoperatorConfig


def _validate_non_negative(name: str, value: float) -> None:
    if value < 0.0:
        raise ValueError(f"`{name}` must be non-negative, got {value}.")


def _validate_gain(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"`{name}` must be in [0, 1], got {value}.")


def _validate_cost(name: str, value: float | Sequence[float], expected_len: int | None = None) -> None:
    if isinstance(value, float | int):
        _validate_non_negative(name, float(value))
        return

    values = list(value)
    if expected_len is not None and len(values) != expected_len:
        raise ValueError(f"`{name}` must be a scalar or a sequence of length {expected_len}, got {value}.")
    for item in values:
        _validate_non_negative(name, float(item))


@TeleoperatorConfig.register_subclass("wheeled_arm_pico")
@dataclass
class WheeledArmPicoConfig(TeleoperatorConfig):
    # 机器人 URDF 路径。默认使用 teleoperator assets 下的 real_robot.urdf。
    urdf_path: Path | None = None

    # PICO 手柄位移到末端目标位移的比例。调大更灵敏，调小更稳。
    scale: float = 1.0
    # grip 输入超过该阈值时，对应手臂才开始跟随手柄。
    activation_threshold: float = 0.9
    # True 时只跟踪末端位置，不跟踪手柄姿态；抖动大时可先打开。
    position_only: bool = False

    # IK 求解频率。通常应与采集 fps 接近，过低会卡顿，过高会增加 CPU/QP 压力。
    solve_frequency_hz: float = 30.0
    # QP solver 名称。None 时自动优先选择 daqp，其次 osqp。
    solver: str | None = None
    # 透传给 qpsolvers.solve_problem 的参数，例如 {"verbose": False}。
    solver_kwargs: dict[str, float | int | bool | str] = field(default_factory=dict)
    # solve_ik 全局 Tikhonov damping。调大可改善数值稳定性，但动作会变慢。
    ik_damping: float = 1e-12
    # True 时如果当前 q 超出模型限位会直接报错；False 时只警告并继续。
    ik_safety_break: bool = False
    # True 时启用模型自带 configuration/velocity limits；调试时可临时关闭。
    enforce_limits: bool = True

    # 是否启用自碰撞 barrier。关闭后可绕过 hpp-fcl/coal 碰撞后端。
    use_self_collision: bool = True
    # 自碰撞最小安全距离，单位米。调大更保守，但可能限制可达空间。
    d_min: float = 0.03
    # 初始姿态下距离小于该值的碰撞对会被忽略；None 时使用 d_min。
    initial_ignore_distance: float | None = None
    # 自碰撞 barrier 约束增益。调大更强硬地避障，过大可能让 QP 更难求。
    self_collision_gain: float = 10.0
    # 安全回退位移的代价增益。调大可加强离开危险区的趋势。
    self_collision_safe_displacement_gain: float = 5.0
    # 可视化/日志中的碰撞 warning 阈值，单位米，不影响实际 barrier d_min。
    collision_warning_distance: float = 0.01

    # mock_xr=True 时不连接 PICO，用内置模拟手柄数据验证 IK/可视化链路。
    mock_xr: bool = False
    # PICO 输入名称，需要与 xrobotoolkit_teleop 暴露的 name 保持一致。
    reset_button: str = "Y"
    left_controller_name: str = "left_controller"
    right_controller_name: str = "right_controller"
    left_grip_name: str = "left_grip"
    right_grip_name: str = "right_grip"
    left_gripper_input_name: str = "left_trigger"
    right_gripper_input_name: str = "right_trigger"
    # trigger=0/1 分别映射到 open/closed；实物夹爪单位不同时改这两个值。
    gripper_open_pos: float = 0.0
    gripper_closed_pos: float = 1.0

    # 左右末端 FrameTask 权重。可传标量，也可传 3 维列表分别调 xyz / rpy 三轴。
    position_cost: float | list[float] = 5.0
    orientation_cost: float | list[float] = 1.0
    # FrameTask 的 Levenberg-Marquardt damping。目标不可达且动作抖时可适当增大。
    frame_lm_damping: float = 0.0
    # FrameTask gain，范围 [0, 1]。调小会低通目标跟踪，动作更慢但更稳。
    task_gain: float = 0.5
    # PostureTask 让手臂保持接近参考姿态，主要用于冗余自由度正则化。
    posture_cost: float = 1e-4
    posture_gain: float = 1.0
    posture_lm_damping: float = 0.0
    # 非双臂关节通过 hard equality constraint 锁在参考位姿。
    locked_joints_gain: float = 1.0
    locked_joints_lm_damping: float = 0.0

    # viser 可视化只显示 PICO/IK/URDF 链路；采集数据流请用 --display_data=true。
    visualize: bool = False
    visualization_host: str = "0.0.0.0"
    visualization_port: int = 8082
    # True 时自动打开浏览器；远程/无桌面环境建议设为 false。
    visualization_open_browser: bool = True
    # viser 刷新频率。0 表示每帧都更新；降低可减少可视化开销。
    visualization_update_hz: float = 10.0

    def __post_init__(self) -> None:
        _validate_non_negative("scale", self.scale)
        _validate_gain("activation_threshold", self.activation_threshold)
        _validate_non_negative("solve_frequency_hz", self.solve_frequency_hz)
        if self.solve_frequency_hz == 0.0:
            raise ValueError("`solve_frequency_hz` must be positive.")
        _validate_non_negative("ik_damping", self.ik_damping)
        _validate_non_negative("d_min", self.d_min)
        if self.initial_ignore_distance is not None:
            _validate_non_negative("initial_ignore_distance", self.initial_ignore_distance)
        _validate_non_negative("self_collision_gain", self.self_collision_gain)
        _validate_non_negative(
            "self_collision_safe_displacement_gain", self.self_collision_safe_displacement_gain
        )
        _validate_non_negative("collision_warning_distance", self.collision_warning_distance)
        _validate_cost("position_cost", self.position_cost, expected_len=3)
        _validate_cost("orientation_cost", self.orientation_cost, expected_len=3)
        _validate_non_negative("frame_lm_damping", self.frame_lm_damping)
        _validate_gain("task_gain", self.task_gain)
        _validate_non_negative("posture_cost", self.posture_cost)
        _validate_gain("posture_gain", self.posture_gain)
        _validate_non_negative("posture_lm_damping", self.posture_lm_damping)
        _validate_gain("locked_joints_gain", self.locked_joints_gain)
        _validate_non_negative("locked_joints_lm_damping", self.locked_joints_lm_damping)
        _validate_non_negative("visualization_update_hz", self.visualization_update_hz)
