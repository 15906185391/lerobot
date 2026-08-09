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

from pathlib import Path

import numpy as np

from lerobot.utils.import_utils import require_package

from .config_wheeled_arm_pico import WheeledArmPicoConfig
from .ik_utils import LEFT_TCP, RIGHT_TCP, se3_to_position_wxyz

LEFT_ARM_PREFIX = "AR5-5_07L-W4C4A2"
RIGHT_ARM_PREFIX = "AR5-5_07R-W4C4A2"


def require_rerun_robot_visualization_dependencies() -> None:
    require_package("rerun-sdk", extra="viz", import_name="rerun")


def _as_xyzw(wxyz: np.ndarray) -> np.ndarray:
    return np.asarray([wxyz[1], wxyz[2], wxyz[3], wxyz[0]], dtype=float)


def _joint_translation(model, data, joint_name: str) -> np.ndarray | None:
    joint_id = model.getJointId(joint_name)
    if joint_id == 0:
        return None
    return np.asarray(data.oMi[joint_id].translation, dtype=float).copy()


def _arm_chain_positions(model, data, configuration, side_prefix: str, tcp_frame: str) -> np.ndarray:
    positions = []
    for idx in range(1, 8):
        position = _joint_translation(model, data, f"{side_prefix}_joint_{idx}")
        if position is not None:
            positions.append(position)

    tcp_pose = configuration.get_transform_frame_to_world(tcp_frame)
    positions.append(np.asarray(tcp_pose.translation, dtype=float).copy())
    return np.asarray(positions, dtype=float)


def _log_transform(rr, path: str, transform, pin, axis_length: float) -> None:
    position, wxyz = se3_to_position_wxyz(transform, pin)
    rr.log(path, rr.Transform3D(translation=position, rotation=rr.Quaternion(xyzw=_as_xyzw(wxyz))))
    rr.log(path, rr.TransformAxes3D(axis_length=axis_length))


def _collision_color(collision_status: str) -> tuple[int, int, int]:
    return {
        "collision": (255, 0, 0),
        "warning": (255, 215, 0),
        "safe": (0, 180, 0),
        "disabled": (120, 120, 120),
    }.get(collision_status, (160, 160, 160))


def log_rerun_robot_visualization(
    config: WheeledArmPicoConfig,
    deps,
    robot,
    configuration,
    left_target_pose,
    right_target_pose,
    xr_ok: bool,
    left_active: bool,
    right_active: bool,
    collision_status: str,
    min_barrier: float | None,
    frame_index: int | None = None,
    timestamp: float | None = None,
    urdf_path: Path | None = None,
) -> None:
    require_rerun_robot_visualization_dependencies()
    import rerun as rr

    if frame_index is not None:
        rr.set_time("frame_index", sequence=frame_index)
    if timestamp is not None:
        rr.set_time("timestamp", timestamp=timestamp)

    pin = deps.pin
    model = robot.model
    data = robot.data
    pin.forwardKinematics(model, data, configuration.q)
    pin.updateFramePlacements(model, data)

    prefix = config.rerun_robot_prefix.rstrip("/")
    rr.log(prefix, rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    if urdf_path is not None:
        rr.log(f"{prefix}/metadata/urdf_path", rr.TextDocument(str(urdf_path)), static=True)

    left_chain = _arm_chain_positions(model, data, configuration, LEFT_ARM_PREFIX, LEFT_TCP)
    right_chain = _arm_chain_positions(model, data, configuration, RIGHT_ARM_PREFIX, RIGHT_TCP)
    rr.log(
        f"{prefix}/left_arm/skeleton",
        rr.LineStrips3D([left_chain], radii=[0.012], colors=[(31, 119, 180)]),
    )
    rr.log(
        f"{prefix}/right_arm/skeleton",
        rr.LineStrips3D([right_chain], radii=[0.012], colors=[(255, 127, 14)]),
    )
    rr.log(
        f"{prefix}/left_arm/joints",
        rr.Points3D(left_chain, radii=[0.025], colors=[(31, 119, 180)]),
    )
    rr.log(
        f"{prefix}/right_arm/joints",
        rr.Points3D(right_chain, radii=[0.025], colors=[(255, 127, 14)]),
    )

    left_tcp_pose = configuration.get_transform_frame_to_world(LEFT_TCP)
    right_tcp_pose = configuration.get_transform_frame_to_world(RIGHT_TCP)
    _log_transform(rr, f"{prefix}/left_tcp", left_tcp_pose, pin, config.rerun_robot_axis_length)
    _log_transform(rr, f"{prefix}/right_tcp", right_tcp_pose, pin, config.rerun_robot_axis_length)
    _log_transform(rr, f"{prefix}/left_target", left_target_pose, pin, config.rerun_robot_axis_length)
    _log_transform(rr, f"{prefix}/right_target", right_target_pose, pin, config.rerun_robot_axis_length)

    target_line_color = (80, 80, 80)
    rr.log(
        f"{prefix}/target_error",
        rr.LineStrips3D(
            [
                np.asarray([left_chain[-1], left_target_pose.translation], dtype=float),
                np.asarray([right_chain[-1], right_target_pose.translation], dtype=float),
            ],
            radii=[0.006],
            colors=[target_line_color],
        ),
    )

    status_label = collision_status
    if min_barrier is not None:
        status_label = f"{collision_status}: {min_barrier:.4f}m"
    rr.log(
        f"{prefix}/status/collision",
        rr.Points3D(
            [[0.0, 0.0, 0.45]],
            radii=[0.045],
            colors=[_collision_color(collision_status)],
            labels=[status_label],
            show_labels=True,
        ),
    )
    rr.log(f"{prefix}/status/xr_ok", rr.Scalars(float(xr_ok)))
    rr.log(f"{prefix}/status/left_active", rr.Scalars(float(left_active)))
    rr.log(f"{prefix}/status/right_active", rr.Scalars(float(right_active)))
    if min_barrier is not None:
        rr.log(f"{prefix}/status/min_barrier", rr.Scalars(float(min_barrier)))
