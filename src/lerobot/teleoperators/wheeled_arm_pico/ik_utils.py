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

import importlib
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

LEFT_TCP = "AR5-5_07L-W4C4A2_tcp"
RIGHT_TCP = "AR5-5_07R-W4C4A2_tcp"
PACKAGE_NAME = "wheeled_robot_sim"
RIGHT_ARM_INITIAL_DEG = [-20.0, 70.0, 75.0, 100.0, 25.0, 0.0, 0.0]
LEFT_ARM_INITIAL_DEG = [20.0, 70.0, -75.0, 100.0, -25.0, 0.0, 0.0]
R_HEADSET_TO_WORLD = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=float)
JOINT_COLORS = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#17becf",
)


def default_urdf_path() -> Path:
    return (
        Path(__file__).parent
        / "assets"
        / PACKAGE_NAME
        / "urdf"
        / "real_robot.urdf"
    ).resolve()


def ensure_vendor_path() -> Path:
    vendor_root = (Path(__file__).parent / "pink").resolve()
    if str(vendor_root) not in sys.path:
        sys.path.insert(0, str(vendor_root))
    return vendor_root


def import_runtime_dependencies(require_collision_backend: bool = True) -> SimpleNamespace:
    ensure_vendor_path()
    missing = []

    try:
        import pinocchio as pin
    except ImportError as exc:
        pin = None
        missing.append(f"pinocchio (install package `pin`): {exc}")

    fcl = None
    if require_collision_backend:
        fcl_errors = []
        for module_name in ("hppfcl", "coal"):
            try:
                fcl = importlib.import_module(module_name)
                break
            except ImportError as exc:
                fcl_errors.append(f"{module_name}: {exc}")
        if fcl is None:
            missing.append(
                "collision geometry backend (install package `hpp-fcl` or `coal-library`; "
                f"tried imports: {'; '.join(fcl_errors)})"
            )

    try:
        import qpsolvers
    except ImportError as exc:
        qpsolvers = None
        missing.append(f"qpsolvers: {exc}")

    try:
        import pink
        from pink import solve_ik
        from pink.barriers import SelfCollisionBarrier
        from pink.exceptions import NoSolutionFound
        from pink.tasks import DampingTask, FrameTask, PostureTask, Task
    except ImportError as exc:
        pink = None
        solve_ik = SelfCollisionBarrier = NoSolutionFound = DampingTask = FrameTask = PostureTask = Task = None
        missing.append(f"bundled pink IK package: {exc}")

    if missing:
        raise ImportError(
            "wheeled_arm_pico requires Pinocchio/Pink IK dependencies plus the PICO SDK. "
            "Missing runtime dependencies in the current Python environment:\n"
            + "\n".join(f"  - {item}" for item in missing)
            + "\nInstall runtime packages such as `pin`, `hpp-fcl` or `coal-library`, "
            "`qpsolvers`, `daqp`, and `xrobotoolkit_sdk`. Python 3.12+ is recommended "
            "for the current LeRobot dependency set. To run without collision checking, "
            "pass `--teleop.use_self_collision=false`."
        )

    return SimpleNamespace(
        fcl=fcl,
        pin=pin,
        qpsolvers=qpsolvers,
        pink=pink,
        solve_ik=solve_ik,
        SelfCollisionBarrier=SelfCollisionBarrier,
        NoSolutionFound=NoSolutionFound,
        DampingTask=DampingTask,
        FrameTask=FrameTask,
        PostureTask=PostureTask,
        Task=Task,
    )


class MockXrClient:
    """Small XR source for validating the pipeline without a PICO device."""

    def __init__(self) -> None:
        self.start_time = time.monotonic()

    def get_pose_by_name(self, name: str) -> np.ndarray:
        t = time.monotonic() - self.start_time
        y = 0.25 if name == "left_controller" else -0.25
        return np.array([0.05 * np.sin(t), y, 0.02 * np.cos(t), 0, 0, 0, 1], dtype=float)

    def get_key_value_by_name(self, name: str) -> float:
        if name in ("left_grip", "right_grip"):
            return 1.0
        if name in ("left_trigger", "right_trigger"):
            t = time.monotonic() - self.start_time
            phase = 0.0 if name == "left_trigger" else np.pi
            return float(0.5 + 0.5 * np.sin(t + phase))
        return 0.0

    def get_button_state_by_name(self, name: str) -> bool:
        return False

    def close(self) -> None:
        pass


def _normalize_xyzw_quaternion(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=float).reshape(4)
    norm = np.linalg.norm(quat)
    if norm < 1e-8 or not np.isfinite(norm):
        raise ValueError(f"Invalid XR quaternion: {quat}")
    return quat / norm


def _quaternion_angular_distance_rad(q_a: np.ndarray, q_b: np.ndarray) -> float:
    q_a = _normalize_xyzw_quaternion(q_a)
    q_b = _normalize_xyzw_quaternion(q_b)
    dot = abs(float(np.clip(np.dot(q_a, q_b), -1.0, 1.0)))
    return 2.0 * float(np.arccos(dot))


def _slerp_xyzw_quaternion(q_a: np.ndarray, q_b: np.ndarray, alpha: float) -> np.ndarray:
    q_a = _normalize_xyzw_quaternion(q_a)
    q_b = _normalize_xyzw_quaternion(q_b)
    dot = float(np.dot(q_a, q_b))
    if dot < 0.0:
        q_b = -q_b
        dot = -dot

    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        return _normalize_xyzw_quaternion(q_a + alpha * (q_b - q_a))

    theta_0 = float(np.arccos(dot))
    theta = theta_0 * alpha
    sin_theta = float(np.sin(theta))
    sin_theta_0 = float(np.sin(theta_0))
    s0 = float(np.cos(theta)) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return _normalize_xyzw_quaternion(s0 * q_a + s1 * q_b)


def _normalize_wxyz_quaternion(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=float).reshape(4)
    norm = np.linalg.norm(quat)
    if norm < 1e-8 or not np.isfinite(norm):
        raise ValueError(f"Invalid quaternion: {quat}")
    return quat / norm


def _rotation_matrix_to_wxyz_quaternion(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=float).reshape(3, 3)
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = float(np.sqrt(trace + 1.0)) * 2.0
        w = 0.25 * s
        x = (rotation[2, 1] - rotation[1, 2]) / s
        y = (rotation[0, 2] - rotation[2, 0]) / s
        z = (rotation[1, 0] - rotation[0, 1]) / s
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = float(np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])) * 2.0
        w = (rotation[2, 1] - rotation[1, 2]) / s
        x = 0.25 * s
        y = (rotation[0, 1] + rotation[1, 0]) / s
        z = (rotation[0, 2] + rotation[2, 0]) / s
    elif rotation[1, 1] > rotation[2, 2]:
        s = float(np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])) * 2.0
        w = (rotation[0, 2] - rotation[2, 0]) / s
        x = (rotation[0, 1] + rotation[1, 0]) / s
        y = 0.25 * s
        z = (rotation[1, 2] + rotation[2, 1]) / s
    else:
        s = float(np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])) * 2.0
        w = (rotation[1, 0] - rotation[0, 1]) / s
        x = (rotation[0, 2] + rotation[2, 0]) / s
        y = (rotation[1, 2] + rotation[2, 1]) / s
        z = 0.25 * s
    return _normalize_wxyz_quaternion(np.array([w, x, y, z], dtype=float))


def _wxyz_quaternion_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = _normalize_wxyz_quaternion(quat)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _wxyz_quaternion_conjugate(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = _normalize_wxyz_quaternion(quat)
    return np.array([w, -x, -y, -z], dtype=float)


def _wxyz_quaternion_multiply(q_a: np.ndarray, q_b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = _normalize_wxyz_quaternion(q_a)
    w2, x2, y2, z2 = _normalize_wxyz_quaternion(q_b)
    return _normalize_wxyz_quaternion(
        np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=float,
        )
    )


def quaternion_to_angle_axis(quat: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    quat = _normalize_wxyz_quaternion(quat)
    if quat[0] < 0.0:
        quat = -quat

    w = float(quat[0])
    vec_part = quat[1:]
    angle = 2.0 * float(np.arccos(np.clip(w, -1.0, 1.0)))
    if angle < eps:
        return np.zeros(3, dtype=float)

    sin_half_angle = float(np.sin(angle / 2.0))
    if abs(sin_half_angle) < eps:
        return np.zeros(3, dtype=float)

    axis = vec_part / sin_half_angle
    return axis * angle


def quat_diff_as_angle_axis(q1: np.ndarray, q2: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    q1_inv = _wxyz_quaternion_conjugate(q1)
    delta_q = _wxyz_quaternion_multiply(q2, q1_inv)
    return quaternion_to_angle_axis(delta_q, eps)


def apply_delta_pose(
    source_pos: np.ndarray,
    source_rot: np.ndarray,
    delta_pos: np.ndarray,
    delta_rot: np.ndarray,
    eps: float = 1.0e-6,
) -> tuple[np.ndarray, np.ndarray]:
    source_pos = np.asarray(source_pos, dtype=float).reshape(3)
    source_rot = _normalize_wxyz_quaternion(source_rot)
    delta_pos = np.asarray(delta_pos, dtype=float).reshape(3)
    delta_rot = np.asarray(delta_rot, dtype=float).reshape(3)

    target_pos = source_pos + delta_pos
    angle = float(np.linalg.norm(delta_rot))
    if angle > eps:
        axis = delta_rot / angle
        rot_delta_quat = np.array(
            [np.cos(angle / 2.0), *(np.sin(angle / 2.0) * axis)],
            dtype=float,
        )
    else:
        rot_delta_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    target_rot = _wxyz_quaternion_multiply(rot_delta_quat, source_rot)
    return target_pos, target_rot


class XrPoseFilter:
    """Filter raw PICO pose arrays [x, y, z, qx, qy, qz, qw] before IK mapping."""

    def __init__(
        self,
        *,
        position_alpha: float,
        orientation_alpha: float,
        position_deadband_m: float,
        orientation_deadband_rad: float,
    ) -> None:
        if not 0.0 <= position_alpha <= 1.0:
            raise ValueError(f"`position_alpha` must be in [0, 1], got {position_alpha}.")
        if not 0.0 <= orientation_alpha <= 1.0:
            raise ValueError(f"`orientation_alpha` must be in [0, 1], got {orientation_alpha}.")
        if position_deadband_m < 0.0:
            raise ValueError(f"`position_deadband_m` must be non-negative, got {position_deadband_m}.")
        if orientation_deadband_rad < 0.0:
            raise ValueError(
                f"`orientation_deadband_rad` must be non-negative, got {orientation_deadband_rad}."
            )

        self.position_alpha = position_alpha
        self.orientation_alpha = orientation_alpha
        self.position_deadband_m = position_deadband_m
        self.orientation_deadband_rad = orientation_deadband_rad
        self._pose: np.ndarray | None = None

    def reset(self) -> None:
        self._pose = None

    def update(self, pose: np.ndarray) -> np.ndarray:
        pose = np.asarray(pose, dtype=float).reshape(-1)
        if pose.shape[0] != 7 or not np.all(np.isfinite(pose)):
            raise ValueError(f"Invalid XR pose: {pose}")

        pose = pose.copy()
        pose[3:] = _normalize_xyzw_quaternion(pose[3:])
        if self._pose is None:
            self._pose = pose
            return pose.copy()

        filtered = self._pose.copy()
        position_delta = pose[:3] - filtered[:3]
        position_distance = float(np.linalg.norm(position_delta))
        if position_distance > self.position_deadband_m:
            deadband_scale = (position_distance - self.position_deadband_m) / position_distance
            filtered[:3] = filtered[:3] + self.position_alpha * deadband_scale * position_delta

        orientation_delta = _quaternion_angular_distance_rad(filtered[3:], pose[3:])
        if orientation_delta > self.orientation_deadband_rad:
            effective_alpha = self.orientation_alpha * (
                orientation_delta - self.orientation_deadband_rad
            ) / orientation_delta
            filtered[3:] = _slerp_xyzw_quaternion(filtered[3:], pose[3:], effective_alpha)

        self._pose = filtered
        return filtered.copy()


class RelativeTeleopTarget:
    """Map relative XR controller motion to a robot end-effector target."""

    def __init__(self) -> None:
        self.ref_controller = None
        self.ref_end_effector = None

    def reset(self) -> None:
        self.ref_controller = None
        self.ref_end_effector = None

    def update(self, controller_pose, current_end_effector, scale: float, orientation_enabled: bool):
        if self.ref_controller is None or self.ref_end_effector is None:
            self.ref_controller = controller_pose.copy()
            self.ref_end_effector = current_end_effector.copy()
            return current_end_effector.copy()

        ref_controller_quat = _rotation_matrix_to_wxyz_quaternion(self.ref_controller.rotation)
        controller_quat = _rotation_matrix_to_wxyz_quaternion(controller_pose.rotation)
        ref_end_effector_quat = _rotation_matrix_to_wxyz_quaternion(self.ref_end_effector.rotation)

        delta_xyz = scale * (controller_pose.translation - self.ref_controller.translation)
        delta_rot = (
            quat_diff_as_angle_axis(ref_controller_quat, controller_quat)
            if orientation_enabled
            else np.zeros(3, dtype=float)
        )
        target_xyz, target_quat = apply_delta_pose(
            self.ref_end_effector.translation,
            ref_end_effector_quat,
            delta_xyz,
            delta_rot,
        )

        target = self.ref_end_effector.copy()
        target.translation = target_xyz
        target.rotation = _wxyz_quaternion_to_rotation_matrix(target_quat)
        return target


def xr_pose_to_world_se3(xr_pose: np.ndarray, pin):
    pose = np.asarray(xr_pose, dtype=float).reshape(-1)
    if pose.shape[0] != 7 or not np.all(np.isfinite(pose)):
        raise ValueError(f"Invalid XR pose: {xr_pose}")

    qx, qy, qz, qw = pose[3], pose[4], pose[5], pose[6]
    quat = pin.Quaternion(qw, qx, qy, qz)
    if quat.norm() < 1e-8:
        raise ValueError(f"Invalid XR quaternion: {xr_pose[3:7]}")
    quat.normalize()

    rotation = R_HEADSET_TO_WORLD @ quat.matrix() @ R_HEADSET_TO_WORLD.T
    translation = R_HEADSET_TO_WORLD @ pose[:3]
    return pin.SE3(rotation, translation)


def initial_configuration(model, pin) -> np.ndarray:
    q_ref = pin.neutral(model)
    for i, (lower, upper) in enumerate(zip(model.lowerPositionLimit, model.upperPositionLimit)):
        if np.isfinite(lower) and np.isfinite(upper):
            q_ref[i] = np.clip(q_ref[i], lower, upper)

    arm_initial_positions = {
        "AR5-5_07R-W4C4A2_joint": np.deg2rad(RIGHT_ARM_INITIAL_DEG),
        "AR5-5_07L-W4C4A2_joint": np.deg2rad(LEFT_ARM_INITIAL_DEG),
    }
    for joint_prefix, joint_values in arm_initial_positions.items():
        for i, joint_value in enumerate(joint_values, start=1):
            joint_id = model.getJointId(f"{joint_prefix}_{i}")
            if joint_id == 0:
                raise ValueError(f"Cannot find joint: {joint_prefix}_{i}")
            joint = model.joints[joint_id]
            q_ref[joint.idx_q] = joint_value
    return q_ref


def arm_joint_indices(model) -> tuple[np.ndarray, np.ndarray]:
    q_indices = []
    v_indices = []
    for side in ("L", "R"):
        prefix = f"AR5-5_07{side}-W4C4A2_joint"
        for i in range(1, 8):
            joint_id = model.getJointId(f"{prefix}_{i}")
            if joint_id == 0:
                raise ValueError(f"Cannot find joint: {prefix}_{i}")
            joint = model.joints[joint_id]
            q_indices.extend(range(joint.idx_q, joint.idx_q + joint.nq))
            v_indices.extend(range(joint.idx_v, joint.idx_v + joint.nv))
    return np.array(q_indices, dtype=int), np.array(v_indices, dtype=int)


def locked_joint_indices(model) -> tuple[np.ndarray, np.ndarray]:
    arm_q_indices, arm_v_indices = arm_joint_indices(model)
    q_mask = np.ones(model.nq, dtype=bool)
    v_mask = np.ones(model.nv, dtype=bool)
    q_mask[arm_q_indices] = False
    v_mask[arm_v_indices] = False
    return np.flatnonzero(q_mask), np.flatnonzero(v_mask)


def make_locked_joints_task_class(task_cls):
    class LockedJointsTask(task_cls):
        def __init__(
            self,
            q_indices: np.ndarray,
            v_indices: np.ndarray,
            target_q: np.ndarray,
            gain: float = 1.0,
            lm_damping: float = 0.0,
        ):
            super().__init__(cost=None, gain=gain, lm_damping=lm_damping)
            self.q_indices = q_indices
            self.v_indices = v_indices
            self.target_q = target_q.copy()

        def compute_error(self, configuration) -> np.ndarray:
            return configuration.q[self.q_indices] - self.target_q[self.q_indices]

        def compute_jacobian(self, configuration) -> np.ndarray:
            jacobian = np.zeros((len(self.v_indices), configuration.model.nv))
            jacobian[np.arange(len(self.v_indices)), self.v_indices] = 1.0
            return jacobian

        def __repr__(self):
            return (
                "LockedJointsTask("
                f"num_locked_joints={len(self.q_indices)}, "
                f"gain={self.gain}, "
                f"lm_damping={self.lm_damping})"
            )

    return LockedJointsTask


def are_adjacent_joints(model, joint_a: int, joint_b: int) -> bool:
    if joint_a == joint_b:
        return True
    if joint_a == 0 and joint_b != 0:
        return model.parents[joint_b] == 0
    if joint_b == 0 and joint_a != 0:
        return model.parents[joint_a] == 0
    return model.parents[joint_a] == joint_b or model.parents[joint_b] == joint_a


def add_collision_geometry(collision_model, name: str, parent_joint: int, geometry, placement, pin) -> None:
    geom = pin.GeometryObject(name, parent_joint, placement, geometry)
    collision_model.addGeometryObject(geom)


def add_box(collision_model, name: str, parent_joint: int, size, xyz, pin, fcl) -> None:
    add_collision_geometry(
        collision_model, name, parent_joint, fcl.Box(*size), pin.SE3(np.eye(3), np.array(xyz)), pin
    )


def add_sphere(collision_model, name: str, parent_joint: int, radius: float, xyz, pin, fcl) -> None:
    add_collision_geometry(
        collision_model,
        name,
        parent_joint,
        fcl.Sphere(radius),
        pin.SE3(np.eye(3), np.array(xyz)),
        pin,
    )


def add_capsule(
    collision_model,
    name: str,
    parent_joint: int,
    radius: float,
    half_length: float,
    xyz,
    pin,
    fcl,
) -> None:
    add_collision_geometry(
        collision_model,
        name,
        parent_joint,
        fcl.Capsule(radius, half_length),
        pin.SE3(np.eye(3), np.array(xyz)),
        pin,
    )


def build_primitive_collision_model(model, pin, fcl):
    collision_model = pin.GeometryModel()

    add_box(collision_model, "torso_box", 0, (0.20, 0.36, 0.42), (0.008, 0.0, -0.083), pin, fcl)
    add_box(
        collision_model,
        "pelvis_box",
        model.getJointId("hip_yaw"),
        (0.32, 0.16, 0.22),
        (-0.053, 0.0, -0.10),
        pin,
        fcl,
    )
    add_box(
        collision_model,
        "head_box",
        model.getJointId("neck_pitch"),
        (0.20, 0.20, 0.17),
        (0.02, 0.035, 0.05),
        pin,
        fcl,
    )

    for side, mount_y in (("L", 0.0725), ("R", -0.0725)):
        prefix = f"AR5-5_07{side}-W4C4A2"
        add_box(
            collision_model,
            f"{prefix}_base_box",
            0,
            (0.15, 0.15, 0.12),
            (0.0, mount_y, 0.0505),
            pin,
            fcl,
        )
        add_capsule(
            collision_model,
            f"{prefix}_link1_capsule",
            model.getJointId(f"{prefix}_joint_1"),
            0.065,
            0.075,
            (0.0, 0.0, 0.15),
            pin,
            fcl,
        )
        add_capsule(
            collision_model,
            f"{prefix}_link2_capsule",
            model.getJointId(f"{prefix}_joint_2"),
            0.060,
            0.13,
            (0.0, 0.0, 0.13),
            pin,
            fcl,
        )
        add_sphere(
            collision_model,
            f"{prefix}_link3_sphere",
            model.getJointId(f"{prefix}_joint_3"),
            0.075,
            (0.0, 0.0, -0.04),
            pin,
            fcl,
        )
        add_capsule(
            collision_model,
            f"{prefix}_link4_capsule",
            model.getJointId(f"{prefix}_joint_4"),
            0.055,
            0.10,
            (0.0, 0.0, 0.07),
            pin,
            fcl,
        )
        add_sphere(
            collision_model,
            f"{prefix}_link5_sphere",
            model.getJointId(f"{prefix}_joint_5"),
            0.065,
            (0.0, 0.0, -0.06),
            pin,
            fcl,
        )
        add_sphere(
            collision_model,
            f"{prefix}_link6_sphere",
            model.getJointId(f"{prefix}_joint_6"),
            0.060,
            (0.0, 0.0, 0.0),
            pin,
            fcl,
        )
        add_capsule(
            collision_model,
            f"{prefix}_link7_capsule",
            model.getJointId(f"{prefix}_joint_7"),
            0.050,
            0.145,
            (0.0, 0.0, 0.155),
            pin,
            fcl,
        )

    return collision_model


def configure_self_collision(robot, q_ref: np.ndarray, initial_ignore_distance: float, pin):
    collision_model = robot.collision_model
    collision_model.removeAllCollisionPairs()
    collision_model.addAllCollisionPairs()

    collision_data = pin.GeometryData(collision_model)
    pin.forwardKinematics(robot.model, robot.data, q_ref)
    pin.updateGeometryPlacements(robot.model, robot.data, collision_model, collision_data, q_ref)
    pin.computeDistances(robot.model, robot.data, collision_model, collision_data, q_ref)

    pairs_to_remove = []
    for k, pair in enumerate(collision_model.collisionPairs):
        geom_a = collision_model.geometryObjects[int(pair.first)]
        geom_b = collision_model.geometryObjects[int(pair.second)]
        initial_distance = collision_data.distanceResults[k].min_distance
        if are_adjacent_joints(robot.model, geom_a.parentJoint, geom_b.parentJoint):
            pairs_to_remove.append(pin.CollisionPair(pair.first, pair.second))
        elif initial_distance <= initial_ignore_distance:
            pairs_to_remove.append(pin.CollisionPair(pair.first, pair.second))

    for pair in pairs_to_remove:
        collision_model.removeCollisionPair(pair)

    collision_data = pin.GeometryData(collision_model)
    collision_data.enable_contact = True
    return collision_data


def select_solver(qpsolvers, requested: str | None) -> str:
    if requested:
        return requested
    if "daqp" in qpsolvers.available_solvers:
        return "daqp"
    if "osqp" in qpsolvers.available_solvers:
        return "osqp"
    if not qpsolvers.available_solvers:
        raise RuntimeError("No qpsolvers backend available. Install one, for example `daqp`.")
    return qpsolvers.available_solvers[0]


def package_dirs_for_urdf(urdf_path: Path) -> list[str]:
    return [str(urdf_path.parents[2])]


def resolve_package_uri(urdf_path: Path):
    try:
        import yourdfpy
    except ImportError as exc:
        raise ImportError("WheeledArmPico visualization requires `yourdfpy`.") from exc

    package_root = urdf_path.parents[1]
    package_prefix = f"package://{PACKAGE_NAME}/"

    def handler(fname: str) -> str:
        if fname.startswith(package_prefix):
            return str(package_root / fname[len(package_prefix) :])
        return yourdfpy.filename_handler_magic(fname, str(urdf_path.parent))

    return handler


def pinocchio_to_yourdfpy_cfg(model, q: np.ndarray) -> dict[str, float]:
    cfg = {}
    for joint_id, joint_name in enumerate(model.names):
        if joint_id == 0:
            continue
        joint = model.joints[joint_id]
        if joint.nq == 1:
            cfg[joint_name] = float(q[joint.idx_q])
    return cfg


def se3_to_position_wxyz(transform, pin) -> tuple[np.ndarray, np.ndarray]:
    quat = pin.Quaternion(transform.rotation)
    return transform.translation.copy(), np.array([quat.w, quat.x, quat.y, quat.z])


def format_joint_table(joint_angles_rad: np.ndarray) -> str:
    joint_angles_deg = np.rad2deg(joint_angles_rad)
    left = joint_angles_deg[:7]
    right = joint_angles_deg[7:]
    lines = [
        "### Arm Joint Angles",
        "",
        "| Joint | Left (deg) | Right (deg) |",
        "| --- | ---: | ---: |",
    ]
    for idx, (left_value, right_value) in enumerate(zip(left, right, strict=True), start=1):
        lines.append(f"| J{idx} | {left_value:7.2f} | {right_value:7.2f} |")
    return "\n".join(lines)


def format_teleop_status(
    enabled: bool,
    xr_ok: bool,
    left_active: bool,
    right_active: bool,
    scale: float,
    status: str,
    min_barrier: float | None,
) -> str:
    collision = "disabled" if min_barrier is None else f"{status} ({min_barrier:.4f})"
    return "\n".join(
        [
            "### PICO Teleop",
            "",
            f"- Enabled: `{enabled}`",
            f"- XR data: `{'ok' if xr_ok else 'waiting/error'}`",
            f"- Left grip active: `{left_active}`",
            f"- Right grip active: `{right_active}`",
            f"- Scale: `{scale:.2f}`",
            f"- Collision: `{collision}`",
            "",
            "Hold left/right grip to drive each arm. Press reset button to re-align.",
        ]
    )


def arm_action_from_q(q: np.ndarray, arm_q_indices: np.ndarray, joint_names: list[str]) -> dict[str, float]:
    arm_q = q[arm_q_indices]
    return {f"{name}.pos": float(value) for name, value in zip(joint_names, arm_q, strict=True)}


def arm_q_from_feedback(feedback: dict[str, Any], joint_names: list[str]) -> np.ndarray | None:
    values = []
    for name in joint_names:
        key = f"{name}.pos"
        if key not in feedback:
            return None
        values.append(float(feedback[key]))
    return np.asarray(values, dtype=float)


def smooth_joint_positions(
    target_q: np.ndarray,
    current_q: np.ndarray,
    previous_step: np.ndarray | None,
    dt: float,
    *,
    alpha: float,
    max_velocity_rad_s: float | None = None,
    max_acceleration_rad_s2: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth joint targets with EMA plus optional velocity and acceleration limits."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"`alpha` must be in [0, 1], got {alpha}.")
    if dt <= 0.0:
        raise ValueError(f"`dt` must be positive, got {dt}.")

    target_q = np.asarray(target_q, dtype=float)
    current_q = np.asarray(current_q, dtype=float)
    if target_q.shape != current_q.shape:
        raise ValueError(f"`target_q` and `current_q` shapes differ: {target_q.shape} vs {current_q.shape}.")
    if max_velocity_rad_s is not None and max_velocity_rad_s < 0.0:
        raise ValueError(f"`max_velocity_rad_s` must be non-negative, got {max_velocity_rad_s}.")
    if max_acceleration_rad_s2 is not None and max_acceleration_rad_s2 < 0.0:
        raise ValueError(
            f"`max_acceleration_rad_s2` must be non-negative, got {max_acceleration_rad_s2}."
        )

    step = target_q - current_q
    if max_velocity_rad_s is not None:
        max_step = float(max_velocity_rad_s) * dt
        step = np.clip(step, -max_step, max_step)

    step = alpha * step

    if max_acceleration_rad_s2 is not None:
        previous_step = (
            np.zeros_like(step)
            if previous_step is None
            else np.asarray(previous_step, dtype=float)
        )
        if previous_step.shape != step.shape:
            raise ValueError(
                f"`previous_step` and target step shapes differ: {previous_step.shape} vs {step.shape}."
            )
        max_step_delta = float(max_acceleration_rad_s2) * dt * dt
        step = previous_step + np.clip(step - previous_step, -max_step_delta, max_step_delta)

    return current_q + step, step
