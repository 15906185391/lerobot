#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
#
# /// script
# dependencies = ["daqp", "loop-rate-limiters", "pin-pink", "qpsolvers",
# "viser", "yourdfpy"]
# ///

"""Wheel-arm robot self-collision avoidance example using G01.urdf."""

from pathlib import Path
import argparse
import sys
import time
import webbrowser

import numpy as np
import hppfcl as fcl
import pinocchio as pin
import qpsolvers
from loop_rate_limiters import RateLimiter

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pink
from pink import solve_ik
from pink.barriers import SelfCollisionBarrier
from pink.tasks import Task
from pink.tasks import FrameTask, PostureTask

import viser
from viser.extras import ViserUrdf
import yourdfpy


LEFT_TCP = "L_tcp"
RIGHT_TCP = "R_tcp"
LEFT_WRIST_FRAME = "L_link7"
RIGHT_WRIST_FRAME = "R_link7"
TCP_OFFSET = np.array([0.0, 0.0, 0.16])
PACKAGE_NAME = "wheeled_arm_sim"
URDF_FILENAME = "G01.urdf"
RIGHT_ARM_INITIAL_DEG = [-20.0, 70.0, 75.0, 100.0, 25.0, 0.0, 0.0]
LEFT_ARM_INITIAL_DEG = [20.0, 70.0, -75.0, 100.0, -25.0, 0.0, 0.0]
TORSO_JOINT_NAMES = ("hip_roll", "hip_yaw")
WAIST_JOINT_NAMES = ("hip_pitch", "hip_roll", "hip_yaw")
SOLVE_FREQUENCY_HZ = 30.0
JOINT_PLOT_WINDOW_S = 10.0
JOINT_PLOT_UPDATE_HZ = 10.0
JOINT_PLOT_ASPECT = 2.2
JOINT_PRIMARY_PLOT_HEIGHT = 360
JOINT_WRIST_PLOT_HEIGHT = 320
JOINT_COLORS = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#17becf",
)


def resolve_package_uri(urdf_path: Path):
    """Return a yourdfpy filename handler for this repository layout."""
    package_root = urdf_path.parents[1]
    package_prefix = f"package://{PACKAGE_NAME}/"

    def handler(fname: str) -> str:
        if fname.startswith(package_prefix):
            return str(package_root / fname[len(package_prefix) :])
        return yourdfpy.filename_handler_magic(fname, str(urdf_path.parent))

    return handler


def package_dirs_for_urdf(urdf_path: Path) -> list[str]:
    """Return mesh search roots needed by both package and relative URDF paths."""
    return [
        str(urdf_path.parent),
        str(urdf_path.parents[1]),
        str(urdf_path.parents[2]),
    ]


def ensure_tcp_frames(model: pin.Model) -> None:
    """Add operational frames at the G01 wrist tips without modifying the URDF."""
    for tcp_name, wrist_frame_name in (
        (LEFT_TCP, LEFT_WRIST_FRAME),
        (RIGHT_TCP, RIGHT_WRIST_FRAME),
    ):
        if model.getFrameId(tcp_name) != model.nframes:
            continue
        wrist_frame_id = require_frame(model, wrist_frame_name)
        wrist_frame = model.frames[wrist_frame_id]
        tcp_placement = wrist_frame.placement * pin.SE3(np.eye(3), TCP_OFFSET)
        model.addFrame(
            pin.Frame(
                tcp_name,
                wrist_frame.parentJoint,
                wrist_frame_id,
                tcp_placement,
                pin.FrameType.OP_FRAME,
            )
        )


def initial_configuration(model: pin.Model) -> np.ndarray:
    """Build a neutral configuration clipped inside finite joint limits."""
    q_ref = pin.neutral(model)
    for i, (lower, upper) in enumerate(
        zip(model.lowerPositionLimit, model.upperPositionLimit)
    ):
        if np.isfinite(lower) and np.isfinite(upper):
            q_ref[i] = np.clip(q_ref[i], lower, upper)

    arm_initial_positions = {
        "R_joint": np.deg2rad(RIGHT_ARM_INITIAL_DEG),
        "L_joint": np.deg2rad(LEFT_ARM_INITIAL_DEG),
    }
    for joint_prefix, joint_values in arm_initial_positions.items():
        for i, joint_value in enumerate(joint_values, start=1):
            joint_name = f"{joint_prefix}{i}"
            joint_id = model.getJointId(joint_name)
            if joint_id == 0:
                raise ValueError(f"Cannot find joint: {joint_name}")
            joint = model.joints[joint_id]
            q_ref[joint.idx_q] = joint_value
    return q_ref


def joint_indices(
    model: pin.Model, joint_names: list[str] | tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Return q/v indices for one-DoF joints, preserving the requested order."""
    q_indices = []
    v_indices = []
    for joint_name in joint_names:
        joint_id = model.getJointId(joint_name)
        if joint_id == 0:
            raise ValueError(f"Cannot find joint: {joint_name}")
        joint = model.joints[joint_id]
        q_indices.extend(range(joint.idx_q, joint.idx_q + joint.nq))
        v_indices.extend(range(joint.idx_v, joint.idx_v + joint.nv))
    return np.array(q_indices, dtype=int), np.array(v_indices, dtype=int)


def arm_joint_names() -> list[str]:
    return [f"{side}_joint{i}" for side in ("L", "R") for i in range(1, 8)]


def arm_joint_indices(model: pin.Model) -> tuple[np.ndarray, np.ndarray]:
    return joint_indices(model, arm_joint_names())


def torso_joint_indices(model: pin.Model) -> tuple[np.ndarray, np.ndarray]:
    return joint_indices(model, TORSO_JOINT_NAMES)


def waist_joint_indices(model: pin.Model) -> tuple[np.ndarray, np.ndarray]:
    return joint_indices(model, WAIST_JOINT_NAMES)


def controlled_joint_indices(model: pin.Model) -> tuple[np.ndarray, np.ndarray]:
    """Default IK control set: G01 hip roll/yaw plus both arms."""
    return joint_indices(model, [*TORSO_JOINT_NAMES, *arm_joint_names()])


def format_joint_table(joint_angles_deg: np.ndarray) -> str:
    left = joint_angles_deg[:7]
    right = joint_angles_deg[7:]
    lines = [
        "### Arm Joint Angles",
        "",
        "| Joint | Left (deg) | Right (deg) |",
        "| --- | ---: | ---: |",
    ]
    for i, (left_value, right_value) in enumerate(zip(left, right), start=1):
        lines.append(f"| J{i} | {left_value:7.2f} | {right_value:7.2f} |")
    return "\n".join(lines)


def format_tcp_pose_table(left_transform: pin.SE3, right_transform: pin.SE3) -> str:
    left_position, left_wxyz = se3_to_viser_pose(left_transform)
    right_position, right_wxyz = se3_to_viser_pose(right_transform)
    lines = [
        "### End-Effector Poses",
        "",
        "| Value | Left TCP | Right TCP |",
        "| --- | ---: | ---: |",
    ]
    for axis, left_value, right_value in zip(
        ("x (m)", "y (m)", "z (m)"), left_position, right_position
    ):
        lines.append(f"| {axis} | {left_value: .4f} | {right_value: .4f} |")
    for axis, left_value, right_value in zip(
        ("qw", "qx", "qy", "qz"), left_wxyz, right_wxyz
    ):
        lines.append(f"| {axis} | {left_value: .4f} | {right_value: .4f} |")
    return "\n".join(lines)


def format_robot_joint_table(model: pin.Model, q: np.ndarray) -> str:
    lines = [
        "### Robot Joint Positions",
        "",
        "| Joint | rad | deg |",
        "| --- | ---: | ---: |",
    ]
    for joint_id, joint_name in enumerate(model.names):
        if joint_id == 0:
            continue
        joint = model.joints[joint_id]
        if joint.nq != 1:
            continue
        value_rad = float(q[joint.idx_q])
        value_deg = float(np.rad2deg(value_rad))
        lines.append(f"| {joint_name} | {value_rad: .4f} | {value_deg: .2f} |")
    return "\n".join(lines)


def locked_joint_indices(model: pin.Model) -> tuple[np.ndarray, np.ndarray]:
    controlled_q_indices, controlled_v_indices = controlled_joint_indices(model)
    q_mask = np.ones(model.nq, dtype=bool)
    v_mask = np.ones(model.nv, dtype=bool)
    q_mask[controlled_q_indices] = False
    v_mask[controlled_v_indices] = False
    return np.flatnonzero(q_mask), np.flatnonzero(v_mask)


class LockedJointsTask(Task):
    """Hard equality task keeping a subset of one-DoF joints fixed."""

    def __init__(self, q_indices: np.ndarray, v_indices: np.ndarray, target_q: np.ndarray):
        super().__init__(cost=None, gain=1.0)
        self.q_indices = q_indices
        self.v_indices = v_indices
        self.target_q = target_q.copy()

    def compute_error(self, configuration: pink.Configuration) -> np.ndarray:
        return configuration.q[self.q_indices] - self.target_q[self.q_indices]

    def compute_jacobian(self, configuration: pink.Configuration) -> np.ndarray:
        jacobian = np.zeros((len(self.v_indices), configuration.model.nv))
        jacobian[np.arange(len(self.v_indices)), self.v_indices] = 1.0
        return jacobian

    def __repr__(self) -> str:
        return f"LockedJointsTask(n_joints={len(self.v_indices)})"


def are_adjacent_joints(model: pin.Model, joint_a: int, joint_b: int) -> bool:
    if joint_a == joint_b:
        return True
    if joint_a == 0 and joint_b != 0:
        return model.parents[joint_b] == 0
    if joint_b == 0 and joint_a != 0:
        return model.parents[joint_a] == 0
    return model.parents[joint_a] == joint_b or model.parents[joint_b] == joint_a


def add_collision_geometry(
    collision_model: pin.GeometryModel,
    name: str,
    parent_joint: int,
    geometry,
    placement: pin.SE3,
) -> None:
    geom = pin.GeometryObject(name, parent_joint, placement, geometry)
    collision_model.addGeometryObject(geom)


def add_geometry_on_frame(
    collision_model: pin.GeometryModel,
    model: pin.Model,
    name: str,
    frame_name: str,
    geometry,
    xyz: tuple[float, float, float],
    rotation: np.ndarray | None = None,
) -> None:
    frame = model.frames[require_frame(model, frame_name)]
    local_placement = pin.SE3(
        np.eye(3) if rotation is None else rotation, np.array(xyz)
    )
    add_collision_geometry(
        collision_model,
        name,
        frame.parentJoint,
        geometry,
        frame.placement * local_placement,
    )


def add_box_on_frame(
    collision_model: pin.GeometryModel,
    model: pin.Model,
    name: str,
    frame_name: str,
    size: tuple[float, float, float],
    xyz: tuple[float, float, float],
    rotation: np.ndarray | None = None,
) -> None:
    add_geometry_on_frame(
        collision_model, model, name, frame_name, fcl.Box(*size), xyz, rotation
    )


def add_sphere_on_frame(
    collision_model: pin.GeometryModel,
    model: pin.Model,
    name: str,
    frame_name: str,
    radius: float,
    xyz: tuple[float, float, float],
) -> None:
    add_geometry_on_frame(
        collision_model, model, name, frame_name, fcl.Sphere(radius), xyz
    )


def add_capsule_on_frame(
    collision_model: pin.GeometryModel,
    model: pin.Model,
    name: str,
    frame_name: str,
    radius: float,
    half_length: float,
    xyz: tuple[float, float, float],
    rotation: np.ndarray | None = None,
) -> None:
    add_geometry_on_frame(
        collision_model,
        model,
        name,
        frame_name,
        fcl.Capsule(radius, half_length),
        xyz,
        rotation,
    )


def build_primitive_collision_model(model: pin.Model) -> pin.GeometryModel:
    """Approximate G01 with low-cost primitives for self-collision."""
    collision_model = pin.GeometryModel()

    add_box_on_frame(
        collision_model,
        model,
        "ankle_pitch_box",
        "ankle_pitch_link",
        (0.22, 0.22, 0.16),
        (-0.06, -0.08, 0.00),
    )
    add_capsule_on_frame(
        collision_model,
        model,
        "knee_pitch_capsule",
        "knee_pitch_link",
        0.075,
        0.22,
        (0.0, -0.08, -0.18),
    )
    add_capsule_on_frame(
        collision_model,
        model,
        "thigh_capsule",
        "thigh",
        0.08,
        0.22,
        (0.0, 0.03, -0.20),
    )
    add_box_on_frame(
        collision_model,
        model,
        "pelvis_box",
        "pelvis",
        (0.24, 0.22, 0.20),
        (-0.02, 0.0, -0.08),
    )
    add_box_on_frame(
        collision_model,
        model,
        "torso_box",
        "torso",
        (0.30, 0.30, 0.24),
        (0.10, 0.0, 0.00),
    )
    add_box_on_frame(
        collision_model,
        model,
        "head_box",
        "head",
        (0.18, 0.18, 0.16),
        (0.03, 0.03, 0.06),
    )

    for side in ("L", "R"):
        add_sphere_on_frame(
            collision_model,
            model,
            f"{side}_base_sphere",
            f"{side}_base",
            0.085,
            (0.0, 0.0, 0.04),
        )
        add_capsule_on_frame(
            collision_model,
            model,
            f"{side}_link1_capsule",
            f"{side}_link1",
            0.060,
            0.095,
            (0.0, 0.0, 0.08),
        )
        add_capsule_on_frame(
            collision_model,
            model,
            f"{side}_link2_capsule",
            f"{side}_link2",
            0.055,
            0.120,
            (0.0, 0.0, 0.11),
        )
        add_capsule_on_frame(
            collision_model,
            model,
            f"{side}_link3_capsule",
            f"{side}_link3",
            0.052,
            0.105,
            (0.0, 0.0, 0.09),
        )
        add_capsule_on_frame(
            collision_model,
            model,
            f"{side}_link4_capsule",
            f"{side}_link4",
            0.050,
            0.095,
            (0.0, 0.0, 0.08),
        )
        add_sphere_on_frame(
            collision_model,
            model,
            f"{side}_link5_sphere",
            f"{side}_link5",
            0.055,
            (0.0, 0.0, 0.05),
        )
        add_sphere_on_frame(
            collision_model,
            model,
            f"{side}_link6_sphere",
            f"{side}_link6",
            0.052,
            (0.0, 0.0, 0.0),
        )
        add_capsule_on_frame(
            collision_model,
            model,
            f"{side}_link7_capsule",
            f"{side}_link7",
            0.045,
            0.120,
            (0.0, 0.0, 0.12),
        )

    return collision_model


def configure_self_collision(
    robot: pin.RobotWrapper,
    q_ref: np.ndarray,
    initial_ignore_distance: float,
) -> pin.GeometryData:
    """Add all collision pairs, then remove adjacent or initially touching pairs."""
    collision_model = robot.collision_model
    collision_model.removeAllCollisionPairs()
    collision_model.addAllCollisionPairs()

    collision_data = pin.GeometryData(collision_model)
    pin.forwardKinematics(robot.model, robot.data, q_ref)
    pin.updateGeometryPlacements(
        robot.model, robot.data, collision_model, collision_data, q_ref
    )
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


def require_frame(model: pin.Model, frame_name: str) -> int:
    frame_id = model.getFrameId(frame_name)
    if frame_id == model.nframes:
        raise ValueError(f"Cannot find frame: {frame_name}")
    return frame_id


def se3_to_viser_pose(transform: pin.SE3) -> tuple[np.ndarray, np.ndarray]:
    quat = pin.Quaternion(transform.rotation)
    return transform.translation.copy(), np.array([quat.w, quat.x, quat.y, quat.z])


def transform_controls_to_se3(control) -> pin.SE3:
    wxyz = np.asarray(control.wxyz)
    quat = pin.Quaternion(wxyz[0], wxyz[1], wxyz[2], wxyz[3])
    quat.normalize()
    return pin.SE3(quat.matrix(), np.asarray(control.position))


def pinocchio_to_yourdfpy_cfg(model: pin.Model, q: np.ndarray) -> dict[str, float]:
    cfg = {}
    for joint_id, joint_name in enumerate(model.names):
        if joint_id == 0:
            continue
        joint = model.joints[joint_id]
        if joint.nq == 1:
            cfg[joint_name] = float(q[joint.idx_q])
    return cfg


def select_solver() -> str:
    if "osqp" in qpsolvers.available_solvers:
        return "osqp"
    return qpsolvers.available_solvers[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=8081, type=int)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--d-min", default=0.03, type=float)
    parser.add_argument("--initial-ignore-distance", default=None, type=float)
    parser.add_argument(
        "--collision-geometry",
        choices=("primitive", "stl"),
        default="primitive",
        help="Geometry used by the self-collision barrier.",
    )
    parser.add_argument(
        "--unlock-non-arm",
        action="store_true",
        help="Allow all non-arm joints to move during IK.",
    )
    args = parser.parse_args()
    if args.initial_ignore_distance is None:
        args.initial_ignore_distance = args.d_min

    urdf_path = (
        Path(__file__).parent.parent
        / "robots"
        / PACKAGE_NAME
        / "urdf"
        / URDF_FILENAME
    ).resolve()
    package_dirs = package_dirs_for_urdf(urdf_path)

    urdf = yourdfpy.URDF.load(
        str(urdf_path),
        build_collision_scene_graph=True,
        load_meshes=True,
        filename_handler=resolve_package_uri(urdf_path),
    )
    robot = pin.RobotWrapper.BuildFromURDF(
        filename=str(urdf_path),
        package_dirs=package_dirs,
        root_joint=None,
    )
    ensure_tcp_frames(robot.model)
    robot.data = robot.model.createData()
    if args.collision_geometry == "primitive":
        robot.collision_model = build_primitive_collision_model(robot.model)

    q_ref = initial_configuration(robot.model)
    locked_q_indices, locked_v_indices = locked_joint_indices(robot.model)
    left_frame_id = require_frame(robot.model, LEFT_TCP)
    right_frame_id = require_frame(robot.model, RIGHT_TCP)

    robot.collision_data = configure_self_collision(
        robot, q_ref, args.initial_ignore_distance
    )
    arm_q_indices, _ = arm_joint_indices(robot.model)
    joint_names = arm_joint_names()
    print(f"URDF: {urdf_path}")
    print(f"nq={robot.model.nq}, collision pairs={len(robot.collision_model.collisionPairs)}")
    if args.unlock_non_arm:
        constraints = []
        print("Controlled joints: full model")
    else:
        constraints = [LockedJointsTask(locked_q_indices, locked_v_indices, q_ref)]
        print(
            "Controlled joints: hip_roll/hip_yaw + left/right arms "
            f"({robot.model.nv - len(locked_v_indices)} moving, "
            f"{len(locked_v_indices)} locked)"
        )

    configuration = pink.Configuration(
        robot.model,
        robot.data,
        q_ref,
        collision_model=robot.collision_model,
        collision_data=robot.collision_data,
    )

    left_task = FrameTask(
        LEFT_TCP,
        position_cost=5.0,
        orientation_cost=1.0,
        gain=0.5,
    )
    right_task = FrameTask(
        RIGHT_TCP,
        position_cost=5.0,
        orientation_cost=1.0,
        gain=0.5,
    )
    posture_task = PostureTask(cost=1e-4)
    tasks = [left_task, right_task, posture_task]

    for task in tasks:
        task.set_target_from_configuration(configuration)

    collision_barrier = SelfCollisionBarrier(
        n_collision_pairs=len(robot.collision_model.collisionPairs),
        gain=10.0,
        safe_displacement_gain=5.0,
        d_min=args.d_min,
    )
    barriers = [collision_barrier]

    solver = select_solver()
    rate = RateLimiter(frequency=SOLVE_FREQUENCY_HZ, warn=False)
    dt = rate.period

    server = viser.ViserServer(host=args.host, port=args.port)
    server.gui.configure_theme(control_layout="fixed", control_width="large")
    if not args.no_browser:
        time.sleep(0.5)
        webbrowser.open(f"http://localhost:{args.port}")

    server.scene.add_grid("/ground", width=2, height=2)
    server.gui.add_number(
        "Solve frequency (Hz)",
        SOLVE_FREQUENCY_HZ,
        disabled=True,
        hint="Fixed IK loop frequency.",
    )
    ik_time_gui = server.gui.add_number(
        "IK time (ms)",
        0.0,
        disabled=True,
        hint="Time spent inside solve_ik only.",
    )
    max_plot_samples = int(JOINT_PLOT_WINDOW_S * SOLVE_FREQUENCY_HZ)
    joint_time_history = np.full(max_plot_samples, np.nan)
    joint_position_history = np.full((len(joint_names), max_plot_samples), np.nan)
    joint_time_axis = np.linspace(-JOINT_PLOT_WINDOW_S, 0.0, max_plot_samples)
    joint_angles_deg = np.rad2deg(configuration.q[arm_q_indices])
    left_tcp_pose = configuration.get_transform_frame_to_world(LEFT_TCP)
    right_tcp_pose = configuration.get_transform_frame_to_world(RIGHT_TCP)
    tcp_pose_gui = server.gui.add_markdown(
        format_tcp_pose_table(left_tcp_pose, right_tcp_pose)
    )
    robot_joint_values_gui = server.gui.add_markdown(
        format_robot_joint_table(robot.model, configuration.q)
    )
    joint_values_gui = server.gui.add_markdown(format_joint_table(joint_angles_deg))
    joint_plot_axes = (
        {"label": "Time (s)"},
        {"label": "Joint angle (deg)"},
    )
    joint_plot_scales = {"x": {"time": False}, "y": {"range": (-180.0, 180.0)}}
    left_primary_plot = server.gui.add_uplot(
        (joint_time_axis, *joint_position_history[0:4]),
        (
            {"label": "time"},
            *(
                {"label": f"L{i}", "stroke": JOINT_COLORS[i - 1], "width": 2.0}
                for i in range(1, 5)
            ),
        ),
        title="Left Arm J1-J4",
        axes=joint_plot_axes,
        scales=joint_plot_scales,
        aspect=JOINT_PLOT_ASPECT,
        height=JOINT_PRIMARY_PLOT_HEIGHT,
    )
    left_wrist_plot = server.gui.add_uplot(
        (joint_time_axis, *joint_position_history[4:7]),
        (
            {"label": "time"},
            *(
                {"label": f"L{i}", "stroke": JOINT_COLORS[i - 1], "width": 2.0}
                for i in range(5, 8)
            ),
        ),
        title="Left Arm J5-J7",
        axes=joint_plot_axes,
        scales=joint_plot_scales,
        aspect=JOINT_PLOT_ASPECT,
        height=JOINT_WRIST_PLOT_HEIGHT,
    )
    right_primary_plot = server.gui.add_uplot(
        (joint_time_axis, *joint_position_history[7:11]),
        (
            {"label": "time"},
            *(
                {"label": f"R{i}", "stroke": JOINT_COLORS[i - 1], "width": 2.0}
                for i in range(1, 5)
            ),
        ),
        title="Right Arm J1-J4",
        axes=joint_plot_axes,
        scales=joint_plot_scales,
        aspect=JOINT_PLOT_ASPECT,
        height=JOINT_PRIMARY_PLOT_HEIGHT,
    )
    right_wrist_plot = server.gui.add_uplot(
        (joint_time_axis, *joint_position_history[11:14]),
        (
            {"label": "time"},
            *(
                {"label": f"R{i}", "stroke": JOINT_COLORS[i - 1], "width": 2.0}
                for i in range(5, 8)
            ),
        ),
        title="Right Arm J5-J7",
        axes=joint_plot_axes,
        scales=joint_plot_scales,
        aspect=JOINT_PLOT_ASPECT,
        height=JOINT_WRIST_PLOT_HEIGHT,
    )
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/G01")
    urdf_vis.update_cfg(pinocchio_to_yourdfpy_cfg(robot.model, configuration.q))

    left_pos, left_wxyz = se3_to_viser_pose(configuration.data.oMf[left_frame_id])
    right_pos, right_wxyz = se3_to_viser_pose(configuration.data.oMf[right_frame_id])
    left_target = server.scene.add_transform_controls(
        "/ik_target_l", scale=0.18, position=left_pos, wxyz=left_wxyz
    )
    right_target = server.scene.add_transform_controls(
        "/ik_target_r", scale=0.18, position=right_pos, wxyz=right_wxyz
    )
    current_left_tcp_frame = server.scene.add_frame(
        "/current_tcp_l", show_axes=True, axes_length=0.12, axes_radius=0.006
    )
    current_right_tcp_frame = server.scene.add_frame(
        "/current_tcp_r", show_axes=True, axes_length=0.12, axes_radius=0.006
    )
    current_left_tcp_frame.position = left_pos
    current_left_tcp_frame.wxyz = left_wxyz
    current_right_tcp_frame.position = right_pos
    current_right_tcp_frame.wxyz = right_wxyz
    server.scene.add_icosphere("/collision_status", radius=0.04, color=(0, 255, 0))

    print(f"Open http://localhost:{args.port}")
    print(f"Left TCP:  position={left_pos}, wxyz={left_wxyz}")
    print(f"Right TCP: position={right_pos}, wxyz={right_wxyz}")

    last_status = None
    last_print_time = 0.0
    last_plot_update_time = 0.0
    t = 0.0

    try:
        while True:
            left_task.transform_target_to_world = transform_controls_to_se3(left_target)
            right_task.transform_target_to_world = transform_controls_to_se3(
                right_target
            )

            h = collision_barrier.compute_barrier(configuration)
            min_barrier = float(np.min(h))
            if min_barrier <= 0.0:
                status = "collision"
                color = (255, 0, 0)
            elif min_barrier < 0.01:
                status = "warning"
                color = (255, 255, 0)
            else:
                status = "safe"
                color = (0, 255, 0)
            now = time.monotonic()
            if status != last_status:
                server.scene.add_icosphere(
                    "/collision_status", radius=0.04, color=color
                )
            if status != last_status or now - last_print_time > 1.0:
                print(f"collision status={status}, min barrier={min_barrier:.4f}")
                last_status = status
                last_print_time = now

            ik_start = time.perf_counter()
            velocity = solve_ik(
                configuration,
                tasks,
                dt,
                solver=solver,
                barriers=barriers,
                constraints=constraints,
                safety_break=False,
            )
            ik_time_gui.value = round((time.perf_counter() - ik_start) * 1000.0, 3)
            configuration.integrate_inplace(velocity, dt)
            if not args.unlock_non_arm:
                q_locked = configuration.q.copy()
                q_locked[locked_q_indices] = q_ref[locked_q_indices]
                configuration.update(q_locked)
            urdf_vis.update_cfg(pinocchio_to_yourdfpy_cfg(robot.model, configuration.q))
            left_tcp_pose = configuration.get_transform_frame_to_world(LEFT_TCP)
            right_tcp_pose = configuration.get_transform_frame_to_world(RIGHT_TCP)
            left_tcp_position, left_tcp_wxyz = se3_to_viser_pose(left_tcp_pose)
            right_tcp_position, right_tcp_wxyz = se3_to_viser_pose(right_tcp_pose)
            current_left_tcp_frame.position = left_tcp_position
            current_left_tcp_frame.wxyz = left_tcp_wxyz
            current_right_tcp_frame.position = right_tcp_position
            current_right_tcp_frame.wxyz = right_tcp_wxyz

            joint_time_history[:-1] = joint_time_history[1:]
            joint_time_history[-1] = t
            joint_position_history[:, :-1] = joint_position_history[:, 1:]
            joint_angles_deg = np.rad2deg(configuration.q[arm_q_indices])
            joint_position_history[:, -1] = joint_angles_deg

            if t - last_plot_update_time >= 1.0 / JOINT_PLOT_UPDATE_HZ:
                valid_times = joint_time_history[np.isfinite(joint_time_history)]
                if valid_times.size > 0:
                    plot_time = joint_time_history - valid_times[-1]
                    tcp_pose_gui.content = format_tcp_pose_table(
                        left_tcp_pose, right_tcp_pose
                    )
                    robot_joint_values_gui.content = format_robot_joint_table(
                        robot.model, configuration.q
                    )
                    joint_values_gui.content = format_joint_table(joint_angles_deg)
                    left_primary_plot.data = (plot_time, *joint_position_history[0:4])
                    left_wrist_plot.data = (plot_time, *joint_position_history[4:7])
                    right_primary_plot.data = (plot_time, *joint_position_history[7:11])
                    right_wrist_plot.data = (plot_time, *joint_position_history[11:14])
                    last_plot_update_time = t

            rate.sleep()
            t += dt
    except KeyboardInterrupt:
        print("Stopped.")


if __name__ == "__main__":
    main()
