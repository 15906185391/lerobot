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

"""Interactive TOPPRA joint-space planner for the LeRobot wheeled arm."""

from __future__ import annotations

import argparse
import ctypes
import os
import sys
import sysconfig
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from lerobot.robots.wheeled_arm.config_wheeled_arm import (
    WHEELED_ARM_DEFAULT_LEFT_END_EFFECTOR,
    WHEELED_ARM_DEFAULT_RIGHT_END_EFFECTOR,
    WHEELED_ARM_END_EFFECTOR_TYPES,
)
from lerobot.teleoperators.wheeled_arm_pico.ik_utils import default_urdf_path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_NAME = "real_robot"
DEFAULT_VISER_HOST = "localhost"
DEFAULT_VISER_PORT = 8000
DEFAULT_LCM_URL = "udpm://239.255.76.67:8880?ttl=1"
DEFAULT_LCM_FEEDBACK_TIMEOUT_S = 1.0
DEFAULT_LCM_START_TIMEOUT_S = 5.0
DEFAULT_EXECUTE_COMMAND_HZ = 250.0
DEFAULT_EXECUTION_DURATION_S = 0.0
DEFAULT_COMMAND_PLOT_PATH = Path("/tmp/lerobot_toppra_joint_commands.png")
TOPPRA_MODE_CHOICES = ("Adaptive", "Hermite", "Cubic", "LinearBlend")
LCM_STATE_DIM = 23
ARM_STATE_DIM = 14


@dataclass(frozen=True)
class JointPlannerModelConfig:
    urdf_path: Path
    srdf_path: Path
    yaml_config_path: Path
    default_joint_group: str
    ee_names: list[str]
    base_link: str
    starting_joint_config: list[float]


def available_models() -> tuple[str, ...]:
    return (DEFAULT_MODEL_NAME,)


def _ensure_matplotlib_config_dir() -> None:
    if os.environ.get("MPLCONFIGDIR"):
        return
    default_config_dir = Path.home() / ".config" / "matplotlib"
    if default_config_dir.exists() and os.access(default_config_dir, os.W_OK):
        return
    cache_dir = Path(os.environ.get("TMPDIR", "/tmp")) / f"lerobot-matplotlib-{os.getuid()}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir)


def _add_cmeel_site_packages() -> None:
    purelib = Path(sysconfig.get_paths()["purelib"])
    cmeel_site = (
        purelib
        / "cmeel.prefix"
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    if cmeel_site.exists() and str(cmeel_site) not in sys.path:
        sys.path.insert(0, str(cmeel_site))


def _add_ros_python_paths() -> None:
    ros_root = Path("/opt/ros")
    if not ros_root.exists():
        return
    python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    for distro_dir in sorted(ros_root.iterdir()):
        for candidate in (
            distro_dir / "local" / "lib" / python_version / "dist-packages",
            distro_dir / "lib" / python_version / "dist-packages",
            distro_dir / "lib" / python_version / "site-packages",
        ):
            if candidate.exists() and str(candidate) not in sys.path:
                sys.path.append(str(candidate))


def _shared_library_roots() -> list[Path]:
    purelib = Path(sysconfig.get_paths()["purelib"])
    roots = [
        Path(sys.prefix) / "lib",
        purelib / "lib",
        purelib / "cmeel.prefix" / "lib",
    ]
    return [root for root in roots if root.exists()]


def _preload_shared_library(name: str) -> bool:
    for root in _shared_library_roots():
        candidate = root / name
        if not candidate.exists():
            continue
        ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
        return True
    return False


def _preload_runtime_libraries() -> None:
    for lib_name in ("libtinyxml2.so.11", "libtinyxml2.so.10", "libtinyxml2.so"):
        try:
            if _preload_shared_library(lib_name):
                return
        except OSError:
            continue


def _load_runtime_dependencies() -> SimpleNamespace:
    _ensure_matplotlib_config_dir()
    _add_cmeel_site_packages()
    _add_ros_python_paths()
    _preload_runtime_libraries()

    try:
        import matplotlib.pyplot as plt
        import pinocchio as pin
        import xacro
        from pinocchio.visualize import ViserVisualizer
        from roboplan.core import JointPath, Scene, collapseContinuousJointPositions
        from roboplan.example_models import get_package_models_dir, get_package_share_dir
        from roboplan.toppra import PathParameterizerTOPPRA, SplineFittingMode, TOPPRAOptions
        from roboplan.visualization import plotJointTrajectory, visualizeJointTrajectory
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "TOPPRA joint planning requires RoboPlan, Pinocchio, xacro, matplotlib, and Viser "
            "runtime dependencies. Run this command from the wheeled-arm RoboPlan environment "
            "or install the missing package shown in the original error."
        ) from exc

    return SimpleNamespace(
        JointPath=JointPath,
        PathParameterizerTOPPRA=PathParameterizerTOPPRA,
        Scene=Scene,
        SplineFittingMode=SplineFittingMode,
        TOPPRAOptions=TOPPRAOptions,
        ViserVisualizer=ViserVisualizer,
        collapseContinuousJointPositions=collapseContinuousJointPositions,
        get_package_models_dir=get_package_models_dir,
        get_package_share_dir=get_package_share_dir,
        pin=pin,
        plotJointTrajectory=plotJointTrajectory,
        plt=plt,
        visualizeJointTrajectory=visualizeJointTrajectory,
        xacro=xacro,
    )


def _patch_viser_visualizer_port_argument(deps: SimpleNamespace) -> None:
    try:
        import viser
    except (ImportError, ModuleNotFoundError):
        return

    def init_viewer(
        self,
        viewer=None,
        open=False,
        loadModel=False,  # noqa: N803 - matches Pinocchio's ViserVisualizer API.
        host=DEFAULT_VISER_HOST,
        port=str(DEFAULT_VISER_PORT),
    ):
        if (viewer is not None) and not isinstance(viewer, viser.ViserServer):
            raise RuntimeError("'viewer' argument must be None or a valid ViserServer instance.")

        self.viewer = viewer or viser.ViserServer(host=host, port=int(port))
        self.frames = {}

        if open:
            import webbrowser

            webbrowser.open(f"http://{self.viewer.get_host()}:{self.viewer.get_port()}")
            while len(self.viewer.get_clients()) == 0:
                time.sleep(0.1)

        if loadModel:
            self.loadViewerModel()

    deps.ViserVisualizer.initViewer = init_viewer


def _pump_matplotlib(deps: SimpleNamespace, delay: float = 0.001) -> None:
    deps.plt.pause(delay)


def _plot_joint_trajectory_from_main_thread(
    deps: SimpleNamespace,
    traj,
    scene,
    group_name: str,
) -> None:
    if threading.current_thread() is not threading.main_thread():
        print("Skipping Matplotlib trajectory plot because Viser callbacks run outside the main thread.")
        return

    fig = deps.plotJointTrajectory(
        traj,
        scene,
        group_name=group_name,
        title="TOPPRA Joint-Space Trajectory",
        positions=True,
        velocities=True,
        accelerations=True,
    )
    fig.canvas.draw()
    fig.canvas.flush_events()


def _make_static_and_preview_visualizers(
    deps: SimpleNamespace,
    pin_model,
    collision_model,
    visual_model,
    host: str,
    port: int,
):
    fixed_viz = deps.ViserVisualizer(pin_model, collision_model, visual_model, copy_models=True)
    fixed_viz.initViewer(open=False, loadModel=False, host=host, port=str(port))
    fixed_viz.loadViewerModel(rootNodeName="initial_robot")
    fixed_viz.displayCollisions(False)

    preview_viz = deps.ViserVisualizer(
        pin_model,
        collision_model=deps.pin.GeometryModel(),
        visual_model=collision_model,
        copy_models=False,
    )
    preview_viz.initViewer(
        viewer=fixed_viz.viewer,
        open=False,
        loadModel=False,
        host=host,
        port=str(port),
    )
    preview_viz.loadViewerModel(
        rootNodeName="preview_robot",
        visual_color=[0.0, 0.48, 1.0, 0.28],
    )
    preview_viz.displayCollisions(False)
    return fixed_viz, preview_viz


def _orthogonal_unit_vector(reference: np.ndarray) -> np.ndarray:
    if len(reference) == 1:
        return np.ones_like(reference)

    reference_norm = float(np.linalg.norm(reference))
    if reference_norm < 1e-12:
        raise ValueError("reference vector must be non-zero.")
    unit_reference = reference / reference_norm

    for idx in np.argsort(np.abs(unit_reference)):
        candidate = np.zeros_like(unit_reference)
        candidate[idx] = 1.0
        candidate -= unit_reference * float(candidate @ unit_reference)
        candidate_norm = float(np.linalg.norm(candidate))
        if candidate_norm > 1e-9:
            return candidate / candidate_norm

    raise RuntimeError("Failed to construct an orthogonal joint-space direction.")


def _make_joint_waypoints(
    deps: SimpleNamespace,
    scene,
    group_name: str,
    q_home_full: np.ndarray,
    waypoint_count: int,
    path_span: float,
    curvature_scale: float,
) -> tuple[object, np.ndarray]:
    if waypoint_count < 2:
        raise ValueError("waypoint_count must be at least 2.")

    group_info = scene.getJointGroupInfo(group_name)
    q_group_home = q_home_full[np.asarray(group_info.q_indices)]
    q_start = deps.collapseContinuousJointPositions(scene, group_name, q_group_home)
    q_lower, q_upper = scene.getPositionLimitVectors(group_name, collapsed=True)

    n_dof = len(q_start)
    if n_dof == 1:
        primary = np.ones(1)
        secondary = np.ones(1)
    else:
        primary = np.where(np.arange(n_dof) % 2 == 0, 1.0, -1.0)
        primary = primary / np.linalg.norm(primary)
        secondary = _orthogonal_unit_vector(primary)

    max_span = float(np.min(np.minimum(q_start - q_lower, q_upper - q_start)))
    if max_span <= 1e-9:
        raise RuntimeError("Home configuration is too close to the joint limits to build a demo path.")

    base_span = min(path_span, 0.8 * max_span)
    curve_span = base_span * curvature_scale

    for scale in (1.0, 0.75, 0.55, 0.35, 0.2):
        span = base_span * scale
        curve = curve_span * scale
        path = deps.JointPath()
        path.joint_names = list(group_info.joint_names)
        positions: list[np.ndarray] = []
        full_waypoints: list[np.ndarray] = []
        safe = True

        for idx in range(waypoint_count):
            s = float(idx) / float(waypoint_count - 1)
            q_group = q_start + s * span * primary + np.sin(np.pi * s) * curve * secondary
            q_group = np.clip(q_group, q_lower, q_upper)
            q_full = scene.toFullJointPositions(group_name, q_group)
            if scene.hasCollisions(q_full):
                safe = False
                break
            positions.append(q_group.copy())
            full_waypoints.append(q_full.copy())

        if safe and len(full_waypoints) == waypoint_count:
            path.positions = positions
            return path, np.array(full_waypoints)

    path = deps.JointPath()
    path.joint_names = list(group_info.joint_names)
    positions = []
    full_waypoints = []
    q_goal = np.clip(q_start + base_span * primary, q_lower, q_upper)
    for idx in range(waypoint_count):
        s = float(idx) / float(waypoint_count - 1)
        q_group = (1.0 - s) * q_start + s * q_goal
        q_full = scene.toFullJointPositions(group_name, q_group)
        positions.append(q_group.copy())
        full_waypoints.append(q_full.copy())
    path.positions = positions
    return path, np.array(full_waypoints)


def _make_goal_joint_waypoints(
    deps: SimpleNamespace,
    scene,
    group_name: str,
    q_home_full: np.ndarray,
    q_goal_group: np.ndarray,
    waypoint_count: int,
    curvature_scale: float,
) -> tuple[object, np.ndarray]:
    if waypoint_count < 2:
        raise ValueError("waypoint_count must be at least 2.")

    group_info = scene.getJointGroupInfo(group_name)
    q_start = deps.collapseContinuousJointPositions(
        scene, group_name, q_home_full[np.asarray(group_info.q_indices)]
    )
    q_goal = deps.collapseContinuousJointPositions(scene, group_name, q_goal_group)
    q_lower, q_upper = scene.getPositionLimitVectors(group_name, collapsed=True)

    delta = q_goal - q_start
    span = float(np.linalg.norm(delta))
    if span < 1e-9:
        primary = np.ones_like(q_start)
        primary = primary / float(np.linalg.norm(primary))
        delta = np.zeros_like(q_start)
    else:
        primary = delta / span

    secondary = _orthogonal_unit_vector(primary)

    for scale in (1.0, 0.75, 0.5, 0.35, 0.2):
        curve = span * curvature_scale * scale
        path = deps.JointPath()
        path.joint_names = list(group_info.joint_names)
        positions: list[np.ndarray] = []
        full_waypoints: list[np.ndarray] = []
        safe = True

        for idx in range(waypoint_count):
            s = float(idx) / float(waypoint_count - 1)
            q_group = q_start + s * delta + np.sin(np.pi * s) * curve * secondary
            q_group = np.clip(q_group, q_lower, q_upper)
            q_full = scene.toFullJointPositions(group_name, q_group)
            if scene.hasCollisions(q_full):
                safe = False
                break
            positions.append(q_group.copy())
            full_waypoints.append(q_full.copy())

        if safe and len(full_waypoints) == waypoint_count:
            path.positions = positions
            return path, np.array(full_waypoints)

    path = deps.JointPath()
    path.joint_names = list(group_info.joint_names)
    positions = []
    full_waypoints = []
    for idx in range(waypoint_count):
        s = float(idx) / float(waypoint_count - 1)
        q_group = (1.0 - s) * q_start + s * q_goal
        q_full = scene.toFullJointPositions(group_name, q_group)
        positions.append(q_group.copy())
        full_waypoints.append(q_full.copy())
    path.positions = positions
    return path, np.array(full_waypoints)


def _segment_samples_are_safe(scene, full_waypoints: np.ndarray, samples: int = 12) -> bool:
    for idx in range(len(full_waypoints) - 1):
        q_start = full_waypoints[idx]
        q_end = full_waypoints[idx + 1]
        for step in range(1, samples):
            fraction = float(step) / float(samples)
            q_interp = scene.interpolate(q_start, q_end, fraction)
            if scene.hasCollisions(q_interp):
                return False
    return True


def _trajectory_samples_are_safe(scene, group_name: str, positions: list[np.ndarray]) -> bool:
    for q in positions:
        q_full = scene.toFullJointPositions(group_name, q)
        if scene.hasCollisions(q_full):
            return False
    return True


def _real_robot_model_config(
    urdf_path: Path | None = None,
    srdf_path: Path | None = None,
    yaml_config_path: Path | None = None,
) -> JointPlannerModelConfig:
    resolved_urdf_path = (urdf_path or default_urdf_path()).expanduser().resolve()
    resolved_srdf_path = (srdf_path or resolved_urdf_path.with_suffix(".srdf")).expanduser().resolve()
    resolved_yaml_path = (
        yaml_config_path or resolved_urdf_path.with_name("real_robot_config.yaml")
    ).expanduser().resolve()

    return JointPlannerModelConfig(
        urdf_path=resolved_urdf_path,
        srdf_path=resolved_srdf_path,
        yaml_config_path=resolved_yaml_path,
        default_joint_group="dual_arm",
        ee_names=["AR5-5_07L-W4C4A2_tcp", "AR5-5_07R-W4C4A2_tcp"],
        base_link="torso",
        starting_joint_config=[
            0.0,
            -0.6,
            0.0,
            1.2,
            0.0,
            0.5,
            0.0,
            0.0,
            -0.6,
            0.0,
            1.2,
            0.0,
            0.5,
            0.0,
            0.0,
            0.0,
            0.0,
            -1.0,
            0.6,
            0.0,
            0.0,
        ],
    )


def get_model_data(
    model: str,
    urdf_path: Path | None = None,
    srdf_path: Path | None = None,
    yaml_config_path: Path | None = None,
) -> JointPlannerModelConfig:
    if model != DEFAULT_MODEL_NAME:
        raise ValueError(
            f"Unsupported model '{model}'. This LeRobot planner currently supports: "
            f"{', '.join(available_models())}."
        )
    model_data = _real_robot_model_config(urdf_path, srdf_path, yaml_config_path)
    missing = [
        (label, path)
        for label, path in (
            ("URDF", model_data.urdf_path),
            ("SRDF", model_data.srdf_path),
            ("RoboPlan YAML", model_data.yaml_config_path),
        )
        if not path.exists()
    ]
    if missing:
        lines = "\n".join(f"  {label}: {path}" for label, path in missing)
        raise FileNotFoundError(f"Missing model file(s):\n{lines}")
    return model_data


def get_package_paths(
    deps: SimpleNamespace,
    model_data: JointPlannerModelConfig,
    extra_package_paths: list[Path] | None = None,
) -> list[Path]:
    candidates: list[Path] = []
    if extra_package_paths:
        candidates.extend(path.expanduser().resolve() for path in extra_package_paths)
    if len(model_data.urdf_path.parents) >= 3:
        candidates.append(model_data.urdf_path.parents[2])
    candidates.extend(
        [
            PROJECT_ROOT / "roboplan_example_models" / "models",
            deps.get_package_models_dir(),
            deps.get_package_share_dir(),
        ]
    )

    paths: list[Path] = []
    for candidate in candidates:
        if candidate.exists() and candidate not in paths:
            paths.append(candidate)
    return paths


def get_home_configuration(scene, model_data: JointPlannerModelConfig) -> np.ndarray:
    q_full = scene.getCurrentJointPositions()
    q_home = np.array(model_data.starting_joint_config)

    if len(q_home) == len(q_full):
        return q_home.copy()

    print(
        f"Warning: starting_joint_config size ({len(q_home)}) does not match "
        f"model configuration size ({len(q_full)}). Using scene default instead."
    )
    return q_full.copy()


def _parse_joint_position_vector(value: str | None, expected_sizes: tuple[int, ...]) -> np.ndarray | None:
    if value is None or not value.strip():
        return None
    raw_items = value.replace(";", ",").replace("\n", ",").split(",")
    try:
        positions = np.array([float(item.strip()) for item in raw_items if item.strip()], dtype=float)
    except ValueError as exc:
        raise ValueError("指定初始关节位置必须是逗号分隔的数字，单位为 rad。") from exc
    if len(positions) not in expected_sizes:
        sizes = " 或 ".join(str(size) for size in expected_sizes)
        raise ValueError(f"指定初始关节位置需要 {sizes} 个数值，当前为 {len(positions)} 个。")
    return positions


def _scene_configuration_from_specified_position(
    scene,
    model_data: JointPlannerModelConfig,
    initial_joint_position: str | None,
) -> np.ndarray:
    q_full = get_home_configuration(scene, model_data)
    specified = _parse_joint_position_vector(initial_joint_position, (ARM_STATE_DIM, len(q_full)))
    if specified is None:
        return q_full
    if len(specified) == ARM_STATE_DIM:
        q_full[:ARM_STATE_DIM] = specified
        return q_full
    return specified.copy()


def _load_lcm_handler_class():
    try:
        from lerobot.robots.wheeled_arm.hardware_interface.lcm_handler import LCMHandler
    except ModuleNotFoundError as exc:
        if exc.name == "lcm":
            raise RuntimeError(
                "缺少 Python lcm 模块，无法从机器人状态读取关节初始位置。"
                "请在机器人运行环境中安装/加载 LCM Python 绑定。"
            ) from exc
        raise
    return LCMHandler


def _make_lcm_handler(args: argparse.Namespace):
    LCMHandler = _load_lcm_handler_class()
    return LCMHandler(
        lcm_url=args.lcm_url,
        left_end_effector=args.left_end_effector,
        right_end_effector=args.right_end_effector,
    )


def _has_fresh_arm_feedback(handler, max_age_s: float) -> bool:
    try:
        return bool(handler.has_arm_state_feedback(max_age_s))
    except TypeError:
        return bool(handler.has_arm_state_feedback())


def _read_lcm_joint_positions(handler) -> np.ndarray:
    with handler.joint_current_pos_lock:
        positions = np.asarray(handler.joint_current_pos, dtype=float).copy()
    if len(positions) < LCM_STATE_DIM:
        raise RuntimeError(
            f"LCM joint_current_pos should have at least {LCM_STATE_DIM} values, got {len(positions)}."
        )
    return positions


def _wait_for_lcm_arm_feedback(
    handler,
    *,
    feedback_timeout_s: float,
    start_timeout_s: float,
    lcm_url: str,
    reason: str,
) -> np.ndarray:
    deadline_s = time.monotonic() + max(start_timeout_s, 0.0)
    while time.monotonic() <= deadline_s:
        if _has_fresh_arm_feedback(handler, feedback_timeout_s):
            return _read_lcm_joint_positions(handler)
        time.sleep(0.05)
    raise RuntimeError(
        f"{reason} 需要新鲜左右臂 LCM 状态，但在 {start_timeout_s:.1f}s 内没有收到。"
        f"请检查机器人控制器、LCM URL ({lcm_url})、组播路由和状态发布程序。"
    )


def _scene_configuration_from_lcm_feedback(
    scene,
    model_data: JointPlannerModelConfig,
    lcm_positions: np.ndarray,
) -> np.ndarray:
    q_full = get_home_configuration(scene, model_data)
    if len(q_full) >= ARM_STATE_DIM:
        q_full[:ARM_STATE_DIM] = lcm_positions[:ARM_STATE_DIM]
    if len(q_full) >= 21:
        q_full[14:19] = lcm_positions[18:23]
        q_full[19:21] = lcm_positions[16:18]
    return q_full


def _set_lcm_arm_moving_flags(handler, moving: bool) -> None:
    for flag_name, value in (
        ("left_arm_moving", moving),
        ("right_arm_moving", moving),
        ("left_gripper_moving", False),
        ("right_gripper_moving", False),
        ("left_suction_moving", False),
        ("right_suction_moving", False),
        ("head_moving", False),
        ("waist_moving", False),
        ("leg_moving", False),
    ):
        if hasattr(handler, flag_name):
            setattr(handler, flag_name, value)


def _trajectory_time_vector(times_s: np.ndarray, count: int, fallback_dt_s: float) -> np.ndarray:
    if len(times_s) == count:
        source_times = np.asarray(times_s, dtype=float).copy()
    else:
        source_times = np.arange(count, dtype=float) * max(fallback_dt_s, 1e-3)
    source_times -= float(source_times[0]) if len(source_times) else 0.0
    return np.maximum.accumulate(source_times)


def _resample_trajectory_positions(
    positions: list[np.ndarray],
    times_s: np.ndarray,
    *,
    command_hz: float,
    fallback_dt_s: float,
    execution_duration_s: float = DEFAULT_EXECUTION_DURATION_S,
    start_idx: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    if not positions:
        return np.zeros(0, dtype=float), np.zeros((0, ARM_STATE_DIM), dtype=float)

    source_positions = np.asarray(positions, dtype=float)
    source_times = _trajectory_time_vector(times_s, len(source_positions), fallback_dt_s)
    start_idx = max(0, min(start_idx, len(source_positions) - 1))
    source_positions = source_positions[start_idx:]
    source_times = source_times[start_idx:] - float(source_times[start_idx])
    source_duration_s = float(source_times[-1]) if len(source_times) > 1 else 0.0
    duration_s = float(execution_duration_s) if execution_duration_s > 0.0 else source_duration_s
    period_s = 1.0 / max(command_hz, 1.0)

    if duration_s <= 0.0:
        return np.array([0.0], dtype=float), source_positions[:1].copy()

    command_times = np.arange(0.0, duration_s, period_s, dtype=float)
    if len(command_times) == 0 or not np.isclose(command_times[-1], duration_s):
        command_times = np.append(command_times, duration_s)
    if source_duration_s <= 0.0:
        command_positions = np.repeat(source_positions[:1], len(command_times), axis=0)
        return command_times, command_positions
    source_query_times = command_times / duration_s * source_duration_s
    command_positions = np.column_stack(
        [
            np.interp(source_query_times, source_times, source_positions[:, idx])
            for idx in range(source_positions.shape[1])
        ]
    )
    return command_times, command_positions


def _resolve_command_plot_path(path: Path) -> Path:
    resolved = path.expanduser()
    if resolved.suffix:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        return resolved
    resolved.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return resolved / f"toppra_joint_commands_{stamp}.png"


def _save_joint_command_plot(
    command_times_s: np.ndarray,
    command_positions: np.ndarray,
    joint_names: list[str],
    plot_path: Path,
) -> Path | None:
    if len(command_times_s) == 0 or len(command_positions) == 0:
        return None

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    output_path = _resolve_command_plot_path(plot_path)
    fig = Figure(figsize=(12, 8), dpi=140, constrained_layout=True)
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 1, sharex=True)
    split = min(7, command_positions.shape[1])
    for idx in range(split):
        axes[0].plot(command_times_s, command_positions[:, idx], label=joint_names[idx])
    for idx in range(split, command_positions.shape[1]):
        axes[1].plot(command_times_s, command_positions[:, idx], label=joint_names[idx])
    axes[0].set_title("Left arm joint command positions")
    axes[1].set_title("Right arm joint command positions")
    axes[1].set_xlabel("time (s)")
    for axis in axes:
        axis.set_ylabel("position (rad)")
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best", fontsize=7)
    fig.savefig(output_path)
    return output_path


def _execute_lcm_arm_trajectory(
    handler,
    scene,
    group_name: str,
    joint_names: list[str],
    positions: list[np.ndarray],
    times_s: np.ndarray,
    args: argparse.Namespace,
    *,
    start_idx: int = 0,
    display_step=None,
) -> bool:
    if args.preview_only:
        print("Preview-only mode: not sending LCM control commands.")
        return False
    if not args.connect_robot:
        print("Robot connection disabled: not sending LCM control commands.")
        return False
    if handler is None:
        print("Execution failed: no LCM handler is available.", file=sys.stderr)
        return False
    if not positions:
        print("Execution failed: trajectory has no samples.", file=sys.stderr)
        return False

    start_idx = max(0, min(start_idx, len(positions) - 1))
    command_times_s, command_positions = _resample_trajectory_positions(
        positions,
        times_s,
        command_hz=float(args.execute_command_hz),
        fallback_dt_s=float(args.dt),
        execution_duration_s=float(args.execution_duration_s),
        start_idx=start_idx,
    )
    plot_path = _save_joint_command_plot(
        command_times_s,
        command_positions,
        list(joint_names),
        args.command_plot_path,
    )
    if plot_path is not None:
        duration_s = float(command_times_s[-1]) if len(command_times_s) else 0.0
        print(
            f"Saved {float(args.execute_command_hz):g} Hz joint command plot "
            f"({duration_s:.3f} s): {plot_path}"
        )

    if not _has_fresh_arm_feedback(handler, args.lcm_feedback_timeout_s):
        print("Execution blocked: no fresh left/right arm LCM feedback.", file=sys.stderr)
        return False

    package = _read_lcm_joint_positions(handler).astype(np.float32, copy=True)
    start_time_s = time.perf_counter()
    display_period = max(1, int(float(args.execute_command_hz) / 30.0))
    _set_lcm_arm_moving_flags(handler, True)
    try:
        for idx, (command_time_s, q_group) in enumerate(zip(command_times_s, command_positions, strict=True)):
            if not _has_fresh_arm_feedback(handler, args.lcm_feedback_timeout_s):
                print("Execution stopped: left/right arm LCM feedback timed out.", file=sys.stderr)
                return False

            wait_s = start_time_s + float(command_time_s) - time.perf_counter()
            if wait_s > 0:
                time.sleep(wait_s)

            package = _read_lcm_joint_positions(handler).astype(np.float32, copy=True)
            q_full = scene.toFullJointPositions(group_name, q_group)
            package[:ARM_STATE_DIM] = np.asarray(q_full[:ARM_STATE_DIM], dtype=np.float32)
            handler.upper_body_data_publisher(package)
            if display_step is not None and (idx % display_period == 0 or idx == len(command_positions) - 1):
                display_step(q_group)
    finally:
        _set_lcm_arm_moving_flags(handler, False)
    return True


def _mode_from_name(deps: SimpleNamespace, name: str):
    try:
        return getattr(deps.SplineFittingMode, name)
    except AttributeError as exc:
        choices = ", ".join(TOPPRA_MODE_CHOICES)
        raise ValueError(f"Unsupported TOPPRA mode '{name}'. Choices: {choices}.") from exc


def run_joint_planner(args: argparse.Namespace) -> int:
    deps = _load_runtime_dependencies()
    _patch_viser_visualizer_port_argument(deps)

    try:
        model_data = get_model_data(args.model, args.urdf_path, args.srdf_path, args.yaml_config_path)
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 2

    toppra_mode = _mode_from_name(deps, args.toppra_mode)
    urdf_xml = deps.xacro.process_file(model_data.urdf_path).toxml()
    srdf_xml = deps.xacro.process_file(model_data.srdf_path).toxml()
    package_paths = get_package_paths(deps, model_data, args.package_path)

    scene = deps.Scene(
        "lerobot_toppra_joint_space_scene",
        urdf=urdf_xml,
        srdf=srdf_xml,
        package_paths=package_paths,
        yaml_config_path=model_data.yaml_config_path,
    )
    group_name = model_data.default_joint_group
    group_info = scene.getJointGroupInfo(group_name)
    lcm_handler = None
    try:
        if args.lcm_initial_state is not None:
            args.connect_robot = bool(args.lcm_initial_state)
        if args.connect_robot:
            lcm_handler = _make_lcm_handler(args)
            print(f"Waiting for current robot joint state from LCM: {args.lcm_url}")
            lcm_positions = _wait_for_lcm_arm_feedback(
                lcm_handler,
                feedback_timeout_s=args.lcm_feedback_timeout_s,
                start_timeout_s=args.lcm_start_timeout_s,
                lcm_url=args.lcm_url,
                reason="连接实物机器人时的关节规划初始姿态",
            )
            q_home_full = _scene_configuration_from_lcm_feedback(scene, model_data, lcm_positions)
            print("Robot connected: using current LCM left/right arm state as the TOPPRA start configuration.")
        else:
            q_home_full = _scene_configuration_from_specified_position(
                scene,
                model_data,
                args.initial_joint_position,
            )
            print("Robot connection disabled: using the specified/model start configuration.")
        scene.setJointPositions(q_home_full)
    except Exception as exc:
        if lcm_handler is not None:
            stop = getattr(lcm_handler, "stop", None)
            if callable(stop):
                stop()
        prefix = "LCM initialization failed" if args.connect_robot else "Initial joint position setup failed"
        print(f"{prefix}: {exc}", file=sys.stderr)
        return 3

    if not args.connect_robot and not args.preview_only:
        print("Robot connection disabled: this run will not send LCM control commands.")

    pin_model = deps.pin.buildModelFromXML(urdf_xml, mimic=True)
    collision_model = deps.pin.buildGeomFromUrdfString(
        pin_model,
        urdf_xml,
        deps.pin.GeometryType.COLLISION,
        package_dirs=package_paths,
    )
    visual_model = deps.pin.buildGeomFromUrdfString(
        pin_model,
        urdf_xml,
        deps.pin.GeometryType.VISUAL,
        package_dirs=package_paths,
    )

    fixed_viz, preview_viz = _make_static_and_preview_visualizers(
        deps,
        pin_model,
        collision_model,
        visual_model,
        args.host,
        args.port,
    )
    fixed_viz.display(q_home_full)
    preview_viz.display(q_home_full)
    time.sleep(0.1)

    try:
        if args.interactive_goal:
            return _run_interactive_goal_planner(
                deps,
                scene,
                model_data,
                group_name,
                group_info,
                q_home_full,
                fixed_viz,
                preview_viz,
                toppra_mode,
                args,
                lcm_handler,
            )

        return _run_generated_demo_path(
            deps,
            scene,
            model_data,
            group_name,
            q_home_full,
            fixed_viz,
            preview_viz,
            toppra_mode,
            args,
            lcm_handler,
        )
    finally:
        if lcm_handler is not None:
            stop = getattr(lcm_handler, "stop", None)
            if callable(stop):
                stop()


def _make_toppra_options(deps: SimpleNamespace, toppra_mode, args: argparse.Namespace):
    return deps.TOPPRAOptions(
        dt=args.dt,
        mode=toppra_mode,
        velocity_scale=args.velocity_scale,
        acceleration_scale=args.acceleration_scale,
        max_adaptive_iterations=args.max_adaptive_iterations,
        max_adaptive_step_size=args.max_adaptive_step_size,
        max_blend_deviation=args.max_blend_deviation,
    )


def _planned_execution_duration_s(traj_times_s: np.ndarray, args: argparse.Namespace) -> float:
    if args.execution_duration_s > 0.0:
        return float(args.execution_duration_s)
    if len(traj_times_s) == 0:
        return 0.0
    return float(traj_times_s[-1] - traj_times_s[0])


def _run_interactive_goal_planner(
    deps: SimpleNamespace,
    scene,
    model_data: JointPlannerModelConfig,
    group_name: str,
    group_info,
    q_home_full: np.ndarray,
    fixed_viz,
    preview_viz,
    toppra_mode,
    args: argparse.Namespace,
    lcm_handler,
) -> int:
    q_lower, q_upper = scene.getPositionLimitVectors(group_name, collapsed=True)
    q_goal = deps.collapseContinuousJointPositions(
        scene, group_name, q_home_full[np.asarray(group_info.q_indices)]
    )
    preview_guard = {"busy": False}
    status_text = fixed_viz.viewer.gui.add_text(
        "Status",
        "Adjust the joint sliders, then plan the path.",
        disabled=True,
    )
    plan_stats = fixed_viz.viewer.gui.add_text(
        "Plan stats",
        "Waiting for the first plan.",
        disabled=True,
    )

    sliders = []
    for idx, joint_name in enumerate(group_info.joint_names):
        limits = scene.getJointInfo(joint_name).limits
        lo = float(q_lower[idx]) if np.isfinite(q_lower[idx]) else float(limits.min_position)
        hi = float(q_upper[idx]) if np.isfinite(q_upper[idx]) else float(limits.max_position)
        if not np.isfinite(lo):
            lo = -np.pi
        if not np.isfinite(hi):
            hi = np.pi
        initial = float(np.clip(q_goal[idx], lo, hi))
        slider = fixed_viz.viewer.gui.add_slider(
            f"{joint_name}",
            min=lo,
            max=hi,
            step=max((hi - lo) / 400.0, 0.001),
            initial_value=initial,
        )
        sliders.append(slider)

    def current_goal_group() -> np.ndarray:
        return np.array([float(slider.value) for slider in sliders], dtype=float)

    def preview_goal(_=None):
        if preview_guard["busy"]:
            return
        q_group = current_goal_group()
        q_full = scene.toFullJointPositions(group_name, q_group)
        scene.setJointPositions(q_full)
        preview_viz.display(q_full)
        status_text.value = "目标已更新，透明预览机器人已更新。"

    for slider in sliders:
        slider.on_update(preview_goal)

    reset_button = fixed_viz.viewer.gui.add_button("Reset Goal")
    plan_button = fixed_viz.viewer.gui.add_button("Plan trajectory")
    animate_button = fixed_viz.viewer.gui.add_button("Animate once")
    execute_button = fixed_viz.viewer.gui.add_button("Execute on robot")
    animate_button.disabled = True
    execute_button.disabled = True
    last_traj: list[np.ndarray] | None = None
    last_times_s: np.ndarray | None = None

    @reset_button.on_click
    def reset_goal(_):
        nonlocal q_goal
        q_goal = deps.collapseContinuousJointPositions(
            scene, group_name, q_home_full[np.asarray(group_info.q_indices)]
        )
        for slider, value in zip(sliders, q_goal, strict=True):
            slider.value = float(value)
        preview_goal()
        status_text.value = "目标已重置到 home。"

    @plan_button.on_click
    def plan_path(_):
        nonlocal last_traj, last_times_s
        preview_guard["busy"] = True
        try:
            q_goal_group = current_goal_group()
            q_goal_full = scene.toFullJointPositions(group_name, q_goal_group)
            scene.setJointPositions(q_goal_full)
            preview_viz.display(q_goal_full)
            path, _ = _make_goal_joint_waypoints(
                deps,
                scene,
                group_name,
                q_home_full,
                q_goal_group,
                args.waypoint_count,
                args.curvature_scale,
            )
            toppra = deps.PathParameterizerTOPPRA(scene, group_name)
            options = _make_toppra_options(deps, toppra_mode, args)
            print(f"Planning {len(path.positions)} joint-space waypoints with TOPPRA...")
            plan_start = time.perf_counter()
            try:
                traj = toppra.generate(path, options)
            except Exception as exc:
                print(f"TOPPRA failed: {exc}")
                status_text.value = f"规划失败：{exc}"
                return
            plan_elapsed = time.perf_counter() - plan_start
            last_traj = list(traj.positions)
            last_times_s = np.asarray(traj.times, dtype=float)
            plan_stats.value = f"toppra {plan_elapsed * 1e3:.1f} ms"
            print(f"Plan stats: toppra {plan_elapsed * 1e3:.1f} ms")
            print(f"Trajectory duration: {traj.times[-1]:.3f} s")
            execution_duration_s = _planned_execution_duration_s(last_times_s, args)
            if args.execution_duration_s > 0.0:
                print(f"Requested execution duration: {execution_duration_s:.3f} s")

            fixed_viz.display(q_home_full)
            deps.visualizeJointTrajectory(
                fixed_viz,
                scene,
                traj,
                model_data.ee_names,
                (0, 140, 220),
                "/toppra_joint_space/trajectory",
            )
            _plot_joint_trajectory_from_main_thread(deps, traj, scene, group_name)
            status_text.value = (
                "预览模式下已生成轨迹。"
                if args.preview_only
                else (
                    f"规划完成，执行时长 {execution_duration_s:.3f}s。可先 Animate once 检查，再点击 Execute on robot。"
                    if args.connect_robot
                    else f"规划完成，执行时长 {execution_duration_s:.3f}s。未连接实物机器人，只能在 Viser 中动画预览。"
                )
            )
            animate_button.disabled = False
            execute_button.disabled = args.preview_only or not args.connect_robot
        finally:
            preview_guard["busy"] = False

    @animate_button.on_click
    def animate_once(_):
        if last_traj is None:
            return
        preview_guard["busy"] = True
        try:
            for q_group in last_traj:
                preview_viz.display(scene.toFullJointPositions(group_name, q_group))
                time.sleep(args.dt)
        finally:
            preview_guard["busy"] = False

    @execute_button.on_click
    def execute_once(_):
        if preview_guard["busy"] or last_traj is None or last_times_s is None:
            return
        preview_guard["busy"] = True
        execute_button.disabled = True
        animate_button.disabled = True
        plan_button.disabled = True
        reset_button.disabled = True
        status_text.value = "正在向实物机器人发送左右臂 LCM 关节命令。"
        print("Executing TOPPRA trajectory on robot via LCM...")
        try:
            ok = _execute_lcm_arm_trajectory(
                lcm_handler,
                scene,
                group_name,
                list(group_info.joint_names),
                last_traj,
                last_times_s,
                args,
                display_step=lambda q_group: preview_viz.display(
                    scene.toFullJointPositions(group_name, q_group)
                ),
            )
            status_text.value = "LCM 执行完成。" if ok else "LCM 执行未完成，请查看终端日志。"
            print("LCM execution complete." if ok else "LCM execution did not complete.")
        finally:
            preview_guard["busy"] = False
            animate_button.disabled = False
            plan_button.disabled = False
            reset_button.disabled = False
            execute_button.disabled = args.preview_only or not args.connect_robot

    preview_goal()
    print(f"Viser server is running at http://{args.host}:{args.port}")
    try:
        while True:
            time.sleep(10.0)
    except KeyboardInterrupt:
        return 0


def _run_generated_demo_path(
    deps: SimpleNamespace,
    scene,
    model_data: JointPlannerModelConfig,
    group_name: str,
    q_home_full: np.ndarray,
    fixed_viz,
    preview_viz,
    toppra_mode,
    args: argparse.Namespace,
    lcm_handler,
) -> int:
    path, full_waypoints = _make_joint_waypoints(
        deps,
        scene,
        group_name,
        q_home_full,
        args.waypoint_count,
        args.path_span,
        args.curvature_scale,
    )
    sparse_path_is_safe = _segment_samples_are_safe(scene, full_waypoints)
    if not sparse_path_is_safe:
        print("Warning: sampled path segments touch collision geometry.")

    toppra = deps.PathParameterizerTOPPRA(scene, group_name)
    options = _make_toppra_options(deps, toppra_mode, args)

    print(f"Planning {len(path.positions)} joint-space waypoints with TOPPRA...")
    plan_start = time.perf_counter()
    try:
        traj = toppra.generate(path, options)
    except Exception as exc:
        print(f"TOPPRA failed: {exc}")
        return 1
    plan_elapsed = time.perf_counter() - plan_start
    print(f"Plan stats: toppra {plan_elapsed * 1e3:.1f} ms")
    print(f"Trajectory duration: {traj.times[-1]:.3f} s")
    execution_duration_s = _planned_execution_duration_s(np.asarray(traj.times, dtype=float), args)
    if args.execution_duration_s > 0.0:
        print(f"Requested execution duration: {execution_duration_s:.3f} s")

    trajectory_is_safe = _trajectory_samples_are_safe(scene, group_name, traj.positions)
    if sparse_path_is_safe and trajectory_is_safe:
        print("Safety check: sampled sparse path and timed trajectory are collision-free.")
    else:
        print("Safety check: collision detected in sampled path or timed trajectory.")

    deps.visualizeJointTrajectory(
        fixed_viz,
        scene,
        traj,
        model_data.ee_names,
        (0, 140, 220),
        "/toppra_joint_space/trajectory",
    )

    deps.plt.figure()
    deps.plt.ion()
    _plot_joint_trajectory_from_main_thread(deps, traj, scene, group_name)
    deps.plt.show(block=False)
    _pump_matplotlib(deps)

    print("Use the Viser GUI controls to preview, scrub, reset, or execute the trajectory.")
    if args.preview_only:
        print("Preview-only mode: execution is disabled for this run.")
    elif not sparse_path_is_safe or not trajectory_is_safe:
        print("Execution disabled: adjust the plan until safety checks pass.")
    elif not args.connect_robot:
        print("Robot connection disabled: execution will not publish LCM commands.")

    trajectory_positions = list(traj.positions)
    trajectory_times_s = np.asarray(traj.times, dtype=float)
    safety_ok = sparse_path_is_safe and trajectory_is_safe
    preview_done = False
    pending_mode: str | None = None
    animating = False

    def display_step(target_step_idx: int, update_slider: bool = True) -> None:
        target_step_idx = max(0, min(target_step_idx, len(trajectory_positions) - 1))
        q_full = scene.toFullJointPositions(group_name, trajectory_positions[target_step_idx])
        scene.setJointPositions(q_full)
        preview_viz.display(q_full)
        if update_slider:
            step_slider.value = target_step_idx

    status_text = fixed_viz.viewer.gui.add_text(
        "Status",
        "Ready to preview." if safety_ok else "Collision detected; execution disabled.",
        disabled=True,
    )
    preview_button = fixed_viz.viewer.gui.add_button("Preview trajectory")
    execute_button = fixed_viz.viewer.gui.add_button("Execute trajectory")
    reset_button = fixed_viz.viewer.gui.add_button("Reset")
    step_slider = fixed_viz.viewer.gui.add_slider(
        "Trajectory step",
        min=0,
        max=len(trajectory_positions) - 1,
        step=1,
        initial_value=0,
    )
    execute_button.disabled = True

    @preview_button.on_click
    def preview_trajectory(_):
        nonlocal pending_mode
        if not animating:
            pending_mode = "preview"

    @execute_button.on_click
    def execute_trajectory(_):
        nonlocal pending_mode
        if animating or args.preview_only or not args.connect_robot or not safety_ok or not preview_done:
            return
        pending_mode = "execute"

    @reset_button.on_click
    def reset(_):
        if animating:
            return
        display_step(0)
        status_text.value = "Reset to trajectory start."

    @step_slider.on_update
    def update_step_from_slider(_):
        if animating:
            return
        display_step(int(step_slider.value), update_slider=False)
        status_text.value = f"Preview step {int(step_slider.value)} / {len(trajectory_positions) - 1}."

    display_step(0)
    print(f"Viser server is running at http://{args.host}:{args.port}")
    try:
        while True:
            if pending_mode is None:
                time.sleep(0.1)
                continue

            mode = pending_mode
            pending_mode = None
            animating = True
            preview_button.disabled = True
            execute_button.disabled = True
            reset_button.disabled = True
            step_slider.disabled = True

            if mode == "preview":
                status_text.value = "Previewing trajectory in Viser."
                print("Previewing trajectory in Viser...")
            else:
                status_text.value = "Executing approved trajectory."
                print("Executing approved trajectory...")

            start_idx = int(step_slider.value)
            if start_idx >= len(trajectory_positions) - 1:
                start_idx = 0
            for idx in range(start_idx, len(trajectory_positions)):
                if mode == "execute":
                    break
                display_step(idx)
                time.sleep(args.dt)
                _pump_matplotlib(deps)
            if mode == "execute":
                ok = _execute_lcm_arm_trajectory(
                    lcm_handler,
                    scene,
                    group_name,
                    list(scene.getJointGroupInfo(group_name).joint_names),
                    trajectory_positions,
                    trajectory_times_s,
                    args,
                    start_idx=start_idx,
                    display_step=lambda q_group: preview_viz.display(
                        scene.toFullJointPositions(group_name, q_group)
                    ),
                )
                if not ok:
                    status_text.value = "Execution failed or stopped; see terminal log."

            animating = False
            if mode == "preview":
                preview_done = True
                status_text.value = (
                    "Preview complete; execution is available."
                    if safety_ok and not args.preview_only and args.connect_robot
                    else "Preview complete."
                )
                print("Preview complete.")
            else:
                if status_text.value != "Execution failed or stopped; see terminal log.":
                    status_text.value = "Execution complete."
                    print("Execution complete.")

            preview_button.disabled = False
            reset_button.disabled = False
            step_slider.disabled = False
            execute_button.disabled = args.preview_only or not args.connect_robot or not safety_ok or not preview_done
    except KeyboardInterrupt:
        return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=available_models(), default=DEFAULT_MODEL_NAME)
    parser.add_argument("--urdf-path", type=Path, default=None)
    parser.add_argument("--srdf-path", type=Path, default=None)
    parser.add_argument("--yaml-config-path", type=Path, default=None)
    parser.add_argument("--package-path", type=Path, action="append", default=[])
    parser.add_argument("--toppra-mode", choices=TOPPRA_MODE_CHOICES, default="Adaptive")
    parser.add_argument("--waypoint-count", type=int, default=6)
    parser.add_argument("--path-span", type=float, default=0.45)
    parser.add_argument("--curvature-scale", type=float, default=0.25)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--velocity-scale", type=float, default=1.0)
    parser.add_argument("--acceleration-scale", type=float, default=1.0)
    parser.add_argument("--max-adaptive-iterations", type=int, default=10)
    parser.add_argument("--max-adaptive-step-size", type=float, default=0.05)
    parser.add_argument("--max-blend-deviation", type=float, default=0.01)
    parser.add_argument("--preview-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--interactive-goal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--connect-robot", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--initial-joint-position",
        default="",
        help=(
            "Comma-separated radian joint positions used when --no-connect-robot is set. "
            "Accepts 14 dual-arm joints or the full model configuration."
        ),
    )
    parser.add_argument("--lcm-url", default=DEFAULT_LCM_URL)
    parser.add_argument("--lcm-feedback-timeout-s", type=float, default=DEFAULT_LCM_FEEDBACK_TIMEOUT_S)
    parser.add_argument("--lcm-start-timeout-s", type=float, default=DEFAULT_LCM_START_TIMEOUT_S)
    parser.add_argument("--lcm-initial-state", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--execute-command-hz", type=float, default=DEFAULT_EXECUTE_COMMAND_HZ)
    parser.add_argument(
        "--execution-duration-s",
        type=float,
        default=DEFAULT_EXECUTION_DURATION_S,
        help="Total command execution duration in seconds. Use 0 to keep the TOPPRA duration.",
    )
    parser.add_argument("--command-plot-path", type=Path, default=DEFAULT_COMMAND_PLOT_PATH)
    parser.add_argument(
        "--left-end-effector",
        choices=tuple(WHEELED_ARM_END_EFFECTOR_TYPES),
        default=WHEELED_ARM_DEFAULT_LEFT_END_EFFECTOR,
    )
    parser.add_argument(
        "--right-end-effector",
        choices=tuple(WHEELED_ARM_END_EFFECTOR_TYPES),
        default=WHEELED_ARM_DEFAULT_RIGHT_END_EFFECTOR,
    )
    parser.add_argument("--host", default=DEFAULT_VISER_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_VISER_PORT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return run_joint_planner(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
