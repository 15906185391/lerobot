#!/usr/bin/env python3

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin
import tyro
import xacro

from common import get_home_configuration, get_model_data
from preview_visualization import make_static_and_preview_visualizers
from roboplan.core import JointPath, Scene, collapseContinuousJointPositions
from roboplan.example_models import get_package_models_dir, get_package_share_dir
from roboplan.toppra import PathParameterizerTOPPRA, SplineFittingMode, TOPPRAOptions
from roboplan.visualization import plotJointTrajectory, visualizeJointTrajectory


def _pump_matplotlib(delay: float = 0.001) -> None:
    plt.pause(delay)


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
    scene: Scene,
    group_name: str,
    q_home_full: np.ndarray,
    waypoint_count: int,
    path_span: float,
    curvature_scale: float,
) -> tuple[JointPath, np.ndarray]:
    if waypoint_count < 2:
        raise ValueError("waypoint_count must be at least 2.")

    group_info = scene.getJointGroupInfo(group_name)
    q_group_home = q_home_full[np.asarray(group_info.q_indices)]
    q_start = collapseContinuousJointPositions(scene, group_name, q_group_home)
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
        raise RuntimeError(
            "Home configuration is too close to the joint limits to build a demo path."
        )

    base_span = min(path_span, 0.8 * max_span)
    curve_span = base_span * curvature_scale

    for scale in (1.0, 0.75, 0.55, 0.35, 0.2):
        span = base_span * scale
        curve = curve_span * scale
        path = JointPath()
        path.joint_names = list(group_info.joint_names)
        positions: list[np.ndarray] = []

        safe = True
        full_waypoints: list[np.ndarray] = []
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

    path = JointPath()
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


def _segment_samples_are_safe(scene: Scene, full_waypoints: np.ndarray, samples: int = 12) -> bool:
    for idx in range(len(full_waypoints) - 1):
        q_start = full_waypoints[idx]
        q_end = full_waypoints[idx + 1]
        for step in range(1, samples):
            fraction = float(step) / float(samples)
            q_interp = scene.interpolate(q_start, q_end, fraction)
            if scene.hasCollisions(q_interp):
                return False
    return True


def _trajectory_samples_are_safe(
    scene: Scene,
    group_name: str,
    positions: list[np.ndarray],
) -> bool:
    for q in positions:
        q_full = scene.toFullJointPositions(group_name, q)
        if scene.hasCollisions(q_full):
            return False
    return True


def _make_goal_joint_waypoints(
    scene: Scene,
    group_name: str,
    q_home_full: np.ndarray,
    q_goal_group: np.ndarray,
    waypoint_count: int,
    curvature_scale: float,
) -> tuple[JointPath, np.ndarray]:
    group_info = scene.getJointGroupInfo(group_name)
    q_start = collapseContinuousJointPositions(
        scene, group_name, q_home_full[np.asarray(group_info.q_indices)]
    )
    q_goal = collapseContinuousJointPositions(scene, group_name, q_goal_group)
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
        path = JointPath()
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

    path = JointPath()
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


def main(
    model: str = "ur5",
    toppra_mode: SplineFittingMode = SplineFittingMode.Adaptive,
    waypoint_count: int = 6,
    path_span: float = 0.45,
    curvature_scale: float = 0.25,
    dt: float = 0.01,
    velocity_scale: float = 1.0,
    acceleration_scale: float = 1.0,
    max_adaptive_iterations: int = 10,
    max_adaptive_step_size: float = 0.05,
    max_blend_deviation: float = 0.01,
    preview_only: bool = False,
    interactive_goal: bool = True,
    host: str = "localhost",
    port: str = "8000",
):
    """
    Generate and replay a joint-space TOPPRA trajectory.

    Parameters:
        model: The name of the robot model to use.
        toppra_mode: The spline fitting mode used by TOPPRA.
        waypoint_count: Number of joint-space waypoints in the sparse path.
        path_span: Maximum joint-space excursion from the home pose.
        curvature_scale: Strength of the mid-path bend used to shape the sparse path.
        dt: Output trajectory sample period in seconds.
        velocity_scale: Scaling applied to the joint velocity limits.
        acceleration_scale: Scaling applied to the joint acceleration limits.
        max_adaptive_iterations: Maximum collision-check iterations for adaptive mode.
        max_adaptive_step_size: Maximum step size used to sample adaptive splines.
        max_blend_deviation: Maximum deviation for linear blend corner rounding.
        preview_only: If True, expose Viser preview controls but keep execution disabled.
        host: The host for the ViserVisualizer.
        port: The port for the ViserVisualizer.
    """
    model_data = get_model_data().get(model)
    if model_data is None:
        print(f"Invalid model requested: {model}")
        sys.exit(1)

    urdf_xml = xacro.process_file(model_data.urdf_path).toxml()
    srdf_xml = xacro.process_file(model_data.srdf_path).toxml()
    source_models_dir = (
        Path(__file__).resolve().parents[2] / "roboplan_example_models" / "models"
    )
    package_paths = [get_package_models_dir(), get_package_share_dir()]
    if source_models_dir.exists() and source_models_dir not in package_paths:
        package_paths.insert(0, source_models_dir)

    scene = Scene(
        "toppra_joint_space_scene",
        urdf=urdf_xml,
        srdf=srdf_xml,
        package_paths=package_paths,
        yaml_config_path=model_data.yaml_config_path,
    )
    group_name = model_data.default_joint_group
    group_info = scene.getJointGroupInfo(group_name)
    q_home_full = get_home_configuration(scene, model_data)
    scene.setJointPositions(q_home_full)

    model = pin.buildModelFromXML(urdf_xml, mimic=True)
    collision_model = pin.buildGeomFromUrdfString(
        model, urdf_xml, pin.GeometryType.COLLISION, package_dirs=package_paths
    )
    visual_model = pin.buildGeomFromUrdfString(
        model, urdf_xml, pin.GeometryType.VISUAL, package_dirs=package_paths
    )

    fixed_viz, preview_viz = make_static_and_preview_visualizers(
        model,
        collision_model,
        visual_model,
        host,
        port,
    )
    fixed_viz.display(q_home_full)
    preview_viz.display(q_home_full)
    time.sleep(0.1)

    if interactive_goal:
        q_lower, q_upper = scene.getPositionLimitVectors(group_name, collapsed=True)
        q_goal = collapseContinuousJointPositions(
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
        animate_button.disabled = True
        last_traj: list[np.ndarray] | None = None

        @reset_button.on_click
        def reset_goal(_):
            nonlocal q_goal
            q_goal = collapseContinuousJointPositions(
                scene, group_name, q_home_full[np.asarray(group_info.q_indices)]
            )
            for slider, value in zip(sliders, q_goal):
                slider.value = float(value)
            preview_goal()
            status_text.value = "目标已重置到 home。"

        @plan_button.on_click
        def plan_path(_):
            nonlocal last_traj
            preview_guard["busy"] = True
            try:
                q_goal_group = current_goal_group()
                q_goal_full = scene.toFullJointPositions(group_name, q_goal_group)
                scene.setJointPositions(q_goal_full)
                preview_viz.display(q_goal_full)
                path, _ = _make_goal_joint_waypoints(
                    scene,
                    group_name,
                    q_home_full,
                    q_goal_group,
                    waypoint_count,
                    curvature_scale,
                )
                toppra = PathParameterizerTOPPRA(scene, group_name)
                options = TOPPRAOptions(
                    dt=dt,
                    mode=toppra_mode,
                    velocity_scale=velocity_scale,
                    acceleration_scale=acceleration_scale,
                    max_adaptive_iterations=max_adaptive_iterations,
                    max_adaptive_step_size=max_adaptive_step_size,
                    max_blend_deviation=max_blend_deviation,
                )
                print(
                    f"Planning {len(path.positions)} joint-space waypoints with TOPPRA..."
                )
                plan_start = time.perf_counter()
                try:
                    traj = toppra.generate(path, options)
                except Exception as exc:
                    print(f"TOPPRA failed: {exc}")
                    status_text.value = f"规划失败：{exc}"
                    return
                plan_elapsed = time.perf_counter() - plan_start
                last_traj = list(traj.positions)
                plan_stats.value = f"toppra {plan_elapsed * 1e3:.1f} ms"
                print(f"Plan stats: toppra {plan_elapsed * 1e3:.1f} ms")
                print(f"Trajectory duration: {traj.times[-1]:.3f} s")

                fixed_viz.display(q_home_full)
                visualizeJointTrajectory(
                    fixed_viz,
                    scene,
                    traj,
                    model_data.ee_names,
                    (0, 140, 220),
                    "/toppra_joint_space/trajectory",
                )
                fig = plotJointTrajectory(
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
                status_text.value = (
                    "规划完成。若需动画播放，可点击 Animate once。"
                    if not preview_only
                    else "预览模式下已生成轨迹。"
                )
                animate_button.disabled = preview_only
                if not preview_only:
                    for q_group in traj.positions:
                        preview_viz.display(scene.toFullJointPositions(group_name, q_group))
                        time.sleep(dt)
            finally:
                preview_guard["busy"] = False

        @animate_button.on_click
        def animate_once(_):
            if last_traj is None:
                return
            for q_group in last_traj:
                preview_viz.display(scene.toFullJointPositions(group_name, q_group))
                time.sleep(dt)

        preview_goal()
        try:
            while True:
                time.sleep(10.0)
        except KeyboardInterrupt:
            pass
        return

    path, full_waypoints = _make_joint_waypoints(
        scene,
        group_name,
        q_home_full,
        waypoint_count,
        path_span,
        curvature_scale,
    )
    sparse_path_is_safe = _segment_samples_are_safe(scene, full_waypoints)
    if not sparse_path_is_safe:
        print("Warning: sampled path segments touch collision geometry.")

    toppra = PathParameterizerTOPPRA(scene, group_name)
    options = TOPPRAOptions(
        dt=dt,
        mode=toppra_mode,
        velocity_scale=velocity_scale,
        acceleration_scale=acceleration_scale,
        max_adaptive_iterations=max_adaptive_iterations,
        max_adaptive_step_size=max_adaptive_step_size,
        max_blend_deviation=max_blend_deviation,
    )

    print(f"Planning {len(path.positions)} joint-space waypoints with TOPPRA...")
    plan_start = time.perf_counter()
    try:
        traj = toppra.generate(path, options)
    except Exception as exc:
        print(f"TOPPRA failed: {exc}")
        sys.exit(1)
    plan_elapsed = time.perf_counter() - plan_start
    print(f"Plan stats: toppra {plan_elapsed * 1e3:.1f} ms")
    print(f"Trajectory duration: {traj.times[-1]:.3f} s")

    trajectory_is_safe = _trajectory_samples_are_safe(scene, group_name, traj.positions)
    if sparse_path_is_safe and trajectory_is_safe:
        print("Safety check: sampled sparse path and timed trajectory are collision-free.")
    else:
        print("Safety check: collision detected in sampled path or timed trajectory.")

    visualizeJointTrajectory(
        fixed_viz,
        scene,
        traj,
        model_data.ee_names,
        (0, 140, 220),
        "/toppra_joint_space/trajectory",
    )

    plt.figure()
    plt.ion()
    fig = plotJointTrajectory(
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
    plt.show(block=False)
    _pump_matplotlib()

    print("Use the Viser GUI controls to preview, scrub, reset, or execute the trajectory.")
    if preview_only:
        print("Preview-only mode: execution is disabled for this run.")
    elif not sparse_path_is_safe or not trajectory_is_safe:
        print("Execution disabled: adjust the plan until safety checks pass.")

    trajectory_positions = list(traj.positions)
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
        if animating:
            return
        pending_mode = "preview"

    @execute_button.on_click
    def execute_trajectory(_):
        nonlocal pending_mode
        if animating:
            return
        if preview_only or not safety_ok or not preview_done:
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
                display_step(idx)
                time.sleep(dt)
                _pump_matplotlib()

            animating = False
            if mode == "preview":
                preview_done = True
                status_text.value = "Preview complete; execution is available." if safety_ok and not preview_only else "Preview complete."
                print("Preview complete.")
            else:
                status_text.value = "Execution complete."
                print("Execution complete.")

            preview_button.disabled = False
            reset_button.disabled = False
            step_slider.disabled = False
            execute_button.disabled = preview_only or not safety_ok or not preview_done
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    tyro.cli(main)
