#!/usr/bin/env python3

import queue
import sys
import time
from collections import deque
import tyro
import xacro

from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin
from pinocchio.visualize import ViserVisualizer

from common import ObstacleConfig, get_model_data
from preview_visualization import make_static_and_preview_visualizers

try:
    import coal
except ModuleNotFoundError:
    import hppfcl as coal

from roboplan.core import CartesianConfiguration, JointConfiguration, Scene
from roboplan.example_models import get_package_share_dir
from roboplan.rrt import (
    ConstraintProjector,
    ConstraintProjectorOptions,
    PoseConstraint,
    RRT,
    RRTOptions,
)
from roboplan.simple_ik import SimpleIk, SimpleIkOptions
from roboplan.toppra import PathParameterizerTOPPRA, SplineFittingMode, TOPPRAOptions
from roboplan.visualization import addPositionPolyline, visualizeJointTrajectory


def _pump_matplotlib(delay: float = 0.001) -> None:
    plt.pause(delay)


# The safe zone the gripper must stay inside, as (min, max) world coordinates in meters.
ZONE_MIN = np.array([0.30, -0.45, 0.25])
ZONE_MAX = np.array([0.75, 0.45, 0.70])
SUPPORTED_MODELS = ("ur5", "franka", "kinova")


def build_obstacles(model_data, inset):
    """Builds the obstacle set: the model's ground plane, and a pillar in the safe zone."""
    ground_plane = next(o for o in model_data.obstacles if o.name == "ground_plane")
    unused = {o.name for o in model_data.obstacles if o.name != "ground_plane"}
    ground_plane = replace(
        ground_plane,
        disabled_collisions=[
            b for b in (ground_plane.disabled_collisions or []) if b not in unused
        ],
    )

    height = 0.85 * ZONE_MAX[2]
    center = [ZONE_MIN[0] + inset[0], 0.5 * (ZONE_MIN[1] + ZONE_MAX[1]), 0.5 * height]
    pillar = ObstacleConfig(
        name="pillar",
        geom=coal.Cylinder(0.06, height),
        parent_frame="universe",
        tform=pin.SE3(np.eye(3), np.array(center)).homogeneous,
        color=np.array([0.8, 0.2, 0.2, 0.8]),
        disabled_collisions=["ground_plane"],
    )
    return [ground_plane, pillar]


def solve_upright_ik(ik, scene, model_data, position, rotation, seed):
    """Solves IK for a world-frame pose, seeded from a given configuration.

    The seed matters: two solutions reached from unrelated seeds can sit in different connected
    components of the constrained space -- elbow up versus elbow down -- and no amount of planning
    time bridges them, since crossing over means tipping the gripper. Returns the full joint
    positions, or None if IK failed.
    """
    goal = CartesianConfiguration()
    goal.base_frame = model_data.base_link
    goal.tip_frame = model_data.ee_names[0]
    goal.tform = (
        np.linalg.inv(scene.forwardKinematics(seed, model_data.base_link))
        @ pin.SE3(rotation, position).homogeneous
    )

    group_name = model_data.default_joint_group
    start = JointConfiguration()
    start.positions = seed[scene.getJointGroupInfo(group_name).q_indices]
    solution = JointConfiguration()
    if not ik.solveIk(goal, start, solution):
        return None
    return scene.toFullJointPositions(group_name, solution.positions)


def sample_goal(
    rng, ik, projector, scene, model_data, rotation, seed, margin, tries=50
):
    """Draws a random goal inside the safe zone with the gripper upright.

    Yaw is sampled too, since the constraint leaves it free. Returns the full joint positions of a
    collision-free, constraint-satisfying goal, or None if no sample worked out.
    """
    for _ in range(tries):
        position = rng.uniform(ZONE_MIN + margin, ZONE_MAX - margin)
        yaw = pin.rpy.rpyToMatrix(0.0, 0.0, rng.uniform(-np.pi, np.pi))
        q_goal = solve_upright_ik(ik, scene, model_data, position, rotation @ yaw, seed)
        if q_goal is None:
            continue

        # IK converges to its own tolerance, so project to the constraint before planning.
        if not projector.satisfies(q_goal):
            q_goal = projector.project(q_goal)
            if q_goal is None:
                continue
        if not scene.hasCollisions(q_goal):
            return q_goal
    return None


def measure_path(scene, path, nominal_z, group_name, ee_name, samples=8):
    """Densely resamples a joint path and measures the gripper along it.

    Sampling more finely than the resolution the planner checked at is what actually tests its
    claim, and picks up the small bulges between check points.

    Returns a dict of arrays:
    path progress from 0 to 1, the gripper's tilt from vertical in degrees, and its position.
    """
    dense = []
    for idx in range(len(path.positions) - 1):
        q_a = scene.toFullJointPositions(group_name, path.positions[idx])
        q_b = scene.toFullJointPositions(group_name, path.positions[idx + 1])
        dense.extend(scene.interpolate(q_a, q_b, s / samples) for s in range(samples))
    dense.append(scene.toFullJointPositions(group_name, path.positions[-1]))

    # Plot against arc length rather than sample index, so the two paths in a comparison are drawn
    # on the same footing even when one of them has many more waypoints.
    arc = np.cumsum(
        [0.0] + [scene.configurationDistance(*p) for p in zip(dense, dense[1:])]
    )

    # Tilt is the angle between the gripper's approach axis and its nominal direction, which is
    # what "upright" means physically.
    tforms = [scene.forwardKinematics(q, ee_name) for q in dense]
    cosines = [np.clip(np.dot(t[:3, 2], nominal_z), -1.0, 1.0) for t in tforms]
    return {
        "progress": arc / arc[-1],
        "tilt": np.degrees(np.arccos(cosines)),
        "positions": np.array([t[:3, 3] for t in tforms]),
    }


def print_metrics(label, metrics, tilt_limit, position_slack):
    """Prints how far a path strays from the constraint.

    Both limits include the constraint's tolerance, which is precisely how far outside its bounds a
    configuration is still allowed to sit.
    """
    positions = metrics["positions"]
    breach = max((ZONE_MIN - positions).max(), (positions - ZONE_MAX).max(), 0.0)
    print(
        f"  {label:>13}: max tilt {metrics['tilt'].max():6.2f} deg (limit {tilt_limit:.2f}), "
        f"max zone breach {breach * 1000.0:6.2f} mm (limit {position_slack * 1000.0:.2f})"
    )


def plot_metrics(metrics, baseline_metrics, tilt_limit):
    """Plots gripper tilt and height against path progress, with the limits drawn in."""
    plt.clf()
    fig = plt.gcf()
    fig.set_size_inches(9, 7)
    axes = fig.subplots(2, 1, sharex=True)

    series = [("constrained", metrics, "tab:green")]
    if baseline_metrics is not None:
        series.append(("unconstrained", baseline_metrics, "tab:red"))
    for label, m, color in series:
        axes[0].plot(m["progress"], m["tilt"], color=color, label=label)
        axes[1].plot(m["progress"], m["positions"][:, 2], color=color, label=label)

    axes[0].axhline(
        tilt_limit,
        color="k",
        linestyle="--",
        linewidth=1,
        label=f"tilt limit ({tilt_limit:.2f} deg)",
    )
    axes[0].set_ylabel("gripper tilt from vertical [deg]")
    axes[0].set_title("Gripper stays upright and inside the safe zone")

    axes[1].axhline(
        ZONE_MIN[2], color="k", linestyle="--", linewidth=1, label="safe zone z"
    )
    axes[1].axhline(ZONE_MAX[2], color="k", linestyle="--", linewidth=1)
    axes[1].set_ylabel("end effector height [m]")
    axes[1].set_xlabel("path progress")

    for ax in axes:
        ax.legend(loc="upper right", fontsize="small")
        ax.grid(alpha=0.3)
    return fig


def main(
    model: str = "ur5",
    max_tilt_degrees: float = 5.0,
    path_step_size: float = 0.1,
    max_connection_distance: float = 0.5,
    max_nodes: int = 40000,
    max_planning_time: float = 1.0,
    collision_check_step_size: float = 0.05,
    rrt_star: bool = True,
    rewire_distance: float = 1.0,
    host: str = "localhost",
    port: str = "8000",
    rng_seed: int = 1234,
    interactive_goal: bool = True,
):
    """
    Plans RRT paths that keep the gripper upright and inside a safe zone.

    Each plan draws a random goal inside a box-shaped safe zone and plans to it without ever tipping
    the gripper over or leaving the box, then plans the same problem unconstrained to show what the
    constraint buys. Both requirements are one `PoseConstraint`: bounds on the end effector's position,
    plus bounds on roll and pitch relative to a straight-down nominal orientation. Yaw is left free.

    Note that bounding roll and pitch to +/- theta admits a total tilt of up to acos(cos^2(theta)),
    because the two rotations compose, so the default 5 degree box allows the gripper to lean at
    most 7.07 degrees off vertical.

    Parameters:
        model: The name of the model to use.
        max_tilt_degrees: The roll and pitch bound on the gripper, in degrees.
        path_step_size: Configuration-space step size taken between constraint projections.
        max_connection_distance: Maximum configuration distance covered by a single extension.
        max_nodes: The maximum number of nodes to add to the search trees. Constrained planning
            needs far more, since each projected hop becomes a node.
        max_planning_time: The planning time budget, all of which is spent optimizing.
        collision_check_step_size: Step size for collision checking, which is also the resolution
            at which constraints are verified along an edge.
        rrt_star: Whether to rewire the trees as they grow, which collapses the zigzag that walking
            in small projected hops leaves behind.
        rewire_distance: The maximum distance between nodes to consider them for rewiring.
        host: The host for the ViserVisualizer.
        port: The port for the ViserVisualizer.
        rng_seed: The seed for the planner, the scene, and the random goal sampling.
        interactive_goal: If true, expose a draggable target marker and preview IK before planning.
    """
    model_data = get_model_data().get(model)
    if model_data is None or model not in SUPPORTED_MODELS:
        print(f"Invalid model requested: {model}. Supported: {SUPPORTED_MODELS}")
        sys.exit(1)

    urdf_xml = xacro.process_file(model_data.urdf_path).toxml()
    srdf_xml = xacro.process_file(model_data.srdf_path).toxml()
    package_paths = [get_package_share_dir()]
    scene = Scene(
        "constrained_rrt_scene",
        urdf=urdf_xml,
        srdf=srdf_xml,
        package_paths=package_paths,
        yaml_config_path=model_data.yaml_config_path,
    )

    # Create a redundant Pinocchio model just for visualization with mimic joints.
    pin_model = pin.buildModelFromXML(urdf_xml, mimic=True)
    collision_model = pin.buildGeomFromUrdfString(
        pin_model, urdf_xml, pin.GeometryType.COLLISION, package_dirs=package_paths
    )
    preview_collision_model = pin.buildGeomFromUrdfString(
        pin_model, urdf_xml, pin.GeometryType.COLLISION, package_dirs=package_paths
    )
    visual_model = pin.buildGeomFromUrdfString(
        pin_model, urdf_xml, pin.GeometryType.VISUAL, package_dirs=package_paths
    )

    group_name = model_data.default_joint_group
    ee_name = model_data.ee_names[0]
    zone_size = ZONE_MAX - ZONE_MIN
    inset = 0.12 * zone_size

    scene.setRngSeed(rng_seed)
    rng = np.random.default_rng(rng_seed)
    q_indices = scene.getJointGroupInfo(group_name).q_indices

    for obstacle in build_obstacles(model_data, inset):
        obstacle.addToScene(scene)
        obstacle.addToPinocchioModels(pin_model, collision_model, visual_model)

    fixed_viz, preview_viz = make_static_and_preview_visualizers(
        pin_model,
        collision_model,
        visual_model,
        host,
        port,
        preview_collision_model=preview_collision_model,
    )

    def add_marker(name, q, color):
        """Marks an end effector position with a small sphere."""
        fixed_viz.viewer.scene.add_icosphere(
            name,
            radius=0.025,
            color=color,
            position=scene.forwardKinematics(q, ee_name)[:3, 3],
        )

    # Draw the safe zone, translucent so the robot stays visible inside it.
    fixed_viz.viewer.scene.add_box(
        "/safe_zone_wireframe",
        dimensions=tuple(zone_size),
        position=0.5 * (ZONE_MIN + ZONE_MAX),
        color=(40, 120, 220),
        opacity=0.35,
        side="back",
    )

    # The nominal "upright" orientation points the gripper's approach (z) axis straight down, the
    # way a top-down grasp holds a cup level. The constraint measures roll, pitch, and yaw relative
    # to this region frame, so a zero displacement means perfectly upright.
    upright = pin.rpy.rpyToMatrix(np.pi, 0.0, 0.0)
    region_tform = np.eye(4)
    region_tform[:3, :3] = upright
    nominal_z = region_tform[:3, 2]

    # Position bounds are expressed in that region frame, not the world. Its half turn about x
    # flips the y and z axes, so a world interval [lo, hi] on those axes becomes [-hi, -lo].
    # Writing the flip out beats rotating the box corners and taking their extent, which would
    # quietly inflate the zone for any nominal orientation that is not axis-aligned.
    region_min = np.array([ZONE_MIN[0], -ZONE_MAX[1], -ZONE_MAX[2]])
    region_max = np.array([ZONE_MAX[0], -ZONE_MIN[1], -ZONE_MIN[2]])

    tilt_bound = np.deg2rad(max_tilt_degrees)
    constraint = PoseConstraint(
        scene,
        group_name,
        ee_name,
        lower_bounds=np.concatenate([region_min, [-tilt_bound, -tilt_bound, -np.pi]]),
        upper_bounds=np.concatenate([region_max, [tilt_bound, tilt_bound, np.pi]]),
        tform=region_tform,
    )
    projection_opts = ConstraintProjectorOptions(path_step_size=path_step_size)
    projector = ConstraintProjector(scene, group_name, [constraint], projection_opts)

    # The limits reported per path add the constraint's tolerance to the bound itself.
    slack = constraint.orientation_tolerance
    tilt_limit = np.degrees(np.arccos(np.cos(tilt_bound + slack) ** 2))
    position_slack = constraint.position_tolerance

    print(f"\n=== Constrained RRT: {model} ===")
    print(f"  safe zone (world) : {ZONE_MIN} to {ZONE_MAX} m")
    print(
        f"  gripper roll/pitch: +/- {max_tilt_degrees:.1f} deg, yaw free (total tilt up to "
        f"{np.degrees(np.arccos(np.cos(tilt_bound) ** 2)):.2f} deg)"
    )
    print(
        f"  tolerances        : {position_slack * 1000.0:.2f} mm, {np.degrees(slack):.3f} deg"
    )

    # Reaching the first upright pose from the home configuration usually needs a few restarts.
    ik = SimpleIk(
        scene, SimpleIkOptions(group_name=group_name, max_restarts=10, max_time=0.5)
    )
    # Anchor the start in a corner of the zone, so the pillar sits between it and many of the goals.
    q_start = solve_upright_ik(
        ik,
        scene,
        model_data,
        ZONE_MIN + inset,
        upright,
        scene.getCurrentJointPositions(),
    )
    if q_start is not None and not projector.satisfies(q_start):
        q_start = projector.project(q_start)
    if q_start is None:
        print("\nCould not reach the start pose on the constraint.")
        sys.exit(1)
    print(f"  start EE position : {scene.forwardKinematics(q_start, ee_name)[:3, 3]}")

    start = JointConfiguration()
    start.positions = q_start[q_indices]

    options = RRTOptions(
        group_name=group_name,
        max_nodes=max_nodes,
        max_connection_distance=max_connection_distance,
        collision_check_step_size=collision_check_step_size,
        max_planning_time=max_planning_time,
        rrt_connect=True,
        rrt_star=rrt_star,
        rewire_distance=rewire_distance,
        fast_return=False,
        constraint_projection=projection_opts,
    )
    rrt = RRT(scene, options)
    rrt.setRngSeed(rng_seed)

    toppra = PathParameterizerTOPPRA(scene, group_name)
    traj_dt = 0.01

    add_marker("/constrained_rrt/start", q_start, (0, 200, 0))
    fixed_viz.display(q_start)
    preview_viz.display(q_start)
    scene.setJointPositions(q_start)

    if interactive_goal:
        q_goal_full = sample_goal(
            rng, ik, projector, scene, model_data, upright, q_start, inset
        )
        if q_goal_full is None:
            print("\nCould not sample an initial reachable goal in the safe zone.")
            sys.exit(1)
        q_goal_initial_full = q_goal_full.copy()
        q_goal_solution = JointConfiguration()
        q_goal_seed = JointConfiguration()
        q_goal_seed.positions = q_goal_full[q_indices].copy()
        goal_target = CartesianConfiguration()
        goal_target.base_frame = model_data.base_link
        goal_target.tip_frame = ee_name
        goal_control = fixed_viz.viewer.scene.add_transform_controls(
            "/constrained_rrt/goal",
            depth_test=False,
            scale=0.2,
            disable_sliders=True,
            visible=True,
        )
        goal_pose = scene.forwardKinematics(q_goal_full, ee_name)
        goal_control.position = goal_pose[:3, 3].copy()
        goal_control.wxyz = pin.Quaternion(goal_pose[:3, :3]).coeffs()[[3, 0, 1, 2]]
        preview_viz.display(q_goal_full)

        solve_times = deque(maxlen=30)
        solve_stats = fixed_viz.viewer.gui.add_text(
            "Solve stats",
            "Waiting for plan-time IK.",
            disabled=True,
        )
        traj_queue = queue.Queue()
        metrics_queue = queue.Queue()
        cur_traj = None
        animate = False
        status_text = fixed_viz.viewer.gui.add_text(
            "Status",
            "Drag the goal marker, then plan the path. IK runs only when planning starts.",
            disabled=True,
        )
        preview_state = {"busy": False, "syncing": False}

        def _sync_goal_control(q_full: np.ndarray) -> None:
            pose = scene.forwardKinematics(q_full, ee_name)
            preview_state["syncing"] = True
            goal_control.position = pose[:3, 3].copy()
            goal_control.wxyz = pin.Quaternion(pose[:3, :3]).coeffs()[[3, 0, 1, 2]]
            preview_state["syncing"] = False

        def _update_goal_preview(_=None) -> bool:
            nonlocal q_goal_full
            if preview_state["busy"] or preview_state["syncing"]:
                return False
            preview_state["busy"] = True
            try:
                world_T_base = scene.forwardKinematics(q_start, model_data.base_link)
                world_T_target = pin.SE3(
                    pin.Quaternion(goal_control.wxyz[[1, 2, 3, 0]]), goal_control.position
                ).homogeneous
                goal_target.tform = np.linalg.inv(world_T_base) @ world_T_target

                q_goal_seed.positions = q_goal_full[q_indices].copy()
                t_start = time.perf_counter()
                result = ik.solveIk(goal_target, q_goal_seed, q_goal_solution)
                elapsed = time.perf_counter() - t_start
                solve_times.append(elapsed)
                avg = sum(solve_times) / len(solve_times)
                freq = 1.0 / avg if avg > 0.0 else 0.0
                stats = (
                    f"last {elapsed * 1000.0:.1f} ms | "
                    f"avg {avg * 1000.0:.1f} ms | "
                    f"{freq:.1f} Hz"
                )
                solve_stats.value = stats
                print(f"Solve stats: {stats}")

                if not result:
                    status_text.value = "目标 IK 失败，请拖动标记后重试。"
                    return False

                q_candidate = scene.toFullJointPositions(
                    model_data.default_joint_group, q_goal_solution.positions
                )
                if not projector.satisfies(q_candidate):
                    projected = projector.project(q_candidate)
                    if projected is None:
                        status_text.value = "目标投影失败，请调整标记位置。"
                        return False
                    q_candidate = projected

                q_goal_full = q_candidate.copy()
                preview_viz.display(q_goal_full)
                q_goal_solution.positions = q_goal_full[q_indices].copy()
                q_goal_seed.positions = q_goal_solution.positions
                _sync_goal_control(q_goal_full)
                status_text.value = "目标已求解并投影到可行约束内。"
                return True
            finally:
                preview_state["busy"] = False

        plan_stats = fixed_viz.viewer.gui.add_text(
            "Plan stats",
            "Waiting for the first plan.",
            disabled=True,
        )
        plan_button = fixed_viz.viewer.gui.add_button("Plan path")
        reset_button = fixed_viz.viewer.gui.add_button("Reset goal")
        animate_button = fixed_viz.viewer.gui.add_button("Animate trajectory")
        animate_button.disabled = True

        @reset_button.on_click
        def reset_goal(_):
            nonlocal q_goal_full
            q_goal_full = q_goal_initial_full.copy()
            goal_pose = scene.forwardKinematics(q_goal_full, ee_name)
            goal_control.position = goal_pose[:3, 3].copy()
            goal_control.wxyz = pin.Quaternion(goal_pose[:3, :3]).coeffs()[[3, 0, 1, 2]]
            scene.setJointPositions(q_start)
            preview_viz.display(q_goal_full)
            status_text.value = "Goal reset to the initial sampled pose."

        @plan_button.on_click
        def plan_path(_):
            nonlocal animate
            animate = False
            plan_button.disabled = True
            animate_button.disabled = True
            try:
                if not _update_goal_preview():
                    return
                plan_start = time.perf_counter()
                q_goal = JointConfiguration()
                q_goal.positions = q_goal_full[q_indices]
                print(
                    f"\nGoal EE position: {scene.forwardKinematics(q_goal_full, ee_name)[:3, 3]}"
                )

                print("Planning with constraints...")
                constrained_start = time.perf_counter()
                try:
                    path = rrt.plan(start, q_goal, [constraint])
                except RuntimeError as ex:
                    print(f"  Constrained planning failed: {ex}")
                    return
                constrained_elapsed = time.perf_counter() - constrained_start
                print(
                    f"  Found {len(path.positions)} waypoints in {constrained_elapsed:.3f} s"
                )
                metrics = measure_path(scene, path, nominal_z, group_name, ee_name)
                print_metrics("constrained", metrics, tilt_limit, position_slack)

                print("Planning without constraints...")
                baseline_metrics = None
                baseline_elapsed = 0.0
                try:
                    baseline_start = time.perf_counter()
                    baseline = rrt.plan(start, q_goal)
                    baseline_elapsed = time.perf_counter() - baseline_start
                    print(
                        f"  Found {len(baseline.positions)} waypoints in {baseline_elapsed:.3f} s"
                    )
                    baseline_metrics = measure_path(
                        scene, baseline, nominal_z, group_name, ee_name
                    )
                    print_metrics(
                        "unconstrained", baseline_metrics, tilt_limit, position_slack
                    )
                    addPositionPolyline(
                        fixed_viz,
                        "/constrained_rrt/unconstrained_path",
                        baseline_metrics["positions"],
                        (200, 60, 60),
                        3.0,
                    )
                except RuntimeError as ex:
                    print(f"  Unconstrained planning failed: {ex}")

                toppra_start = time.perf_counter()
                traj = toppra.generate(
                    path, TOPPRAOptions(dt=traj_dt, mode=SplineFittingMode.Adaptive)
                )
                toppra_elapsed = time.perf_counter() - toppra_start
                fixed_viz.display(q_start)
                visualizeJointTrajectory(
                    fixed_viz, scene, traj, [ee_name], (0, 180, 0), "/constrained_rrt/path"
                )

                total_elapsed = time.perf_counter() - plan_start
                stats = (
                    f"constrained {constrained_elapsed:.3f} s | "
                    f"baseline {baseline_elapsed:.3f} s | "
                    f"toppra {toppra_elapsed:.3f} s | "
                    f"total {total_elapsed:.3f} s"
                )
                plan_stats.value = stats
                print(f"Plan stats: {stats}")

                traj_queue.put(traj)
                metrics_queue.put((metrics, baseline_metrics))
                status_text.value = "Planning complete. Use Animate trajectory to replay it."
            finally:
                plan_button.disabled = False
                animate_button.disabled = False

        @animate_button.on_click
        def animate_trajectory(_):
            nonlocal animate
            plan_button.disabled = True
            animate_button.disabled = True
            animate = True

        add_marker("/constrained_rrt/goal_marker", q_goal_full, (200, 0, 0))
        status_text.value = "Ready. Drag the goal marker; IK runs when planning starts."

        plt.figure()
        plt.ion()
        plt.show(block=False)
        _pump_matplotlib()
        while True:
            if not metrics_queue.empty():
                fig = plot_metrics(*metrics_queue.get(), tilt_limit)
                cur_traj = traj_queue.get()
                plt.draw()
                fig.canvas.draw()
                fig.canvas.flush_events()
                _pump_matplotlib()
            elif animate and cur_traj is not None:
                print("\nAnimating trajectory...")
                for q in cur_traj.positions:
                    preview_viz.display(scene.toFullJointPositions(group_name, q))
                    time.sleep(traj_dt)
                    _pump_matplotlib()
                preview_viz.display(q_start)
                animate = False
                plan_button.disabled = False
                animate_button.disabled = False
                print("...done!")
            else:
                _pump_matplotlib(0.05)

        return

    # Plotting has to happen on the main thread, so the viser callback below hands its results over
    # through these queues rather than drawing from inside the click handler.
    traj_queue = queue.Queue()
    metrics_queue = queue.Queue()
    cur_traj = None
    animate = False
    plan_stats = fixed_viz.viewer.gui.add_text(
        "Plan stats",
        "Waiting for the first plan.",
        disabled=True,
    )

    plan_button = fixed_viz.viewer.gui.add_button("Plan path")
    animate_button = fixed_viz.viewer.gui.add_button("Animate trajectory")
    animate_button.disabled = True

    @plan_button.on_click
    def plan_path(_):
        nonlocal animate
        animate = False
        plan_button.disabled = True
        animate_button.disabled = True
        try:
            plan_start = time.perf_counter()
            q_goal = sample_goal(
                rng, ik, projector, scene, model_data, upright, q_start, inset
            )
            if q_goal is None:
                print(
                    "\nCould not sample a reachable goal in the safe zone. Try again."
                )
                return
            add_marker("/constrained_rrt/goal", q_goal, (200, 0, 0))
            print(
                f"\nGoal EE position: {scene.forwardKinematics(q_goal, ee_name)[:3, 3]}"
            )

            goal = JointConfiguration()
            goal.positions = q_goal[q_indices]

            print("Planning with constraints...")
            constrained_start = time.perf_counter()
            try:
                path = rrt.plan(start, goal, [constraint])
            except RuntimeError as ex:
                print(f"  Constrained planning failed: {ex}")
                return
            constrained_elapsed = time.perf_counter() - constrained_start
            print(f"  Found {len(path.positions)} waypoints in {constrained_elapsed:.3f} s")
            metrics = measure_path(scene, path, nominal_z, group_name, ee_name)
            print_metrics("constrained", metrics, tilt_limit, position_slack)

            # Plan the same problem again with no constraints, for comparison.
            print("Planning without constraints...")
            baseline_metrics = None
            baseline_elapsed = 0.0
            try:
                baseline_start = time.perf_counter()
                baseline = rrt.plan(start, goal)
                baseline_elapsed = time.perf_counter() - baseline_start
                print(
                    f"  Found {len(baseline.positions)} waypoints in {baseline_elapsed:.3f} s"
                )
                baseline_metrics = measure_path(
                    scene, baseline, nominal_z, group_name, ee_name
                )
                print_metrics(
                    "unconstrained", baseline_metrics, tilt_limit, position_slack
                )
                addPositionPolyline(
                    fixed_viz,
                    "/constrained_rrt/unconstrained_path",
                    baseline_metrics["positions"],
                    (200, 60, 60),
                    3.0,
                )
            except RuntimeError as ex:
                print(f"  Unconstrained planning failed: {ex}")

            toppra_start = time.perf_counter()
            traj = toppra.generate(
                path, TOPPRAOptions(dt=traj_dt, mode=SplineFittingMode.Adaptive)
            )
            toppra_elapsed = time.perf_counter() - toppra_start
            fixed_viz.display(q_start)
            visualizeJointTrajectory(
                fixed_viz, scene, traj, [ee_name], (0, 180, 0), "/constrained_rrt/path"
            )

            total_elapsed = time.perf_counter() - plan_start
            stats = (
                f"constrained {constrained_elapsed:.3f} s | "
                f"baseline {baseline_elapsed:.3f} s | "
                f"toppra {toppra_elapsed:.3f} s | "
                f"total {total_elapsed:.3f} s"
            )
            plan_stats.value = stats
            print(f"Plan stats: {stats}")

            traj_queue.put(traj)
            metrics_queue.put((metrics, baseline_metrics))
        finally:
            plan_button.disabled = False
            animate_button.disabled = False

    @animate_button.on_click
    def animate_trajectory(_):
        nonlocal animate
        plan_button.disabled = True
        animate_button.disabled = True
        animate = True

    # Main display and animation loop.
    plt.figure()
    plt.ion()
    plt.show(block=False)
    _pump_matplotlib()
    while True:
        if not metrics_queue.empty():
            fig = plot_metrics(*metrics_queue.get(), tilt_limit)
            cur_traj = traj_queue.get()
            plt.draw()
            fig.canvas.draw()
            fig.canvas.flush_events()
            _pump_matplotlib()
        elif animate and cur_traj is not None:
            print("\nAnimating trajectory...")
            for q in cur_traj.positions:
                preview_viz.display(scene.toFullJointPositions(group_name, q))
                time.sleep(traj_dt)
                _pump_matplotlib()
            preview_viz.display(q_start)
            animate = False
            plan_button.disabled = False
            animate_button.disabled = False
            print("...done!")
        else:
            _pump_matplotlib(0.05)


if __name__ == "__main__":
    tyro.cli(main)
