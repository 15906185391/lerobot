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

import importlib.util
import time
import webbrowser
from pathlib import Path

import numpy as np

from .config_wheeled_arm_pico import WheeledArmPicoConfig
from .ik_utils import (
    JOINT_COLORS,
    format_joint_table,
    format_teleop_status,
    pinocchio_to_yourdfpy_cfg,
    resolve_package_uri,
    se3_to_position_wxyz,
)


def require_visualization_dependencies() -> None:
    missing = [
        package_name
        for module_name, package_name in (("viser", "viser"), ("yourdfpy", "yourdfpy"))
        if importlib.util.find_spec(module_name) is None
    ]
    if missing:
        raise ImportError(
            "WheeledArmPico visualization requires "
            f"{', '.join(f'`{package}`' for package in missing)}. "
            "Install the missing package(s), or set `--teleop.visualize=false` if you only need "
            "the Rerun data view from `--display_data=true --display_mode=rerun`."
        )


class WheeledArmPicoVisualizer:
    def __init__(
        self,
        config: WheeledArmPicoConfig,
        deps,
        robot,
        configuration,
        arm_q_indices: np.ndarray,
        urdf_path: Path,
    ) -> None:
        require_visualization_dependencies()
        try:
            import viser
            import yourdfpy
            from viser.extras import ViserUrdf
        except ImportError as exc:
            raise ImportError(
                "WheeledArmPico visualization requires `viser` and `yourdfpy`."
            ) from exc

        self.config = config
        self.deps = deps
        self.robot = robot
        self.configuration = configuration
        self.arm_q_indices = arm_q_indices
        self._last_update_t = 0.0
        self._update_period_s = (
            1.0 / config.visualization_update_hz if config.visualization_update_hz > 0 else 0.0
        )

        self.server = viser.ViserServer(
            host=config.visualization_host,
            port=config.visualization_port,
        )
        self.server.gui.configure_theme(control_layout="fixed", control_width="large")
        self.server.scene.add_grid("/ground", width=2, height=2)

        urdf = yourdfpy.URDF.load(
            str(urdf_path),
            build_collision_scene_graph=True,
            load_meshes=True,
            filename_handler=resolve_package_uri(urdf_path),
        )
        self.urdf_vis = ViserUrdf(self.server, urdf, root_node_name="/real_robot")
        self.status_gui = self.server.gui.add_markdown(
            format_teleop_status(True, False, False, False, config.scale, "unknown", None)
        )
        self.joint_values_gui = self.server.gui.add_markdown(
            format_joint_table(configuration.q[arm_q_indices])
        )

        self._joint_history_len = max(2, int(10.0 * config.solve_frequency_hz))
        self._joint_time_history = np.full(self._joint_history_len, np.nan)
        self._joint_position_history = np.full((len(arm_q_indices), self._joint_history_len), np.nan)
        joint_time_axis = np.linspace(-10.0, 0.0, self._joint_history_len)
        axes = ({"label": "Time (s)"}, {"label": "Joint angle (deg)"})
        scales = {"x": {"time": False}, "y": {"range": (-180.0, 180.0)}}
        self.left_primary_plot = self.server.gui.add_uplot(
            (joint_time_axis, *self._joint_position_history[0:4]),
            ({"label": "time"}, *({"label": f"L{i}", "stroke": JOINT_COLORS[i - 1]} for i in range(1, 5))),
            title="Left Arm J1-J4",
            axes=axes,
            scales=scales,
            height=300,
        )
        self.left_wrist_plot = self.server.gui.add_uplot(
            (joint_time_axis, *self._joint_position_history[4:7]),
            ({"label": "time"}, *({"label": f"L{i}", "stroke": JOINT_COLORS[i - 1]} for i in range(5, 8))),
            title="Left Arm J5-J7",
            axes=axes,
            scales=scales,
            height=260,
        )
        self.right_primary_plot = self.server.gui.add_uplot(
            (joint_time_axis, *self._joint_position_history[7:11]),
            ({"label": "time"}, *({"label": f"R{i}", "stroke": JOINT_COLORS[i - 1]} for i in range(1, 5))),
            title="Right Arm J1-J4",
            axes=axes,
            scales=scales,
            height=300,
        )
        self.right_wrist_plot = self.server.gui.add_uplot(
            (joint_time_axis, *self._joint_position_history[11:14]),
            ({"label": "time"}, *({"label": f"R{i}", "stroke": JOINT_COLORS[i - 1]} for i in range(5, 8))),
            title="Right Arm J5-J7",
            axes=axes,
            scales=scales,
            height=260,
        )

        left_pos, left_wxyz = se3_to_position_wxyz(
            configuration.get_transform_frame_to_world("AR5-5_07L-W4C4A2_tcp"),
            deps.pin,
        )
        right_pos, right_wxyz = se3_to_position_wxyz(
            configuration.get_transform_frame_to_world("AR5-5_07R-W4C4A2_tcp"),
            deps.pin,
        )
        self.left_target = self.server.scene.add_transform_controls(
            "/pico_target_l", scale=0.16, fixed=True, position=left_pos, wxyz=left_wxyz
        )
        self.right_target = self.server.scene.add_transform_controls(
            "/pico_target_r", scale=0.16, fixed=True, position=right_pos, wxyz=right_wxyz
        )
        self.collision_handle = self.server.scene.add_icosphere(
            "/collision_status", radius=0.04, color=(0, 255, 0)
        )

        self.update(
            left_target_pose=configuration.get_transform_frame_to_world("AR5-5_07L-W4C4A2_tcp"),
            right_target_pose=configuration.get_transform_frame_to_world("AR5-5_07R-W4C4A2_tcp"),
            xr_ok=False,
            left_active=False,
            right_active=False,
            collision_status="unknown",
            min_barrier=None,
            force=True,
        )
        if config.visualization_open_browser:
            time.sleep(0.5)
            webbrowser.open(f"http://localhost:{config.visualization_port}")

    def update(
        self,
        left_target_pose,
        right_target_pose,
        xr_ok: bool,
        left_active: bool,
        right_active: bool,
        collision_status: str,
        min_barrier: float | None,
        force: bool = False,
    ) -> None:
        now = time.monotonic()
        if not force and now - self._last_update_t < self._update_period_s:
            return
        self._last_update_t = now

        self.urdf_vis.update_cfg(pinocchio_to_yourdfpy_cfg(self.robot.model, self.configuration.q))
        left_pos, left_wxyz = se3_to_position_wxyz(left_target_pose, self.deps.pin)
        right_pos, right_wxyz = se3_to_position_wxyz(right_target_pose, self.deps.pin)
        self.left_target.position = left_pos
        self.left_target.wxyz = left_wxyz
        self.right_target.position = right_pos
        self.right_target.wxyz = right_wxyz

        color = {
            "collision": (255, 0, 0),
            "warning": (255, 255, 0),
            "safe": (0, 255, 0),
        }.get(collision_status, (160, 160, 160))
        self.collision_handle.color = color
        self.status_gui.content = format_teleop_status(
            True,
            xr_ok,
            left_active,
            right_active,
            self.config.scale,
            collision_status,
            min_barrier,
        )
        arm_q = self.configuration.q[self.arm_q_indices]
        self.joint_values_gui.content = format_joint_table(arm_q)
        self._update_joint_plots(arm_q)

    def _update_joint_plots(self, arm_q: np.ndarray) -> None:
        self._joint_time_history[:-1] = self._joint_time_history[1:]
        self._joint_time_history[-1] = time.monotonic()
        self._joint_position_history[:, :-1] = self._joint_position_history[:, 1:]
        self._joint_position_history[:, -1] = np.rad2deg(arm_q)

        valid_times = self._joint_time_history[np.isfinite(self._joint_time_history)]
        if valid_times.size == 0:
            return
        plot_time = self._joint_time_history - valid_times[-1]
        self.left_primary_plot.data = (plot_time, *self._joint_position_history[0:4])
        self.left_wrist_plot.data = (plot_time, *self._joint_position_history[4:7])
        self.right_primary_plot.data = (plot_time, *self._joint_position_history[7:11])
        self.right_wrist_plot.data = (plot_time, *self._joint_position_history[11:14])

    def close(self) -> None:
        for method_name in ("stop", "close"):
            method = getattr(self.server, method_name, None)
            if callable(method):
                method()
                return
