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

"""Manual joint jog console for recovering a wheeled_arm from abnormal poses."""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time
import webbrowser
import xml.etree.ElementTree as ET
from ctypes.util import find_library
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lerobot.robots.wheeled_arm.config_wheeled_arm import (
    WHEELED_ARM_DEFAULT_LEFT_END_EFFECTOR,
    WHEELED_ARM_DEFAULT_RIGHT_END_EFFECTOR,
    WHEELED_ARM_END_EFFECTOR_TYPES,
)

QT_XCB_INSTALL_HINT = (
    "sudo apt-get update && sudo apt-get install -y "
    "libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 "
    "libxcb-render-util0 libxcb-xinerama0 libxcb-xkb1 libxkbcommon-x11-0 libegl1 libgl1"
)
QT_LD_PATH_PATCHED_ENV = "LEROBOT_QT_LD_LIBRARY_PATH_PATCHED"


def _find_xcb_cursor_library() -> Path | None:
    found = find_library("xcb-cursor")
    if found:
        path = Path(found)
        if path.is_absolute():
            return path

    for lib_dir in (Path(sys.prefix) / "lib", Path(os.environ.get("CONDA_PREFIX", "")) / "lib"):
        if not lib_dir:
            continue
        for candidate in lib_dir.glob("libxcb-cursor.so*"):
            return candidate
    return None


def _ensure_qt_library_path(xcb_cursor_library: Path) -> None:
    lib_dir = str(xcb_cursor_library.parent)
    paths = [path for path in os.environ.get("LD_LIBRARY_PATH", "").split(":") if path]
    if lib_dir in paths or os.environ.get(QT_LD_PATH_PATCHED_ENV) == "1":
        return

    env = os.environ.copy()
    env[QT_LD_PATH_PATCHED_ENV] = "1"
    env["LD_LIBRARY_PATH"] = f"{lib_dir}:{env.get('LD_LIBRARY_PATH', '')}".rstrip(":")
    os.execvpe(sys.executable, [sys.executable, "-m", "lerobot.scripts.wheeled_arm_joint_jog", *sys.argv[1:]], env)


def _prepare_linux_qt_platform() -> None:
    if sys.platform != "linux":
        return
    requested_platform = os.environ.get("QT_QPA_PLATFORM", "").strip()
    if requested_platform and requested_platform != "xcb":
        return
    if not requested_platform and os.environ.get("WAYLAND_DISPLAY"):
        os.environ["QT_QPA_PLATFORM"] = "wayland"
        return
    xcb_cursor_library = _find_xcb_cursor_library()
    if xcb_cursor_library is not None:
        _ensure_qt_library_path(xcb_cursor_library)
        return
    raise SystemExit(
        "启动 PySide6 关节点动控制台需要 Ubuntu 的 Qt/xcb 系统库，但当前缺少 libxcb-cursor0。\n\n"
        f"请先执行：\n  {QT_XCB_INSTALL_HINT}\n"
    )


_prepare_linux_qt_platform()

try:
    from PySide6.QtCore import QObject, QSize, Qt, QTimer, QUrl, Signal, Slot
    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QScrollArea,
        QSpinBox,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:  # pragma: no cover - optional GUI dependency.
    raise SystemExit(
        "wheeled_arm_joint_jog requires PySide6. "
        "Run: bash scripts/setup_wheeled_arm_pico_conda.bash --env xr"
    ) from exc

try:
    from PySide6.QtWebEngineWidgets import QWebEngineView
except ImportError:  # pragma: no cover - optional embedded browser dependency.
    QWebEngineView = None


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_URDF_PATH = (
    PROJECT_ROOT
    / "src/lerobot/teleoperators/wheeled_arm_pico/assets/wheeled_robot_sim/urdf/real_robot.urdf"
)
DEFAULT_LCM_URL = "udpm://239.255.76.67:8880?ttl=1"
DEFAULT_COMMAND_HZ = 100.0
MAX_COMMAND_HZ = 100.0
DEFAULT_MAX_SPEED_DEG_S = 8.0
MAX_SPEED_DEG_S = 20.0
DEFAULT_MAX_ACCEL_DEG_S2 = 30.0
MAX_ACCEL_DEG_S2 = 120.0
DEFAULT_TARGET_TOLERANCE_DEG = 0.03
DEFAULT_GRIPPER_MAX_SPEED = 30.0
MAX_GRIPPER_MAX_SPEED = 130.0
DEFAULT_GRIPPER_MAX_ACCELERATION = 120.0
MAX_GRIPPER_MAX_ACCELERATION = 520.0
DEFAULT_GRIPPER_TARGET_TOLERANCE = 0.05
DEFAULT_GRIPPER_JOG_STEP = 5.0
MIN_GRIPPER_JOG_STEP = 0.1
MAX_GRIPPER_JOG_STEP = 30.0
GRIPPER_POSITION_LOWER = 0.0
GRIPPER_POSITION_UPPER = 130.0
SUCTION_POSITION_LOWER = 0.0
SUCTION_POSITION_UPPER = 1.0
ARM_JOINT_NAMES = [f"left_arm_{i}" for i in range(7)] + [f"right_arm_{i}" for i in range(7)]
GRIPPER_JOINT_NAMES = ["left_gripper", "right_gripper"]
HEAD_JOINT_NAMES = ["neck_yaw", "neck_pitch"]
JOINT_NAMES = [*ARM_JOINT_NAMES, *GRIPPER_JOINT_NAMES, *HEAD_JOINT_NAMES]
HEAD_JOINT_INDICES = {16, 17}
END_EFFECTOR_JOINT_INDICES = {14, 15}
RESET_LEFT_ARM_DEG = np.array([20.0, 70.0, -75.0, 100.0, -25.0, 0.0, 0.0], dtype=np.float32)
RESET_RIGHT_ARM_DEG = np.array([-20.0, 70.0, 75.0, 100.0, 25.0, 0.0, 0.0], dtype=np.float32)


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _rad_to_deg(value: float) -> float:
    return float(np.rad2deg(value))


def _deg_to_rad(value: float) -> float:
    return float(np.deg2rad(value))


def _viser_url(host: str, port: int) -> str:
    browser_host = "localhost" if host.strip() in {"", "0.0.0.0", "::"} else host.strip()
    if ":" in browser_host and not browser_host.startswith("["):
        browser_host = f"[{browser_host}]"
    return f"http://{browser_host}:{port}"


def _embedded_viser_placeholder_html() -> str:
    return """
    <html>
      <body style="margin:0;display:flex;align-items:center;justify-content:center;height:100vh;
                   font-family:sans-serif;color:#526174;background:#f8fafc;">
        <div>等待 viser 启动...</div>
      </body>
    </html>
    """


def _part_for_index(index: int, left_end_effector: str, right_end_effector: str) -> str:
    if index < 7:
        return "left_arm"
    if index < 14:
        return "right_arm"
    if index == 14:
        return f"left_{left_end_effector}"
    if index == 15:
        return f"right_{right_end_effector}"
    if index in HEAD_JOINT_INDICES:
        return "head"
    raise ValueError(f"Unknown joint index for jogging: {index}")


def _end_effector_joint_name(index: int, left_end_effector: str, right_end_effector: str) -> str:
    if index == 14:
        return f"left_{left_end_effector}"
    if index == 15:
        return f"right_{right_end_effector}"
    return JOINT_NAMES[index]


def _urdf_joint_name(index: int) -> str | None:
    if index < 7:
        return f"AR5-5_07L-W4C4A2_joint_{index + 1}"
    if index < 14:
        return f"AR5-5_07R-W4C4A2_joint_{index - 6}"
    if index == 16:
        return "neck_yaw"
    if index == 17:
        return "neck_pitch"
    return None


def _read_joint_limits_deg(urdf_path: Path) -> list[tuple[float, float]]:
    default_limits = [(-180.0, 180.0) for _ in JOINT_NAMES]
    if not urdf_path.exists():
        return default_limits
    try:
        root = ET.parse(urdf_path).getroot()
    except ET.ParseError:
        return default_limits

    limits_by_name: dict[str, tuple[float, float]] = {}
    for joint in root.findall("joint"):
        name = joint.attrib.get("name", "")
        limit = joint.find("limit")
        if limit is None:
            continue
        try:
            lower = _rad_to_deg(float(limit.attrib["lower"]))
            upper = _rad_to_deg(float(limit.attrib["upper"]))
        except (KeyError, ValueError):
            continue
        limits_by_name[name] = (lower, upper)

    limits = []
    for index in range(len(JOINT_NAMES)):
        limits.append(limits_by_name.get(_urdf_joint_name(index) or "", default_limits[index]))
    return limits


@dataclass
class JointCommand:
    target: np.ndarray
    moving_indices: set[int]


class JointJogVisualizer:
    def __init__(self, urdf_path: Path, host: str, port: int, open_browser: bool) -> None:
        self.url = _viser_url(host, port)
        try:
            import viser
            import yourdfpy
            from viser.extras import ViserUrdf
        except ImportError as exc:
            raise ImportError("关节点动 URDF 可视化需要安装 `viser` 和 `yourdfpy`。") from exc

        from lerobot.teleoperators.wheeled_arm_pico.ik_utils import resolve_package_uri

        self.server = viser.ViserServer(host=host, port=port)
        self.server.gui.configure_theme(control_layout="fixed", control_width="large")
        self.server.scene.add_grid("/ground", width=2, height=2)
        urdf = yourdfpy.URDF.load(
            str(urdf_path),
            build_collision_scene_graph=False,
            load_meshes=True,
            filename_handler=resolve_package_uri(urdf_path),
        )
        self.urdf_vis = ViserUrdf(self.server, urdf, root_node_name="/joint_jog_robot")
        self.status_gui = self.server.gui.add_markdown("等待 LCM 反馈...")
        if open_browser:
            QTimer.singleShot(500, lambda: webbrowser.open(self.url))

    def update(self, positions: np.ndarray, mode: str) -> None:
        cfg: dict[str, float] = {}
        for index in range(len(JOINT_NAMES)):
            urdf_name = _urdf_joint_name(index)
            if urdf_name is not None:
                cfg[urdf_name] = float(positions[index])
        self.urdf_vis.update_cfg(cfg)
        left = ", ".join(f"{v:.1f}" for v in np.rad2deg(positions[:7]))
        right = ", ".join(f"{v:.1f}" for v in np.rad2deg(positions[7:14]))
        head = ", ".join(f"{v:.1f}" for v in np.rad2deg(positions[16:18]))
        self.status_gui.content = (
            f"**关节点动状态**: {mode}\n\n"
            f"左臂 deg: `{left}`\n\n"
            f"右臂 deg: `{right}`\n\n"
            f"头部 deg: `{head}`"
        )

    def close(self) -> None:
        for method_name in ("stop", "close"):
            method = getattr(self.server, method_name, None)
            if callable(method):
                method()
                return


class JointJogSession(QObject):
    feedback = Signal(object)
    status = Signal(str)
    error = Signal(str)
    connectedChanged = Signal(bool)
    visualizationUrlChanged = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.handler = None
        self.visualizer: JointJogVisualizer | None = None
        self.connected = False
        self.active_target: JointCommand | None = None
        self.target_position = np.zeros(23, dtype=np.float32)
        self.last_feedback = np.zeros(23, dtype=np.float32)
        self.last_command_position = np.zeros(23, dtype=np.float32)
        self.last_command_velocity = np.zeros(23, dtype=np.float32)
        self.command_position_initialized = False
        self.feedback_timeout_s = 1.0
        self.command_hz = DEFAULT_COMMAND_HZ
        self.max_speed_rad_s = _deg_to_rad(DEFAULT_MAX_SPEED_DEG_S)
        self.max_accel_rad_s2 = _deg_to_rad(DEFAULT_MAX_ACCEL_DEG_S2)
        self.target_tolerance_rad = _deg_to_rad(DEFAULT_TARGET_TOLERANCE_DEG)
        self.gripper_max_speed = DEFAULT_GRIPPER_MAX_SPEED
        self.gripper_max_accel = DEFAULT_GRIPPER_MAX_ACCELERATION
        self.gripper_target_tolerance = DEFAULT_GRIPPER_TARGET_TOLERANCE
        self.left_end_effector = WHEELED_ARM_DEFAULT_LEFT_END_EFFECTOR
        self.right_end_effector = WHEELED_ARM_DEFAULT_RIGHT_END_EFFECTOR
        self.command_lock = threading.Lock()
        self.command_stop_event = threading.Event()
        self.command_thread: threading.Thread | None = None
        self.command_generation = 0
        self.last_command_time = 0.0
        self.last_status_text = ""
        self.last_status_emit_time = 0.0
        self.last_visualization_update_time = 0.0
        self.visualization_update_interval_s = 0.2

        self.poll_timer = QTimer(self)
        self.poll_timer.setInterval(100)
        self.poll_timer.timeout.connect(self.poll_feedback)

    def start(
        self,
        *,
        lcm_url: str,
        feedback_timeout_s: float,
        command_hz: float,
        max_speed_deg_s: float,
        max_acceleration_deg_s2: float,
        visualize: bool,
        urdf_path: Path,
        left_end_effector: str = WHEELED_ARM_DEFAULT_LEFT_END_EFFECTOR,
        right_end_effector: str = WHEELED_ARM_DEFAULT_RIGHT_END_EFFECTOR,
        visualization_host: str,
        visualization_port: int,
        visualization_open_browser: bool,
    ) -> None:
        if self.connected:
            self._emit_status("关节点动已经启动。", force=True)
            return
        try:
            from lerobot.robots.wheeled_arm.hardware_interface.lcm_handler import LCMHandler
        except ModuleNotFoundError as exc:
            if exc.name == "lcm":
                self.error.emit(
                    "缺少 Python lcm 模块。请确认在 conda xr 环境中安装机器人 SDK 依赖。"
                )
                return
            self.error.emit(str(exc))
            return

        try:
            self.handler = LCMHandler(
                lcm_url=lcm_url,
                left_end_effector=left_end_effector,
                right_end_effector=right_end_effector,
            )
            self.left_end_effector = left_end_effector
            self.right_end_effector = right_end_effector
            self.feedback_timeout_s = feedback_timeout_s
            self.command_hz = _clip(command_hz, 1.0, MAX_COMMAND_HZ)
            self.max_speed_rad_s = _deg_to_rad(_clip(max_speed_deg_s, 0.1, MAX_SPEED_DEG_S))
            self.max_accel_rad_s2 = _deg_to_rad(
                _clip(max_acceleration_deg_s2, 1.0, MAX_ACCEL_DEG_S2)
            )
            if visualize:
                self.visualizer = JointJogVisualizer(
                    urdf_path=urdf_path,
                    host=visualization_host,
                    port=visualization_port,
                    open_browser=visualization_open_browser,
                )
                self.visualizationUrlChanged.emit(self.visualizer.url)
            else:
                self.visualizationUrlChanged.emit("")
        except Exception as exc:  # noqa: BLE001 - surface hardware/optional dependency failures in GUI.
            self.error.emit(f"启动关节点动失败：{exc}")
            self.stop()
            return

        self.connected = True
        self.connectedChanged.emit(True)
        self._emit_status("已启动 LCM，正在等待新鲜左右臂反馈。", force=True)
        self.poll_timer.start()
        self.poll_feedback()

    def stop(self) -> None:
        self._stop_command_thread()
        self.poll_timer.stop()
        if self.visualizer is not None:
            self.visualizer.close()
            self.visualizer = None
        self.visualizationUrlChanged.emit("")
        if self.handler is not None:
            stop = getattr(self.handler, "stop", None)
            if callable(stop):
                stop()
            self.handler = None
        if self.connected:
            self.connected = False
            self.connectedChanged.emit(False)
        self._emit_status("关节点动已停止。", force=True)

    def has_fresh_feedback(self) -> bool:
        if self.handler is None:
            return False
        try:
            return bool(self.handler.has_arm_state_feedback(self.feedback_timeout_s))
        except TypeError:
            return bool(self.handler.has_arm_state_feedback(self.feedback_timeout_s))

    def _is_gripper_joint(self, index: int) -> bool:
        return (index == 14 and self.left_end_effector == "gripper") or (
            index == 15 and self.right_end_effector == "gripper"
        )

    def _is_suction_joint(self, index: int) -> bool:
        return (index == 14 and self.left_end_effector == "suction") or (
            index == 15 and self.right_end_effector == "suction"
        )

    @Slot()
    def poll_feedback(self) -> None:
        if self.handler is None:
            return
        with self.handler.joint_current_pos_lock:
            self.last_feedback = np.asarray(self.handler.joint_current_pos, dtype=np.float32).copy()
        if not self.command_position_initialized:
            self.last_command_position = self.last_feedback.copy()
            self.target_position = self.last_feedback.copy()
            self.command_position_initialized = True
        self.feedback.emit(self.last_feedback.copy())
        mode = "反馈新鲜" if self.has_fresh_feedback() else "等待反馈/反馈超时"
        now = time.monotonic()
        if (
            self.visualizer is not None
            and now - self.last_visualization_update_time >= self.visualization_update_interval_s
        ):
            self.visualizer.update(self.last_feedback, mode)
            self.last_visualization_update_time = now
        self._emit_status(mode, min_interval_s=1.0)

    def sync_target_to_feedback(self) -> None:
        if self.handler is None:
            return
        self.poll_feedback()
        self.target_position = self.last_feedback.copy()
        self.last_command_position = self.last_feedback.copy()
        self.last_command_velocity = np.zeros_like(self.last_command_velocity)
        self.command_position_initialized = True
        self._stop_command_thread()
        self._emit_status("已将目标同步到当前 LCM 反馈。", force=True)

    def jog_to(self, index: int, target_value: float) -> None:
        moving_indices = {index}
        if not self._can_publish(moving_indices):
            return
        target = self.last_feedback.copy()
        target[index] = target_value
        self._start_target(target, moving_indices)

    def move_to_target(self, target_position: np.ndarray, moving_indices: set[int]) -> None:
        if not self._can_publish(moving_indices):
            return
        if not moving_indices:
            self._emit_status("没有选择要移动的关节。", force=True)
            return
        target = self.last_feedback.copy()
        for index in moving_indices:
            target[index] = target_position[index]
        self._start_target(target, moving_indices)

    def stop_publish(self) -> None:
        self._stop_command_thread()
        self._emit_status("已停止继续发布关节命令。", force=True)

    def _can_publish(self, moving_indices: set[int]) -> bool:
        if self.handler is None or not self.connected:
            self.error.emit("请先启动关节点动连接。")
            return False
        self.poll_feedback()
        ok, message = self._has_required_feedback(moving_indices)
        if not ok:
            self.error.emit(message)
            return False
        return True

    def _has_required_feedback(self, moving_indices: set[int]) -> tuple[bool, str]:
        if not self.has_fresh_feedback():
            return False, "没有新鲜左右臂 LCM 反馈，禁止发送关节命令。"
        if moving_indices & HEAD_JOINT_INDICES and not self.has_fresh_head_feedback():
            return False, "没有新鲜头部 LCM 反馈，禁止发送头部点动命令。"
        return True, ""

    def has_fresh_head_feedback(self) -> bool:
        if self.handler is None:
            return False
        has_head_state_feedback = getattr(self.handler, "has_head_state_feedback", None)
        if callable(has_head_state_feedback):
            try:
                return bool(has_head_state_feedback(self.feedback_timeout_s))
            except TypeError:
                return bool(has_head_state_feedback())
        head_state_updated = getattr(self.handler, "head_state_updated", None)
        return bool(head_state_updated and head_state_updated.is_set())

    def _start_target(self, target: np.ndarray, moving_indices: set[int]) -> None:
        moving_indices = set(moving_indices)
        if not moving_indices:
            return

        start_thread = False
        with self.command_lock:
            target = target.astype(np.float32, copy=True)
            if self.active_target is None:
                merged_target = self.last_command_position.copy()
                if not self.command_position_initialized:
                    merged_target = self.last_feedback.copy()
                    self.last_command_position = self.last_feedback.copy()
                    self.command_position_initialized = True
                merged_indices = set(moving_indices)
                self.last_command_velocity = np.zeros_like(self.last_command_velocity)
            else:
                merged_target = self.active_target.target.copy()
                merged_indices = set(self.active_target.moving_indices) | moving_indices

            for index in moving_indices:
                merged_target[index] = target[index]

            self.active_target = JointCommand(target=merged_target, moving_indices=merged_indices)
            if self.last_command_time <= 0.0:
                self.last_command_time = time.perf_counter()
            self.command_stop_event.clear()
            start_thread = self.command_thread is None or not self.command_thread.is_alive()

        if start_thread:
            self.command_thread = threading.Thread(
                target=self._command_loop,
                name="wheeled-arm-joint-jog-command",
                daemon=True,
            )
            self.command_thread.start()
        else:
            self._emit_status(f"已更新点动目标，关节数：{len(moving_indices)}。", min_interval_s=0.25)

    def _stop_command_thread(self, *, clear_target: bool = True) -> None:
        self.command_stop_event.set()
        thread = self.command_thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        self.command_thread = None
        with self.command_lock:
            self.command_generation += 1
            if clear_target:
                self.active_target = None
            self.last_command_velocity = np.zeros_like(self.last_command_velocity)
            self.last_command_time = 0.0
        self._set_moving_flags(set())

    def _command_loop(self) -> None:
        period_s = 1.0 / self.command_hz
        try:
            while not self.command_stop_event.is_set():
                loop_start = time.perf_counter()
                with self.command_lock:
                    moving_indices = (
                        set()
                        if self.active_target is None
                        else self.active_target.moving_indices.copy()
                    )
                ok, message = self._has_required_feedback(moving_indices)
                if not ok:
                    with self.command_lock:
                        self.active_target = None
                        self.last_command_velocity = np.zeros_like(self.last_command_velocity)
                    self._set_moving_flags(set())
                    self.error.emit(f"反馈超时，已停止关节点动发布：{message}")
                    return

                done = self._command_step(time.perf_counter())
                if done:
                    return

                elapsed_s = time.perf_counter() - loop_start
                self.command_stop_event.wait(max(period_s - elapsed_s, 0.0))
        finally:
            with self.command_lock:
                if threading.current_thread() is self.command_thread:
                    self.command_thread = None

    def _command_step(self, now: float) -> bool:
        with self.command_lock:
            if self.handler is None or self.active_target is None:
                return True

            dt = max(0.001, min(now - self.last_command_time, 0.1))
            package = self.last_command_position.copy()
            velocity = self.last_command_velocity.copy()
            moving_indices = self.active_target.moving_indices.copy()
            target = self.active_target.target.copy()
            done = True

            for index in moving_indices:
                if self._is_suction_joint(index):
                    package[index] = float(_clip(target[index], SUCTION_POSITION_LOWER, SUCTION_POSITION_UPPER))
                    velocity[index] = 0.0
                    continue
                is_gripper = self._is_gripper_joint(index)
                target_tolerance = (
                    self.gripper_target_tolerance if is_gripper else self.target_tolerance_rad
                )
                max_speed = self.gripper_max_speed if is_gripper else self.max_speed_rad_s
                max_accel_delta = (
                    self.gripper_max_accel if is_gripper else self.max_accel_rad_s2
                ) * dt

                delta = float(target[index] - package[index])
                if abs(delta) <= target_tolerance:
                    package[index] = target[index]
                    velocity[index] = 0.0
                    continue

                desired_velocity = float(np.clip(delta / dt, -max_speed, max_speed))
                velocity[index] = float(
                    np.clip(
                        desired_velocity,
                        velocity[index] - max_accel_delta,
                        velocity[index] + max_accel_delta,
                    )
                )
                step = velocity[index] * dt
                if abs(step) >= abs(delta):
                    package[index] = target[index]
                    velocity[index] = 0.0
                else:
                    package[index] += step
                    done = False

            self.last_command_position = package.copy()
            self.last_command_velocity = velocity.copy()
            self.last_command_time = now
            handler = self.handler

        self._set_moving_flags(moving_indices)
        if not self.command_stop_event.is_set() and handler is not None:
            handler.upper_body_data_publisher(package)

        if done:
            with self.command_lock:
                target_unchanged = self.active_target is not None and np.allclose(
                    self.active_target.target, target
                )
                if target_unchanged:
                    self.active_target = None
                    self.last_command_velocity = np.zeros_like(self.last_command_velocity)
                    self.last_command_time = 0.0
                else:
                    self.last_command_time = now
            if target_unchanged:
                self._set_moving_flags(set())
                self._emit_status(f"点动完成，关节数：{len(moving_indices)}。", force=True)
                return True
            return False
        return False

    def _emit_status(self, text: str, *, min_interval_s: float = 0.0, force: bool = False) -> None:
        now = time.monotonic()
        if force or text != self.last_status_text or now - self.last_status_emit_time >= min_interval_s:
            self.last_status_text = text
            self.last_status_emit_time = now
            self.status.emit(text)

    def _set_moving_flags(self, moving_indices: set[int]) -> None:
        if self.handler is None:
            return
        parts = {
            _part_for_index(index, self.left_end_effector, self.right_end_effector)
            for index in moving_indices
        }
        for part, flag_name in (
            ("left_arm", "left_arm_moving"),
            ("right_arm", "right_arm_moving"),
            ("left_gripper", "left_gripper_moving"),
            ("right_gripper", "right_gripper_moving"),
            ("left_suction", "left_suction_moving"),
            ("right_suction", "right_suction_moving"),
            ("head", "head_moving"),
            ("waist", "waist_moving"),
            ("leg", "leg_moving"),
        ):
            if hasattr(self.handler, flag_name):
                setattr(self.handler, flag_name, part in parts)


class JointRow(QWidget):
    jogRequested = Signal(int, float)
    targetChanged = Signal()

    def __init__(self, index: int, name: str, lower: float, upper: float, joint_kind: str = "joint") -> None:
        super().__init__()
        self.index = index
        self.name = name
        self.lower = lower
        self.upper = upper
        self.joint_kind = joint_kind
        self.is_gripper = joint_kind == "gripper"
        self.is_suction = joint_kind == "suction"
        self.current_value = 0.0
        self.target_initialized = False
        self.setMinimumHeight(44)

        layout = QGridLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(6)

        self.enable = QCheckBox()
        self.label = QLabel(name)
        self.label.setMinimumWidth(118)
        self.current_label = QLabel("-")
        self.current_label.setMinimumWidth(96)
        self.target = QDoubleSpinBox()
        self.target.setDecimals(0 if self.is_suction else (3 if self.is_gripper else 2))
        self.target.setRange(lower, upper)
        self.target.setSingleStep(1.0 if self.is_suction else (5.0 if self.is_gripper else 1.0))
        self.target.setSuffix("" if self.is_gripper or self.is_suction else " deg")
        self.target.setMinimumWidth(132)
        self.target.valueChanged.connect(lambda _value: self.targetChanged.emit())
        self.minus_btn = QPushButton("-")
        self.plus_btn = QPushButton("+")
        self.minus_btn.setFixedSize(38, 34)
        self.plus_btn.setFixedSize(38, 34)
        self.minus_btn.clicked.connect(lambda: self._request_jog(-1.0))
        self.plus_btn.clicked.connect(lambda: self._request_jog(1.0))

        layout.addWidget(self.enable, 0, 0)
        layout.addWidget(self.label, 0, 1)
        layout.addWidget(self.current_label, 0, 2)
        layout.addWidget(self.target, 0, 3)
        layout.addWidget(self.minus_btn, 0, 4)
        layout.addWidget(self.plus_btn, 0, 5)
        layout.setColumnStretch(3, 1)

    def set_current(self, raw_value: float) -> None:
        if self.is_suction:
            self.current_value = float(raw_value)
            self.current_label.setText(f"{self.current_value:.2f} kPa")
        elif self.is_gripper:
            self.current_value = float(raw_value)
            self.current_label.setText(f"{self.current_value:.3f}")
        else:
            self.current_value = float(_rad_to_deg(raw_value))
            self.current_label.setText(f"{self.current_value:.2f} deg")
        if not self.target_initialized:
            self.sync_target()

    def sync_target(self) -> None:
        value = _clip(self.current_value, self.lower, self.upper)
        self.target.blockSignals(True)
        self.target.setValue(value)
        self.target.blockSignals(False)
        self.target_initialized = True

    def raw_target(self) -> float:
        value = _clip(float(self.target.value()), self.lower, self.upper)
        return value if self.is_gripper or self.is_suction else _deg_to_rad(value)

    def _request_jog(self, direction: float) -> None:
        window = self.window()
        step = getattr(window, "jog_step", None)
        default_step = 1.0 if self.is_suction else (5.0 if self.is_gripper else 1.0)
        jog_step = float(step(self.joint_kind)) if callable(step) else default_step
        target = _clip(float(self.target.value()) + direction * jog_step, self.lower, self.upper)
        self.target.setValue(target)
        self.target_initialized = True
        self.jogRequested.emit(self.index, target if self.is_gripper or self.is_suction else _deg_to_rad(target))


class PathPicker(QWidget):
    changed = Signal()

    def __init__(self, placeholder: str = "") -> None:
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self.edit = QLineEdit()
        self.edit.setPlaceholderText(placeholder)
        self.edit.textChanged.connect(self.changed)
        self.button = QPushButton("浏览")
        self.button.setFixedWidth(68)
        self.button.clicked.connect(self._browse)
        layout.addWidget(self.edit, 1)
        layout.addWidget(self.button)

    def text(self) -> str:
        return self.edit.text().strip()

    def setText(self, value: str) -> None:
        self.edit.setText(value)

    def _browse(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(self, "选择 URDF 文件", self.text() or str(Path.home()))
        if selected:
            self.edit.setText(selected)


class JointJogWindow(QMainWindow):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.args = args
        self.session = JointJogSession()
        self.joint_rows: list[JointRow] = []
        self.setWindowTitle("Wheeled Arm 关节点动控制台")
        self.setMinimumSize(QSize(980, 720))
        self._build_ui()
        self._connect_session()
        self._load_args()

    def _build_ui(self) -> None:
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(APP_STYLESHEET)

        root = QWidget()
        root.setMinimumWidth(900)
        layout = QGridLayout(root)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setHorizontalSpacing(16)
        layout.setVerticalSpacing(12)

        title = QLabel("Wheeled Arm 关节点动控制台")
        title.setObjectName("Title")
        subtitle = QLabel("仅用于异常姿态救援：必须手动启动连接，所有运动从新鲜 LCM 反馈开始。")
        subtitle.setObjectName("Subtitle")
        subtitle.setWordWrap(True)
        title_block = QVBoxLayout()
        title_block.addWidget(title)
        title_block.addWidget(subtitle)
        layout.addLayout(title_block, 0, 0, 1, 2)

        settings = QGroupBox("连接与安全")
        form = QFormLayout(settings)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setVerticalSpacing(10)
        self.lcm_url = QLineEdit(DEFAULT_LCM_URL)
        self.urdf_path = PathPicker("real_robot.urdf")
        self.urdf_path.setText(str(DEFAULT_URDF_PATH))
        self.visualize = QCheckBox("启动 viser URDF 可视化")
        self.visualize.setChecked(True)
        self.visualization_host = QLineEdit("0.0.0.0")
        self.visualization_port = QSpinBox()
        self.visualization_port.setRange(1, 65535)
        self.visualization_port.setValue(8092)
        self.open_browser = QCheckBox("自动打开浏览器")
        self.open_browser.setChecked(False)
        self.feedback_timeout = QDoubleSpinBox()
        self.feedback_timeout.setRange(0.1, 30.0)
        self.feedback_timeout.setValue(1.0)
        self.feedback_timeout.setSuffix(" s")
        self.command_hz = QDoubleSpinBox()
        self.command_hz.setRange(1.0, MAX_COMMAND_HZ)
        self.command_hz.setValue(DEFAULT_COMMAND_HZ)
        self.command_hz.setSuffix(" Hz")
        self.max_speed = QDoubleSpinBox()
        self.max_speed.setRange(0.1, MAX_SPEED_DEG_S)
        self.max_speed.setValue(DEFAULT_MAX_SPEED_DEG_S)
        self.max_speed.setSuffix(" deg/s")
        self.max_accel = QDoubleSpinBox()
        self.max_accel.setRange(1.0, MAX_ACCEL_DEG_S2)
        self.max_accel.setValue(DEFAULT_MAX_ACCEL_DEG_S2)
        self.max_accel.setSuffix(" deg/s²")
        self.jog_step_deg = QDoubleSpinBox()
        self.jog_step_deg.setRange(0.05, 10.0)
        self.jog_step_deg.setValue(1.0)
        self.jog_step_deg.setSuffix(" deg")
        self.jog_step_gripper = QDoubleSpinBox()
        self.jog_step_gripper.setRange(MIN_GRIPPER_JOG_STEP, MAX_GRIPPER_JOG_STEP)
        self.jog_step_gripper.setValue(DEFAULT_GRIPPER_JOG_STEP)
        self.left_end_effector = QComboBox()
        self.left_end_effector.addItems(list(WHEELED_ARM_END_EFFECTOR_TYPES))
        self.left_end_effector.setCurrentText(WHEELED_ARM_DEFAULT_LEFT_END_EFFECTOR)
        self.right_end_effector = QComboBox()
        self.right_end_effector.addItems(list(WHEELED_ARM_END_EFFECTOR_TYPES))
        self.right_end_effector.setCurrentText(WHEELED_ARM_DEFAULT_RIGHT_END_EFFECTOR)
        form.addRow("LCM URL", self.lcm_url)
        form.addRow("URDF", self.urdf_path)
        form.addRow("左臂末端", self.left_end_effector)
        form.addRow("右臂末端", self.right_end_effector)
        form.addRow("", self.visualize)
        form.addRow("viser host", self.visualization_host)
        form.addRow("viser port", self.visualization_port)
        form.addRow("", self.open_browser)
        form.addRow("反馈超时", self.feedback_timeout)
        form.addRow("发布频率", self.command_hz)
        form.addRow("最大速度", self.max_speed)
        form.addRow("最大加速度", self.max_accel)
        form.addRow("点动步长", self.jog_step_deg)
        form.addRow("夹爪步长", self.jog_step_gripper)
        layout.addWidget(settings, 1, 0)

        status_box = QGroupBox("状态与操作")
        status_layout = QVBoxLayout(status_box)
        self.status_label = QLabel("未启动。启动后会先等待左右臂新鲜反馈。")
        self.status_label.setObjectName("StatusPill")
        self.status_label.setWordWrap(True)
        status_layout.addWidget(self.status_label)
        button_row = QGridLayout()
        button_row.setHorizontalSpacing(8)
        button_row.setVerticalSpacing(8)
        self.start_btn = QPushButton("启动连接")
        self.start_btn.setObjectName("PrimaryButton")
        self.stop_btn = QPushButton("停止连接")
        self.stop_btn.setObjectName("DangerButton")
        self.stop_btn.setEnabled(False)
        self.stop_publish_btn = QPushButton("停止发布")
        self.stop_publish_btn.setObjectName("DangerButton")
        self.stop_publish_btn.setEnabled(False)
        self.sync_btn = QPushButton("同步当前为目标")
        self.sync_btn.setEnabled(False)
        self.move_selected_btn = QPushButton("点动到勾选目标")
        self.move_selected_btn.setObjectName("PrimaryButton")
        self.move_selected_btn.setEnabled(False)
        button_row.addWidget(self.start_btn, 0, 0)
        button_row.addWidget(self.stop_btn, 0, 1)
        button_row.addWidget(self.stop_publish_btn, 0, 2)
        button_row.addWidget(self.sync_btn, 1, 0)
        button_row.addWidget(self.move_selected_btn, 1, 1, 1, 2)
        button_row.setColumnStretch(0, 1)
        button_row.setColumnStretch(1, 1)
        button_row.setColumnStretch(2, 1)
        status_layout.addLayout(button_row)

        preset_row = QGridLayout()
        preset_row.setHorizontalSpacing(8)
        preset_row.setVerticalSpacing(8)
        self.reset_left_btn = QPushButton("左臂默认目标")
        self.reset_right_btn = QPushButton("右臂默认目标")
        self.reset_both_btn = QPushButton("双臂默认目标")
        self.reset_head_btn = QPushButton("头部默认目标")
        self.select_left_btn = QPushButton("勾选左臂")
        self.select_right_btn = QPushButton("勾选右臂")
        self.select_left_gripper_btn = QPushButton("勾选左末端")
        self.select_right_gripper_btn = QPushButton("勾选右末端")
        self.select_head_btn = QPushButton("勾选头部")
        self.clear_selection_btn = QPushButton("取消勾选")
        preset_buttons = (
            self.reset_left_btn,
            self.reset_right_btn,
            self.reset_both_btn,
            self.reset_head_btn,
            self.select_left_btn,
            self.select_right_btn,
            self.select_left_gripper_btn,
            self.select_right_gripper_btn,
            self.select_head_btn,
            self.clear_selection_btn,
        )
        for index, button in enumerate(preset_buttons):
            button.setEnabled(False)
            preset_row.addWidget(button, index // 3, index % 3)
        status_layout.addLayout(preset_row)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setObjectName("Log")
        self.log.setFont(QFont("monospace", 10))
        self.log.setMinimumHeight(150)
        status_layout.addWidget(self.log, 1)
        layout.addWidget(status_box, 1, 1)

        joints = QGroupBox("关节")
        joints_layout = QGridLayout(joints)
        joints_layout.setHorizontalSpacing(14)
        joints_layout.setVerticalSpacing(10)
        header = QLabel(
            "勾选关节后可批量点动到目标；单行 +/- 会按当前反馈做小步点动。"
            "手臂和头部角度单位为 degree，夹爪目标范围为 [0, 130]，吸盘目标范围为 [0, 1]。"
        )
        header.setObjectName("Hint")
        header.setWordWrap(True)
        joints_layout.addWidget(header, 0, 0, 1, 2)

        self.joint_container = QWidget()
        self.joint_layout = QGridLayout(self.joint_container)
        self.joint_layout.setVerticalSpacing(2)
        self.joint_layout.setColumnStretch(3, 1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(360)
        scroll.setWidget(self.joint_container)
        joints_layout.addWidget(scroll, 1, 0)

        self.visualization_url = ""
        self.visualization_panel = QWidget()
        self.visualization_panel.setObjectName("ViserPanel")
        visualization_layout = QVBoxLayout(self.visualization_panel)
        visualization_layout.setContentsMargins(10, 10, 10, 10)
        visualization_layout.setSpacing(8)

        visualization_header = QHBoxLayout()
        self.visualization_title = QLabel("viser 可视化")
        self.visualization_title.setObjectName("PanelTitle")
        self.open_visualization_btn = QPushButton("外部打开")
        self.open_visualization_btn.setEnabled(False)
        self.open_visualization_btn.clicked.connect(self.open_visualization_browser)
        visualization_header.addWidget(self.visualization_title)
        visualization_header.addStretch(1)
        visualization_header.addWidget(self.open_visualization_btn)
        visualization_layout.addLayout(visualization_header)

        self.visualization_url_label = QLabel("未启动")
        self.visualization_url_label.setObjectName("Hint")
        self.visualization_url_label.setTextInteractionFlags(Qt.TextBrowserInteraction)
        self.visualization_url_label.setOpenExternalLinks(True)
        visualization_layout.addWidget(self.visualization_url_label)

        if QWebEngineView is not None:
            self.visualization_view = QWebEngineView()
            self.visualization_view.setMinimumSize(420, 360)
            self.visualization_view.setHtml(_embedded_viser_placeholder_html())
            visualization_layout.addWidget(self.visualization_view, 1)
        else:
            self.visualization_view = None
            self.visualization_placeholder = QLabel(
                "当前 PySide6 环境缺少 QtWebEngine，启动 viser 后可在此处打开本地页面。"
            )
            self.visualization_placeholder.setObjectName("Hint")
            self.visualization_placeholder.setWordWrap(True)
            self.visualization_placeholder.setMinimumHeight(360)
            self.visualization_placeholder.setAlignment(Qt.AlignCenter)
            visualization_layout.addWidget(self.visualization_placeholder, 1)

        joints_layout.addWidget(self.visualization_panel, 1, 1)
        joints_layout.setColumnStretch(0, 3)
        joints_layout.setColumnStretch(1, 4)
        layout.addWidget(joints, 2, 0, 1, 2)
        layout.setColumnStretch(0, 2)
        layout.setColumnStretch(1, 3)
        layout.setRowStretch(2, 1)

        outer_scroll = QScrollArea()
        outer_scroll.setWidgetResizable(True)
        outer_scroll.setObjectName("WindowScroll")
        outer_scroll.setWidget(root)
        self.setCentralWidget(outer_scroll)

        self._rebuild_joint_rows()
        self.start_btn.clicked.connect(self.start_session)
        self.stop_btn.clicked.connect(self.session.stop)
        self.stop_publish_btn.clicked.connect(self.session.stop_publish)
        self.sync_btn.clicked.connect(self.sync_targets)
        self.move_selected_btn.clicked.connect(self.move_selected)
        self.reset_left_btn.clicked.connect(lambda: self.set_reset_targets("left"))
        self.reset_right_btn.clicked.connect(lambda: self.set_reset_targets("right"))
        self.reset_both_btn.clicked.connect(lambda: self.set_reset_targets("both"))
        self.reset_head_btn.clicked.connect(lambda: self.set_reset_targets("head"))
        self.select_left_btn.clicked.connect(lambda: self.select_arm("left"))
        self.select_right_btn.clicked.connect(lambda: self.select_arm("right"))
        self.select_left_gripper_btn.clicked.connect(lambda: self.select_end_effector("left"))
        self.select_right_gripper_btn.clicked.connect(lambda: self.select_end_effector("right"))
        self.select_head_btn.clicked.connect(lambda: self.select_arm("head"))
        self.clear_selection_btn.clicked.connect(lambda: self.select_arm("none"))
        self.left_end_effector.currentTextChanged.connect(lambda *_: self._rebuild_joint_rows())
        self.right_end_effector.currentTextChanged.connect(lambda *_: self._rebuild_joint_rows())
        self.urdf_path.changed.connect(self._rebuild_joint_rows)

    def _connect_session(self) -> None:
        self.session.feedback.connect(self.update_feedback)
        self.session.status.connect(self.set_status)
        self.session.error.connect(self.show_error)
        self.session.connectedChanged.connect(self.set_connected)
        self.session.visualizationUrlChanged.connect(self.update_visualization_url)

    def _load_args(self) -> None:
        self.lcm_url.setText(self.args.lcm_url)
        self.urdf_path.setText(str(self.args.urdf_path))
        self.visualization_host.setText(self.args.visualization_host)
        self.visualization_port.setValue(self.args.visualization_port)
        self.visualize.setChecked(self.args.visualize)
        self.open_browser.setChecked(self.args.open_browser)
        self.left_end_effector.setCurrentText(self.args.left_end_effector)
        self.right_end_effector.setCurrentText(self.args.right_end_effector)
        self.command_hz.setValue(self.args.command_hz)
        self.max_speed.setValue(self.args.max_speed_deg_s)
        self.max_accel.setValue(self.args.max_acceleration_deg_s2)

    def _rebuild_joint_rows(self) -> None:
        for row in self.joint_rows:
            row.setParent(None)
        self.joint_rows = []

        limits = _read_joint_limits_deg(Path(self.urdf_path.text() or DEFAULT_URDF_PATH).expanduser())
        left_end_effector = self.left_end_effector.currentText()
        right_end_effector = self.right_end_effector.currentText()
        for index, name in enumerate(JOINT_NAMES):
            if index == 14:
                joint_kind = left_end_effector
                name = _end_effector_joint_name(index, left_end_effector, right_end_effector)
                lower, upper = (
                    (SUCTION_POSITION_LOWER, SUCTION_POSITION_UPPER)
                    if joint_kind == "suction"
                    else (GRIPPER_POSITION_LOWER, GRIPPER_POSITION_UPPER)
                )
            elif index == 15:
                joint_kind = right_end_effector
                name = _end_effector_joint_name(index, left_end_effector, right_end_effector)
                lower, upper = (
                    (SUCTION_POSITION_LOWER, SUCTION_POSITION_UPPER)
                    if joint_kind == "suction"
                    else (GRIPPER_POSITION_LOWER, GRIPPER_POSITION_UPPER)
                )
            else:
                joint_kind = "joint"
                lower, upper = limits[index]
            row = JointRow(index, name, lower, upper, joint_kind=joint_kind)
            row.jogRequested.connect(self.session.jog_to)
            self.joint_rows.append(row)
            self.joint_layout.addWidget(row, index, 0)

    def jog_step(self, joint_kind: str) -> float:
        if joint_kind == "suction":
            return 1.0
        if joint_kind == "gripper":
            return self.jog_step_gripper.value()
        return self.jog_step_deg.value()

    @Slot()
    def start_session(self) -> None:
        if not Path(self.urdf_path.text()).expanduser().exists():
            QMessageBox.warning(self, "URDF 不存在", "请确认 URDF 路径正确。")
            return
        self.append_log("启动关节点动连接。")
        self.session.start(
            lcm_url=self.lcm_url.text().strip() or DEFAULT_LCM_URL,
            feedback_timeout_s=self.feedback_timeout.value(),
            command_hz=self.command_hz.value(),
            max_speed_deg_s=self.max_speed.value(),
            max_acceleration_deg_s2=self.max_accel.value(),
            visualize=self.visualize.isChecked(),
            urdf_path=Path(self.urdf_path.text()).expanduser(),
            left_end_effector=self.left_end_effector.currentText(),
            right_end_effector=self.right_end_effector.currentText(),
            visualization_host=self.visualization_host.text().strip() or "0.0.0.0",
            visualization_port=self.visualization_port.value(),
            visualization_open_browser=self.open_browser.isChecked(),
        )

    @Slot(bool)
    def set_connected(self, connected: bool) -> None:
        self.start_btn.setEnabled(not connected)
        self.stop_btn.setEnabled(connected)
        self.stop_publish_btn.setEnabled(connected)
        self.sync_btn.setEnabled(connected)
        self.move_selected_btn.setEnabled(connected)
        self.left_end_effector.setEnabled(not connected)
        self.right_end_effector.setEnabled(not connected)
        for button in (
            self.reset_left_btn,
            self.reset_right_btn,
            self.reset_both_btn,
            self.reset_head_btn,
            self.select_left_btn,
            self.select_right_btn,
            self.select_left_gripper_btn,
            self.select_right_gripper_btn,
            self.select_head_btn,
            self.clear_selection_btn,
        ):
            button.setEnabled(connected)

    @Slot(str)
    def set_status(self, text: str) -> None:
        self.status_label.setText(text)
        self.append_log(text)

    @Slot(str)
    def show_error(self, text: str) -> None:
        self.append_log(f"错误：{text}")
        QMessageBox.warning(self, "关节点动异常", text)

    @Slot(object)
    def update_feedback(self, positions: np.ndarray) -> None:
        for index, row in enumerate(self.joint_rows):
            row.set_current(float(positions[index]))

    def sync_targets(self) -> None:
        self.session.sync_target_to_feedback()
        for row in self.joint_rows:
            row.sync_target()

    def target_array(self) -> np.ndarray:
        target = self.session.last_feedback.copy()
        for row in self.joint_rows:
            target[row.index] = row.raw_target()
        return target

    def selected_indices(self) -> set[int]:
        return {row.index for row in self.joint_rows if row.enable.isChecked()}

    def move_selected(self) -> None:
        indices = self.selected_indices()
        if not indices:
            QMessageBox.information(self, "未勾选关节", "请先勾选要移动的关节。")
            return
        self.session.move_to_target(self.target_array(), indices)

    def set_reset_targets(self, side: str) -> None:
        if side in {"left", "both"}:
            for i, value in enumerate(RESET_LEFT_ARM_DEG):
                self.joint_rows[i].target.setValue(float(value))
                self.joint_rows[i].enable.setChecked(True)
        if side in {"right", "both"}:
            for i, value in enumerate(RESET_RIGHT_ARM_DEG, start=7):
                self.joint_rows[i].target.setValue(float(value))
                self.joint_rows[i].enable.setChecked(True)
        if side == "head":
            for i in sorted(HEAD_JOINT_INDICES):
                self.joint_rows[i].target.setValue(0.0)
                self.joint_rows[i].enable.setChecked(True)

    def select_arm(self, side: str) -> None:
        for row in self.joint_rows:
            checked = (
                (side == "left" and row.index < 7)
                or (side == "right" and 7 <= row.index < 14)
                or (side == "head" and row.index in HEAD_JOINT_INDICES)
            )
            row.enable.setChecked(checked)
        if side == "none":
            for row in self.joint_rows:
                row.enable.setChecked(False)

    def select_end_effector(self, side: str) -> None:
        end_effector_index_by_side = {"left": 14, "right": 15}
        if side not in end_effector_index_by_side and side != "both":
            raise ValueError(f"Unknown end effector side: {side}")
        for row in self.joint_rows:
            checked = (
                (side == "left" and row.index == end_effector_index_by_side["left"])
                or (side == "right" and row.index == end_effector_index_by_side["right"])
                or (side == "both" and row.index in end_effector_index_by_side.values())
            )
            row.enable.setChecked(checked)

    def append_log(self, text: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.log.append(f"[{timestamp}] {text}")
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    @Slot(str)
    def update_visualization_url(self, url: str) -> None:
        self.visualization_url = url
        self.open_visualization_btn.setEnabled(bool(url))
        if not url:
            self.visualization_url_label.setText("未启动")
            if self.visualization_view is not None:
                self.visualization_view.setHtml(_embedded_viser_placeholder_html())
            return

        self.visualization_url_label.setText(f'<a href="{url}">{url}</a>')
        if self.visualization_view is not None:
            self.visualization_view.load(QUrl(url))

    def open_visualization_browser(self) -> None:
        if self.visualization_url:
            webbrowser.open(self.visualization_url)

    def closeEvent(self, event) -> None:  # noqa: N802
        self.session.stop()
        super().closeEvent(event)


APP_STYLESHEET = """
QMainWindow, QWidget {
    background: #f5f7fb;
    color: #172033;
    font-size: 13px;
}
#Title {
    font-size: 24px;
    font-weight: 800;
    color: #101827;
}
#Subtitle, #Hint {
    color: #526174;
}
QGroupBox {
    background: #ffffff;
    border: 1px solid #d7e1ec;
    border-radius: 10px;
    margin-top: 12px;
    padding: 12px;
    font-weight: 700;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 8px;
    background: #ffffff;
    color: #2d3a4e;
}
QLineEdit, QDoubleSpinBox, QSpinBox, QTextEdit {
    background: #ffffff;
    color: #172033;
    border: 1px solid #cbd7e4;
    border-radius: 8px;
    padding: 7px 9px;
    min-height: 22px;
    selection-background-color: #2f6fed;
    selection-color: #ffffff;
}
QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
    width: 0px;
    border: none;
}
QSpinBox::up-arrow, QSpinBox::down-arrow, QDoubleSpinBox::up-arrow, QDoubleSpinBox::down-arrow {
    width: 0px;
    height: 0px;
}
QPushButton {
    background: #ffffff;
    border: 1px solid #c8d5e2;
    border-radius: 8px;
    padding: 7px 12px;
    color: #1e2a3d;
    font-weight: 650;
    min-height: 24px;
}
QPushButton:hover {
    background: #f3f7fb;
    border-color: #9fb1c7;
}
QPushButton:disabled {
    color: #9aa8ba;
    background: #edf2f7;
    border-color: #dbe4ee;
}
#PrimaryButton {
    background: #2f6fed;
    color: #ffffff;
    border-color: #2f6fed;
}
#DangerButton {
    background: #d94949;
    color: #ffffff;
    border-color: #d94949;
}
#StatusPill {
    background: #e8f0ff;
    color: #1d4ed8;
    border: 1px solid #b8cdfb;
    border-radius: 8px;
    padding: 10px 12px;
    font-weight: 700;
}
#Log {
    background: #111a27;
    color: #d8e7f3;
    border: 1px solid #223147;
    border-radius: 8px;
}
#ViserPanel {
    background: #f8fafc;
    border: 1px solid #d7e1ec;
    border-radius: 8px;
}
#PanelTitle {
    color: #253247;
    font-weight: 800;
}
QScrollArea {
    background: transparent;
    border: none;
}
QScrollArea > QWidget > QWidget {
    background: transparent;
}
QScrollBar:vertical {
    background: transparent;
    width: 10px;
    margin: 2px;
}
QScrollBar::handle:vertical {
    background: #bfccd9;
    border-radius: 5px;
    min-height: 28px;
}
QScrollBar::handle:vertical:hover {
    background: #9dafc2;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
QScrollBar:horizontal {
    background: transparent;
    height: 10px;
    margin: 2px;
}
QScrollBar::handle:horizontal {
    background: #bfccd9;
    border-radius: 5px;
    min-width: 28px;
}
QScrollBar::handle:horizontal:hover {
    background: #9dafc2;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}
QCheckBox {
    spacing: 8px;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border-radius: 5px;
    border: 1px solid #a8b8ca;
    background: #ffffff;
}
QCheckBox::indicator:checked {
    background: #2f6fed;
    border-color: #2f6fed;
}
QMessageBox {
    background: #f7f9fc;
    color: #172033;
}
QMessageBox QLabel {
    color: #172033;
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lcm-url", default=DEFAULT_LCM_URL)
    parser.add_argument("--urdf-path", type=Path, default=DEFAULT_URDF_PATH)
    parser.add_argument(
        "--left-end-effector",
        choices=WHEELED_ARM_END_EFFECTOR_TYPES,
        default=WHEELED_ARM_DEFAULT_LEFT_END_EFFECTOR,
    )
    parser.add_argument(
        "--right-end-effector",
        choices=WHEELED_ARM_END_EFFECTOR_TYPES,
        default=WHEELED_ARM_DEFAULT_RIGHT_END_EFFECTOR,
    )
    parser.add_argument("--visualize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--visualization-host", default="0.0.0.0")
    parser.add_argument("--visualization-port", type=int, default=8092)
    parser.add_argument("--open-browser", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--command-hz", type=float, default=DEFAULT_COMMAND_HZ)
    parser.add_argument("--max-speed-deg-s", type=float, default=DEFAULT_MAX_SPEED_DEG_S)
    parser.add_argument("--max-acceleration-deg-s2", type=float, default=DEFAULT_MAX_ACCEL_DEG_S2)
    return parser.parse_args()


def main() -> None:
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    args = parse_args()
    app = QApplication(sys.argv)
    app.setApplicationName("LeRobot Wheeled Arm Joint Jog")
    window = JointJogWindow(args)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
