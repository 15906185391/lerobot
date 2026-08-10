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

"""PySide6 desktop launcher for wheeled_arm + wheeled_arm_pico workflows."""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
import threading
from ctypes.util import find_library
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.utils.constants import HF_LEROBOT_HOME


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
    os.execvpe(sys.executable, [sys.executable, "-m", "lerobot.scripts.wheeled_arm_gui", *sys.argv[1:]], env)


def _prepare_linux_qt_platform() -> None:
    """Avoid Qt's hard abort when the xcb platform plugin lacks system libraries."""
    if sys.platform != "linux":
        return

    requested_platform = os.environ.get("QT_QPA_PLATFORM", "").strip()
    if requested_platform and requested_platform != "xcb":
        return

    # Native Wayland sessions can avoid the xcb plugin entirely. Keep an explicit
    # user choice untouched, but pick Wayland automatically when it is clearly available.
    if not requested_platform and os.environ.get("WAYLAND_DISPLAY"):
        os.environ["QT_QPA_PLATFORM"] = "wayland"
        return

    xcb_cursor_library = _find_xcb_cursor_library()
    if xcb_cursor_library is not None:
        _ensure_qt_library_path(xcb_cursor_library)
        return

    raise SystemExit(
        "启动 PySide6 GUI 需要 Ubuntu 的 Qt/xcb 系统库，但当前缺少 libxcb-cursor0。\n\n"
        f"请先执行：\n  {QT_XCB_INSTALL_HINT}\n\n"
        "然后重新运行：\n  lerobot-wheeled-arm-gui\n\n"
        "如果你在 Wayland 桌面中，也可以临时尝试：\n"
        "  QT_QPA_PLATFORM=wayland lerobot-wheeled-arm-gui"
    )


_prepare_linux_qt_platform()

try:
    from PySide6.QtCore import QObject, QRect, QSettings, QSize, Qt, QTimer, Signal, Slot
    from PySide6.QtGui import QAction, QBrush, QClipboard, QColor, QFont, QIcon, QImage, QPainter, QPen, QPixmap
    from PySide6.QtWidgets import (
        QApplication,
        QButtonGroup,
        QCheckBox,
        QComboBox,
        QDialog,
        QFileDialog,
        QFormLayout,
        QFrame,
        QGraphicsDropShadowEffect,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QPlainTextEdit,
        QDoubleSpinBox,
        QScrollArea,
        QSizePolicy,
        QSpinBox,
        QStackedWidget,
        QTextBrowser,
        QTabWidget,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:  # pragma: no cover - depends on optional GUI extra.
    raise SystemExit(
        "lerobot-wheeled-arm-gui requires PySide6.\n"
        "Install it with: python -m pip install -e '.[gui]'\n"
        "Or run: bash scripts/setup_wheeled_arm_pico_conda.bash --env xr"
    ) from exc


def _iter_path_with_parents(path: Path) -> list[Path]:
    resolved = path.expanduser().resolve()
    return [resolved, *resolved.parents]


def _find_project_root() -> Path:
    env_root = os.environ.get("LEROBOT_PROJECT_ROOT")
    search_roots: list[Path] = []
    if env_root:
        search_roots.extend(_iter_path_with_parents(Path(env_root)))
    search_roots.extend(_iter_path_with_parents(Path(__file__)))
    search_roots.extend(_iter_path_with_parents(Path.cwd()))

    common_root = Path.home() / "Documents/lerobot"
    if common_root.exists():
        search_roots.extend(_iter_path_with_parents(common_root))

    seen: set[Path] = set()
    for root in search_roots:
        if root in seen:
            continue
        seen.add(root)
        if (root / "pyproject.toml").exists() and (root / "src/lerobot").exists():
            return root
        if (root / "GUI_reference/Any4LeRobotGUI/backend").exists():
            return root

    return Path(__file__).resolve().parents[3]


PROJECT_ROOT = _find_project_root()
EDIT_OPERATION_LABELS = {
    "查看信息": "info",
    "删除 Episode": "delete_episodes",
    "拆分数据集": "split",
    "合并数据集": "merge",
    "删除 Feature": "remove_feature",
    "修改任务文本": "modify_tasks",
    "图片转视频": "convert_image_to_video",
    "重算统计": "recompute_stats",
    "重编码视频": "reencode_videos",
}
EDIT_OPERATION_ORDER = list(EDIT_OPERATION_LABELS)
ANY4LEROBOT_BACKEND = PROJECT_ROOT / "GUI_reference/Any4LeRobotGUI/backend"
CONVERSION_LABELS = {
    "OpenX → LeRobot": "openx_to_lerobot",
    "AgiBot → LeRobot": "agibot_to_lerobot",
    "RoboMIND → LeRobot": "robomind_to_lerobot",
    "LIBERO → LeRobot": "libero_to_lerobot",
    "LeRobot → RLDS": "lerobot_to_rlds",
    "LeRobot v1.6 → v2.0": "v16_to_v20",
    "LeRobot v2.0 → v2.1": "v20_to_v21",
    "LeRobot v2.1 → v2.0": "v21_to_v20",
    "LeRobot v2.1 → v3.0": "v21_to_v30",
    "LeRobot v3.0 → v2.1": "v30_to_v21",
}
CONVERSION_ORDER = list(CONVERSION_LABELS)
CONVERSION_SCRIPTS = {
    "openx_to_lerobot": "openx2lerobot/openx_rlds.py",
    "agibot_to_lerobot": "agibot2lerobot/agibot_h5.py",
    "robomind_to_lerobot": "robomind2lerobot/robomind_h5.py",
    "libero_to_lerobot": "libero2lerobot/libero_h5.py",
    "lerobot_to_rlds": "lerobot2rlds/lerobot2rlds.py",
    "v16_to_v20": "ds_version_convert/v16_to_v20/convert_dataset_v16_to_v20.py",
    "v20_to_v21": "ds_version_convert/v20_to_v21/convert_dataset_v20_to_v21.py",
    "v21_to_v20": "ds_version_convert/v21_to_v20/convert_dataset_v21_to_v20.py",
    "v21_to_v30": PROJECT_ROOT / "src/lerobot/scripts/convert_dataset_v21_to_v30.py",
    "v30_to_v21": "ds_version_convert/v30_to_v21/convert_dataset_v30_to_v21.py",
}
CONVERSION_STACK_INDEX = {
    "openx_to_lerobot": 0,
    "agibot_to_lerobot": 1,
    "robomind_to_lerobot": 2,
    "libero_to_lerobot": 3,
    "lerobot_to_rlds": 4,
    "v16_to_v20": 5,
    "v20_to_v21": 6,
    "v21_to_v20": 6,
    "v21_to_v30": 7,
    "v30_to_v21": 8,
}


def _is_conversion_backend(path: str | Path) -> bool:
    backend = Path(path).expanduser()
    return all(
        script.exists() if isinstance(script, Path) else (backend / script).exists()
        for script in CONVERSION_SCRIPTS.values()
    )


COMMON_COMMAND_LABELS = {
    "系统信息": "info",
    "查找相机": "find_cameras",
    "查找串口": "find_port",
    "遥操作": "teleoperate",
    "回放 Episode": "replay",
    "校准设备": "calibrate",
    "设置电机": "setup_motors",
    "查关节限位": "find_joint_limits",
    "设置/测试 CAN": "setup_can",
    "训练策略": "train",
    "评估策略": "eval",
    "策略 Rollout": "rollout",
    "数据标注": "annotate",
    "图像增强预览": "imgtransform_viz",
    "补充分位数统计": "augment_quantile_stats",
    "转换 DCP Checkpoint": "convert_dcp",
    "训练 FAST Tokenizer": "train_tokenizer",
    "自定义脚本": "custom",
}
COMMON_COMMAND_ORDER = list(COMMON_COMMAND_LABELS)
COMMON_SCRIPT_MODULES = {
    "info": "lerobot.scripts.lerobot_info",
    "find_cameras": "lerobot.scripts.lerobot_find_cameras",
    "find_port": "lerobot.scripts.lerobot_find_port",
    "teleoperate": "lerobot.scripts.lerobot_teleoperate",
    "replay": "lerobot.scripts.lerobot_replay",
    "calibrate": "lerobot.scripts.lerobot_calibrate",
    "setup_motors": "lerobot.scripts.lerobot_setup_motors",
    "find_joint_limits": "lerobot.scripts.lerobot_find_joint_limits",
    "setup_can": "lerobot.scripts.lerobot_setup_can",
    "train": "lerobot.scripts.lerobot_train",
    "eval": "lerobot.scripts.lerobot_eval",
    "rollout": "lerobot.scripts.lerobot_rollout",
    "annotate": "lerobot.scripts.lerobot_annotate",
    "imgtransform_viz": "lerobot.scripts.lerobot_imgtransform_viz",
    "augment_quantile_stats": "lerobot.scripts.augment_dataset_quantile_stats",
    "convert_dcp": "lerobot.scripts.lerobot_convert_dcp",
    "train_tokenizer": "lerobot.scripts.lerobot_train_tokenizer",
}
DEFAULT_WHEELED_ARM_URDF = (
    PROJECT_ROOT
    / "src/lerobot/teleoperators/wheeled_arm_pico/assets/wheeled_robot_sim/urdf/real_robot.urdf"
)


def _bool_arg(value: bool) -> str:
    return "true" if value else "false"


def _settings_bool(value, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "on"}


def _module_command(module: str) -> list[str]:
    return [sys.executable, "-m", module]


def _format_command(command: list[str]) -> str:
    return shlex.join(command)


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _list_arg(value: str, item_type: type = str) -> str | None:
    value = value.strip()
    if not value:
        return None
    if value.startswith("["):
        return value
    items = _split_csv(value)
    if item_type is int:
        return json.dumps([int(item) for item in items])
    return json.dumps(items, ensure_ascii=False)


def _json_or_none(value: str) -> str | None:
    value = value.strip()
    if not value:
        return None
    json.loads(value)
    return value


def _readable_exit(code: int) -> str:
    if code == 0:
        return "已正常结束"
    if code < 0:
        return f"收到信号 {-code} 后结束"
    return f"退出码 {code}"


def _meta_info_path(root: Path) -> Path:
    return root / "meta" / "info.json"


def _read_dataset_counts(root: Path) -> tuple[int, int] | None:
    info_path = _meta_info_path(root)
    if not info_path.exists():
        return None
    try:
        with info_path.open("r", encoding="utf-8") as f:
            info = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return int(info.get("total_episodes", 0)), int(info.get("total_frames", 0))


def _dataset_root(repo_id: str, root: str | None) -> Path:
    if root:
        return Path(root).expanduser()
    return HF_LEROBOT_HOME / repo_id


def describe_local_dataset(repo_id: str, root: str | None) -> tuple[Path, int, int] | None:
    repo_id = repo_id.strip()
    if not repo_id:
        return None
    dataset_root = _dataset_root(repo_id, root)
    counts = _read_dataset_counts(dataset_root)
    if counts is None:
        return None
    total_episodes, total_frames = counts
    return dataset_root, total_episodes, total_frames


def find_latest_local_dataset(repo_id: str, root: str | None, no_stamp: bool, resume: bool) -> str | None:
    """Return the most likely repo_id to visualize after a recording run."""
    repo_id = repo_id.strip()
    if not repo_id:
        return None

    if root:
        root_path = Path(root).expanduser()
        root_counts = _read_dataset_counts(root_path)
        if root_counts is not None and root_counts[0] > 0:
            return repo_id
        return None

    exact = HF_LEROBOT_HOME / repo_id
    exact_counts = _read_dataset_counts(exact)
    if (no_stamp or resume) and exact_counts is not None and exact_counts[0] > 0:
        return repo_id

    if "/" not in repo_id:
        candidates_root = HF_LEROBOT_HOME
        prefix = f"{repo_id}_"
    else:
        namespace, name = repo_id.split("/", 1)
        candidates_root = HF_LEROBOT_HOME / namespace
        prefix = f"{name}_"

    if not candidates_root.exists():
        return repo_id if exact_counts is not None and exact_counts[0] > 0 else None

    candidates = [
        path
        for path in candidates_root.iterdir()
        if path.is_dir() and path.name.startswith(prefix) and _meta_info_path(path).exists()
    ]
    if not candidates:
        return repo_id if exact_counts is not None and exact_counts[0] > 0 else None

    non_empty_candidates = [
        path for path in candidates if (counts := _read_dataset_counts(path)) is not None and counts[0] > 0
    ]
    if not non_empty_candidates:
        return repo_id if exact_counts is not None and exact_counts[0] > 0 else None

    newest = max(non_empty_candidates, key=lambda path: path.stat().st_mtime)
    if "/" not in repo_id:
        return newest.name
    namespace = repo_id.split("/", 1)[0]
    return f"{namespace}/{newest.name}"


@dataclass
class ProcessHandle:
    process: subprocess.Popen
    command: list[str]


class ProcessRunner(QObject):
    output = Signal(str)
    started = Signal(str)
    finished = Signal(int)
    failed = Signal(str)

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name
        self._handle: ProcessHandle | None = None
        self._wait_thread: threading.Thread | None = None

    @property
    def is_running(self) -> bool:
        return self._handle is not None and self._handle.process.poll() is None

    def start(self, command: list[str], cwd: Path = PROJECT_ROOT) -> bool:
        if self.is_running:
            self.failed.emit(f"{self.name} 已在运行。")
            return False

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        kwargs: dict[str, object] = {
            "cwd": str(cwd),
            "env": env,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "stdin": subprocess.PIPE,
            "text": True,
            "bufsize": 1,
        }
        if os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            kwargs["start_new_session"] = True

        try:
            process = subprocess.Popen(command, **kwargs)
        except OSError as exc:
            self.failed.emit(f"无法启动 {self.name}: {exc}")
            return False

        self._handle = ProcessHandle(process=process, command=command)
        self.started.emit(_format_command(command))
        self._wait_thread = threading.Thread(target=self._pump_output, name=f"{self.name}-runner", daemon=True)
        self._wait_thread.start()
        return True

    def interrupt(self) -> None:
        if not self.is_running or self._handle is None:
            return

        process = self._handle.process
        try:
            if os.name == "nt":
                process.send_signal(getattr(signal, "CTRL_BREAK_EVENT", signal.SIGTERM))
            else:
                os.killpg(process.pid, signal.SIGINT)
        except ProcessLookupError:
            return
        except OSError as exc:
            self.failed.emit(f"发送停止信号失败: {exc}")

    def kill(self) -> None:
        if not self.is_running or self._handle is None:
            return
        try:
            self._handle.process.kill()
        except OSError as exc:
            self.failed.emit(f"强制结束失败: {exc}")

    def write_stdin(self, text: str) -> None:
        if not self.is_running or self._handle is None or self._handle.process.stdin is None:
            self.failed.emit(f"{self.name} 没有可写入的输入流。")
            return
        try:
            self._handle.process.stdin.write(text)
            self._handle.process.stdin.flush()
        except OSError as exc:
            self.failed.emit(f"发送输入失败: {exc}")

    def _pump_output(self) -> None:
        assert self._handle is not None
        process = self._handle.process
        if process.stdout is not None:
            for line in process.stdout:
                self.output.emit(line.rstrip("\n"))
        code = process.wait()
        self._handle = None
        self.finished.emit(code)


class DatasetImageLabel(QLabel):
    """自适应缩放的数据集图像预览标签。"""

    def __init__(self, title: str) -> None:
        super().__init__()
        self._title = title
        self._pixmap = QPixmap()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(220, 160)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setObjectName("DatasetImageLabel")
        self.set_placeholder("等待加载")

    def set_placeholder(self, text: str | None = None) -> None:
        self._pixmap = QPixmap()
        self.setPixmap(QPixmap())
        self.setText(f"{self._title}\n{text or '无图像'}")

    def set_preview_pixmap(self, pixmap: QPixmap) -> None:
        self._pixmap = pixmap
        self.setText("")
        self._update_scaled_pixmap()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._update_scaled_pixmap()

    def _update_scaled_pixmap(self) -> None:
        if self._pixmap.isNull():
            return
        self.setPixmap(
            self._pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )


class FrameRangeSlider(QFrame):
    """帧定位与区间选择条。"""

    rangeChanged = Signal(int, int)
    frameChanged = Signal(int)
    sliderPressed = Signal()
    sliderReleased = Signal(int)

    def __init__(self) -> None:
        super().__init__()
        self.total_frames = 0
        self.current_frame = 0
        self.sel_start = 0
        self.sel_end = 0
        self._drag_mode: str | None = None
        self._drag_anchor = 0
        self.setMinimumHeight(82)
        self.setObjectName("FrameRangeSlider")

    def set_total_frames(self, total_frames: int) -> None:
        self.total_frames = max(0, total_frames)
        max_index = max(0, self.total_frames - 1)
        self.current_frame = min(self.current_frame, max_index)
        self.sel_start = min(self.sel_start, max_index)
        self.sel_end = min(self.sel_end, max_index)
        self.update()

    def set_current_frame(self, frame_index: int) -> None:
        self.current_frame = max(0, min(frame_index, max(0, self.total_frames - 1)))
        self.update()

    def reset_selection(self) -> None:
        self.sel_start = 0
        self.sel_end = max(0, self.total_frames - 1)
        self.rangeChanged.emit(*self.selected_range())
        self.update()

    def selected_range(self) -> tuple[int, int]:
        return min(self.sel_start, self.sel_end), max(self.sel_start, self.sel_end)

    def _event_x(self, event) -> int:
        if hasattr(event, "position"):
            return int(event.position().x())
        return int(event.x())

    def frame_from_x(self, x: int) -> int:
        if self.total_frames <= 1:
            return 0
        left = 18
        right = max(left + 1, self.width() - 18)
        x = max(left, min(x, right))
        return round((x - left) / (right - left) * (self.total_frames - 1))

    def x_from_frame(self, frame: int) -> int:
        if self.total_frames <= 1:
            return 18
        left = 18
        right = max(left + 1, self.width() - 18)
        return round(left + frame / (self.total_frames - 1) * (right - left))

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton or self.total_frames <= 0:
            return
        frame = self.frame_from_x(self._event_x(event))
        self.sliderPressed.emit()
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            self._drag_mode = "select"
            self._drag_anchor = frame
            self.sel_start = frame
            self.sel_end = frame
            self.rangeChanged.emit(*self.selected_range())
        else:
            self._drag_mode = "scrub"
            self.current_frame = frame
            self.frameChanged.emit(frame)
        self.update()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag_mode is None or self.total_frames <= 0:
            return
        frame = self.frame_from_x(self._event_x(event))
        if self._drag_mode == "select":
            self.sel_end = frame
            self.rangeChanged.emit(*self.selected_range())
        else:
            self.current_frame = frame
            self.frameChanged.emit(frame)
        self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton or self._drag_mode is None:
            return
        frame = self.frame_from_x(self._event_x(event))
        if self._drag_mode == "select":
            self.sel_end = frame
            self.rangeChanged.emit(*self.selected_range())
        else:
            self.current_frame = frame
            self.frameChanged.emit(frame)
            self.sliderReleased.emit(frame)
        self._drag_mode = None
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        left = 18
        right = max(left + 1, self.width() - 18)
        bar_y = 34
        bar_h = 10
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#d7dee9"))
        painter.drawRoundedRect(left, bar_y, right - left, bar_h, 5, 5)

        if self.total_frames > 0:
            start, end = self.selected_range()
            sel_x1 = self.x_from_frame(start)
            sel_x2 = self.x_from_frame(end)
            painter.setBrush(QColor("#93c5fd"))
            painter.drawRoundedRect(min(sel_x1, sel_x2), bar_y, max(6, abs(sel_x2 - sel_x1)), bar_h, 5, 5)

            current_x = self.x_from_frame(self.current_frame)
            painter.setPen(QPen(QColor("#dc2626"), 2))
            painter.drawLine(current_x, 17, current_x, 57)

            for frame in (start, end):
                x = self.x_from_frame(frame)
                painter.setPen(QPen(QColor("#2563eb"), 2))
                painter.setBrush(QBrush(QColor("#ffffff")))
                painter.drawEllipse(x - 5, bar_y - 5, 10, 20)

        painter.setPen(QColor("#4b5563"))
        painter.drawText(
            QRect(8, 5, max(10, self.width() - 16), 20),
            Qt.AlignmentFlag.AlignCenter,
            "拖动定位帧    Shift+拖动选择区间",
        )
        if self.total_frames > 0:
            start, end = self.selected_range()
            text = f"当前帧 {self.current_frame} / {self.total_frames - 1}    选择区间 {start} - {end}"
        else:
            text = "无可用帧"
        painter.drawText(
            QRect(8, 58, max(10, self.width() - 16), 18),
            Qt.AlignmentFlag.AlignCenter,
            text,
        )


def _frame_value_to_pixmap(value: Any) -> QPixmap:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if getattr(value, "ndim", 0) == 3 and value.shape[0] in (1, 3, 4):
            value = value.permute(1, 2, 0)
        value = value.numpy()
    elif hasattr(value, "convert") and hasattr(value, "tobytes") and hasattr(value, "size"):
        image = value.convert("RGB")
        width, height = image.size
        qimage = QImage(image.tobytes(), width, height, width * 3, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimage.copy())

    array = np.asarray(value)
    if array.ndim == 4:
        array = array[0]
    if array.ndim == 3 and array.shape[0] in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    if array.ndim != 3:
        raise ValueError(f"无法显示形状为 {array.shape} 的图像数据。")
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    elif array.shape[2] > 3:
        array = array[:, :, :3]

    if np.issubdtype(array.dtype, np.floating):
        max_value = float(np.nanmax(array)) if array.size else 1.0
        if max_value <= 1.0:
            array = array * 255.0
    array = np.nan_to_num(array, copy=False)
    array = np.clip(array, 0, 255).astype(np.uint8, copy=False)
    array = np.ascontiguousarray(array)
    height, width, channels = array.shape
    qimage = QImage(array.data, width, height, width * channels, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimage.copy())


class LeRobotDatasetPreview(QWidget):
    deleteEpisodeRequested = Signal(int)

    def __init__(self) -> None:
        super().__init__()
        self.repo_id = ""
        self.root_text = ""
        self.dataset_root: Path | None = None
        self.dataset = None
        self.total_episodes = 0
        self.total_dataset_frames = 0
        self.current_episode = 0
        self.current_frame = 0
        self.camera_keys: list[str] = []
        self.is_playing = False
        self.play_selection_only = False
        self.was_playing_before_drag = False

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.play_next_frame)
        self._build_ui()
        self._set_enabled(False)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 14, 12, 12)
        layout.setSpacing(10)

        top_row = QHBoxLayout()
        self.load_btn = QPushButton("加载预览")
        self.load_btn.setObjectName("PrimaryButton")
        self.prev_episode_btn = QPushButton("上一集")
        self.next_episode_btn = QPushButton("下一集")
        self.delete_episode_btn = QPushButton("填入删除当前 Episode")
        self.delete_episode_btn.setObjectName("DangerButton")
        self.load_btn.clicked.connect(self.load_preview)
        self.prev_episode_btn.clicked.connect(self.prev_episode)
        self.next_episode_btn.clicked.connect(self.next_episode)
        self.delete_episode_btn.clicked.connect(self.request_delete_current_episode)
        top_row.addWidget(self.load_btn)
        top_row.addWidget(self.prev_episode_btn)
        top_row.addWidget(self.next_episode_btn)
        top_row.addWidget(self.delete_episode_btn)
        top_row.addStretch(1)
        layout.addLayout(top_row)

        self.source_label = QLabel("未选择数据集")
        self.source_label.setWordWrap(True)
        self.source_label.setObjectName("MutedLabel")
        self.episode_label = QLabel("Episode：-")
        self.episode_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.episode_label.setObjectName("PreviewEpisodeLabel")
        self.info_label = QLabel("填写输入 Repo ID 后点击加载预览。")
        self.info_label.setWordWrap(True)
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_label.setObjectName("MutedLabel")
        layout.addWidget(self.source_label)
        layout.addWidget(self.episode_label)
        layout.addWidget(self.info_label)

        self.image_grid = QGridLayout()
        self.image_grid.setSpacing(8)
        self.image_labels = [DatasetImageLabel(f"相机 {index + 1}") for index in range(4)]
        for index, label in enumerate(self.image_labels):
            self.image_grid.addWidget(label, index // 2, index % 2)
        layout.addLayout(self.image_grid, 1)

        self.slider = FrameRangeSlider()
        self.slider.frameChanged.connect(self.show_frame)
        self.slider.sliderPressed.connect(self.pause_for_drag)
        self.slider.sliderReleased.connect(self.restore_after_drag)
        self.slider.rangeChanged.connect(self.update_range_info)
        layout.addWidget(self.slider)

        controls = QHBoxLayout()
        self.play_btn = QPushButton("播放")
        self.play_all_btn = QPushButton("整段循环")
        self.play_selection_btn = QPushButton("仅播区间")
        self.reset_range_btn = QPushButton("重置区间")
        self.trim_hint_btn = QPushButton("裁剪区间")
        self.play_btn.clicked.connect(self.toggle_play)
        self.play_all_btn.clicked.connect(self.set_play_all)
        self.play_selection_btn.clicked.connect(self.set_play_selection)
        self.reset_range_btn.clicked.connect(self.reset_range)
        self.trim_hint_btn.clicked.connect(self.show_trim_hint)
        controls.addStretch(1)
        controls.addWidget(self.play_btn)
        controls.addWidget(self.play_all_btn)
        controls.addWidget(self.play_selection_btn)
        controls.addWidget(self.reset_range_btn)
        controls.addWidget(self.trim_hint_btn)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.range_info_label = QLabel("未选择区间")
        self.range_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.range_info_label.setObjectName("MutedLabel")
        layout.addWidget(self.range_info_label)

    def set_source(self, repo_id: str, root_text: str) -> None:
        repo_id = repo_id.strip()
        root_text = root_text.strip()
        if repo_id == self.repo_id and root_text == self.root_text:
            return
        self.repo_id = repo_id
        self.root_text = root_text
        self.dataset = None
        self.total_episodes = 0
        self.total_dataset_frames = 0
        self.current_episode = 0
        self.current_frame = 0
        self.camera_keys = []
        self.timer.stop()
        self.is_playing = False
        self.update_play_button_text()
        self._set_enabled(False)

        if not self.repo_id:
            self.dataset_root = None
            self.source_label.setText("未选择数据集")
            self.clear_images("等待加载")
            self.info_label.setText("填写输入 Repo ID 后点击加载预览。")
            return

        dataset_info = describe_local_dataset(self.repo_id, self.root_text or None)
        if dataset_info is None:
            self.dataset_root = _dataset_root(self.repo_id, self.root_text or None)
            self.source_label.setText(f"本地路径：{self.dataset_root}")
            self.clear_images("未找到本地数据")
            self.info_label.setText("没有找到 meta/info.json，请检查 Repo ID 或 root。")
            return

        self.dataset_root, self.total_episodes, self.total_dataset_frames = dataset_info
        self.source_label.setText(f"本地路径：{self.dataset_root}")
        if self.total_episodes <= 0:
            self.clear_images("空数据集")
            self.info_label.setText("该数据集当前没有 episode。")
            return
        self._set_enabled(True, loaded=False)
        self.clear_images("点击加载预览")
        self.info_label.setText(
            f"共 {self.total_episodes} 个 episode，{self.total_dataset_frames} 帧。点击加载预览。"
        )

    def load_preview(self) -> None:
        if not self.repo_id:
            self.info_label.setText("请先填写输入 Repo ID。")
            return
        dataset_info = describe_local_dataset(self.repo_id, self.root_text or None)
        if dataset_info is None:
            self.info_label.setText("没有找到本地数据集，无法预览。")
            return
        self.dataset_root, self.total_episodes, self.total_dataset_frames = dataset_info
        if self.total_episodes <= 0:
            self.info_label.setText("该数据集当前没有 episode。")
            self._set_enabled(False)
            return

        self.current_episode = max(0, min(self.current_episode, self.total_episodes - 1))
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            self.dataset = LeRobotDataset(
                self.repo_id,
                root=self.dataset_root,
                episodes=[self.current_episode],
                return_uint8=True,
                token=False,
            )
        except Exception as exc:  # noqa: BLE001 - GUI must surface optional decoder/import errors.
            self.dataset = None
            self.clear_images("加载失败")
            self.info_label.setText(f"加载失败：{exc}")
            return

        self.camera_keys = list(self.dataset.meta.camera_keys)
        fps = max(1, int(getattr(self.dataset, "fps", 30)))
        self.timer.setInterval(max(15, int(1000 / fps)))
        frame_count = len(self.dataset)
        self.current_frame = 0
        self.slider.current_frame = 0
        self.slider.sel_start = 0
        self.slider.sel_end = max(0, frame_count - 1)
        self.slider.set_total_frames(frame_count)
        self._set_enabled(True, loaded=frame_count > 0)
        self.update_episode_info()
        self.update_range_info()
        self.show_frame(0)

    def clear_images(self, message: str) -> None:
        for label in self.image_labels:
            label.set_placeholder(message)
        self.episode_label.setText("Episode：-")
        self.range_info_label.setText("未选择区间")
        self.slider.set_total_frames(0)

    def update_episode_info(self) -> None:
        frame_count = len(self.dataset) if self.dataset is not None else 0
        cameras = "、".join(self.camera_keys[:4]) if self.camera_keys else "无图像键"
        hidden = "" if len(self.camera_keys) <= 4 else f"；另有 {len(self.camera_keys) - 4} 路未显示"
        self.episode_label.setText(
            f"Episode {self.current_episode} / {self.total_episodes - 1}    {frame_count} 帧"
        )
        self.info_label.setText(f"相机：{cameras}{hidden}")
        self.prev_episode_btn.setEnabled(self.current_episode > 0)
        self.next_episode_btn.setEnabled(self.current_episode < self.total_episodes - 1)

    def _set_enabled(self, enabled: bool, loaded: bool = False) -> None:
        self.load_btn.setEnabled(bool(self.repo_id))
        for button in (
            self.prev_episode_btn,
            self.next_episode_btn,
            self.delete_episode_btn,
            self.play_btn,
            self.play_all_btn,
            self.play_selection_btn,
            self.reset_range_btn,
            self.trim_hint_btn,
        ):
            button.setEnabled(enabled and loaded)
        if not enabled or not loaded:
            self.prev_episode_btn.setEnabled(False)
            self.next_episode_btn.setEnabled(False)
        self.slider.setEnabled(enabled and loaded)

    def prev_episode(self) -> None:
        if self.current_episode <= 0:
            return
        self.current_episode -= 1
        self.load_preview()

    def next_episode(self) -> None:
        if self.current_episode >= self.total_episodes - 1:
            return
        self.current_episode += 1
        self.load_preview()

    def show_frame(self, frame_index: int) -> None:
        if self.dataset is None or len(self.dataset) <= 0:
            return
        self.current_frame = max(0, min(frame_index, len(self.dataset) - 1))
        self.slider.set_current_frame(self.current_frame)
        try:
            frame = self.dataset[self.current_frame]
            for index, label in enumerate(self.image_labels):
                if index >= len(self.camera_keys):
                    label.set_placeholder("无此路相机")
                    continue
                camera_key = self.camera_keys[index]
                label._title = camera_key
                label.set_preview_pixmap(_frame_value_to_pixmap(frame[camera_key]))
        except Exception as exc:  # noqa: BLE001 - show the error in the preview instead of crashing Qt.
            self.pause()
            self.info_label.setText(f"读取第 {self.current_frame} 帧失败：{exc}")
        self.update_range_info()

    def play_next_frame(self) -> None:
        if self.dataset is None or len(self.dataset) <= 0:
            self.pause()
            return
        start, end = self.slider.selected_range()
        if self.play_selection_only and end > start:
            next_frame = self.current_frame + 1
            if next_frame > end or next_frame < start:
                next_frame = start
        else:
            next_frame = (self.current_frame + 1) % len(self.dataset)
        self.show_frame(next_frame)

    def toggle_play(self) -> None:
        if self.is_playing:
            self.pause()
        else:
            self.play()

    def play(self) -> None:
        if self.dataset is None or len(self.dataset) <= 0:
            return
        self.is_playing = True
        self.timer.start()
        self.update_play_button_text()
        self.update_range_info()

    def pause(self) -> None:
        self.is_playing = False
        self.timer.stop()
        self.update_play_button_text()
        self.update_range_info()

    def update_play_button_text(self) -> None:
        self.play_btn.setText("暂停" if self.is_playing else "播放")

    def pause_for_drag(self) -> None:
        self.was_playing_before_drag = self.is_playing
        self.pause()

    def restore_after_drag(self, frame_index: int) -> None:
        self.show_frame(frame_index)
        if self.was_playing_before_drag:
            self.play()

    def set_play_all(self) -> None:
        self.play_selection_only = False
        self.update_range_info()

    def set_play_selection(self) -> None:
        self.play_selection_only = True
        start, _end = self.slider.selected_range()
        if self.current_frame < start:
            self.show_frame(start)
        self.update_range_info()

    def reset_range(self) -> None:
        self.slider.reset_selection()
        self.play_selection_only = False
        self.update_range_info()

    def update_range_info(self, *_args) -> None:
        if self.dataset is None or len(self.dataset) <= 0:
            self.range_info_label.setText("未选择区间")
            return
        start, end = self.slider.selected_range()
        mode = "仅播区间" if self.play_selection_only else "整段循环"
        state = "播放中" if self.is_playing else "已暂停"
        self.range_info_label.setText(
            f"{mode}，{state}；当前帧 {self.current_frame}；选择区间 {start} - {end}"
        )

    def request_delete_current_episode(self) -> None:
        if self.dataset is None:
            return
        self.deleteEpisodeRequested.emit(self.current_episode)

    def show_trim_hint(self) -> None:
        start, end = self.slider.selected_range()
        QMessageBox.information(
            self,
            "暂未执行帧裁剪",
            "当前 LeRobot 编辑脚本没有安全的按帧裁剪接口，GUI 不会直接删除 parquet/video 中的帧。\n\n"
            f"已选区间：Episode {self.current_episode}，帧 {start} - {end}\n\n"
            "如后续需要帧级裁剪，应先在项目中新增专用 dataset operation，再由 GUI 调用。"
        )


class PathPicker(QWidget):
    changed = Signal()

    def __init__(self, placeholder: str = "", directory: bool = True) -> None:
        super().__init__()
        self.directory = directory
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
        if self.directory:
            selected = QFileDialog.getExistingDirectory(self, "选择目录", self.text() or str(Path.home()))
        else:
            selected, _ = QFileDialog.getOpenFileName(self, "选择文件", self.text() or str(Path.home()))
        if selected:
            self.edit.setText(selected)


class HelpDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Wheeled Arm PICO 使用说明")
        self.setMinimumSize(QSize(860, 640))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        title = QLabel("Wheeled Arm PICO 使用说明")
        title.setObjectName("DialogTitle")
        subtitle = QLabel("采集、遥操作、复位、查看和数据处理的现场操作速查。")
        subtitle.setObjectName("MutedLabel")
        subtitle.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(subtitle)

        tabs = QTabWidget()
        tabs.addTab(self._page(_HELP_RECORDING_HTML), "采集")
        tabs.addTab(self._page(_HELP_TELEOP_HTML), "PICO 遥操作")
        tabs.addTab(self._page(_HELP_DATASET_HTML), "数据集")
        tabs.addTab(self._page(_HELP_COMMANDS_HTML), "常用命令")
        tabs.addTab(self._page(_HELP_TROUBLESHOOTING_HTML), "故障排查")
        layout.addWidget(tabs, 1)

        button_row = QHBoxLayout()
        copy_btn = QPushButton("复制说明")
        close_btn = QPushButton("关闭")
        close_btn.setObjectName("PrimaryButton")
        copy_btn.clicked.connect(self._copy_all)
        close_btn.clicked.connect(self.accept)
        button_row.addStretch(1)
        button_row.addWidget(copy_btn)
        button_row.addWidget(close_btn)
        layout.addLayout(button_row)

    def _page(self, html: str) -> QTextBrowser:
        page = QTextBrowser()
        page.setReadOnly(True)
        page.setObjectName("HelpText")
        page.setStyleSheet(
            "QTextBrowser#HelpText, QTextBrowser#HelpText viewport {"
            "background-color: #ffffff; color: #172033;"
            "}"
        )
        page.viewport().setStyleSheet("background-color: #ffffff; color: #172033;")
        page.document().setDefaultStyleSheet(
            "body { background-color: #ffffff; color: #172033; } "
            "h2 { color: #101827; } "
            "pre, code { background-color: #eef3f8; color: #172033; }"
        )
        page.setHtml(f"<body>{html}</body>")
        return page

    def _copy_all(self) -> None:
        QApplication.clipboard().setText(HELP_PLAIN_TEXT)


_HELP_RECORDING_HTML = """
<h2>数据采集流程</h2>
<ol>
  <li>确认机器人控制器、PICO 服务、相机和 LCM 网络已经启动。</li>
  <li>在“采集”页填写 Repo ID、任务、集数、单集秒数、FPS 和 LCM URL。</li>
  <li>保持“等待 PICO 开始每集”开启时，程序会先遥操作但不保存，按 PICO A 开始保存当前集。</li>
  <li>采集过程中右侧“命令预览”会展示实际执行命令，“运行日志”会持续输出进程日志。</li>
  <li>每集结束后，程序会调用 wheeled_arm 的 movej 复位，再进入重置等待阶段。</li>
  <li>复位 movej 过程中如发现异常，按住 PICO X 会停止继续发布复位轨迹并结束采集。</li>
  <li>采集完成后可在“查看数据集”页点击“使用最近采集”并打开数据集。</li>
</ol>
<p><b>默认数据目录：</b>GUI 使用 LeRobot 的 HF_LEROBOT_HOME。自定义 root 时，查看和编辑也要填写同一个 root。</p>
"""

_HELP_TELEOP_HTML = """
<h2>PICO 遥操作</h2>
<ul>
  <li>开始遥操作时，record 会从 LCM 读取当前机器人左右臂关节状态，并同步到 PICO IK 初始位姿。</li>
  <li>左右 grip 超过阈值后，对应手臂才跟随控制器移动；松开后保持当前机器人末端位置。</li>
  <li>trigger 映射到左右夹爪开合；开合范围可通过高级参数覆盖。</li>
  <li>Y 默认重置 PICO 相对位姿基线，不直接移动机器人。</li>
  <li>A 进入下一阶段或开始当前 episode；B 丢弃并重录当前 episode；X 停止整次采集。</li>
  <li>机器人自动复位期间，X 也是急停按钮；按下后 movej 插补会中断并关闭 moving flags。</li>
</ul>
<h2>复位姿态</h2>
<p>episode 间复位使用 movej。目标角度会从度转换为弧度：</p>
<pre>左臂: [20, 70, -75, 100, -25, 0, 0]
右臂: [-20, 70, 75, 100, 25, 0, 0]</pre>
"""

_HELP_DATASET_HTML = """
<h2>查看与编辑数据集</h2>
<ul>
  <li>“查看数据集”用于启动 Rerun 或 Foxglove 查看指定 episode。</li>
  <li>“编辑数据集”支持查看信息、删除 episode、拆分、合并、删除 feature、修改任务文本和重算统计。</li>
  <li>编辑前先确认输入 Repo ID/root；会生成新数据集的操作需要填写输出 Repo ID/root。</li>
  <li>内置预览可播放 episode，并把当前 episode 填入“删除 Episode”。</li>
  <li>转换页面向 OpenX、AgiBot、RoboMIND、LIBERO、RLDS 和 LeRobot 版本转换。</li>
</ul>
<p>执行删除、覆盖、重编码等操作前，GUI 会弹出确认框；重要数据建议先备份。</p>
"""

_HELP_COMMANDS_HTML = """
<h2>常用命令</h2>
<ul>
  <li>“系统信息”检查 Python、LeRobot、PyTorch、CUDA、FFmpeg 等环境。</li>
  <li>“查找相机”和“查找串口”用于采集前确认硬件。</li>
  <li>“遥操作”可不保存数据，只测试 PICO、IK、LCM 和可视化链路。</li>
  <li>“回放 Episode”会把数据集动作重新下发给 wheeled_arm。</li>
  <li>“训练策略”“评估策略”“策略 Rollout”用于数据采集后的训练和部署验证。</li>
</ul>
<p>所有页签都会生成命令预览；不确定时先复制命令到终端运行，便于定位环境问题。</p>
"""

_HELP_TROUBLESHOOTING_HTML = """
<h2>故障排查</h2>
<ul>
  <li><b>GUI 无法启动：</b>运行 setup 脚本安装 PySide6 和 Qt/xcb 系统库。</li>
  <li><b>缺少 lcm：</b>在当前 conda 环境执行 <code>python -m pip install lcm</code>。</li>
  <li><b>没有机器人反馈：</b>确认 LCM URL、组播路由、控制器状态话题和左右臂状态是否正常。</li>
  <li><b>PICO 不动：</b>确认 XRoboToolkit SDK 已安装，PICO 服务运行，控制器名称与配置一致。</li>
  <li><b>IK 报碰撞或无解：</b>可临时在高级参数加入 <code>--teleop.use_self_collision=false</code> 排查。</li>
  <li><b>相机无图：</b>先用“常用命令 > 查找相机”确认 topic、分辨率和 FPS。</li>
  <li><b>数据集找不到：</b>确认 Repo ID 是否带用户名，以及 root 是否和采集时一致。</li>
</ul>
"""

HELP_PLAIN_TEXT = """Wheeled Arm PICO 使用说明

数据采集:
1. 确认机器人控制器、PICO 服务、相机和 LCM 网络已经启动。
2. 在“采集”页填写 Repo ID、任务、集数、单集秒数、FPS 和 LCM URL。
3. “等待 PICO 开始每集”开启时，按 PICO A 开始保存当前集。
4. 每集结束后会用 movej 复位，再进入重置等待阶段。
5. 复位过程中如发现异常，按住 PICO X 会中断 movej 复位并停止采集。

PICO 按键:
A 进入下一阶段或开始当前 episode。
B 丢弃并重录当前 episode。
X 停止整次采集；复位 movej 期间也作为急停按钮。
Y 重置 PICO 相对位姿基线，不直接移动机器人。

复位姿态:
左臂: [20, 70, -75, 100, -25, 0, 0] 度
右臂: [-20, 70, 75, 100, 25, 0, 0] 度
程序会转换为弧度并通过 movej 下发。

故障排查:
- GUI 无法启动: 运行 setup 脚本安装 PySide6 和 Qt/xcb 系统库。
- 缺少 lcm: python -m pip install lcm
- 没有机器人反馈: 检查 LCM URL、组播路由和左右臂状态话题。
- PICO 不动: 检查 XRoboToolkit SDK、PICO 服务和控制器名称。
- IK 碰撞/无解: 可临时加 --teleop.use_self_collision=false 排查。
"""


class WheeledArmGui(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.settings = QSettings("LeRobot", "WheeledArmPicoGui")
        self.record_runner = ProcessRunner("数据采集")
        self.viewer_runner = ProcessRunner("数据集查看")
        self.edit_runner = ProcessRunner("数据集编辑")
        self.conversion_runner = ProcessRunner("格式转换")
        self.common_runner = ProcessRunner("常用命令")
        self._last_record_base_repo_id = ""
        self._last_record_root = ""
        self._last_record_no_stamp = False
        self._last_record_resume = False

        self.setWindowTitle("LeRobot Wheeled Arm 控制台")
        self.setMinimumSize(QSize(1280, 800))
        self._build_ui()
        self._connect_runners()
        self._load_settings()
        self.update_record_preview()
        self.update_viewer_preview()
        self.update_edit_preview()
        self.update_conversion_preview()
        self.update_common_preview()

    def closeEvent(self, event) -> None:  # noqa: N802
        if (
            self.record_runner.is_running
            or self.viewer_runner.is_running
            or self.edit_runner.is_running
            or self.conversion_runner.is_running
            or self.common_runner.is_running
        ):
            answer = QMessageBox.question(
                self,
                "仍有任务运行",
                "采集、查看、编辑、转换或常用命令进程仍在运行。是否先发送停止信号并关闭窗口？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.record_runner.interrupt()
            self.viewer_runner.interrupt()
            self.edit_runner.interrupt()
            self.conversion_runner.interrupt()
            self.common_runner.interrupt()
        self._save_settings()
        super().closeEvent(event)

    def _build_ui(self) -> None:
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(APP_STYLESHEET)

        root = QWidget()
        root.setObjectName("AppRoot")
        root_layout = QGridLayout(root)
        root_layout.setContentsMargins(22, 20, 22, 20)
        root_layout.setHorizontalSpacing(18)
        root_layout.setVerticalSpacing(14)

        title = QLabel("Wheeled Arm PICO 数据采集与管理")
        title.setObjectName("TitleLabel")
        subtitle = QLabel("把采集、查看和编辑数据集封装为图形界面，同时保留完整命令预览和运行日志。")
        subtitle.setObjectName("SubtitleLabel")
        title_block = QVBoxLayout()
        title_block.addWidget(title)
        title_block.addWidget(subtitle)
        root_layout.addLayout(title_block, 0, 0, 1, 2)

        self.help_btn = QPushButton("使用说明")
        self.help_btn.setObjectName("HelpButton")
        self.help_btn.clicked.connect(self.show_help)
        root_layout.addWidget(
            self.help_btn,
            0,
            2,
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop,
        )

        self.tabs = QTabWidget()
        self.record_tab = self._make_record_tab()
        self.viewer_tab = self._make_viewer_tab()
        self.edit_tab = self._make_edit_tab()
        self.conversion_tab = self._make_conversion_tab()
        self.common_tab = self._make_common_tab()
        self.tabs.addTab(self.record_tab, "采集")
        self.tabs.addTab(self.viewer_tab, "查看数据集")
        self.tabs.addTab(self.edit_tab, "编辑数据集")
        self.tabs.addTab(self.conversion_tab, "格式转换")
        self.tabs.addTab(self.common_tab, "常用命令")

        sidebar = self._make_sidebar()
        root_layout.addWidget(sidebar, 1, 0)
        root_layout.addWidget(self.tabs, 1, 1)

        right_panel = self._make_right_panel()
        self.tabs.currentChanged.connect(self.preview_tabs.setCurrentIndex)
        self.tabs.currentChanged.connect(self._sync_sidebar)
        root_layout.addWidget(right_panel, 1, 2)
        root_layout.setColumnStretch(0, 0)
        root_layout.setColumnStretch(1, 5)
        root_layout.setColumnStretch(2, 4)
        root_layout.setRowStretch(1, 1)

        self.setCentralWidget(root)
        self.statusBar().showMessage("就绪")

        self._build_menu_bar()

    @Slot()
    def show_help(self) -> None:
        dialog = HelpDialog(self)
        dialog.exec()

    def _make_sidebar(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("SidebarPanel")
        panel.setFixedWidth(178)
        self._apply_panel_shadow(panel)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 16, 14, 16)
        layout.setSpacing(8)

        label = QLabel("工作区")
        label.setObjectName("SidebarTitle")
        layout.addWidget(label)

        self.sidebar_group = QButtonGroup(self)
        self.sidebar_group.setExclusive(True)
        self.sidebar_buttons: list[QPushButton] = []
        for index, label_text in enumerate(("采集", "查看", "编辑", "转换", "常用")):
            button = QPushButton(label_text)
            button.setObjectName("SidebarButton")
            button.setCheckable(True)
            button.clicked.connect(
                lambda _checked=False, tab_index=index: self.tabs.setCurrentIndex(tab_index)
            )
            self.sidebar_group.addButton(button, index)
            self.sidebar_buttons.append(button)
            layout.addWidget(button)

        self.sidebar_buttons[0].setChecked(True)
        layout.addStretch(1)

        quick_help = QPushButton("F1 帮助")
        quick_help.setObjectName("SidebarGhostButton")
        quick_help.clicked.connect(self.show_help)
        layout.addWidget(quick_help)
        return panel

    def _sync_sidebar(self, index: int) -> None:
        if hasattr(self, "sidebar_buttons") and 0 <= index < len(self.sidebar_buttons):
            self.sidebar_buttons[index].setChecked(True)

    def _apply_panel_shadow(self, widget: QWidget) -> None:
        shadow = QGraphicsDropShadowEffect(widget)
        shadow.setBlurRadius(28)
        shadow.setOffset(0, 10)
        shadow.setColor(QColor(31, 45, 70, 42))
        widget.setGraphicsEffect(shadow)

    @Slot()
    def start_active_tab(self) -> None:
        callbacks = (
            self.start_recording,
            self.start_viewer,
            self.start_edit,
            self.start_conversion,
            self.start_common_command,
        )
        callbacks[self.tabs.currentIndex()]()

    @Slot()
    def stop_active_tab(self) -> None:
        callbacks = (
            self.stop_recording,
            self.stop_viewer,
            self.stop_edit,
            self.stop_conversion,
            self.stop_common_command,
        )
        callbacks[self.tabs.currentIndex()]()

    def _build_menu_bar(self) -> None:
        file_menu = self.menuBar().addMenu("文件")
        view_menu = self.menuBar().addMenu("视图")
        run_menu = self.menuBar().addMenu("运行")
        help_menu = self.menuBar().addMenu("帮助")

        copy_action = QAction("复制当前命令", self)
        copy_action.setShortcut("Ctrl+Shift+C")
        copy_action.triggered.connect(self.copy_active_command)
        self.addAction(copy_action)
        file_menu.addAction(copy_action)

        clear_log_action = QAction("清空运行日志", self)
        clear_log_action.setShortcut("Ctrl+L")
        clear_log_action.triggered.connect(self.log_view.clear)
        self.addAction(clear_log_action)
        file_menu.addAction(clear_log_action)

        file_menu.addSeparator()
        exit_action = QAction("退出", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        self.addAction(exit_action)
        file_menu.addAction(exit_action)

        for index, label in enumerate(("采集", "查看数据集", "编辑数据集", "格式转换", "常用命令")):
            action = QAction(label, self)
            action.triggered.connect(lambda _checked=False, tab_index=index: self.tabs.setCurrentIndex(tab_index))
            view_menu.addAction(action)

        run_current_action = QAction("运行当前页命令", self)
        run_current_action.setShortcut("Ctrl+R")
        run_current_action.triggered.connect(self.start_active_tab)
        self.addAction(run_current_action)
        run_menu.addAction(run_current_action)

        stop_current_action = QAction("停止当前页任务", self)
        stop_current_action.setShortcut("Ctrl+.")
        stop_current_action.triggered.connect(self.stop_active_tab)
        self.addAction(stop_current_action)
        run_menu.addAction(stop_current_action)

        help_action = QAction("打开使用说明", self)
        help_action.setShortcut("F1")
        help_action.triggered.connect(self.show_help)
        self.addAction(help_action)
        help_menu.addAction(help_action)

    def _make_record_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 12, 12)
        layout.setSpacing(14)

        dataset_box = QGroupBox("数据集")
        dataset_form = QFormLayout(dataset_box)
        self._setup_form_layout(dataset_form)
        self.repo_id = QLineEdit("kuanli/wheeled_arm_pico_test")
        self.repo_id.setPlaceholderText("例如 kuanli/wheeled_arm_pico_test")
        self.task = QLineEdit("PICO teleoperate wheeled arm")
        self.dataset_root = PathPicker("留空时使用 HF_LEROBOT_HOME/数据集名")
        self.num_episodes = self._spin(1, 10000, 1)
        self.episode_time_s = self._spin(1, 36000, 30)
        self.reset_time_s = self._spin(0, 36000, 5)
        self.fps = self._spin(1, 240, 30)
        self.resume = QCheckBox("继续写入已有数据集")
        self.no_stamp = QCheckBox("固定数据集名")
        self.video = QCheckBox("保存为视频")
        self.video.setChecked(True)
        self.streaming_encoding = QCheckBox("边采集边编码")
        self.streaming_encoding.setChecked(True)
        self.push_to_hub = QCheckBox("完成后上传 Hub")
        self.private = QCheckBox("私有数据集")
        self.play_sounds = QCheckBox("英文语音提示")
        self.play_sounds.setChecked(True)
        self.wait_for_episode_start = QCheckBox("等待 PICO 开始每集")
        self.wait_for_episode_start.setChecked(True)

        dataset_form.addRow("Repo ID", self.repo_id)
        dataset_form.addRow("任务", self.task)
        dataset_form.addRow("本地 root", self.dataset_root)
        dataset_form.addRow("集数", self.num_episodes)
        dataset_form.addRow("单集秒数", self.episode_time_s)
        dataset_form.addRow("重置秒数", self.reset_time_s)
        dataset_form.addRow("FPS", self.fps)
        dataset_form.addRow("", self.resume)
        dataset_form.addRow("", self.no_stamp)
        dataset_form.addRow("", self.video)
        dataset_form.addRow("", self.streaming_encoding)
        dataset_form.addRow("", self.push_to_hub)
        dataset_form.addRow("", self.private)
        dataset_form.addRow("", self.play_sounds)
        dataset_form.addRow("", self.wait_for_episode_start)
        layout.addWidget(dataset_box)

        robot_box = QGroupBox("机器人与可视化")
        robot_form = QFormLayout(robot_box)
        self._setup_form_layout(robot_form)
        self.display_data = QCheckBox("Rerun 数据窗口")
        self.display_data.setChecked(True)
        self.display_mode = QComboBox()
        self.display_mode.addItems(["rerun", "foxglove"])
        self.display_ip = QLineEdit()
        self.display_ip.setPlaceholderText("远程可视化时填写，通常留空")
        self.display_port = self._spin(0, 65535, 0)
        self.display_port.setSpecialValueText("默认")
        self.display_compressed_images = QCheckBox("压缩图像流")
        self.viser = QCheckBox("viser URDF 窗口")
        self.viser.setChecked(True)
        self.rerun_robot = QCheckBox("Rerun 机器人 3D")
        self.rerun_robot.setChecked(True)
        self.mock_xr = QCheckBox("模拟 PICO 输入")
        self.lcm_url = QLineEdit("udpm://239.255.76.67:8880?ttl=1")
        self.camera_override = QCheckBox("覆盖 front 相机参数")
        self.camera_topic = QLineEdit("/camera/color/image_raw")
        self.camera_width = self._spin(1, 8192, 640)
        self.camera_height = self._spin(1, 8192, 480)
        self.camera_fps = self._spin(1, 240, 30)
        self.advanced_args = QPlainTextEdit()
        self.advanced_args.setPlaceholderText("--teleop.scale=0.8 --teleop.use_self_collision=false")
        self.advanced_args.setFixedHeight(74)

        robot_form.addRow("", self.display_data)
        robot_form.addRow("显示模式", self.display_mode)
        robot_form.addRow("显示 IP", self.display_ip)
        robot_form.addRow("显示端口", self.display_port)
        robot_form.addRow("", self.display_compressed_images)
        robot_form.addRow("", self.viser)
        robot_form.addRow("", self.rerun_robot)
        robot_form.addRow("", self.mock_xr)
        robot_form.addRow("LCM URL", self.lcm_url)
        robot_form.addRow("", self.camera_override)
        robot_form.addRow("相机 topic", self.camera_topic)
        robot_form.addRow("相机宽", self.camera_width)
        robot_form.addRow("相机高", self.camera_height)
        robot_form.addRow("相机 FPS", self.camera_fps)
        robot_form.addRow("高级参数", self.advanced_args)
        layout.addWidget(robot_box)

        button_row = QHBoxLayout()
        self.start_record_btn = QPushButton("开始采集")
        self.start_record_btn.setObjectName("PrimaryButton")
        self.stop_record_btn = QPushButton("停止采集")
        self.stop_record_btn.setObjectName("DangerButton")
        self.stop_record_btn.setEnabled(False)
        self.copy_record_btn = QPushButton("复制命令")
        self.start_record_btn.clicked.connect(self.start_recording)
        self.stop_record_btn.clicked.connect(self.stop_recording)
        self.copy_record_btn.clicked.connect(lambda: self.copy_command(self.record_command_preview.toPlainText()))
        button_row.addWidget(self.start_record_btn)
        button_row.addWidget(self.stop_record_btn)
        button_row.addWidget(self.copy_record_btn)
        button_row.addStretch(1)
        layout.addLayout(button_row)
        layout.addStretch(1)

        for widget in self._record_widgets():
            self._connect_preview_signal(widget, self.update_record_preview)
        self.dataset_root.changed.connect(self.update_record_preview)

        return self._scrollable(page)

    def _make_viewer_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 12, 12)
        layout.setSpacing(14)

        viewer_box = QGroupBox("Rerun/Foxglove 查看")
        form = QFormLayout(viewer_box)
        self._setup_form_layout(form)
        self.viewer_repo_id = QLineEdit()
        self.viewer_repo_id.setPlaceholderText("例如 kuanli/wheeled_arm_pico_test_20260809_110636")
        self.viewer_root = PathPicker("仅当数据集 root 为自定义路径时填写")
        self.viewer_episode = self._spin(0, 1000000, 0)
        self.viewer_display_mode = QComboBox()
        self.viewer_display_mode.addItems(["rerun", "foxglove"])
        self.viewer_mode = QComboBox()
        self.viewer_mode.addItems(["local", "distant"])
        self.viewer_web_port = self._spin(0, 65535, 0)
        self.viewer_web_port.setSpecialValueText("默认")
        self.viewer_grpc_port = self._spin(1, 65535, 9876)
        self.viewer_host = QLineEdit("127.0.0.1")
        self.viewer_compressed = QCheckBox("压缩图像")
        self.viewer_autoplay = QCheckBox("自动播放")
        self.viewer_autoplay.setChecked(True)
        form.addRow("Repo ID", self.viewer_repo_id)
        form.addRow("本地 root", self.viewer_root)
        form.addRow("Episode", self.viewer_episode)
        form.addRow("显示后端", self.viewer_display_mode)
        form.addRow("Rerun 模式", self.viewer_mode)
        form.addRow("Web 端口", self.viewer_web_port)
        form.addRow("gRPC 端口", self.viewer_grpc_port)
        form.addRow("Host", self.viewer_host)
        form.addRow("", self.viewer_compressed)
        form.addRow("", self.viewer_autoplay)
        layout.addWidget(viewer_box)

        button_row = QHBoxLayout()
        self.start_viewer_btn = QPushButton("打开数据集")
        self.start_viewer_btn.setObjectName("PrimaryButton")
        self.stop_viewer_btn = QPushButton("停止查看")
        self.stop_viewer_btn.setObjectName("DangerButton")
        self.stop_viewer_btn.setEnabled(False)
        self.copy_viewer_btn = QPushButton("复制命令")
        self.use_latest_btn = QPushButton("使用最近采集")
        self.start_viewer_btn.clicked.connect(self.start_viewer)
        self.stop_viewer_btn.clicked.connect(self.stop_viewer)
        self.copy_viewer_btn.clicked.connect(lambda: self.copy_command(self.viewer_command_preview.toPlainText()))
        self.use_latest_btn.clicked.connect(self.fill_latest_dataset)
        button_row.addWidget(self.start_viewer_btn)
        button_row.addWidget(self.stop_viewer_btn)
        button_row.addWidget(self.copy_viewer_btn)
        button_row.addWidget(self.use_latest_btn)
        button_row.addStretch(1)
        layout.addLayout(button_row)
        layout.addStretch(1)

        for widget in (
            self.viewer_repo_id,
            self.viewer_episode,
            self.viewer_display_mode,
            self.viewer_mode,
            self.viewer_web_port,
            self.viewer_grpc_port,
            self.viewer_host,
            self.viewer_compressed,
            self.viewer_autoplay,
        ):
            self._connect_preview_signal(widget, self.update_viewer_preview)
        self.viewer_root.changed.connect(self.update_viewer_preview)

        return self._scrollable(page)

    def _make_edit_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 12, 12)
        layout.setSpacing(14)

        common_box = QGroupBox("输入与输出")
        common_form = QFormLayout(common_box)
        self._setup_form_layout(common_form)
        self.edit_repo_id = QLineEdit()
        self.edit_repo_id.setPlaceholderText("输入数据集 Repo ID，例如 kuanli/wheeled_arm_pico_test_20260809_230821")
        self.edit_root = PathPicker("仅当输入数据集 root 为自定义路径时填写")
        self.edit_new_repo_id = QLineEdit()
        self.edit_new_repo_id.setPlaceholderText("留空时使用操作默认输出；原地操作按脚本规则执行")
        self.edit_new_root = PathPicker("输出数据集目录，可留空")
        self.edit_push_to_hub = QCheckBox("完成后上传 Hub")
        self.edit_operation = QComboBox()
        self.edit_operation.addItems(EDIT_OPERATION_ORDER)
        common_form.addRow("输入 Repo ID", self.edit_repo_id)
        common_form.addRow("输入 root", self.edit_root)
        common_form.addRow("输出 Repo ID", self.edit_new_repo_id)
        common_form.addRow("输出 root", self.edit_new_root)
        common_form.addRow("操作", self.edit_operation)
        common_form.addRow("", self.edit_push_to_hub)
        layout.addWidget(common_box)

        self.edit_stack = QStackedWidget()
        self.edit_stack.addWidget(self._make_info_edit_panel())
        self.edit_stack.addWidget(self._make_delete_episodes_edit_panel())
        self.edit_stack.addWidget(self._make_split_edit_panel())
        self.edit_stack.addWidget(self._make_merge_edit_panel())
        self.edit_stack.addWidget(self._make_remove_feature_edit_panel())
        self.edit_stack.addWidget(self._make_modify_tasks_edit_panel())
        self.edit_stack.addWidget(self._make_convert_video_edit_panel())
        self.edit_stack.addWidget(self._make_recompute_stats_edit_panel())
        self.edit_stack.addWidget(self._make_reencode_videos_edit_panel())
        layout.addWidget(self.edit_stack)

        preview_box = QGroupBox("数据集预览")
        preview_layout = QVBoxLayout(preview_box)
        self.edit_dataset_preview = LeRobotDatasetPreview()
        self.edit_dataset_preview.deleteEpisodeRequested.connect(self._fill_delete_episode_from_preview)
        preview_layout.addWidget(self.edit_dataset_preview)
        layout.addWidget(preview_box, 1)

        advanced_box = QGroupBox("高级参数")
        advanced_layout = QVBoxLayout(advanced_box)
        self.edit_advanced_args = QPlainTextEdit()
        self.edit_advanced_args.setPlaceholderText("--operation.rgb_encoder.preset=fast --operation.depth_encoder.use_log=true")
        self.edit_advanced_args.setFixedHeight(74)
        advanced_layout.addWidget(self.edit_advanced_args)
        layout.addWidget(advanced_box)

        button_row = QHBoxLayout()
        self.start_edit_btn = QPushButton("执行编辑")
        self.start_edit_btn.setObjectName("PrimaryButton")
        self.stop_edit_btn = QPushButton("停止编辑")
        self.stop_edit_btn.setObjectName("DangerButton")
        self.stop_edit_btn.setEnabled(False)
        self.copy_edit_btn = QPushButton("复制命令")
        self.fill_edit_from_viewer_btn = QPushButton("使用查看数据集")
        self.start_edit_btn.clicked.connect(self.start_edit)
        self.stop_edit_btn.clicked.connect(self.stop_edit)
        self.copy_edit_btn.clicked.connect(lambda: self.copy_command(self.edit_command_preview.toPlainText()))
        self.fill_edit_from_viewer_btn.clicked.connect(self.fill_edit_from_viewer)
        button_row.addWidget(self.start_edit_btn)
        button_row.addWidget(self.stop_edit_btn)
        button_row.addWidget(self.copy_edit_btn)
        button_row.addWidget(self.fill_edit_from_viewer_btn)
        button_row.addStretch(1)
        layout.addLayout(button_row)
        layout.addStretch(1)

        self.edit_operation.currentIndexChanged.connect(self.edit_stack.setCurrentIndex)
        self.edit_operation.currentIndexChanged.connect(self.update_edit_preview)
        for widget in self._edit_widgets():
            self._connect_preview_signal(widget, self.update_edit_preview)
        self.edit_root.changed.connect(self.update_edit_preview)
        self.edit_new_root.changed.connect(self.update_edit_preview)

        return self._scrollable(page)

    def _make_conversion_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 12, 12)
        layout.setSpacing(14)

        type_box = QGroupBox("转换类型")
        type_form = QFormLayout(type_box)
        self._setup_form_layout(type_form)
        self.conversion_type = QComboBox()
        self.conversion_type.addItems(CONVERSION_ORDER)
        self.conversion_python = PathPicker("默认使用当前环境 Python", directory=False)
        self.conversion_python.setText(sys.executable)
        self.conversion_backend = PathPicker("Any4LeRobot backend 路径")
        self.conversion_backend.setText(str(ANY4LEROBOT_BACKEND))
        self.conversion_script_label = QLabel("")
        self.conversion_script_label.setWordWrap(True)
        self.conversion_script_label.setObjectName("MutedLabel")
        type_form.addRow("类型", self.conversion_type)
        type_form.addRow("Python", self.conversion_python)
        type_form.addRow("Backend", self.conversion_backend)
        type_form.addRow("脚本", self.conversion_script_label)
        layout.addWidget(type_box)

        self.conversion_stack = QStackedWidget()
        self.conversion_stack.addWidget(self._make_openx_conversion_panel())
        self.conversion_stack.addWidget(self._make_agibot_conversion_panel())
        self.conversion_stack.addWidget(self._make_robomind_conversion_panel())
        self.conversion_stack.addWidget(self._make_libero_conversion_panel())
        self.conversion_stack.addWidget(self._make_lerobot_rlds_conversion_panel())
        self.conversion_stack.addWidget(self._make_v16_v20_conversion_panel())
        self.conversion_stack.addWidget(self._make_v20_v21_conversion_panel())
        self.conversion_stack.addWidget(self._make_v21_v30_conversion_panel())
        self.conversion_stack.addWidget(self._make_v30_v21_conversion_panel())
        layout.addWidget(self.conversion_stack)

        advanced_box = QGroupBox("高级参数")
        advanced_layout = QVBoxLayout(advanced_box)
        self.conversion_advanced_args = QPlainTextEdit()
        self.conversion_advanced_args.setPlaceholderText("--extra-flag value")
        self.conversion_advanced_args.setFixedHeight(74)
        advanced_layout.addWidget(self.conversion_advanced_args)
        layout.addWidget(advanced_box)

        button_row = QHBoxLayout()
        self.start_conversion_btn = QPushButton("开始转换")
        self.start_conversion_btn.setObjectName("PrimaryButton")
        self.stop_conversion_btn = QPushButton("停止转换")
        self.stop_conversion_btn.setObjectName("DangerButton")
        self.stop_conversion_btn.setEnabled(False)
        self.copy_conversion_btn = QPushButton("复制命令")
        self.start_conversion_btn.clicked.connect(self.start_conversion)
        self.stop_conversion_btn.clicked.connect(self.stop_conversion)
        self.copy_conversion_btn.clicked.connect(
            lambda: self.copy_command(self.conversion_command_preview.toPlainText())
        )
        button_row.addWidget(self.start_conversion_btn)
        button_row.addWidget(self.stop_conversion_btn)
        button_row.addWidget(self.copy_conversion_btn)
        button_row.addStretch(1)
        layout.addLayout(button_row)
        layout.addStretch(1)

        self.conversion_type.currentIndexChanged.connect(self._on_conversion_type_changed)
        for widget in self._conversion_widgets():
            self._connect_preview_signal(widget, self.update_conversion_preview)
        self.conversion_python.changed.connect(self.update_conversion_preview)
        self.conversion_backend.changed.connect(self.update_conversion_preview)
        self.openx_raw_dir.changed.connect(self.update_conversion_preview)
        self.openx_local_dir.changed.connect(self.update_conversion_preview)
        self.agibot_src_path.changed.connect(self.update_conversion_preview)
        self.agibot_output_path.changed.connect(self.update_conversion_preview)
        self.robomind_src_path.changed.connect(self.update_conversion_preview)
        self.robomind_output_path.changed.connect(self.update_conversion_preview)
        self.libero_output_path.changed.connect(self.update_conversion_preview)
        self.libero_resume_dir.changed.connect(self.update_conversion_preview)
        self.rlds_src_dir.changed.connect(self.update_conversion_preview)
        self.rlds_output_dir.changed.connect(self.update_conversion_preview)
        self.v16_local_dir.changed.connect(self.update_conversion_preview)
        self.v16_tasks_path.changed.connect(self.update_conversion_preview)
        self.version_root.changed.connect(self.update_conversion_preview)
        self.v30_root.changed.connect(self.update_conversion_preview)
        self.v3021_root.changed.connect(self.update_conversion_preview)
        self._on_conversion_type_changed()

        return self._scrollable(page)

    def _make_common_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 12, 12)
        layout.setSpacing(14)

        command_box = QGroupBox("命令")
        command_form = QFormLayout(command_box)
        self._setup_form_layout(command_form)
        self.common_command = QComboBox()
        self.common_command.addItems(COMMON_COMMAND_ORDER)
        self.common_advanced_args = QPlainTextEdit()
        self.common_advanced_args.setPlaceholderText("--extra.arg=value")
        self.common_advanced_args.setFixedHeight(74)
        command_form.addRow("选择命令", self.common_command)
        command_form.addRow("高级参数", self.common_advanced_args)
        layout.addWidget(command_box)

        self.common_stack = QStackedWidget()
        self.common_stack.addWidget(self._make_system_info_common_panel())
        self.common_stack.addWidget(self._make_find_cameras_common_panel())
        self.common_stack.addWidget(self._make_find_port_common_panel())
        self.common_stack.addWidget(self._make_teleoperate_common_panel())
        self.common_stack.addWidget(self._make_replay_common_panel())
        self.common_stack.addWidget(self._make_calibrate_common_panel())
        self.common_stack.addWidget(self._make_setup_motors_common_panel())
        self.common_stack.addWidget(self._make_find_joint_limits_common_panel())
        self.common_stack.addWidget(self._make_setup_can_common_panel())
        self.common_stack.addWidget(self._make_train_common_panel())
        self.common_stack.addWidget(self._make_eval_common_panel())
        self.common_stack.addWidget(self._make_rollout_common_panel())
        self.common_stack.addWidget(self._make_annotate_common_panel())
        self.common_stack.addWidget(self._make_imgtransform_common_panel())
        self.common_stack.addWidget(self._make_quantile_common_panel())
        self.common_stack.addWidget(self._make_convert_dcp_common_panel())
        self.common_stack.addWidget(self._make_train_tokenizer_common_panel())
        self.common_stack.addWidget(self._make_custom_common_panel())
        layout.addWidget(self.common_stack)

        button_row = QHBoxLayout()
        self.start_common_btn = QPushButton("运行命令")
        self.start_common_btn.setObjectName("PrimaryButton")
        self.stop_common_btn = QPushButton("停止命令")
        self.stop_common_btn.setObjectName("DangerButton")
        self.stop_common_btn.setEnabled(False)
        self.common_enter_btn = QPushButton("发送 Enter")
        self.common_enter_btn.setEnabled(False)
        self.copy_common_btn = QPushButton("复制命令")
        self.start_common_btn.clicked.connect(self.start_common_command)
        self.stop_common_btn.clicked.connect(self.stop_common_command)
        self.common_enter_btn.clicked.connect(lambda: self.common_runner.write_stdin("\n"))
        self.copy_common_btn.clicked.connect(lambda: self.copy_command(self.common_command_preview.toPlainText()))
        button_row.addWidget(self.start_common_btn)
        button_row.addWidget(self.stop_common_btn)
        button_row.addWidget(self.common_enter_btn)
        button_row.addWidget(self.copy_common_btn)
        button_row.addStretch(1)
        layout.addLayout(button_row)
        layout.addStretch(1)

        self.common_command.currentIndexChanged.connect(self.common_stack.setCurrentIndex)
        self.common_command.currentIndexChanged.connect(self.update_common_preview)
        for widget in self._common_widgets():
            self._connect_preview_signal(widget, self.update_common_preview)
        self.common_camera_output_dir.changed.connect(self.update_common_preview)
        self.common_replay_root.changed.connect(self.update_common_preview)
        self.common_joint_urdf_path.changed.connect(self.update_common_preview)
        self.common_train_output_dir.changed.connect(self.update_common_preview)
        self.common_eval_output_dir.changed.connect(self.update_common_preview)
        self.common_annotate_root.changed.connect(self.update_common_preview)
        self.common_imgtransform_output_dir.changed.connect(self.update_common_preview)
        self.common_quantile_root.changed.connect(self.update_common_preview)
        self.common_dcp_checkpoint_dir.changed.connect(self.update_common_preview)
        self.common_tokenizer_root.changed.connect(self.update_common_preview)
        self.common_tokenizer_output_dir.changed.connect(self.update_common_preview)

        return self._scrollable(page)

    def _make_openx_conversion_panel(self) -> QWidget:
        box = QGroupBox("OpenX → LeRobot")
        form = QFormLayout(box)
        self._setup_form_layout(form)
        self.openx_raw_dir = PathPicker("OpenX / TFDS 原始数据目录")
        self.openx_local_dir = PathPicker("输出 LeRobot 数据集目录")
        self.openx_repo_id = QLineEdit()
        self.openx_repo_id.setPlaceholderText("可选；上传 Hub 时必填，例如 user/openx_dataset")
        self.openx_robot_type = QLineEdit()
        self.openx_robot_type.setPlaceholderText("可选，例如 franka")
        self.openx_fps = self._spin(0, 240, 0)
        self.openx_fps.setSpecialValueText("自动")
        self.openx_use_videos = QCheckBox("保存为视频")
        self.openx_use_videos.setChecked(True)
        self.openx_push_to_hub = QCheckBox("完成后上传 Hub")
        self.openx_image_processes = self._spin(0, 128, 5)
        self.openx_image_threads = self._spin(1, 128, 10)
        form.addRow("raw dir", self.openx_raw_dir)
        form.addRow("local dir", self.openx_local_dir)
        form.addRow("Repo ID", self.openx_repo_id)
        form.addRow("robot type", self.openx_robot_type)
        form.addRow("FPS", self.openx_fps)
        form.addRow("", self.openx_use_videos)
        form.addRow("", self.openx_push_to_hub)
        form.addRow("图片进程", self.openx_image_processes)
        form.addRow("图片线程", self.openx_image_threads)
        return box

    def _make_agibot_conversion_panel(self) -> QWidget:
        box = QGroupBox("AgiBot → LeRobot")
        form = QFormLayout(box)
        self._setup_form_layout(form)
        self.agibot_src_path = PathPicker("AgiBot 原始数据目录")
        self.agibot_output_path = PathPicker("输出 LeRobot 数据集目录")
        self.agibot_eef_type = QComboBox()
        self.agibot_eef_type.addItems(["gripper", "dexhand", "tactile"])
        self.agibot_task_ids = QLineEdit()
        self.agibot_task_ids.setPlaceholderText("可选：task_327 task_351 或 task_327,task_351")
        self.agibot_cpus = self._spin(1, 128, 3)
        self.agibot_save_depth = QCheckBox("保存深度")
        self.agibot_debug = QCheckBox("debug 模式")
        form.addRow("src path", self.agibot_src_path)
        form.addRow("output path", self.agibot_output_path)
        form.addRow("eef type", self.agibot_eef_type)
        form.addRow("task ids", self.agibot_task_ids)
        form.addRow("CPU/任务", self.agibot_cpus)
        form.addRow("", self.agibot_save_depth)
        form.addRow("", self.agibot_debug)
        return box

    def _make_robomind_conversion_panel(self) -> QWidget:
        box = QGroupBox("RoboMIND → LeRobot")
        form = QFormLayout(box)
        self._setup_form_layout(form)
        self.robomind_src_path = PathPicker("RoboMIND 原始数据目录")
        self.robomind_output_path = PathPicker("输出 LeRobot 数据集目录")
        self.robomind_benchmark = QComboBox()
        self.robomind_benchmark.addItems(["benchmark1_1_release", "benchmark1_0_release", "benchmark1_2_release"])
        self.robomind_embodiments = QLineEdit("agilex_3rgb")
        self.robomind_embodiments.setPlaceholderText("空格或逗号分隔，例如 agilex_3rgb franka_3rgb")
        self.robomind_cpus = self._spin(1, 128, 2)
        self.robomind_save_depth = QCheckBox("保存深度")
        self.robomind_debug = QCheckBox("debug 模式")
        form.addRow("src path", self.robomind_src_path)
        form.addRow("output path", self.robomind_output_path)
        form.addRow("benchmark", self.robomind_benchmark)
        form.addRow("embodiments", self.robomind_embodiments)
        form.addRow("CPU/任务", self.robomind_cpus)
        form.addRow("", self.robomind_save_depth)
        form.addRow("", self.robomind_debug)
        return box

    def _make_libero_conversion_panel(self) -> QWidget:
        box = QGroupBox("LIBERO → LeRobot")
        form = QFormLayout(box)
        self._setup_form_layout(form)
        self.libero_src_paths = QLineEdit()
        self.libero_src_paths.setPlaceholderText("一个或多个目录，空格或逗号分隔")
        self.libero_output_path = PathPicker("输出 LeRobot 数据集目录")
        self.libero_executor = QComboBox()
        self.libero_executor.addItems(["local", "ray"])
        self.libero_cpus = self._spin(1, 128, 1)
        self.libero_tasks_per_job = self._spin(1, 10000, 1)
        self.libero_workers = self._spin(-1, 10000, -1)
        self.libero_resume_dir = PathPicker("可选：继续转换的 logs 目录")
        self.libero_repo_id = QLineEdit()
        self.libero_repo_id.setPlaceholderText("上传 Hub 时必填")
        self.libero_push_to_hub = QCheckBox("完成后上传 Hub")
        self.libero_debug = QCheckBox("debug 模式")
        form.addRow("src paths", self.libero_src_paths)
        form.addRow("output path", self.libero_output_path)
        form.addRow("executor", self.libero_executor)
        form.addRow("CPU/任务", self.libero_cpus)
        form.addRow("tasks/job", self.libero_tasks_per_job)
        form.addRow("workers", self.libero_workers)
        form.addRow("resume dir", self.libero_resume_dir)
        form.addRow("Repo ID", self.libero_repo_id)
        form.addRow("", self.libero_push_to_hub)
        form.addRow("", self.libero_debug)
        return box

    def _make_lerobot_rlds_conversion_panel(self) -> QWidget:
        box = QGroupBox("LeRobot → RLDS")
        form = QFormLayout(box)
        self._setup_form_layout(form)
        self.rlds_src_dir = PathPicker("输入 LeRobot 数据集目录")
        self.rlds_output_dir = PathPicker("输出 RLDS/TFDS 目录")
        self.rlds_task_name = QLineEdit("default_task")
        self.rlds_encoding = QComboBox()
        self.rlds_encoding.addItems(["jpeg", "png"])
        self.rlds_version = QLineEdit("0.1.0")
        self.rlds_enable_beam = QCheckBox("启用 Beam")
        self.rlds_beam_mode = QComboBox()
        self.rlds_beam_mode.addItems(["multi_processing", "multi_threading"])
        self.rlds_beam_workers = self._spin(1, 128, 5)
        self.rlds_homepage = QLineEdit()
        self.rlds_citation = QPlainTextEdit()
        self.rlds_citation.setFixedHeight(58)
        self.rlds_description = QPlainTextEdit()
        self.rlds_description.setFixedHeight(58)
        form.addRow("src dir", self.rlds_src_dir)
        form.addRow("output dir", self.rlds_output_dir)
        form.addRow("task name", self.rlds_task_name)
        form.addRow("图片格式", self.rlds_encoding)
        form.addRow("版本号", self.rlds_version)
        form.addRow("", self.rlds_enable_beam)
        form.addRow("Beam 模式", self.rlds_beam_mode)
        form.addRow("Beam workers", self.rlds_beam_workers)
        form.addRow("homepage", self.rlds_homepage)
        form.addRow("citation", self.rlds_citation)
        form.addRow("description", self.rlds_description)
        return box

    def _make_v16_v20_conversion_panel(self) -> QWidget:
        box = QGroupBox("LeRobot v1.6 → v2.0")
        form = QFormLayout(box)
        self._setup_form_layout(form)
        self.v16_repo_id = QLineEdit()
        self.v16_repo_id.setPlaceholderText("输入数据集 Repo ID")
        self.v16_task_mode = QComboBox()
        self.v16_task_mode.addItems(["single-task", "tasks-col", "tasks-path"])
        self.v16_task_value = QLineEdit()
        self.v16_task_value.setPlaceholderText("任务文本、列名或 JSON 路径")
        self.v16_tasks_path = PathPicker("tasks JSON 文件", directory=False)
        self.v16_robot = QLineEdit()
        self.v16_robot.setPlaceholderText("可选：koch / aloha / so100 ...")
        self.v16_local_dir = PathPicker("默认 /tmp/lerobot_dataset_v2")
        self.v16_license = QLineEdit("apache-2.0")
        self.v16_test_branch = QLineEdit()
        warning = QLabel("注意：v1.6 → v2.0 依赖旧版 LeRobot 的 lerobot.common.* 模块，当前环境可能只能查看命令。")
        warning.setWordWrap(True)
        warning.setObjectName("MutedLabel")
        form.addRow("Repo ID", self.v16_repo_id)
        form.addRow("任务来源", self.v16_task_mode)
        form.addRow("任务值", self.v16_task_value)
        form.addRow("tasks path", self.v16_tasks_path)
        form.addRow("robot", self.v16_robot)
        form.addRow("local dir", self.v16_local_dir)
        form.addRow("license", self.v16_license)
        form.addRow("test branch", self.v16_test_branch)
        form.addRow("", warning)
        return box

    def _make_v20_v21_conversion_panel(self) -> QWidget:
        box = QGroupBox("LeRobot v2.0/v2.1 互转")
        form = QFormLayout(box)
        self._setup_form_layout(form)
        self.version_repo_id = QLineEdit()
        self.version_repo_id.setPlaceholderText("输入数据集 Repo ID")
        self.version_root = PathPicker("可选：本地数据集 root")
        self.version_push_to_hub = QCheckBox("完成后上传 Hub")
        self.version_delete_old_stats = QCheckBox("删除旧 stats.json")
        self.version_branch = QLineEdit()
        self.version_num_workers = self._spin(1, 128, 4)
        form.addRow("Repo ID", self.version_repo_id)
        form.addRow("root", self.version_root)
        form.addRow("", self.version_push_to_hub)
        form.addRow("", self.version_delete_old_stats)
        form.addRow("branch", self.version_branch)
        form.addRow("workers", self.version_num_workers)
        return box

    def _make_v21_v30_conversion_panel(self) -> QWidget:
        box = QGroupBox("LeRobot v2.1 → v3.0")
        form = QFormLayout(box)
        self._setup_form_layout(form)
        self.v30_repo_id = QLineEdit()
        self.v30_repo_id.setPlaceholderText("输入数据集 Repo ID")
        self.v30_root = PathPicker("可选：本地数据集 root")
        self.v30_push_to_hub = QCheckBox("完成后上传 Hub")
        self.v30_push_to_hub.setChecked(True)
        self.v30_force = QCheckBox("强制转换")
        self.v30_branch = QLineEdit()
        self.v30_data_size = self._spin(0, 100000, 0)
        self.v30_data_size.setSpecialValueText("默认")
        self.v30_video_size = self._spin(0, 100000, 0)
        self.v30_video_size.setSpecialValueText("默认")
        form.addRow("Repo ID", self.v30_repo_id)
        form.addRow("root", self.v30_root)
        form.addRow("", self.v30_push_to_hub)
        form.addRow("", self.v30_force)
        form.addRow("branch", self.v30_branch)
        form.addRow("data MB", self.v30_data_size)
        form.addRow("video MB", self.v30_video_size)
        return box

    def _make_v30_v21_conversion_panel(self) -> QWidget:
        box = QGroupBox("LeRobot v3.0 → v2.1")
        form = QFormLayout(box)
        self._setup_form_layout(form)
        self.v3021_repo_id = QLineEdit()
        self.v3021_repo_id.setPlaceholderText("输入数据集 Repo ID")
        self.v3021_root = PathPicker("可选：本地数据集 root")
        warning = QLabel(
            "注意：该转换会原地重建数据集，并把原 v3.0 目录备份为同级 *_v3.0。"
            "参考脚本还提示某些环境可能需要 datasets<4.0.0。"
        )
        warning.setWordWrap(True)
        warning.setObjectName("MutedLabel")
        form.addRow("Repo ID", self.v3021_repo_id)
        form.addRow("root", self.v3021_root)
        form.addRow("", warning)
        return box

    def _make_system_info_common_panel(self) -> QWidget:
        box = QGroupBox("系统信息")
        layout = QVBoxLayout(box)
        label = QLabel("运行 lerobot-info，输出 Python、LeRobot、PyTorch、CUDA、FFmpeg 等环境信息。")
        label.setWordWrap(True)
        label.setObjectName("MutedLabel")
        layout.addWidget(label)
        return box

    def _make_find_cameras_common_panel(self) -> QWidget:
        box = QGroupBox("查找相机")
        form = QFormLayout(box)
        self._setup_form_layout(form)
        self.common_camera_type = QComboBox()
        self.common_camera_type.addItems(["全部", "opencv", "realsense", "ros2"])
        self.common_camera_output_dir = PathPicker("outputs/captured_images")
        self.common_camera_record_time = self._double_spin(0.1, 3600.0, 2.0, 1)
        self.common_camera_warmup = self._spin(0, 60, 1)
        form.addRow("类型", self.common_camera_type)
        form.addRow("图片目录", self.common_camera_output_dir)
        form.addRow("采样秒数", self.common_camera_record_time)
        form.addRow("预热秒数", self.common_camera_warmup)
        return box

    def _make_find_port_common_panel(self) -> QWidget:
        box = QGroupBox("查找串口")
        layout = QVBoxLayout(box)
        label = QLabel("运行后按日志提示拔掉 USB，再点击“发送 Enter”。该脚本会比较拔线前后的串口列表。")
        label.setWordWrap(True)
        label.setObjectName("MutedLabel")
        layout.addWidget(label)
        return box

    def _make_teleoperate_common_panel(self) -> QWidget:
        box = QGroupBox("遥操作")
        form = QFormLayout(box)
        self._setup_form_layout(form)
        self.common_teleop_fps = self._spin(1, 240, 30)
        self.common_teleop_time_s = self._spin(0, 36000, 0)
        self.common_teleop_time_s.setSpecialValueText("不限时")
        self.common_teleop_display_data = QCheckBox("显示 Rerun/Foxglove 数据")
        self.common_teleop_display_mode = QComboBox()
        self.common_teleop_display_mode.addItems(["rerun", "foxglove"])
        self.common_teleop_viser = QCheckBox("viser URDF 窗口")
        self.common_teleop_viser.setChecked(True)
        self.common_teleop_rerun_robot = QCheckBox("Rerun 机器人 3D")
        self.common_teleop_rerun_robot.setChecked(True)
        self.common_teleop_mock_xr = QCheckBox("模拟 PICO 输入")
        self.common_teleop_lcm_url = QLineEdit("udpm://239.255.76.67:8880?ttl=1")
        form.addRow("FPS", self.common_teleop_fps)
        form.addRow("运行秒数", self.common_teleop_time_s)
        form.addRow("", self.common_teleop_display_data)
        form.addRow("显示后端", self.common_teleop_display_mode)
        form.addRow("", self.common_teleop_viser)
        form.addRow("", self.common_teleop_rerun_robot)
        form.addRow("", self.common_teleop_mock_xr)
        form.addRow("LCM URL", self.common_teleop_lcm_url)
        return box

    def _make_replay_common_panel(self) -> QWidget:
        box = QGroupBox("回放 Episode")
        form = QFormLayout(box)
        self._setup_form_layout(form)
        self.common_replay_repo_id = QLineEdit()
        self.common_replay_repo_id.setPlaceholderText("要回放的数据集 Repo ID")
        self.common_replay_root = PathPicker("仅当数据集 root 为自定义路径时填写")
        self.common_replay_episode = self._spin(0, 1000000, 0)
        self.common_replay_fps = self._spin(1, 240, 30)
        self.common_replay_play_sounds = QCheckBox("英文语音提示")
        self.common_replay_play_sounds.setChecked(True)
        self.common_replay_lcm_url = QLineEdit("udpm://239.255.76.67:8880?ttl=1")
        form.addRow("Repo ID", self.common_replay_repo_id)
        form.addRow("root", self.common_replay_root)
        form.addRow("Episode", self.common_replay_episode)
        form.addRow("FPS", self.common_replay_fps)
        form.addRow("", self.common_replay_play_sounds)
        form.addRow("LCM URL", self.common_replay_lcm_url)
        return box

    def _make_calibrate_common_panel(self) -> QWidget:
        box = QGroupBox("校准设备")
        form = QFormLayout(box)
        self._setup_form_layout(form)
        self.common_calibrate_target = QComboBox()
        self.common_calibrate_target.addItems(["robot", "teleop"])
        self.common_calibrate_robot_type = QLineEdit("wheeled_arm")
        self.common_calibrate_teleop_type = QLineEdit("wheeled_arm_pico")
        form.addRow("目标", self.common_calibrate_target)
        form.addRow("robot.type", self.common_calibrate_robot_type)
        form.addRow("teleop.type", self.common_calibrate_teleop_type)
        return box

    def _make_setup_motors_common_panel(self) -> QWidget:
        box = QGroupBox("设置电机")
        form = QFormLayout(box)
        self._setup_form_layout(form)
        self.common_setup_motors_target = QComboBox()
        self.common_setup_motors_target.addItems(["robot", "teleop"])
        self.common_setup_motors_robot_type = QLineEdit("wheeled_arm")
        self.common_setup_motors_teleop_type = QLineEdit("wheeled_arm_pico")
        form.addRow("目标", self.common_setup_motors_target)
        form.addRow("robot.type", self.common_setup_motors_robot_type)
        form.addRow("teleop.type", self.common_setup_motors_teleop_type)
        return box

    def _make_find_joint_limits_common_panel(self) -> QWidget:
        box = QGroupBox("查关节限位")
        form = QFormLayout(box)
        self._setup_form_layout(form)
        self.common_joint_urdf_path = PathPicker("real_robot.urdf", directory=False)
        self.common_joint_urdf_path.setText(str(DEFAULT_WHEELED_ARM_URDF))
        self.common_joint_target_frame = QLineEdit("gripper")
        self.common_joint_teleop_time = self._spin(1, 36000, 30)
        self.common_joint_warmup_time = self._spin(0, 36000, 5)
        self.common_joint_fps = self._spin(1, 240, 30)
        self.common_joint_lcm_url = QLineEdit("udpm://239.255.76.67:8880?ttl=1")
        form.addRow("URDF", self.common_joint_urdf_path)
        form.addRow("目标 frame", self.common_joint_target_frame)
        form.addRow("采样秒数", self.common_joint_teleop_time)
        form.addRow("预热秒数", self.common_joint_warmup_time)
        form.addRow("FPS", self.common_joint_fps)
        form.addRow("LCM URL", self.common_joint_lcm_url)
        return box

    def _make_setup_can_common_panel(self) -> QWidget:
        box = QGroupBox("设置/测试 CAN")
        form = QFormLayout(box)
        self._setup_form_layout(form)
        self.common_can_mode = QComboBox()
        self.common_can_mode.addItems(["test", "setup", "speed"])
        self.common_can_interfaces = QLineEdit("can0")
        self.common_can_bitrate = self._spin(1, 10000000, 1000000)
        self.common_can_data_bitrate = self._spin(1, 20000000, 5000000)
        self.common_can_use_fd = QCheckBox("CAN FD")
        self.common_can_use_fd.setChecked(True)
        self.common_can_motor_ids = QLineEdit()
        self.common_can_motor_ids.setPlaceholderText("可选：1,2,3,4,5,6,7,8")
        self.common_can_timeout = self._double_spin(0.1, 30.0, 1.0, 1)
        self.common_can_speed_iterations = self._spin(1, 1000000, 100)
        form.addRow("mode", self.common_can_mode)
        form.addRow("interfaces", self.common_can_interfaces)
        form.addRow("bitrate", self.common_can_bitrate)
        form.addRow("data bitrate", self.common_can_data_bitrate)
        form.addRow("", self.common_can_use_fd)
        form.addRow("motor ids", self.common_can_motor_ids)
        form.addRow("timeout", self.common_can_timeout)
        form.addRow("speed iters", self.common_can_speed_iterations)
        return box

    def _make_train_common_panel(self) -> QWidget:
        box = QGroupBox("训练策略")
        form = QFormLayout(box)
        self._setup_form_layout(form)
        self.common_train_dataset_repo_id = QLineEdit()
        self.common_train_dataset_repo_id.setPlaceholderText("训练数据集 Repo ID，例如 user/my_dataset")
        self.common_train_policy_type = QLineEdit("act")
        self.common_train_output_dir = PathPicker("outputs/train/wheeled_arm_policy")
        self.common_train_output_dir.setText("outputs/train/wheeled_arm_policy")
        self.common_train_steps = self._spin(1, 100000000, 10000)
        self.common_train_batch_size = self._spin(1, 1000000, 8)
        self.common_train_device = QLineEdit("cuda")
        self.common_train_wandb = QCheckBox("启用 WandB")
        self.common_train_push_to_hub = QCheckBox("完成后上传 Hub")
        form.addRow("dataset.repo_id", self.common_train_dataset_repo_id)
        form.addRow("policy.type", self.common_train_policy_type)
        form.addRow("output_dir", self.common_train_output_dir)
        form.addRow("steps", self.common_train_steps)
        form.addRow("batch_size", self.common_train_batch_size)
        form.addRow("policy.device", self.common_train_device)
        form.addRow("", self.common_train_wandb)
        form.addRow("", self.common_train_push_to_hub)
        return box

    def _make_eval_common_panel(self) -> QWidget:
        box = QGroupBox("评估策略")
        form = QFormLayout(box)
        self._setup_form_layout(form)
        self.common_eval_policy_path = QLineEdit()
        self.common_eval_policy_path.setPlaceholderText("Hub repo 或 pretrained_model 路径")
        self.common_eval_env_type = QLineEdit("pusht")
        self.common_eval_n_episodes = self._spin(1, 1000000, 10)
        self.common_eval_batch_size = self._spin(1, 1000000, 10)
        self.common_eval_device = QLineEdit("cuda")
        self.common_eval_output_dir = PathPicker("outputs/eval")
        form.addRow("policy.path", self.common_eval_policy_path)
        form.addRow("env.type", self.common_eval_env_type)
        form.addRow("episodes", self.common_eval_n_episodes)
        form.addRow("batch_size", self.common_eval_batch_size)
        form.addRow("policy.device", self.common_eval_device)
        form.addRow("output_dir", self.common_eval_output_dir)
        return box

    def _make_rollout_common_panel(self) -> QWidget:
        box = QGroupBox("策略 Rollout")
        form = QFormLayout(box)
        self._setup_form_layout(form)
        self.common_rollout_strategy = QComboBox()
        self.common_rollout_strategy.addItems(["base", "sentry", "highlight", "dagger", "episodic"])
        self.common_rollout_policy_path = QLineEdit()
        self.common_rollout_policy_path.setPlaceholderText("Hub repo 或 pretrained_model 路径")
        self.common_rollout_robot_type = QLineEdit("wheeled_arm")
        self.common_rollout_teleop_type = QLineEdit("wheeled_arm_pico")
        self.common_rollout_dataset_repo_id = QLineEdit()
        self.common_rollout_dataset_repo_id.setPlaceholderText("记录 rollout 数据时填写，例如 user/rollout_data")
        self.common_rollout_task = QLineEdit("PICO rollout wheeled arm")
        self.common_rollout_duration = self._spin(0, 360000, 30)
        self.common_rollout_duration.setSpecialValueText("不限时")
        self.common_rollout_fps = self._spin(1, 240, 30)
        self.common_rollout_inference = QComboBox()
        self.common_rollout_inference.addItems(["sync", "rtc"])
        self.common_rollout_display_data = QCheckBox("显示 Rerun/Foxglove 数据")
        self.common_rollout_display_mode = QComboBox()
        self.common_rollout_display_mode.addItems(["rerun", "foxglove"])
        form.addRow("strategy.type", self.common_rollout_strategy)
        form.addRow("policy.path", self.common_rollout_policy_path)
        form.addRow("robot.type", self.common_rollout_robot_type)
        form.addRow("teleop.type", self.common_rollout_teleop_type)
        form.addRow("dataset.repo_id", self.common_rollout_dataset_repo_id)
        form.addRow("task", self.common_rollout_task)
        form.addRow("duration", self.common_rollout_duration)
        form.addRow("fps", self.common_rollout_fps)
        form.addRow("inference.type", self.common_rollout_inference)
        form.addRow("", self.common_rollout_display_data)
        form.addRow("显示后端", self.common_rollout_display_mode)
        return box

    def _make_annotate_common_panel(self) -> QWidget:
        box = QGroupBox("数据标注")
        form = QFormLayout(box)
        self._setup_form_layout(form)
        self.common_annotate_repo_id = QLineEdit()
        self.common_annotate_repo_id.setPlaceholderText("Hub 数据集 Repo ID；root 为空时使用")
        self.common_annotate_root = PathPicker("本地数据集 root")
        self.common_annotate_new_repo_id = QLineEdit()
        self.common_annotate_new_repo_id.setPlaceholderText("可选：上传到新数据集")
        self.common_annotate_vlm_model = QLineEdit("Qwen/Qwen3.6-27B")
        self.common_annotate_camera_key = QLineEdit()
        self.common_annotate_camera_key.setPlaceholderText("可选：observation.images.front")
        self.common_annotate_episode_parallelism = self._spin(1, 1024, 16)
        self.common_annotate_push_to_hub = QCheckBox("完成后上传 Hub")
        self.common_annotate_skip_validation = QCheckBox("跳过校验")
        form.addRow("repo_id", self.common_annotate_repo_id)
        form.addRow("root", self.common_annotate_root)
        form.addRow("new_repo_id", self.common_annotate_new_repo_id)
        form.addRow("vlm.model_id", self.common_annotate_vlm_model)
        form.addRow("vlm.camera_key", self.common_annotate_camera_key)
        form.addRow("并发 episode", self.common_annotate_episode_parallelism)
        form.addRow("", self.common_annotate_push_to_hub)
        form.addRow("", self.common_annotate_skip_validation)
        return box

    def _make_imgtransform_common_panel(self) -> QWidget:
        box = QGroupBox("图像增强预览")
        form = QFormLayout(box)
        self._setup_form_layout(form)
        self.common_imgtransform_repo_id = QLineEdit()
        self.common_imgtransform_repo_id.setPlaceholderText("数据集 Repo ID")
        self.common_imgtransform_episodes = QLineEdit("0")
        self.common_imgtransform_episodes.setPlaceholderText("0,1,2 或留空")
        self.common_imgtransform_output_dir = PathPicker("outputs/image_transforms")
        self.common_imgtransform_output_dir.setText("outputs/image_transforms")
        self.common_imgtransform_n_examples = self._spin(1, 1000, 5)
        self.common_imgtransform_enable = QCheckBox("启用 image_transforms")
        self.common_imgtransform_enable.setChecked(True)
        form.addRow("repo_id", self.common_imgtransform_repo_id)
        form.addRow("episodes", self.common_imgtransform_episodes)
        form.addRow("output_dir", self.common_imgtransform_output_dir)
        form.addRow("examples", self.common_imgtransform_n_examples)
        form.addRow("", self.common_imgtransform_enable)
        return box

    def _make_quantile_common_panel(self) -> QWidget:
        box = QGroupBox("补充分位数统计")
        form = QFormLayout(box)
        self._setup_form_layout(form)
        self.common_quantile_repo_id = QLineEdit()
        self.common_quantile_repo_id.setPlaceholderText("数据集 Repo ID")
        self.common_quantile_root = PathPicker("可选：本地数据集 root")
        self.common_quantile_overwrite = QCheckBox("覆盖已有分位数统计")
        self.common_quantile_no_sampling = QCheckBox("逐帧计算图像/视频")
        self.common_quantile_skip_images = QCheckBox("跳过图像/视频")
        form.addRow("Repo ID", self.common_quantile_repo_id)
        form.addRow("root", self.common_quantile_root)
        form.addRow("", self.common_quantile_overwrite)
        form.addRow("", self.common_quantile_no_sampling)
        form.addRow("", self.common_quantile_skip_images)
        return box

    def _make_convert_dcp_common_panel(self) -> QWidget:
        box = QGroupBox("转换 DCP Checkpoint")
        form = QFormLayout(box)
        self._setup_form_layout(form)
        self.common_dcp_checkpoint_dir = PathPicker("checkpoint step 或 pretrained_model 目录")
        self.common_dcp_delete = QCheckBox("转换成功后删除 DCP 分片")
        self.common_dcp_push_repo = QLineEdit()
        self.common_dcp_push_repo.setPlaceholderText("可选：上传模型 Repo ID")
        self.common_dcp_private = QCheckBox("私有模型 Repo")
        form.addRow("checkpoint_dir", self.common_dcp_checkpoint_dir)
        form.addRow("", self.common_dcp_delete)
        form.addRow("push_to_hub", self.common_dcp_push_repo)
        form.addRow("", self.common_dcp_private)
        return box

    def _make_train_tokenizer_common_panel(self) -> QWidget:
        box = QGroupBox("训练 FAST Tokenizer")
        form = QFormLayout(box)
        self._setup_form_layout(form)
        self.common_tokenizer_repo_id = QLineEdit()
        self.common_tokenizer_repo_id.setPlaceholderText("训练数据集 Repo ID")
        self.common_tokenizer_root = PathPicker("可选：本地数据集 root")
        self.common_tokenizer_output_dir = PathPicker("fast_tokenizer_xxx")
        self.common_tokenizer_action_horizon = self._spin(1, 10000, 10)
        self.common_tokenizer_sample_fraction = self._double_spin(0.001, 1.0, 0.1, 3)
        self.common_tokenizer_encoded_dims = QLineEdit("0:6,7:23")
        self.common_tokenizer_relative_dims = QLineEdit()
        self.common_tokenizer_relative_dims.setPlaceholderText("可选：0,1,2,3,4,5")
        self.common_tokenizer_vocab_size = self._spin(1, 1000000, 1024)
        self.common_tokenizer_scale = self._double_spin(0.01, 10000.0, 10.0, 2)
        self.common_tokenizer_push_to_hub = QCheckBox("完成后上传 Hub")
        self.common_tokenizer_hub_repo_id = QLineEdit()
        self.common_tokenizer_hub_repo_id.setPlaceholderText("可选：user/fast_tokenizer")
        form.addRow("repo_id", self.common_tokenizer_repo_id)
        form.addRow("root", self.common_tokenizer_root)
        form.addRow("output_dir", self.common_tokenizer_output_dir)
        form.addRow("action_horizon", self.common_tokenizer_action_horizon)
        form.addRow("sample_fraction", self.common_tokenizer_sample_fraction)
        form.addRow("encoded_dims", self.common_tokenizer_encoded_dims)
        form.addRow("relative_dims", self.common_tokenizer_relative_dims)
        form.addRow("vocab_size", self.common_tokenizer_vocab_size)
        form.addRow("scale", self.common_tokenizer_scale)
        form.addRow("", self.common_tokenizer_push_to_hub)
        form.addRow("hub_repo_id", self.common_tokenizer_hub_repo_id)
        return box

    def _make_custom_common_panel(self) -> QWidget:
        box = QGroupBox("自定义脚本")
        form = QFormLayout(box)
        self._setup_form_layout(form)
        self.common_custom_module = QLineEdit("lerobot.scripts.lerobot_info")
        self.common_custom_module.setPlaceholderText("例如 lerobot.scripts.lerobot_train")
        form.addRow("Python 模块", self.common_custom_module)
        return box

    def _make_info_edit_panel(self) -> QWidget:
        box = QGroupBox("查看信息")
        form = QFormLayout(box)
        self._setup_form_layout(form)
        self.edit_info_show_features = QCheckBox("显示 feature 详情")
        form.addRow("", self.edit_info_show_features)
        return box

    def _make_delete_episodes_edit_panel(self) -> QWidget:
        box = QGroupBox("删除 Episode")
        form = QFormLayout(box)
        self._setup_form_layout(form)
        self.edit_delete_episode_indices = QLineEdit()
        self.edit_delete_episode_indices.setPlaceholderText("例如 0,2,5 或 [0, 2, 5]")
        form.addRow("Episode", self.edit_delete_episode_indices)
        return box

    def _make_split_edit_panel(self) -> QWidget:
        box = QGroupBox("拆分数据集")
        form = QFormLayout(box)
        self._setup_form_layout(form)
        self.edit_split_splits = QPlainTextEdit()
        self.edit_split_splits.setPlaceholderText('{"train": 0.8, "val": 0.2} 或 {"train": [0,1], "val": [2]}')
        self.edit_split_splits.setFixedHeight(86)
        form.addRow("splits JSON", self.edit_split_splits)
        return box

    def _make_merge_edit_panel(self) -> QWidget:
        box = QGroupBox("合并数据集")
        form = QFormLayout(box)
        self._setup_form_layout(form)
        self.edit_merge_repo_ids = QLineEdit()
        self.edit_merge_repo_ids.setPlaceholderText("repo1,repo2 或 [\"repo1\", \"repo2\"]")
        self.edit_merge_roots = QLineEdit()
        self.edit_merge_roots.setPlaceholderText("可选：/path/to/repo1,/path/to/repo2")
        self.edit_merge_concat_videos = QCheckBox("合并视频文件")
        self.edit_merge_concat_videos.setChecked(True)
        self.edit_merge_concat_data = QCheckBox("合并 parquet 数据")
        self.edit_merge_concat_data.setChecked(True)
        form.addRow("Repo IDs", self.edit_merge_repo_ids)
        form.addRow("Roots", self.edit_merge_roots)
        form.addRow("", self.edit_merge_concat_videos)
        form.addRow("", self.edit_merge_concat_data)
        return box

    def _make_remove_feature_edit_panel(self) -> QWidget:
        box = QGroupBox("删除 Feature")
        form = QFormLayout(box)
        self._setup_form_layout(form)
        self.edit_remove_features = QLineEdit()
        self.edit_remove_features.setPlaceholderText("例如 observation.images.front 或 [\"observation.images.front\"]")
        form.addRow("Feature", self.edit_remove_features)
        return box

    def _make_modify_tasks_edit_panel(self) -> QWidget:
        box = QGroupBox("修改任务文本")
        form = QFormLayout(box)
        self._setup_form_layout(form)
        self.edit_modify_new_task = QLineEdit()
        self.edit_modify_new_task.setPlaceholderText("所有 episode 使用的新任务文本，可留空")
        self.edit_modify_episode_tasks = QPlainTextEdit()
        self.edit_modify_episode_tasks.setPlaceholderText('可选：{"0": "任务 A", "1": "任务 B"}')
        self.edit_modify_episode_tasks.setFixedHeight(72)
        self.edit_modify_replacements = QPlainTextEdit()
        self.edit_modify_replacements.setPlaceholderText('可选：{"旧任务": "新任务"}')
        self.edit_modify_replacements.setFixedHeight(72)
        form.addRow("默认任务", self.edit_modify_new_task)
        form.addRow("按集任务 JSON", self.edit_modify_episode_tasks)
        form.addRow("替换 JSON", self.edit_modify_replacements)
        return box

    def _make_convert_video_edit_panel(self) -> QWidget:
        box = QGroupBox("图片转视频")
        form = QFormLayout(box)
        self._setup_form_layout(form)
        self.edit_convert_episode_indices = QLineEdit()
        self.edit_convert_episode_indices.setPlaceholderText("可选：0,1,2")
        self.edit_convert_num_workers = self._spin(0, 128, 4)
        self.edit_convert_max_episodes = self._spin(0, 1000000, 0)
        self.edit_convert_max_episodes.setSpecialValueText("默认")
        self.edit_convert_max_frames = self._spin(0, 100000000, 0)
        self.edit_convert_max_frames.setSpecialValueText("默认")
        form.addRow("Episode", self.edit_convert_episode_indices)
        form.addRow("进程数", self.edit_convert_num_workers)
        form.addRow("每批集数", self.edit_convert_max_episodes)
        form.addRow("每批帧数", self.edit_convert_max_frames)
        return box

    def _make_recompute_stats_edit_panel(self) -> QWidget:
        box = QGroupBox("重算统计")
        form = QFormLayout(box)
        self._setup_form_layout(form)
        self.edit_stats_skip_image_video = QCheckBox("跳过图像/视频统计")
        self.edit_stats_skip_image_video.setChecked(True)
        self.edit_stats_relative_action = QCheckBox("计算 relative action 统计")
        self.edit_stats_exclude_joints = QLineEdit()
        self.edit_stats_exclude_joints.setPlaceholderText("可选：gripper,left_gripper")
        self.edit_stats_chunk_size = self._spin(1, 1000000, 50)
        self.edit_stats_num_workers = self._spin(0, 128, 0)
        self.edit_stats_overwrite = QCheckBox("允许原地覆盖")
        form.addRow("", self.edit_stats_skip_image_video)
        form.addRow("", self.edit_stats_relative_action)
        form.addRow("排除关节", self.edit_stats_exclude_joints)
        form.addRow("chunk size", self.edit_stats_chunk_size)
        form.addRow("进程数", self.edit_stats_num_workers)
        form.addRow("", self.edit_stats_overwrite)
        return box

    def _make_reencode_videos_edit_panel(self) -> QWidget:
        box = QGroupBox("重编码视频")
        form = QFormLayout(box)
        self._setup_form_layout(form)
        self.edit_reencode_vcodec = QLineEdit("h264")
        self.edit_reencode_pix_fmt = QLineEdit("yuv420p")
        self.edit_reencode_crf = self._spin(0, 63, 23)
        self.edit_reencode_num_workers = self._spin(0, 128, 0)
        self.edit_reencode_encoder_threads = self._spin(0, 128, 0)
        self.edit_reencode_encoder_threads.setSpecialValueText("默认")
        self.edit_reencode_overwrite = QCheckBox("允许原地覆盖")
        form.addRow("vcodec", self.edit_reencode_vcodec)
        form.addRow("pix fmt", self.edit_reencode_pix_fmt)
        form.addRow("crf", self.edit_reencode_crf)
        form.addRow("进程数", self.edit_reencode_num_workers)
        form.addRow("编码线程", self.edit_reencode_encoder_threads)
        form.addRow("", self.edit_reencode_overwrite)
        return box

    def _make_right_panel(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("RightPanel")
        self._apply_panel_shadow(panel)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        status_box = QGroupBox("状态")
        status_layout = QVBoxLayout(status_box)
        self.status_label = QLabel("未运行")
        self.status_label.setObjectName("StatusPill")
        self.dataset_hint = QLabel(f"默认数据目录：{HF_LEROBOT_HOME}")
        self.dataset_hint.setWordWrap(True)
        self.dataset_hint.setObjectName("MutedLabel")
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.dataset_hint)
        layout.addWidget(status_box)

        preview_box = QGroupBox("命令预览")
        preview_layout = QVBoxLayout(preview_box)
        self.preview_tabs = QTabWidget()
        self.record_command_preview = QPlainTextEdit()
        self.record_command_preview.setReadOnly(True)
        self.viewer_command_preview = QPlainTextEdit()
        self.viewer_command_preview.setReadOnly(True)
        self.edit_command_preview = QPlainTextEdit()
        self.edit_command_preview.setReadOnly(True)
        self.conversion_command_preview = QPlainTextEdit()
        self.conversion_command_preview.setReadOnly(True)
        self.common_command_preview = QPlainTextEdit()
        self.common_command_preview.setReadOnly(True)
        for edit in (
            self.record_command_preview,
            self.viewer_command_preview,
            self.edit_command_preview,
            self.conversion_command_preview,
            self.common_command_preview,
        ):
            edit.setObjectName("CommandPreview")
            edit.setFont(QFont("monospace", 10))
            edit.setMaximumBlockCount(1000)
        self.preview_tabs.addTab(self.record_command_preview, "采集")
        self.preview_tabs.addTab(self.viewer_command_preview, "查看")
        self.preview_tabs.addTab(self.edit_command_preview, "编辑")
        self.preview_tabs.addTab(self.conversion_command_preview, "转换")
        self.preview_tabs.addTab(self.common_command_preview, "常用")
        preview_layout.addWidget(self.preview_tabs)
        layout.addWidget(preview_box, 1)

        log_box = QGroupBox("运行日志")
        log_layout = QVBoxLayout(log_box)
        self.log_view = QTextEdit()
        self.log_view.setObjectName("LogView")
        self.log_view.setReadOnly(True)
        self.log_view.setFont(QFont("monospace", 10))
        self.log_view.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        log_layout.addWidget(self.log_view)

        log_buttons = QHBoxLayout()
        clear_btn = QPushButton("清空日志")
        clear_btn.clicked.connect(self.log_view.clear)
        copy_log_btn = QPushButton("复制日志")
        copy_log_btn.clicked.connect(lambda: self.copy_command(self.log_view.toPlainText()))
        log_buttons.addWidget(clear_btn)
        log_buttons.addWidget(copy_log_btn)
        log_buttons.addStretch(1)
        log_layout.addLayout(log_buttons)
        layout.addWidget(log_box, 2)

        return panel

    def _spin(self, minimum: int, maximum: int, value: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        return spin

    def _double_spin(
        self, minimum: float, maximum: float, value: float, decimals: int = 1
    ) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setValue(value)
        return spin

    def _setup_form_layout(self, form: QFormLayout) -> None:
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        form.setFormAlignment(Qt.AlignmentFlag.AlignTop)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(10)
        form.setContentsMargins(12, 26, 12, 14)

    def _scrollable(self, content: QWidget) -> QScrollArea:
        content.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.MinimumExpanding)
        scroll = QScrollArea()
        scroll.setWidget(content)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        return scroll

    def _record_widgets(self) -> tuple[QWidget, ...]:
        return (
            self.repo_id,
            self.task,
            self.num_episodes,
            self.episode_time_s,
            self.reset_time_s,
            self.fps,
            self.resume,
            self.no_stamp,
            self.video,
            self.streaming_encoding,
            self.push_to_hub,
            self.private,
            self.play_sounds,
            self.wait_for_episode_start,
            self.display_data,
            self.display_mode,
            self.display_ip,
            self.display_port,
            self.display_compressed_images,
            self.viser,
            self.rerun_robot,
            self.mock_xr,
            self.lcm_url,
            self.camera_override,
            self.camera_topic,
            self.camera_width,
            self.camera_height,
            self.camera_fps,
            self.advanced_args,
        )

    def _edit_widgets(self) -> tuple[QWidget, ...]:
        return (
            self.edit_repo_id,
            self.edit_new_repo_id,
            self.edit_push_to_hub,
            self.edit_info_show_features,
            self.edit_delete_episode_indices,
            self.edit_split_splits,
            self.edit_merge_repo_ids,
            self.edit_merge_roots,
            self.edit_merge_concat_videos,
            self.edit_merge_concat_data,
            self.edit_remove_features,
            self.edit_modify_new_task,
            self.edit_modify_episode_tasks,
            self.edit_modify_replacements,
            self.edit_convert_episode_indices,
            self.edit_convert_num_workers,
            self.edit_convert_max_episodes,
            self.edit_convert_max_frames,
            self.edit_stats_skip_image_video,
            self.edit_stats_relative_action,
            self.edit_stats_exclude_joints,
            self.edit_stats_chunk_size,
            self.edit_stats_num_workers,
            self.edit_stats_overwrite,
            self.edit_reencode_vcodec,
            self.edit_reencode_pix_fmt,
            self.edit_reencode_crf,
            self.edit_reencode_num_workers,
            self.edit_reencode_encoder_threads,
            self.edit_reencode_overwrite,
            self.edit_advanced_args,
        )

    def _conversion_widgets(self) -> tuple[QWidget, ...]:
        return (
            self.conversion_type,
            self.openx_repo_id,
            self.openx_robot_type,
            self.openx_fps,
            self.openx_use_videos,
            self.openx_push_to_hub,
            self.openx_image_processes,
            self.openx_image_threads,
            self.agibot_eef_type,
            self.agibot_task_ids,
            self.agibot_cpus,
            self.agibot_save_depth,
            self.agibot_debug,
            self.robomind_benchmark,
            self.robomind_embodiments,
            self.robomind_cpus,
            self.robomind_save_depth,
            self.robomind_debug,
            self.libero_src_paths,
            self.libero_executor,
            self.libero_cpus,
            self.libero_tasks_per_job,
            self.libero_workers,
            self.libero_repo_id,
            self.libero_push_to_hub,
            self.libero_debug,
            self.rlds_task_name,
            self.rlds_encoding,
            self.rlds_version,
            self.rlds_enable_beam,
            self.rlds_beam_mode,
            self.rlds_beam_workers,
            self.rlds_homepage,
            self.rlds_citation,
            self.rlds_description,
            self.v16_repo_id,
            self.v16_task_mode,
            self.v16_task_value,
            self.v16_robot,
            self.v16_license,
            self.v16_test_branch,
            self.version_repo_id,
            self.version_push_to_hub,
            self.version_delete_old_stats,
            self.version_branch,
            self.version_num_workers,
            self.v30_repo_id,
            self.v30_push_to_hub,
            self.v30_force,
            self.v30_branch,
            self.v30_data_size,
            self.v30_video_size,
            self.v3021_repo_id,
            self.conversion_advanced_args,
        )

    def _common_widgets(self) -> tuple[QWidget, ...]:
        return (
            self.common_command,
            self.common_advanced_args,
            self.common_camera_type,
            self.common_camera_record_time,
            self.common_camera_warmup,
            self.common_teleop_fps,
            self.common_teleop_time_s,
            self.common_teleop_display_data,
            self.common_teleop_display_mode,
            self.common_teleop_viser,
            self.common_teleop_rerun_robot,
            self.common_teleop_mock_xr,
            self.common_teleop_lcm_url,
            self.common_replay_repo_id,
            self.common_replay_episode,
            self.common_replay_fps,
            self.common_replay_play_sounds,
            self.common_replay_lcm_url,
            self.common_calibrate_target,
            self.common_calibrate_robot_type,
            self.common_calibrate_teleop_type,
            self.common_setup_motors_target,
            self.common_setup_motors_robot_type,
            self.common_setup_motors_teleop_type,
            self.common_joint_target_frame,
            self.common_joint_teleop_time,
            self.common_joint_warmup_time,
            self.common_joint_fps,
            self.common_joint_lcm_url,
            self.common_can_mode,
            self.common_can_interfaces,
            self.common_can_bitrate,
            self.common_can_data_bitrate,
            self.common_can_use_fd,
            self.common_can_motor_ids,
            self.common_can_timeout,
            self.common_can_speed_iterations,
            self.common_train_dataset_repo_id,
            self.common_train_policy_type,
            self.common_train_steps,
            self.common_train_batch_size,
            self.common_train_device,
            self.common_train_wandb,
            self.common_train_push_to_hub,
            self.common_eval_policy_path,
            self.common_eval_env_type,
            self.common_eval_n_episodes,
            self.common_eval_batch_size,
            self.common_eval_device,
            self.common_rollout_strategy,
            self.common_rollout_policy_path,
            self.common_rollout_robot_type,
            self.common_rollout_teleop_type,
            self.common_rollout_dataset_repo_id,
            self.common_rollout_task,
            self.common_rollout_duration,
            self.common_rollout_fps,
            self.common_rollout_inference,
            self.common_rollout_display_data,
            self.common_rollout_display_mode,
            self.common_annotate_repo_id,
            self.common_annotate_new_repo_id,
            self.common_annotate_vlm_model,
            self.common_annotate_camera_key,
            self.common_annotate_episode_parallelism,
            self.common_annotate_push_to_hub,
            self.common_annotate_skip_validation,
            self.common_imgtransform_repo_id,
            self.common_imgtransform_episodes,
            self.common_imgtransform_n_examples,
            self.common_imgtransform_enable,
            self.common_quantile_repo_id,
            self.common_quantile_overwrite,
            self.common_quantile_no_sampling,
            self.common_quantile_skip_images,
            self.common_dcp_delete,
            self.common_dcp_push_repo,
            self.common_dcp_private,
            self.common_tokenizer_repo_id,
            self.common_tokenizer_action_horizon,
            self.common_tokenizer_sample_fraction,
            self.common_tokenizer_encoded_dims,
            self.common_tokenizer_relative_dims,
            self.common_tokenizer_vocab_size,
            self.common_tokenizer_scale,
            self.common_tokenizer_push_to_hub,
            self.common_tokenizer_hub_repo_id,
            self.common_custom_module,
        )

    def _connect_preview_signal(self, widget: QWidget, callback) -> None:
        if isinstance(widget, QLineEdit):
            widget.textChanged.connect(callback)
        elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            widget.valueChanged.connect(callback)
        elif isinstance(widget, QCheckBox):
            widget.toggled.connect(callback)
        elif isinstance(widget, QComboBox):
            widget.currentIndexChanged.connect(callback)
        elif isinstance(widget, QPlainTextEdit):
            widget.textChanged.connect(callback)

    def _connect_runners(self) -> None:
        self.record_runner.started.connect(lambda command: self._on_started("采集", command))
        self.record_runner.output.connect(lambda line: self.append_log("采集", line))
        self.record_runner.failed.connect(lambda message: self.append_log("采集", message))
        self.record_runner.finished.connect(self._on_record_finished)

        self.viewer_runner.started.connect(lambda command: self._on_started("查看", command))
        self.viewer_runner.output.connect(lambda line: self.append_log("查看", line))
        self.viewer_runner.failed.connect(lambda message: self.append_log("查看", message))
        self.viewer_runner.finished.connect(self._on_viewer_finished)

        self.edit_runner.started.connect(lambda command: self._on_started("编辑", command))
        self.edit_runner.output.connect(lambda line: self.append_log("编辑", line))
        self.edit_runner.failed.connect(lambda message: self.append_log("编辑", message))
        self.edit_runner.finished.connect(self._on_edit_finished)

        self.conversion_runner.started.connect(lambda command: self._on_started("转换", command))
        self.conversion_runner.output.connect(lambda line: self.append_log("转换", line))
        self.conversion_runner.failed.connect(lambda message: self.append_log("转换", message))
        self.conversion_runner.finished.connect(self._on_conversion_finished)

        self.common_runner.started.connect(lambda command: self._on_started("常用", command))
        self.common_runner.output.connect(lambda line: self.append_log("常用", line))
        self.common_runner.failed.connect(lambda message: self.append_log("常用", message))
        self.common_runner.finished.connect(self._on_common_finished)

    def _load_settings(self) -> None:
        self.repo_id.setText(self.settings.value("record/repo_id", self.repo_id.text()))
        self.task.setText(self.settings.value("record/task", self.task.text()))
        self.dataset_root.setText(self.settings.value("record/root", ""))
        self.lcm_url.setText(self.settings.value("record/lcm_url", self.lcm_url.text()))
        self.advanced_args.setPlainText(self.settings.value("record/advanced_args", ""))
        self.wait_for_episode_start.setChecked(
            _settings_bool(
                self.settings.value(
                    "record/wait_for_episode_start", self.wait_for_episode_start.isChecked()
                ),
                self.wait_for_episode_start.isChecked(),
            )
        )
        self.viewer_repo_id.setText(self.settings.value("viewer/repo_id", ""))
        self.viewer_root.setText(self.settings.value("viewer/root", ""))
        self.edit_repo_id.setText(self.settings.value("edit/repo_id", ""))
        self.edit_root.setText(self.settings.value("edit/root", ""))
        self.edit_new_repo_id.setText(self.settings.value("edit/new_repo_id", ""))
        self.edit_new_root.setText(self.settings.value("edit/new_root", ""))
        self.edit_advanced_args.setPlainText(self.settings.value("edit/advanced_args", ""))
        self.conversion_python.setText(self.settings.value("conversion/python", self.conversion_python.text()))
        saved_backend = str(self.settings.value("conversion/backend", self.conversion_backend.text()) or "")
        backend_path = saved_backend if _is_conversion_backend(saved_backend) else str(ANY4LEROBOT_BACKEND)
        self.conversion_backend.setText(backend_path)
        if saved_backend != backend_path:
            self.settings.setValue("conversion/backend", backend_path)
        self.openx_repo_id.setText(self.settings.value("conversion/openx_repo_id", ""))
        self.openx_local_dir.setText(self.settings.value("conversion/openx_local_dir", ""))
        self.rlds_task_name.setText(self.settings.value("conversion/rlds_task_name", self.rlds_task_name.text()))
        self.version_repo_id.setText(self.settings.value("conversion/version_repo_id", ""))
        self.v30_repo_id.setText(self.settings.value("conversion/v30_repo_id", ""))
        self.v3021_repo_id.setText(self.settings.value("conversion/v3021_repo_id", ""))
        self.conversion_advanced_args.setPlainText(self.settings.value("conversion/advanced_args", ""))
        self.common_replay_repo_id.setText(self.settings.value("common/replay_repo_id", ""))
        self.common_replay_root.setText(self.settings.value("common/replay_root", ""))
        self.common_custom_module.setText(
            self.settings.value("common/custom_module", self.common_custom_module.text())
        )
        self.common_advanced_args.setPlainText(self.settings.value("common/advanced_args", ""))

    def _save_settings(self) -> None:
        self.settings.setValue("record/repo_id", self.repo_id.text())
        self.settings.setValue("record/task", self.task.text())
        self.settings.setValue("record/root", self.dataset_root.text())
        self.settings.setValue("record/lcm_url", self.lcm_url.text())
        self.settings.setValue("record/advanced_args", self.advanced_args.toPlainText())
        self.settings.setValue(
            "record/wait_for_episode_start", self.wait_for_episode_start.isChecked()
        )
        self.settings.setValue("viewer/repo_id", self.viewer_repo_id.text())
        self.settings.setValue("viewer/root", self.viewer_root.text())
        self.settings.setValue("edit/repo_id", self.edit_repo_id.text())
        self.settings.setValue("edit/root", self.edit_root.text())
        self.settings.setValue("edit/new_repo_id", self.edit_new_repo_id.text())
        self.settings.setValue("edit/new_root", self.edit_new_root.text())
        self.settings.setValue("edit/advanced_args", self.edit_advanced_args.toPlainText())
        self.settings.setValue("conversion/python", self.conversion_python.text())
        self.settings.setValue("conversion/backend", self.conversion_backend.text())
        self.settings.setValue("conversion/openx_repo_id", self.openx_repo_id.text())
        self.settings.setValue("conversion/openx_local_dir", self.openx_local_dir.text())
        self.settings.setValue("conversion/rlds_task_name", self.rlds_task_name.text())
        self.settings.setValue("conversion/version_repo_id", self.version_repo_id.text())
        self.settings.setValue("conversion/v30_repo_id", self.v30_repo_id.text())
        self.settings.setValue("conversion/v3021_repo_id", self.v3021_repo_id.text())
        self.settings.setValue("conversion/advanced_args", self.conversion_advanced_args.toPlainText())
        self.settings.setValue("common/replay_repo_id", self.common_replay_repo_id.text())
        self.settings.setValue("common/replay_root", self.common_replay_root.text())
        self.settings.setValue("common/custom_module", self.common_custom_module.text())
        self.settings.setValue("common/advanced_args", self.common_advanced_args.toPlainText())

    def build_record_command(self) -> list[str]:
        command = _module_command("lerobot.scripts.lerobot_record")
        command.extend(
            [
                "--robot.type=wheeled_arm",
                "--teleop.type=wheeled_arm_pico",
                f"--dataset.repo_id={self.repo_id.text().strip()}",
                f"--dataset.single_task={self.task.text().strip()}",
                f"--dataset.num_episodes={self.num_episodes.value()}",
                f"--dataset.episode_time_s={self.episode_time_s.value()}",
                f"--dataset.reset_time_s={self.reset_time_s.value()}",
                f"--dataset.fps={self.fps.value()}",
                f"--dataset.video={_bool_arg(self.video.isChecked())}",
                f"--dataset.streaming_encoding={_bool_arg(self.streaming_encoding.isChecked())}",
                f"--dataset.push_to_hub={_bool_arg(self.push_to_hub.isChecked())}",
                f"--display_data={_bool_arg(self.display_data.isChecked())}",
                f"--display_mode={self.display_mode.currentText()}",
                f"--display_compressed_images={_bool_arg(self.display_compressed_images.isChecked())}",
                f"--play_sounds={_bool_arg(self.play_sounds.isChecked())}",
                f"--resume={_bool_arg(self.resume.isChecked())}",
                f"--wait_for_episode_start={_bool_arg(self.wait_for_episode_start.isChecked())}",
                f"--teleop.visualize={_bool_arg(self.viser.isChecked())}",
                f"--teleop.rerun_visualize_robot={_bool_arg(self.rerun_robot.isChecked())}",
                f"--teleop.mock_xr={_bool_arg(self.mock_xr.isChecked())}",
            ]
        )

        if self.dataset_root.text():
            command.append(f"--dataset.root={self.dataset_root.text()}")
        if self.private.isChecked():
            command.append("--dataset.private=true")
        if self.no_stamp.isChecked():
            command.append("--dataset.no_stamp=true")
        if self.display_ip.text().strip():
            command.append(f"--display_ip={self.display_ip.text().strip()}")
        if self.display_port.value() > 0:
            command.append(f"--display_port={self.display_port.value()}")
        if self.lcm_url.text().strip():
            command.append(f"--robot.lcm_url={self.lcm_url.text().strip()}")
        if self.camera_override.isChecked():
            camera_config = (
                "{front: {type: ros2, "
                f"topic_name: {self.camera_topic.text().strip()}, "
                "node_name: wheeled_arm_front_camera, "
                f"width: {self.camera_width.value()}, "
                f"height: {self.camera_height.value()}, "
                f"fps: {self.camera_fps.value()}}}"
            )
            command.append(f"--robot.cameras={camera_config}")

        extra = self.advanced_args.toPlainText().strip()
        if extra:
            command.extend(shlex.split(extra))
        return command

    def build_viewer_command(self) -> list[str]:
        command = _module_command("lerobot.scripts.lerobot_dataset_viz")
        command.extend(
            [
                "--repo-id",
                self.viewer_repo_id.text().strip(),
                "--episode-index",
                str(self.viewer_episode.value()),
                "--display-mode",
                self.viewer_display_mode.currentText(),
            ]
        )
        if self.viewer_root.text():
            command.extend(["--root", self.viewer_root.text()])
        if self.viewer_display_mode.currentText() == "rerun":
            command.extend(["--mode", self.viewer_mode.currentText()])
            command.extend(["--grpc-port", str(self.viewer_grpc_port.value())])
        else:
            command.extend(["--host", self.viewer_host.text().strip() or "127.0.0.1"])
            if not self.viewer_autoplay.isChecked():
                command.append("--no-autoplay")
        if self.viewer_web_port.value() > 0:
            command.extend(["--web-port", str(self.viewer_web_port.value())])
        if self.viewer_compressed.isChecked():
            command.append("--display-compressed-images")
        return command

    def build_edit_command(self) -> list[str]:
        operation = EDIT_OPERATION_LABELS[self.edit_operation.currentText()]
        command = _module_command("lerobot.scripts.lerobot_edit_dataset")

        if operation != "merge":
            command.extend(["--repo_id", self.edit_repo_id.text().strip()])
            if self.edit_root.text():
                command.extend(["--root", self.edit_root.text()])

        if self.edit_new_repo_id.text().strip():
            command.extend(["--new_repo_id", self.edit_new_repo_id.text().strip()])
        if self.edit_new_root.text():
            command.extend(["--new_root", self.edit_new_root.text()])
        if self.edit_push_to_hub.isChecked():
            command.extend(["--push_to_hub", "true"])

        command.extend(["--operation.type", operation])

        if operation == "info":
            command.extend(["--operation.show_features", _bool_arg(self.edit_info_show_features.isChecked())])
        elif operation == "delete_episodes":
            indices = _list_arg(self.edit_delete_episode_indices.text(), int)
            if indices:
                command.extend(["--operation.episode_indices", indices])
        elif operation == "split":
            splits = _json_or_none(self.edit_split_splits.toPlainText())
            if splits:
                command.extend(["--operation.splits", splits])
        elif operation == "merge":
            repo_ids = _list_arg(self.edit_merge_repo_ids.text(), str)
            if repo_ids:
                command.extend(["--operation.repo_ids", repo_ids])
            roots = _list_arg(self.edit_merge_roots.text(), str)
            if roots:
                command.extend(["--operation.roots", roots])
            command.extend(["--operation.concatenate_videos", _bool_arg(self.edit_merge_concat_videos.isChecked())])
            command.extend(["--operation.concatenate_data", _bool_arg(self.edit_merge_concat_data.isChecked())])
        elif operation == "remove_feature":
            features = _list_arg(self.edit_remove_features.text(), str)
            if features:
                command.extend(["--operation.feature_names", features])
        elif operation == "modify_tasks":
            if self.edit_modify_new_task.text().strip():
                command.extend(["--operation.new_task", self.edit_modify_new_task.text().strip()])
            episode_tasks = _json_or_none(self.edit_modify_episode_tasks.toPlainText())
            if episode_tasks:
                command.extend(["--operation.episode_tasks", episode_tasks])
            replacements = _json_or_none(self.edit_modify_replacements.toPlainText())
            if replacements:
                command.extend(["--operation.task_replacements", replacements])
        elif operation == "convert_image_to_video":
            indices = _list_arg(self.edit_convert_episode_indices.text(), int)
            if indices:
                command.extend(["--operation.episode_indices", indices])
            command.extend(["--operation.num_workers", str(self.edit_convert_num_workers.value())])
            if self.edit_convert_max_episodes.value() > 0:
                command.extend(
                    ["--operation.max_episodes_per_batch", str(self.edit_convert_max_episodes.value())]
                )
            if self.edit_convert_max_frames.value() > 0:
                command.extend(["--operation.max_frames_per_batch", str(self.edit_convert_max_frames.value())])
        elif operation == "recompute_stats":
            command.extend(
                ["--operation.skip_image_video", _bool_arg(self.edit_stats_skip_image_video.isChecked())]
            )
            command.extend(["--operation.relative_action", _bool_arg(self.edit_stats_relative_action.isChecked())])
            excluded = _list_arg(self.edit_stats_exclude_joints.text(), str)
            if excluded:
                command.extend(["--operation.relative_exclude_joints", excluded])
            command.extend(["--operation.chunk_size", str(self.edit_stats_chunk_size.value())])
            command.extend(["--operation.num_workers", str(self.edit_stats_num_workers.value())])
            command.extend(["--operation.overwrite", _bool_arg(self.edit_stats_overwrite.isChecked())])
        elif operation == "reencode_videos":
            if self.edit_reencode_vcodec.text().strip():
                command.extend(["--operation.rgb_encoder.vcodec", self.edit_reencode_vcodec.text().strip()])
            if self.edit_reencode_pix_fmt.text().strip():
                command.extend(["--operation.rgb_encoder.pix_fmt", self.edit_reencode_pix_fmt.text().strip()])
            command.extend(["--operation.rgb_encoder.crf", str(self.edit_reencode_crf.value())])
            command.extend(["--operation.num_workers", str(self.edit_reencode_num_workers.value())])
            if self.edit_reencode_encoder_threads.value() > 0:
                command.extend(["--operation.encoder_threads", str(self.edit_reencode_encoder_threads.value())])
            command.extend(["--operation.overwrite", _bool_arg(self.edit_reencode_overwrite.isChecked())])

        extra = self.edit_advanced_args.toPlainText().strip()
        if extra:
            command.extend(shlex.split(extra))
        return command

    def _conversion_key(self) -> str:
        return CONVERSION_LABELS[self.conversion_type.currentText()]

    def _conversion_backend_root(self) -> Path:
        return Path(self.conversion_backend.text() or ANY4LEROBOT_BACKEND).expanduser()

    def _conversion_script_path(self) -> Path:
        script = CONVERSION_SCRIPTS[self._conversion_key()]
        return script if isinstance(script, Path) else self._conversion_backend_root() / script

    def _append_space_list_args(self, command: list[str], option: str, value: str) -> None:
        items = shlex.split(value.replace(",", " "))
        if items:
            command.append(option)
            command.extend(items)

    def build_conversion_command(self) -> list[str]:
        key = self._conversion_key()
        python = self.conversion_python.text() or sys.executable
        script_path = self._conversion_script_path()
        command = [python, str(script_path)]

        if key == "openx_to_lerobot":
            command.extend(["--raw-dir", self.openx_raw_dir.text()])
            command.extend(["--local-dir", self.openx_local_dir.text()])
            if self.openx_repo_id.text().strip():
                command.extend(["--repo-id", self.openx_repo_id.text().strip()])
            if self.openx_push_to_hub.isChecked():
                command.append("--push-to-hub")
            if self.openx_robot_type.text().strip():
                command.extend(["--robot-type", self.openx_robot_type.text().strip()])
            if self.openx_fps.value() > 0:
                command.extend(["--fps", str(self.openx_fps.value())])
            if self.openx_use_videos.isChecked():
                command.append("--use-videos")
            command.extend(["--image-writer-process", str(self.openx_image_processes.value())])
            command.extend(["--image-writer-threads", str(self.openx_image_threads.value())])
        elif key == "agibot_to_lerobot":
            command.extend(["--src-path", self.agibot_src_path.text()])
            command.extend(["--output-path", self.agibot_output_path.text()])
            command.extend(["--eef-type", self.agibot_eef_type.currentText()])
            self._append_space_list_args(command, "--task-ids", self.agibot_task_ids.text())
            command.extend(["--cpus-per-task", str(self.agibot_cpus.value())])
            if self.agibot_save_depth.isChecked():
                command.append("--save-depth")
            if self.agibot_debug.isChecked():
                command.append("--debug")
        elif key == "robomind_to_lerobot":
            command.extend(["--src-path", self.robomind_src_path.text()])
            command.extend(["--benchmark", self.robomind_benchmark.currentText()])
            command.extend(["--output-path", self.robomind_output_path.text()])
            self._append_space_list_args(command, "--embodiments", self.robomind_embodiments.text())
            command.extend(["--cpus-per-task", str(self.robomind_cpus.value())])
            if self.robomind_save_depth.isChecked():
                command.append("--save-depth")
            if self.robomind_debug.isChecked():
                command.append("--debug")
        elif key == "libero_to_lerobot":
            self._append_space_list_args(command, "--src-paths", self.libero_src_paths.text())
            command.extend(["--output-path", self.libero_output_path.text()])
            command.extend(["--executor", self.libero_executor.currentText()])
            command.extend(["--cpus-per-task", str(self.libero_cpus.value())])
            command.extend(["--tasks-per-job", str(self.libero_tasks_per_job.value())])
            command.extend(["--workers", str(self.libero_workers.value())])
            if self.libero_resume_dir.text():
                command.extend(["--resume-dir", self.libero_resume_dir.text()])
            if self.libero_repo_id.text().strip():
                command.extend(["--repo-id", self.libero_repo_id.text().strip()])
            if self.libero_push_to_hub.isChecked():
                command.append("--push-to-hub")
            if self.libero_debug.isChecked():
                command.append("--debug")
        elif key == "lerobot_to_rlds":
            command.extend(["--src-dir", self.rlds_src_dir.text()])
            command.extend(["--output-dir", self.rlds_output_dir.text()])
            command.extend(["--task-name", self.rlds_task_name.text().strip()])
            command.extend(["--encoding-format", self.rlds_encoding.currentText()])
            command.extend(["--version", self.rlds_version.text().strip()])
            if self.rlds_enable_beam.isChecked():
                command.append("--enable-beam")
            command.extend(["--beam-run-mode", self.rlds_beam_mode.currentText()])
            command.extend(["--beam-num-workers", str(self.rlds_beam_workers.value())])
            if self.rlds_homepage.text().strip():
                command.extend(["--homepage", self.rlds_homepage.text().strip()])
            if self.rlds_citation.toPlainText().strip():
                command.extend(["--citation", self.rlds_citation.toPlainText().strip()])
            if self.rlds_description.toPlainText().strip():
                command.extend(["--description", self.rlds_description.toPlainText().strip()])
        elif key == "v16_to_v20":
            command.extend(["--repo-id", self.v16_repo_id.text().strip()])
            task_option = f"--{self.v16_task_mode.currentText()}"
            task_value = self.v16_tasks_path.text() if self.v16_task_mode.currentText() == "tasks-path" else self.v16_task_value.text().strip()
            command.extend([task_option, task_value])
            if self.v16_robot.text().strip():
                command.extend(["--robot", self.v16_robot.text().strip()])
            if self.v16_local_dir.text():
                command.extend(["--local-dir", self.v16_local_dir.text()])
            if self.v16_license.text().strip():
                command.extend(["--license", self.v16_license.text().strip()])
            if self.v16_test_branch.text().strip():
                command.extend(["--test-branch", self.v16_test_branch.text().strip()])
        elif key in {"v20_to_v21", "v21_to_v20"}:
            command.extend(["--repo-id", self.version_repo_id.text().strip()])
            if self.version_root.text():
                command.extend(["--root", self.version_root.text()])
            if self.version_push_to_hub.isChecked():
                command.append("--push-to-hub")
            if self.version_delete_old_stats.isChecked():
                command.append("--delete-old-stats")
            if self.version_branch.text().strip():
                command.extend(["--branch", self.version_branch.text().strip()])
            if key == "v20_to_v21":
                command.extend(["--num-workers", str(self.version_num_workers.value())])
        elif key == "v21_to_v30":
            command.extend(["--repo-id", self.v30_repo_id.text().strip()])
            if self.v30_root.text():
                command.extend(["--root", self.v30_root.text()])
            command.extend(["--push-to-hub", _bool_arg(self.v30_push_to_hub.isChecked())])
            if self.v30_force.isChecked():
                command.append("--force-conversion")
            if self.v30_branch.text().strip():
                command.extend(["--branch", self.v30_branch.text().strip()])
            if self.v30_data_size.value() > 0:
                command.extend(["--data-file-size-in-mb", str(self.v30_data_size.value())])
            if self.v30_video_size.value() > 0:
                command.extend(["--video-file-size-in-mb", str(self.v30_video_size.value())])
        elif key == "v30_to_v21":
            command.extend(["--repo-id", self.v3021_repo_id.text().strip()])
            if self.v3021_root.text():
                command.extend(["--root", self.v3021_root.text()])

        extra = self.conversion_advanced_args.toPlainText().strip()
        if extra:
            command.extend(shlex.split(extra))
        return command

    def build_common_command(self) -> list[str]:
        command_type = COMMON_COMMAND_LABELS[self.common_command.currentText()]
        if command_type == "custom":
            module = self.common_custom_module.text().strip()
            command = _module_command(module)
        else:
            command = _module_command(COMMON_SCRIPT_MODULES[command_type])

        if command_type == "find_cameras":
            camera_type = self.common_camera_type.currentText()
            if camera_type != "全部":
                command.append(camera_type)
            if self.common_camera_output_dir.text():
                command.extend(["--output-dir", self.common_camera_output_dir.text()])
            command.extend(["--record-time-s", str(self.common_camera_record_time.value())])
            command.extend(["--warmup-s", str(self.common_camera_warmup.value())])
        elif command_type == "teleoperate":
            command.extend(
                [
                    "--robot.type=wheeled_arm",
                    "--teleop.type=wheeled_arm_pico",
                    f"--fps={self.common_teleop_fps.value()}",
                    f"--display_data={_bool_arg(self.common_teleop_display_data.isChecked())}",
                    f"--display_mode={self.common_teleop_display_mode.currentText()}",
                    f"--display_compressed_images={_bool_arg(False)}",
                    f"--teleop.visualize={_bool_arg(self.common_teleop_viser.isChecked())}",
                    f"--teleop.rerun_visualize_robot={_bool_arg(self.common_teleop_rerun_robot.isChecked())}",
                    f"--teleop.mock_xr={_bool_arg(self.common_teleop_mock_xr.isChecked())}",
                ]
            )
            if self.common_teleop_time_s.value() > 0:
                command.append(f"--teleop_time_s={self.common_teleop_time_s.value()}")
            if self.common_teleop_lcm_url.text().strip():
                command.append(f"--robot.lcm_url={self.common_teleop_lcm_url.text().strip()}")
        elif command_type == "replay":
            command.extend(
                [
                    "--robot.type=wheeled_arm",
                    f"--dataset.repo_id={self.common_replay_repo_id.text().strip()}",
                    f"--dataset.episode={self.common_replay_episode.value()}",
                    f"--dataset.fps={self.common_replay_fps.value()}",
                    f"--play_sounds={_bool_arg(self.common_replay_play_sounds.isChecked())}",
                ]
            )
            if self.common_replay_root.text():
                command.append(f"--dataset.root={self.common_replay_root.text()}")
            if self.common_replay_lcm_url.text().strip():
                command.append(f"--robot.lcm_url={self.common_replay_lcm_url.text().strip()}")
        elif command_type == "calibrate":
            if self.common_calibrate_target.currentText() == "robot":
                command.append(f"--robot.type={self.common_calibrate_robot_type.text().strip()}")
            else:
                command.append(f"--teleop.type={self.common_calibrate_teleop_type.text().strip()}")
        elif command_type == "setup_motors":
            if self.common_setup_motors_target.currentText() == "robot":
                command.append(f"--robot.type={self.common_setup_motors_robot_type.text().strip()}")
            else:
                command.append(f"--teleop.type={self.common_setup_motors_teleop_type.text().strip()}")
        elif command_type == "find_joint_limits":
            command.extend(
                [
                    "--robot.type=wheeled_arm",
                    "--teleop.type=wheeled_arm_pico",
                    f"--urdf_path={self.common_joint_urdf_path.text()}",
                    f"--target_frame_name={self.common_joint_target_frame.text().strip()}",
                    f"--teleop_time_s={self.common_joint_teleop_time.value()}",
                    f"--warmup_time_s={self.common_joint_warmup_time.value()}",
                    f"--control_loop_fps={self.common_joint_fps.value()}",
                ]
            )
            if self.common_joint_lcm_url.text().strip():
                command.append(f"--robot.lcm_url={self.common_joint_lcm_url.text().strip()}")
        elif command_type == "setup_can":
            command.extend(
                [
                    f"--mode={self.common_can_mode.currentText()}",
                    f"--interfaces={self.common_can_interfaces.text().strip()}",
                    f"--bitrate={self.common_can_bitrate.value()}",
                    f"--data_bitrate={self.common_can_data_bitrate.value()}",
                    f"--use_fd={_bool_arg(self.common_can_use_fd.isChecked())}",
                    f"--timeout={self.common_can_timeout.value()}",
                    f"--speed_iterations={self.common_can_speed_iterations.value()}",
                ]
            )
            motor_ids = _list_arg(self.common_can_motor_ids.text(), int)
            if motor_ids:
                command.append(f"--motor_ids={motor_ids}")
        elif command_type == "train":
            command.extend(
                [
                    f"--dataset.repo_id={self.common_train_dataset_repo_id.text().strip()}",
                    f"--policy.type={self.common_train_policy_type.text().strip()}",
                    f"--output_dir={self.common_train_output_dir.text()}",
                    f"--steps={self.common_train_steps.value()}",
                    f"--batch_size={self.common_train_batch_size.value()}",
                    f"--wandb.enable={_bool_arg(self.common_train_wandb.isChecked())}",
                ]
            )
            if self.common_train_device.text().strip():
                command.append(f"--policy.device={self.common_train_device.text().strip()}")
            if self.common_train_push_to_hub.isChecked():
                command.append("--policy.push_to_hub=true")
        elif command_type == "eval":
            command.extend(
                [
                    f"--policy.path={self.common_eval_policy_path.text().strip()}",
                    f"--env.type={self.common_eval_env_type.text().strip()}",
                    f"--eval.n_episodes={self.common_eval_n_episodes.value()}",
                    f"--eval.batch_size={self.common_eval_batch_size.value()}",
                ]
            )
            if self.common_eval_device.text().strip():
                command.append(f"--policy.device={self.common_eval_device.text().strip()}")
            if self.common_eval_output_dir.text():
                command.append(f"--output_dir={self.common_eval_output_dir.text()}")
        elif command_type == "rollout":
            command.extend(
                [
                    f"--strategy.type={self.common_rollout_strategy.currentText()}",
                    f"--policy.path={self.common_rollout_policy_path.text().strip()}",
                    f"--robot.type={self.common_rollout_robot_type.text().strip()}",
                    f"--task={self.common_rollout_task.text().strip()}",
                    f"--duration={self.common_rollout_duration.value()}",
                    f"--fps={self.common_rollout_fps.value()}",
                    f"--inference.type={self.common_rollout_inference.currentText()}",
                    f"--display_data={_bool_arg(self.common_rollout_display_data.isChecked())}",
                    f"--display_mode={self.common_rollout_display_mode.currentText()}",
                ]
            )
            if self.common_rollout_teleop_type.text().strip():
                command.append(f"--teleop.type={self.common_rollout_teleop_type.text().strip()}")
            if self.common_rollout_dataset_repo_id.text().strip():
                command.extend(
                    [
                        f"--dataset.repo_id={self.common_rollout_dataset_repo_id.text().strip()}",
                        f"--dataset.single_task={self.common_rollout_task.text().strip()}",
                    ]
                )
        elif command_type == "annotate":
            if self.common_annotate_repo_id.text().strip():
                command.append(f"--repo_id={self.common_annotate_repo_id.text().strip()}")
            if self.common_annotate_root.text():
                command.append(f"--root={self.common_annotate_root.text()}")
            if self.common_annotate_new_repo_id.text().strip():
                command.append(f"--new_repo_id={self.common_annotate_new_repo_id.text().strip()}")
            command.extend(
                [
                    f"--vlm.model_id={self.common_annotate_vlm_model.text().strip()}",
                    f"--executor.episode_parallelism={self.common_annotate_episode_parallelism.value()}",
                    f"--push_to_hub={_bool_arg(self.common_annotate_push_to_hub.isChecked())}",
                    f"--skip_validation={_bool_arg(self.common_annotate_skip_validation.isChecked())}",
                ]
            )
            if self.common_annotate_camera_key.text().strip():
                command.append(f"--vlm.camera_key={self.common_annotate_camera_key.text().strip()}")
        elif command_type == "imgtransform_viz":
            command.append(f"--repo_id={self.common_imgtransform_repo_id.text().strip()}")
            episodes = _list_arg(self.common_imgtransform_episodes.text(), int)
            if episodes:
                command.append(f"--episodes={episodes}")
            if self.common_imgtransform_output_dir.text():
                command.append(f"--output_dir={self.common_imgtransform_output_dir.text()}")
            command.extend(
                [
                    f"--n_examples={self.common_imgtransform_n_examples.value()}",
                    f"--image_transforms.enable={_bool_arg(self.common_imgtransform_enable.isChecked())}",
                ]
            )
        elif command_type == "augment_quantile_stats":
            command.extend(["--repo-id", self.common_quantile_repo_id.text().strip()])
            if self.common_quantile_root.text():
                command.extend(["--root", self.common_quantile_root.text()])
            if self.common_quantile_overwrite.isChecked():
                command.append("--overwrite")
            if self.common_quantile_no_sampling.isChecked():
                command.append("--no-sampling")
            if self.common_quantile_skip_images.isChecked():
                command.append("--skip-images")
        elif command_type == "convert_dcp":
            command.append(f"--checkpoint_dir={self.common_dcp_checkpoint_dir.text()}")
            if self.common_dcp_delete.isChecked():
                command.append("--delete_dcp=true")
            if self.common_dcp_push_repo.text().strip():
                command.append(f"--push_to_hub={self.common_dcp_push_repo.text().strip()}")
                command.append(f"--private={_bool_arg(self.common_dcp_private.isChecked())}")
        elif command_type == "train_tokenizer":
            command.extend(
                [
                    f"--repo_id={self.common_tokenizer_repo_id.text().strip()}",
                    f"--action_horizon={self.common_tokenizer_action_horizon.value()}",
                    f"--sample_fraction={self.common_tokenizer_sample_fraction.value()}",
                    f"--encoded_dims={self.common_tokenizer_encoded_dims.text().strip()}",
                    f"--vocab_size={self.common_tokenizer_vocab_size.value()}",
                    f"--scale={self.common_tokenizer_scale.value()}",
                    f"--push_to_hub={_bool_arg(self.common_tokenizer_push_to_hub.isChecked())}",
                ]
            )
            if self.common_tokenizer_root.text():
                command.append(f"--root={self.common_tokenizer_root.text()}")
            if self.common_tokenizer_output_dir.text():
                command.append(f"--output_dir={self.common_tokenizer_output_dir.text()}")
            if self.common_tokenizer_relative_dims.text().strip():
                command.extend(
                    [
                        f"--relative_dims={self.common_tokenizer_relative_dims.text().strip()}",
                        "--use_relative_transform=true",
                    ]
                )
            if self.common_tokenizer_hub_repo_id.text().strip():
                command.append(f"--hub_repo_id={self.common_tokenizer_hub_repo_id.text().strip()}")

        extra = self.common_advanced_args.toPlainText().strip()
        if extra:
            command.extend(shlex.split(extra))
        return command

    @Slot()
    def update_record_preview(self, *_args) -> None:
        try:
            text = _format_command(self.build_record_command())
        except ValueError as exc:
            text = f"高级参数解析失败：{exc}"
        self.record_command_preview.setPlainText(text)

    @Slot()
    def update_viewer_preview(self, *_args) -> None:
        self.viewer_command_preview.setPlainText(_format_command(self.build_viewer_command()))

    @Slot()
    def update_edit_preview(self, *_args) -> None:
        try:
            text = _format_command(self.build_edit_command())
        except (ValueError, json.JSONDecodeError) as exc:
            text = f"编辑参数解析失败：{exc}"
        self.edit_command_preview.setPlainText(text)
        if hasattr(self, "edit_dataset_preview"):
            self.edit_dataset_preview.set_source(self.edit_repo_id.text(), self.edit_root.text())

    @Slot()
    def update_conversion_preview(self, *_args) -> None:
        if not hasattr(self, "conversion_command_preview"):
            return
        script_path = self._conversion_script_path()
        self.conversion_script_label.setText(str(script_path))
        try:
            text = _format_command(self.build_conversion_command())
        except ValueError as exc:
            text = f"转换参数解析失败：{exc}"
        self.conversion_command_preview.setPlainText(text)

    @Slot()
    def _on_conversion_type_changed(self, *_args) -> None:
        key = self._conversion_key()
        self.conversion_stack.setCurrentIndex(CONVERSION_STACK_INDEX[key])
        self.update_conversion_preview()

    @Slot()
    def update_common_preview(self, *_args) -> None:
        try:
            text = _format_command(self.build_common_command())
        except (ValueError, json.JSONDecodeError) as exc:
            text = f"常用命令参数解析失败：{exc}"
        self.common_command_preview.setPlainText(text)

    def validate_record_form(self) -> bool:
        if not self.repo_id.text().strip():
            QMessageBox.warning(self, "缺少 Repo ID", "请填写数据集 Repo ID。")
            return False
        if not self.task.text().strip():
            QMessageBox.warning(self, "缺少任务描述", "请填写本次采集的任务描述。")
            return False
        try:
            shlex.split(self.advanced_args.toPlainText().strip())
        except ValueError as exc:
            QMessageBox.warning(self, "高级参数错误", f"高级参数无法解析：{exc}")
            return False
        return True

    def validate_viewer_form(self) -> bool:
        repo_id = self.viewer_repo_id.text().strip()
        root = self.viewer_root.text() or None
        if not repo_id:
            QMessageBox.warning(self, "缺少 Repo ID", "请填写要查看的数据集 Repo ID。")
            return False
        dataset_info = describe_local_dataset(repo_id, root)
        if dataset_info is None:
            QMessageBox.warning(
                self,
                "没有找到数据集",
                "本地没有找到该数据集的 meta/info.json。\n\n"
                "请确认 Repo ID 是否正确；如果数据集使用了自定义 root，请在“本地 root”中填写数据集根目录。",
            )
            return False
        dataset_root, total_episodes, total_frames = dataset_info
        if total_episodes <= 0:
            QMessageBox.warning(
                self,
                "数据集为空",
                f"该数据集存在，但还没有保存任何 episode。\n\n"
                f"路径：{dataset_root}\n"
                f"total_episodes={total_episodes}, total_frames={total_frames}\n\n"
                "请先完成一次成功采集，或选择另一个已有 episode 的数据集。",
            )
            return False
        if self.viewer_episode.value() >= total_episodes:
            QMessageBox.warning(
                self,
                "Episode 不存在",
                f"该数据集只有 {total_episodes} 个 episode，可选范围是 0 到 {total_episodes - 1}。\n"
                f"当前填写的是 {self.viewer_episode.value()}。",
            )
            return False
        return True

    def validate_edit_form(self) -> bool:
        operation = EDIT_OPERATION_LABELS[self.edit_operation.currentText()]
        if operation != "merge" and not self.edit_repo_id.text().strip():
            QMessageBox.warning(self, "缺少输入 Repo ID", "请填写要编辑的数据集 Repo ID。")
            return False
        if operation == "merge" and not self.edit_new_repo_id.text().strip():
            QMessageBox.warning(self, "缺少输出 Repo ID", "合并数据集需要填写输出 Repo ID。")
            return False

        try:
            if operation == "delete_episodes" and not _list_arg(self.edit_delete_episode_indices.text(), int):
                QMessageBox.warning(self, "缺少 Episode", "请填写要删除的 episode 编号。")
                return False
            if operation == "split" and not _json_or_none(self.edit_split_splits.toPlainText()):
                QMessageBox.warning(self, "缺少拆分配置", "请填写 splits JSON。")
                return False
            if operation == "merge" and not _list_arg(self.edit_merge_repo_ids.text(), str):
                QMessageBox.warning(self, "缺少输入数据集", "请填写要合并的 Repo IDs。")
                return False
            if operation == "remove_feature" and not _list_arg(self.edit_remove_features.text(), str):
                QMessageBox.warning(self, "缺少 Feature", "请填写要删除的 feature 名称。")
                return False
            if (
                operation == "modify_tasks"
                and not self.edit_modify_new_task.text().strip()
                and not self.edit_modify_episode_tasks.toPlainText().strip()
                and not self.edit_modify_replacements.toPlainText().strip()
            ):
                QMessageBox.warning(self, "缺少任务修改内容", "请至少填写默认任务、按集任务 JSON 或替换 JSON。")
                return False

            # Force parsing now so malformed JSON/list parameters are caught before launching the process.
            self.build_edit_command()
        except (ValueError, json.JSONDecodeError) as exc:
            QMessageBox.warning(self, "编辑参数错误", f"编辑参数无法解析：{exc}")
            return False

        if self._edit_operation_needs_confirmation(operation):
            answer = QMessageBox.question(
                self,
                "确认编辑数据集",
                "该操作可能修改原数据集，或移动原目录生成备份。\n\n"
                f"将执行：{self.edit_operation.currentText()}\n\n"
                "确认继续吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return False
        return True

    def validate_conversion_form(self) -> bool:
        key = self._conversion_key()
        script_path = self._conversion_script_path()
        if not self.conversion_python.text():
            QMessageBox.warning(self, "缺少 Python", "请填写用于运行转换脚本的 Python 路径。")
            return False
        if not script_path.exists():
            QMessageBox.warning(
                self,
                "没有找到转换脚本",
                f"当前转换类型对应脚本不存在：\n{script_path}\n\n"
                "请确认 Backend 路径是否指向 Any4LeRobotGUI/backend。",
            )
            return False

        def require_text(widget, title: str, message: str) -> bool:
            if not widget.text().strip():
                QMessageBox.warning(self, title, message)
                return False
            return True

        try:
            shlex.split(self.conversion_advanced_args.toPlainText().strip())
            if key == "openx_to_lerobot":
                if not require_text(self.openx_raw_dir, "缺少输入目录", "请填写 OpenX raw dir。"):
                    return False
                if not require_text(self.openx_local_dir, "缺少输出目录", "请填写输出 local dir。"):
                    return False
                if self.openx_push_to_hub.isChecked() and not self.openx_repo_id.text().strip():
                    QMessageBox.warning(self, "缺少 Repo ID", "OpenX 转换上传 Hub 时需要填写 Repo ID。")
                    return False
            elif key == "agibot_to_lerobot":
                if not require_text(self.agibot_src_path, "缺少输入目录", "请填写 AgiBot src path。"):
                    return False
                if not require_text(self.agibot_output_path, "缺少输出目录", "请填写输出 output path。"):
                    return False
                self._append_space_list_args([], "--task-ids", self.agibot_task_ids.text())
            elif key == "robomind_to_lerobot":
                if not require_text(self.robomind_src_path, "缺少输入目录", "请填写 RoboMIND src path。"):
                    return False
                if not require_text(self.robomind_output_path, "缺少输出目录", "请填写输出 output path。"):
                    return False
                if not shlex.split(self.robomind_embodiments.text().replace(",", " ")):
                    QMessageBox.warning(self, "缺少 embodiment", "请至少填写一个 embodiment。")
                    return False
            elif key == "libero_to_lerobot":
                if not shlex.split(self.libero_src_paths.text().replace(",", " ")):
                    QMessageBox.warning(self, "缺少输入目录", "请填写至少一个 LIBERO src path。")
                    return False
                if not require_text(self.libero_output_path, "缺少输出目录", "请填写输出 output path。"):
                    return False
                if self.libero_push_to_hub.isChecked() and not self.libero_repo_id.text().strip():
                    QMessageBox.warning(self, "缺少 Repo ID", "LIBERO 转换上传 Hub 时需要填写 Repo ID。")
                    return False
            elif key == "lerobot_to_rlds":
                if not require_text(self.rlds_src_dir, "缺少输入目录", "请填写输入 LeRobot src dir。"):
                    return False
                if not require_text(self.rlds_output_dir, "缺少输出目录", "请填写输出 RLDS output dir。"):
                    return False
                if not self.rlds_task_name.text().strip():
                    QMessageBox.warning(self, "缺少任务名", "请填写 task name。")
                    return False
                if not self.rlds_version.text().strip():
                    QMessageBox.warning(self, "缺少版本号", "请填写 RLDS 版本号，例如 0.1.0。")
                    return False
            elif key == "v16_to_v20":
                if not self.v16_repo_id.text().strip():
                    QMessageBox.warning(self, "缺少 Repo ID", "请填写输入数据集 Repo ID。")
                    return False
                task_value = (
                    self.v16_tasks_path.text()
                    if self.v16_task_mode.currentText() == "tasks-path"
                    else self.v16_task_value.text().strip()
                )
                if not task_value:
                    QMessageBox.warning(self, "缺少任务信息", "请填写 single-task、tasks-col 或 tasks-path。")
                    return False
            elif key in {"v20_to_v21", "v21_to_v20"}:
                if not self.version_repo_id.text().strip():
                    QMessageBox.warning(self, "缺少 Repo ID", "请填写输入数据集 Repo ID。")
                    return False
            elif key == "v21_to_v30":
                if not self.v30_repo_id.text().strip():
                    QMessageBox.warning(self, "缺少 Repo ID", "请填写输入数据集 Repo ID。")
                    return False
            elif key == "v30_to_v21":
                if not self.v3021_repo_id.text().strip():
                    QMessageBox.warning(self, "缺少 Repo ID", "请填写输入数据集 Repo ID。")
                    return False

            self.build_conversion_command()
        except ValueError as exc:
            QMessageBox.warning(self, "转换参数错误", f"转换参数无法解析：{exc}")
            return False

        if key.startswith("v") or key == "lerobot_to_rlds":
            return True
        answer = QMessageBox.question(
            self,
            "确认开始转换",
            "外部格式转换通常会占用较长时间，并可能需要额外依赖如 tensorflow、h5py、ray 或 datatrove。\n\n"
            f"将执行：{self.conversion_type.currentText()}\n\n"
            "确认继续吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def validate_common_form(self) -> bool:
        command_type = COMMON_COMMAND_LABELS[self.common_command.currentText()]
        try:
            if command_type == "custom" and not self.common_custom_module.text().strip():
                QMessageBox.warning(self, "缺少模块名", "请填写要运行的 Python 模块。")
                return False
            if command_type == "replay" and not self.common_replay_repo_id.text().strip():
                QMessageBox.warning(self, "缺少数据集", "回放 Episode 需要填写数据集 Repo ID。")
                return False
            if command_type == "find_joint_limits" and not self.common_joint_urdf_path.text():
                QMessageBox.warning(self, "缺少 URDF", "查关节限位需要填写 URDF 路径。")
                return False
            if command_type == "find_joint_limits" and not self.common_joint_target_frame.text().strip():
                QMessageBox.warning(self, "缺少目标 frame", "请填写目标 frame 名称。")
                return False
            if command_type == "setup_can" and not self.common_can_interfaces.text().strip():
                QMessageBox.warning(self, "缺少 CAN 接口", "请填写 CAN interface，例如 can0。")
                return False
            if command_type == "train":
                if not self.common_train_dataset_repo_id.text().strip():
                    QMessageBox.warning(self, "缺少数据集", "训练策略需要填写 dataset.repo_id。")
                    return False
                if not self.common_train_policy_type.text().strip():
                    QMessageBox.warning(self, "缺少策略类型", "训练策略需要填写 policy.type。")
                    return False
            if command_type == "eval":
                if not self.common_eval_policy_path.text().strip():
                    QMessageBox.warning(self, "缺少策略路径", "评估策略需要填写 policy.path。")
                    return False
                if not self.common_eval_env_type.text().strip():
                    QMessageBox.warning(self, "缺少环境类型", "评估策略需要填写 env.type。")
                    return False
            if command_type == "rollout":
                if not self.common_rollout_policy_path.text().strip():
                    QMessageBox.warning(self, "缺少策略路径", "Rollout 需要填写 policy.path。")
                    return False
                if not self.common_rollout_robot_type.text().strip():
                    QMessageBox.warning(self, "缺少机器人类型", "Rollout 需要填写 robot.type。")
                    return False
                if self.common_rollout_strategy.currentText() != "base" and not self.common_rollout_dataset_repo_id.text().strip():
                    QMessageBox.warning(self, "缺少数据集", "非 base rollout 需要填写 dataset.repo_id。")
                    return False
            if command_type == "annotate":
                if not self.common_annotate_repo_id.text().strip() and not self.common_annotate_root.text():
                    QMessageBox.warning(self, "缺少数据集", "数据标注需要填写 repo_id 或 root。")
                    return False
                if not self.common_annotate_vlm_model.text().strip():
                    QMessageBox.warning(self, "缺少 VLM 模型", "数据标注需要填写 vlm.model_id。")
                    return False
            if command_type == "imgtransform_viz" and not self.common_imgtransform_repo_id.text().strip():
                QMessageBox.warning(self, "缺少数据集", "图像增强预览需要填写 repo_id。")
                return False
            if command_type == "augment_quantile_stats" and not self.common_quantile_repo_id.text().strip():
                QMessageBox.warning(self, "缺少数据集", "补充分位数统计需要填写 Repo ID。")
                return False
            if command_type == "convert_dcp" and not self.common_dcp_checkpoint_dir.text():
                QMessageBox.warning(self, "缺少 checkpoint", "转换 DCP 需要填写 checkpoint_dir。")
                return False
            if command_type == "train_tokenizer":
                if not self.common_tokenizer_repo_id.text().strip():
                    QMessageBox.warning(self, "缺少数据集", "训练 FAST Tokenizer 需要填写 repo_id。")
                    return False
                if not self.common_tokenizer_encoded_dims.text().strip():
                    QMessageBox.warning(self, "缺少维度", "训练 FAST Tokenizer 需要填写 encoded_dims。")
                    return False
            self.build_common_command()
        except (ValueError, json.JSONDecodeError) as exc:
            QMessageBox.warning(self, "命令参数错误", f"常用命令参数无法解析：{exc}")
            return False
        return True

    def _edit_operation_needs_confirmation(self, operation: str) -> bool:
        if operation == "modify_tasks":
            return True
        if operation in {"delete_episodes", "remove_feature"}:
            return not self.edit_new_repo_id.text().strip() and not self.edit_new_root.text()
        if operation == "recompute_stats":
            return self.edit_stats_overwrite.isChecked()
        if operation == "reencode_videos":
            return self.edit_reencode_overwrite.isChecked()
        return False

    @Slot()
    def start_recording(self) -> None:
        if not self.validate_record_form():
            return
        self._save_settings()
        command = self.build_record_command()
        self._last_record_base_repo_id = self.repo_id.text().strip()
        self._last_record_root = self.dataset_root.text()
        self._last_record_no_stamp = self.no_stamp.isChecked()
        self._last_record_resume = self.resume.isChecked()
        if self.record_runner.start(command):
            self.start_record_btn.setEnabled(False)
            self.stop_record_btn.setEnabled(True)
            self.status_label.setText("采集中")
            self.preview_tabs.setCurrentWidget(self.record_command_preview)

    @Slot()
    def stop_recording(self) -> None:
        self.record_runner.interrupt()
        self.statusBar().showMessage("已请求停止采集，正在等待清理设备和保存数据。")
        QTimer.singleShot(8000, self._offer_record_kill_if_running)

    @Slot()
    def start_viewer(self) -> None:
        if not self.validate_viewer_form():
            return
        self._save_settings()
        if self.viewer_runner.start(self.build_viewer_command()):
            self.start_viewer_btn.setEnabled(False)
            self.stop_viewer_btn.setEnabled(True)
            self.status_label.setText("查看中")
            self.preview_tabs.setCurrentWidget(self.viewer_command_preview)

    @Slot()
    def stop_viewer(self) -> None:
        self.viewer_runner.interrupt()
        QTimer.singleShot(3000, self._offer_viewer_kill_if_running)

    @Slot()
    def start_edit(self) -> None:
        if not self.validate_edit_form():
            return
        self._save_settings()
        if self.edit_runner.start(self.build_edit_command()):
            self.start_edit_btn.setEnabled(False)
            self.stop_edit_btn.setEnabled(True)
            self.status_label.setText("编辑中")
            self.preview_tabs.setCurrentWidget(self.edit_command_preview)

    @Slot()
    def stop_edit(self) -> None:
        self.edit_runner.interrupt()
        QTimer.singleShot(3000, self._offer_edit_kill_if_running)

    @Slot()
    def start_conversion(self) -> None:
        if not self.validate_conversion_form():
            return
        self._save_settings()
        script_path = self._conversion_script_path()
        if self.conversion_runner.start(self.build_conversion_command(), cwd=script_path.parent):
            self.start_conversion_btn.setEnabled(False)
            self.stop_conversion_btn.setEnabled(True)
            self.status_label.setText("转换中")
            self.preview_tabs.setCurrentWidget(self.conversion_command_preview)

    @Slot()
    def stop_conversion(self) -> None:
        self.conversion_runner.interrupt()
        QTimer.singleShot(3000, self._offer_conversion_kill_if_running)

    @Slot()
    def fill_edit_from_viewer(self) -> None:
        self.edit_repo_id.setText(self.viewer_repo_id.text())
        self.edit_root.setText(self.viewer_root.text())
        self.tabs.setCurrentWidget(self.edit_tab)
        self.update_edit_preview()

    @Slot(int)
    def _fill_delete_episode_from_preview(self, episode_index: int) -> None:
        self.edit_operation.setCurrentText("删除 Episode")
        self.edit_delete_episode_indices.setText(str(episode_index))
        self.update_edit_preview()
        self.statusBar().showMessage(f"已填入待删除 Episode：{episode_index}", 3000)

    @Slot()
    def start_common_command(self) -> None:
        if not self.validate_common_form():
            return
        self._save_settings()
        if self.common_runner.start(self.build_common_command()):
            self.start_common_btn.setEnabled(False)
            self.stop_common_btn.setEnabled(True)
            self.common_enter_btn.setEnabled(True)
            self.status_label.setText("常用命令运行中")
            self.preview_tabs.setCurrentWidget(self.common_command_preview)

    @Slot()
    def stop_common_command(self) -> None:
        self.common_runner.interrupt()
        QTimer.singleShot(3000, self._offer_common_kill_if_running)

    @Slot()
    def fill_latest_dataset(self) -> None:
        latest = find_latest_local_dataset(
            self.repo_id.text(),
            self.dataset_root.text() or None,
            self.no_stamp.isChecked(),
            self.resume.isChecked(),
        )
        if latest is None:
            QMessageBox.information(
                self,
                "没有找到可查看的数据集",
                "还没有在默认缓存中找到匹配且已保存 episode 的数据集。\n\n"
                "如果刚才采集失败或中途退出，可能只创建了空数据集目录；请先完成一次成功采集。",
            )
            return
        self.viewer_repo_id.setText(latest)
        if self.dataset_root.text():
            self.viewer_root.setText(self.dataset_root.text())
        self.tabs.setCurrentWidget(self.viewer_tab)
        self.update_viewer_preview()

    def _offer_record_kill_if_running(self) -> None:
        if not self.record_runner.is_running:
            return
        answer = QMessageBox.question(
            self,
            "采集仍在运行",
            "采集进程还没有退出。是否强制结束？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.record_runner.kill()

    def _offer_viewer_kill_if_running(self) -> None:
        if not self.viewer_runner.is_running:
            return
        self.viewer_runner.kill()

    def _offer_edit_kill_if_running(self) -> None:
        if not self.edit_runner.is_running:
            return
        self.edit_runner.kill()

    def _offer_conversion_kill_if_running(self) -> None:
        if not self.conversion_runner.is_running:
            return
        self.conversion_runner.kill()

    def _offer_common_kill_if_running(self) -> None:
        if not self.common_runner.is_running:
            return
        self.common_runner.kill()

    def _on_started(self, name: str, command: str) -> None:
        self.append_log(name, f"$ {command}")
        self.statusBar().showMessage(f"{name} 已启动")

    @Slot(int)
    def _on_record_finished(self, code: int) -> None:
        self.start_record_btn.setEnabled(True)
        self.stop_record_btn.setEnabled(False)
        latest = find_latest_local_dataset(
            self._last_record_base_repo_id,
            self._last_record_root or None,
            self._last_record_no_stamp,
            self._last_record_resume,
        )
        if latest:
            self.viewer_repo_id.setText(latest)
            self.edit_repo_id.setText(latest)
            if self._last_record_root:
                self.viewer_root.setText(self._last_record_root)
                self.edit_root.setText(self._last_record_root)
            self.dataset_hint.setText(f"最近数据集：{latest}")
        elif self._last_record_base_repo_id:
            self.dataset_hint.setText("最近采集没有找到已保存 episode 的数据集。")
        self.status_label.setText(self._current_status_text())
        self.statusBar().showMessage(f"采集{_readable_exit(code)}")
        self.append_log("采集", f"进程{_readable_exit(code)}")

    @Slot(int)
    def _on_viewer_finished(self, code: int) -> None:
        self.start_viewer_btn.setEnabled(True)
        self.stop_viewer_btn.setEnabled(False)
        self.status_label.setText(self._current_status_text())
        self.statusBar().showMessage(f"查看{_readable_exit(code)}")
        self.append_log("查看", f"进程{_readable_exit(code)}")

    @Slot(int)
    def _on_edit_finished(self, code: int) -> None:
        self.start_edit_btn.setEnabled(True)
        self.stop_edit_btn.setEnabled(False)
        self.status_label.setText(self._current_status_text())
        self.statusBar().showMessage(f"编辑{_readable_exit(code)}")
        self.append_log("编辑", f"进程{_readable_exit(code)}")

    @Slot(int)
    def _on_conversion_finished(self, code: int) -> None:
        self.start_conversion_btn.setEnabled(True)
        self.stop_conversion_btn.setEnabled(False)
        self.status_label.setText(self._current_status_text())
        self.statusBar().showMessage(f"转换{_readable_exit(code)}")
        self.append_log("转换", f"进程{_readable_exit(code)}")

    @Slot(int)
    def _on_common_finished(self, code: int) -> None:
        self.start_common_btn.setEnabled(True)
        self.stop_common_btn.setEnabled(False)
        self.common_enter_btn.setEnabled(False)
        self.status_label.setText(self._current_status_text())
        self.statusBar().showMessage(f"常用命令{_readable_exit(code)}")
        self.append_log("常用", f"进程{_readable_exit(code)}")

    def _current_status_text(self) -> str:
        if self.record_runner.is_running:
            return "采集中"
        if self.viewer_runner.is_running:
            return "查看中"
        if self.edit_runner.is_running:
            return "编辑中"
        if self.conversion_runner.is_running:
            return "转换中"
        if self.common_runner.is_running:
            return "常用命令运行中"
        return "未运行"

    def append_log(self, source: str, line: str) -> None:
        self.log_view.append(f"[{source}] {line}")
        self.log_view.verticalScrollBar().setValue(self.log_view.verticalScrollBar().maximum())

    @Slot()
    def copy_active_command(self) -> None:
        active = self.preview_tabs.currentWidget()
        if isinstance(active, QPlainTextEdit):
            self.copy_command(active.toPlainText())

    def copy_command(self, text: str) -> None:
        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(text, QClipboard.Mode.Clipboard)
            self.statusBar().showMessage("已复制到剪贴板", 2500)


APP_STYLESHEET = """
QWidget {
    background: transparent;
    color: #172033;
    font-family: "Inter", "Noto Sans CJK SC", "Microsoft YaHei", "Segoe UI", sans-serif;
    font-size: 14px;
}
QMainWindow, #AppRoot {
    background: qlineargradient(
        x1: 0, y1: 0, x2: 1, y2: 1,
        stop: 0 #e9f3f7,
        stop: 0.46 #f4f7fb,
        stop: 1 #e8edf6
    );
}
QToolTip {
    background: #ffffff;
    color: #172033;
    border: 1px solid #cbd7e4;
    border-radius: 6px;
    padding: 6px 8px;
}
#TitleLabel {
    font-size: 27px;
    font-weight: 750;
    color: #101827;
}
#DialogTitle {
    font-size: 22px;
    font-weight: 750;
    color: #101827;
}
#SubtitleLabel, #MutedLabel {
    color: #647084;
}
#HelpText {
    background-color: #ffffff;
    color: #172033;
    border: 1px solid #cbd7e4;
    border-radius: 8px;
    padding: 12px;
    line-height: 1.45;
}
QTextBrowser#HelpText {
    background-color: #ffffff;
    color: #172033;
}
QTextBrowser#HelpText QWidget {
    background-color: #ffffff;
    color: #172033;
}
#SidebarPanel, #RightPanel {
    background: rgba(255, 255, 255, 184);
    border: 1px solid rgba(255, 255, 255, 210);
    border-radius: 14px;
}
#SidebarTitle {
    color: #647084;
    font-size: 12px;
    font-weight: 800;
    padding: 0 4px 8px 4px;
}
#SidebarButton {
    background: transparent;
    border: 1px solid transparent;
    border-radius: 10px;
    color: #344259;
    padding: 10px 12px;
    text-align: left;
}
#SidebarButton:hover {
    background: rgba(255, 255, 255, 170);
    border-color: rgba(137, 161, 189, 120);
}
#SidebarButton:checked {
    background: rgba(47, 111, 237, 226);
    color: #ffffff;
    border-color: rgba(47, 111, 237, 240);
}
#SidebarGhostButton {
    background: rgba(255, 255, 255, 142);
    border-color: rgba(168, 199, 210, 180);
    color: #2f5f72;
}
#PreviewEpisodeLabel {
    font-size: 17px;
    font-weight: 750;
    color: #101827;
}
#DatasetImageLabel {
    background: #0f1720;
    color: #d9e6f2;
    border: 1px solid #233247;
    border-radius: 8px;
    padding: 8px;
}
#FrameRangeSlider {
    background: #f7fafc;
    border: 1px solid #d6e0ea;
    border-radius: 8px;
}
QGroupBox {
    background: rgba(255, 255, 255, 214);
    border: 1px solid rgba(216, 226, 236, 210);
    border-radius: 10px;
    margin-top: 18px;
    padding: 16px;
    font-weight: 650;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 8px;
    color: #2d3a4e;
    background: rgba(255, 255, 255, 230);
}
QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background: #ffffff;
    border: 1px solid #cbd7e4;
    border-radius: 8px;
    padding: 8px 10px;
    selection-background-color: #2f6fed;
    selection-color: #ffffff;
}
QLineEdit:hover, QPlainTextEdit:hover, QTextEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover {
    border-color: #9fb1c7;
}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border: 1px solid #2f6fed;
}
QLineEdit:disabled, QPlainTextEdit:disabled, QTextEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {
    background: #edf2f7;
    color: #91a0b3;
    border-color: #dbe4ee;
}
QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
    width: 0px;
    border: none;
}
QSpinBox::up-arrow, QSpinBox::down-arrow, QDoubleSpinBox::up-arrow, QDoubleSpinBox::down-arrow {
    width: 0px;
    height: 0px;
}
#CommandPreview, #LogView {
    background: #111a27;
    color: #d8e7f3;
    border: 1px solid #223147;
    border-radius: 8px;
    padding: 10px;
}
#CommandPreview:focus, #LogView:focus {
    border: 1px solid #3b82f6;
}
QComboBox::drop-down {
    border: none;
    width: 24px;
}
QComboBox::down-arrow {
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid #607086;
    width: 0px;
    height: 0px;
    margin-right: 8px;
}
QComboBox QAbstractItemView {
    background: #ffffff;
    color: #172033;
    border: 1px solid #b9c8d8;
    border-radius: 8px;
    padding: 6px;
    outline: none;
    selection-background-color: #e8f0ff;
    selection-color: #102036;
}
QComboBox QAbstractItemView::item {
    min-height: 28px;
    padding: 6px 10px;
    border-radius: 6px;
}
QComboBox QAbstractItemView::item:hover {
    background: #f2f6fb;
}
QComboBox QAbstractItemView::item:selected {
    background: #e8f0ff;
    color: #102036;
}
QTabWidget::pane {
    border: 1px solid rgba(216, 226, 236, 210);
    border-radius: 12px;
    background: rgba(255, 255, 255, 178);
    top: -1px;
}
QTabBar::tab {
    background: #e7edf4;
    color: #516075;
    border: 1px solid #d6e0ea;
    padding: 9px 18px;
    margin-right: 6px;
    border-radius: 8px;
}
QTabBar::tab:selected {
    background: #ffffff;
    color: #102036;
    border-color: #b9c8d8;
}
QTabBar::tab:!selected:hover {
    background: #f2f6fa;
    color: #243247;
}
QPushButton {
    background: #ffffff;
    border: 1px solid #c8d5e2;
    border-radius: 8px;
    padding: 8px 15px;
    font-weight: 600;
    color: #1e2a3d;
}
QPushButton:hover {
    background: #f3f7fb;
    border-color: #9fb1c7;
}
QPushButton:pressed {
    background: #e6edf5;
}
QPushButton:disabled {
    color: #9aa8ba;
    background: #edf2f7;
    border-color: #dbe4ee;
}
QMenuBar {
    background: rgba(255, 255, 255, 190);
    color: #243247;
    border-bottom: 1px solid rgba(196, 211, 226, 160);
}
QMenuBar::item {
    background: transparent;
    padding: 7px 12px;
    border-radius: 7px;
}
QMenuBar::item:selected {
    background: rgba(232, 240, 255, 210);
}
#PrimaryButton {
    background: #2f6fed;
    color: #ffffff;
    border-color: #2f6fed;
}
#PrimaryButton:hover {
    background: #255ed0;
    border-color: #255ed0;
}
#PrimaryButton:pressed {
    background: #1f4fb0;
}
#DangerButton {
    background: #d94949;
    color: #ffffff;
    border-color: #d94949;
}
#DangerButton:hover {
    background: #bf3030;
    border-color: #bf3030;
}
#DangerButton:pressed {
    background: #a82626;
}
#HelpButton {
    background: #ffffff;
    color: #2f5f72;
    border-color: #a8c7d2;
}
#HelpButton:hover {
    background: #eef8fa;
    border-color: #72aebb;
}
#HelpButton:pressed {
    background: #dceff4;
}
#StatusPill {
    background: #e9f8f1;
    color: #08704f;
    border: 1px solid #b6ead1;
    border-radius: 8px;
    padding: 9px 12px;
    font-weight: 700;
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
QCheckBox::indicator:hover {
    border-color: #2f6fed;
}
QCheckBox::indicator:checked {
    background: #2f6fed;
    border: 1px solid #2f6fed;
}
QCheckBox::indicator:disabled {
    background: #edf2f7;
    border-color: #d5dee9;
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
QMenu {
    background: #ffffff;
    color: #172033;
    border: 1px solid #b9c8d8;
    border-radius: 8px;
    padding: 6px;
}
QMenu::item {
    padding: 7px 18px;
    border-radius: 6px;
}
QMenu::item:selected {
    background: #e8f0ff;
    color: #102036;
}
QStatusBar {
    background: #eef3f8;
    color: #647084;
}
"""


def main() -> None:
    _prepare_linux_qt_platform()
    app = QApplication(sys.argv)
    app.setApplicationName("LeRobot Wheeled Arm GUI")
    app.setWindowIcon(QIcon())
    window = WheeledArmGui()
    window.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
