#!/usr/bin/env python3
"""PySide6 launcher for the RoboPlan Python examples."""

from __future__ import annotations

import json
import os
import signal
import sys
import sysconfig
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _append_env_tokens(name: str, tokens: tuple[str, ...]) -> None:
    existing = os.environ.get(name, "")
    parts = existing.split() if existing else []
    for token in tokens:
        if token not in parts:
            parts.append(token)
    os.environ[name] = " ".join(parts)


def _append_qt_logging_rules(rules: tuple[str, ...]) -> None:
    existing = os.environ.get("QT_LOGGING_RULES", "")
    parts = [part for part in existing.split(";") if part]
    for rule in rules:
        if rule not in parts:
            parts.append(rule)
    os.environ["QT_LOGGING_RULES"] = ";".join(parts)


def _configure_qt_webengine_logging() -> None:
    _append_env_tokens(
        "QTWEBENGINE_CHROMIUM_FLAGS",
        (
            "--disable-logging",
            "--log-level=3",
        ),
    )
    _append_qt_logging_rules(
        (
            "qt.webenginecontext.debug=false",
            "qt.webenginecontext.info=false",
        ),
    )


_configure_qt_webengine_logging()

from PySide6.QtCore import QAbstractAnimation, QEasingCurve, QProcess, QProcessEnvironment, QPropertyAnimation, QSize, QTimer, Qt, QUrl
from PySide6.QtGui import QAction, QColor, QDesktopServices, QFontDatabase, QKeySequence, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGraphicsDropShadowEffect,
    QGraphicsOpacityEffect,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QStyle,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
    QStackedWidget,
)

try:
    from PySide6.QtWebEngineCore import QWebEnginePage
    from PySide6.QtWebEngineWidgets import QWebEngineView
except ImportError:  # pragma: no cover - optional GUI extra.
    QWebEnginePage = None
    QWebEngineView = None


if QWebEnginePage is not None:

    class QuietWebEnginePage(QWebEnginePage):
        def javaScriptConsoleMessage(self, level, message, line_number, source_id):
            if "THREE.THREE.Clock" in message or "THREE.Clock" in message:
                return
            super().javaScriptConsoleMessage(level, message, line_number, source_id)

else:
    QuietWebEnginePage = None


EXAMPLE_DIR = Path(__file__).resolve().parent
RUNNER = EXAMPLE_DIR / "example_runner.py"


def _add_shadow(
    widget: QWidget,
    color: str = "#8ea7cf",
    blur_radius: float = 20.0,
    y_offset: float = 8.0,
) -> None:
    shadow = QGraphicsDropShadowEffect(widget)
    shadow.setBlurRadius(blur_radius)
    shadow.setOffset(0.0, y_offset)
    shadow.setColor(QColor(color))
    widget.setGraphicsEffect(shadow)


CATEGORY_ZH = {
    "Core": "核心",
    "IK": "逆解",
    "Planning": "规划",
    "Tracking": "跟踪",
}

EXAMPLE_TITLE_ZH = {
    "example_scene": "场景基础",
    "example_ik": "基础逆解",
    "example_oink": "OInK 跟踪",
    "example_oink_position_barriers": "OInK 位置约束",
    "example_rrt": "RRT 规划器",
    "example_toppra_joint_planning": "关节空间 TOPPRA 预览",
    "example_constrained_rrt": "约束 RRT",
    "example_cartesian_planning": "笛卡尔规划器",
    "example_action_chunk_tracking": "动作片段跟踪",
    "example_teleop": "键盘遥操作",
}

EXAMPLE_DESC_ZH = {
    "example_scene": "创建关节配置、载入模型场景并打印场景信息。",
    "example_ik": "带可拖拽末端标记的交互式逆运动学。",
    "example_oink": "带可拖拽 Viser 目标和可选碰撞约束的 OInK 伺服循环。",
    "example_oink_position_barriers": "带控制屏障函数位置安全约束的 OInK。",
    "example_rrt": "拖拽末端目标后使用 RRT 规划，再做时间参数化和动画播放。",
    "example_toppra_joint_planning": "生成平滑的关节空间路点路径，并用 TOPPRA 做时间参数化。",
    "example_constrained_rrt": "拖拽末端目标后规划保持夹爪竖直并位于安全区域内的路径。",
    "example_cartesian_planning": "通过可拖拽的 Viser 目标规划锯齿形笛卡尔路径，同时绘制生成的关节轨迹。",
    "example_action_chunk_tracking": "通过 OInK 跟踪模拟学习策略生成的笛卡尔或关节动作片段。",
    "example_teleop": "使用键盘指令在终端中遥操作一个或多个末端执行器。",
}

PARAM_LABEL_ZH = {
    "model": "机器人型号",
    "toppra_mode": "TOPPRA 模式",
    "waypoint_count": "路点数量",
    "path_span": "路径跨度",
    "curvature_scale": "曲率系数",
    "dt": "采样步长",
    "velocity_scale": "速度缩放",
    "acceleration_scale": "加速度缩放",
    "max_adaptive_iterations": "自适应迭代",
    "max_adaptive_step_size": "自适应步长",
    "max_blend_deviation": "融合偏差",
    "max_iters": "最大迭代",
    "step_size": "步长",
    "max_linear_error_norm": "线性误差阈值",
    "max_angular_error_norm": "角度误差阈值",
    "check_collisions": "检查碰撞",
    "task_gain": "任务增益",
    "lm_damping": "LM 阻尼",
    "regularization": "正则化",
    "control_freq": "控制频率",
    "reference_filter_tau": "参考滤波时间常数",
    "self_collision_num_pairs": "自碰撞对数",
    "self_collision_d_min": "自碰撞最小距离",
    "self_collision_d_max": "自碰撞最大距离",
    "self_collision_gain": "自碰撞增益",
    "limit_acceleration": "限制加速度",
    "max_connection_distance": "连接距离",
    "collision_check_step_size": "碰撞检查步长",
    "collision_check_use_bisection": "二分碰撞检查",
    "goal_biasing_probability": "目标偏置概率",
    "max_nodes": "最大节点数",
    "max_planning_time": "规划时间",
    "rrt_connect": "RRT-Connect",
    "rrt_star": "RRT*",
    "rewire_distance": "重连距离",
    "fast_return": "快速返回",
    "include_shortcutting": "包含捷径优化",
    "max_shortcutting_iters": "捷径迭代",
    "rng_seed": "随机种子",
    "include_obstacles": "包含障碍物",
    "include_octrees": "包含八叉树",
    "speed_mode": "速度模式",
    "max_linear_speed": "最大线速度",
    "max_angular_speed": "最大角速度",
    "max_linear_acceleration": "最大线加速度",
    "max_angular_acceleration": "最大角加速度",
    "max_position_error": "最大位置误差",
    "max_orientation_error": "最大姿态误差",
    "path_size": "路径尺寸",
    "path_num_passes": "来回次数",
    "path_corner_radius": "拐角半径",
    "path_corner_arc_step_deg": "拐角弧步长",
    "interactive_goal": "交互目标",
    "action_space": "动作空间",
    "chunk_horizon": "片段时域",
    "action_scale": "动作缩放",
    "segment_time": "片段时长",
    "sleep": "生成时休眠",
    "playback_speed": "回放速度",
    "device": "设备",
    "ee_mode": "末端模式",
    "linear_sensitivity": "线速度灵敏度",
    "angular_sensitivity": "角速度灵敏度",
    "config_task_gain": "构型任务增益",
    "target_axes_length": "目标坐标轴长度",
    "target_axes_radius": "目标坐标轴半径",
    "max_tilt_degrees": "最大倾角",
    "path_step_size": "路径步长",
    "barrier_gain": "屏障增益",
    "barrier_size": "屏障尺寸",
    "safety_margin": "安全裕度",
    "reference_filter_tau": "参考滤波时间常数",
}

CHOICE_ZH = {
    "toppra_mode": {
        "Adaptive": "自适应",
        "Hermite": "Hermite 插值",
        "Cubic": "三次样条",
        "LinearBlend": "线性混合",
    },
    "speed_mode": {
        "TimeOptimal": "时间最优",
        "Bounded": "受限",
    },
    "action_space": {
        "cartesian": "笛卡尔",
        "joint": "关节",
    },
    "device": {
        "keyboard": "键盘控制",
    },
    "ee_mode": {
        "first": "首个末端",
        "all": "全部末端",
    },
}

ADVANCED_PARAMETER_NAMES = {
    "barrier_gain",
    "barrier_size",
    "check_collisions",
    "collision_check_step_size",
    "collision_check_use_bisection",
    "config_task_gain",
    "control_freq",
    "fast_return",
    "goal_biasing_probability",
    "include_obstacles",
    "include_octrees",
    "include_shortcutting",
    "limit_acceleration",
    "lm_damping",
    "max_adaptive_iterations",
    "max_adaptive_step_size",
    "max_blend_deviation",
    "max_connection_distance",
    "max_linear_acceleration",
    "max_linear_error_norm",
    "max_linear_speed",
    "max_nodes",
    "max_angular_acceleration",
    "max_angular_error_norm",
    "max_angular_speed",
    "max_orientation_error",
    "max_planning_time",
    "max_position_error",
    "max_rotation_error",
    "max_shortcutting_iters",
    "max_tilt_degrees",
    "path_corner_arc_step_deg",
    "path_corner_radius",
    "path_num_passes",
    "path_size",
    "path_span",
    "path_step_size",
    "reference_filter_tau",
    "regularization",
    "rewire_distance",
    "rng_seed",
    "self_collision_d_max",
    "self_collision_d_min",
    "self_collision_gain",
    "self_collision_num_pairs",
    "sleep",
    "speed_mode",
    "step_size",
    "task_gain",
    "toppra_mode",
    "velocity_scale",
    "acceleration_scale",
}

USAGE_TEXT_ZH = """1. 从左侧选择示例，或在搜索框中快速筛选。
2. 在参数区调整模型、路点和约束。
3. 关节规划类示例先点“预览”，在 Viser 中检查轨迹。
4. 确认安全后再点“执行”，运行中可随时停止。
5. 打开可视化窗口后，可直接查看机器人状态和轨迹。"""


def _zh_example_title(spec: ExampleSpec) -> str:
    return EXAMPLE_TITLE_ZH.get(spec.module, spec.title)


def _zh_example_description(spec: ExampleSpec) -> str:
    return EXAMPLE_DESC_ZH.get(spec.module, spec.description)


def _zh_category(category: str) -> str:
    return CATEGORY_ZH.get(category, category)


def _zh_parameter_label(parameter: ParameterSpec) -> str:
    return PARAM_LABEL_ZH.get(parameter.name, parameter.label)


def _zh_choice(parameter: ParameterSpec, choice: Any) -> str:
    return CHOICE_ZH.get(parameter.name, {}).get(str(choice), str(choice))


def _default_parameter_value(parameter: ParameterSpec) -> Any:
    if parameter.kind == "bool":
        return bool(parameter.default)
    if parameter.kind == "choice":
        return parameter.default
    if parameter.kind == "optional_int":
        return None if parameter.default is None else int(parameter.default)
    if parameter.kind == "int":
        return int(parameter.default)
    if parameter.kind == "float":
        return float(parameter.default)
    return str(parameter.default)


PROCESS_ERROR_ZH = {
    "FailedToStart": "启动失败",
    "Crashed": "进程崩溃",
    "Timedout": "等待超时",
    "WriteError": "写入错误",
    "ReadError": "读取错误",
    "UnknownError": "未知错误",
}


def _zh_process_error(error: QProcess.ProcessError) -> str:
    return PROCESS_ERROR_ZH.get(error.name, error.name)


def _cmeel_site_packages() -> Path:
    purelib = Path(sysconfig.get_paths()["purelib"])
    return (
        purelib
        / "cmeel.prefix"
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )


def _shared_library_paths() -> list[str]:
    purelib = Path(sysconfig.get_paths()["purelib"])
    paths = [
        str(Path(sys.prefix) / "lib"),
        str(purelib / "lib"),
        str(purelib / "cmeel.prefix" / "lib"),
    ]
    unique_paths: list[str] = []
    for path in paths:
        if Path(path).exists() and path not in unique_paths:
            unique_paths.append(path)
    return unique_paths


def _available_models() -> tuple[str, ...]:
    try:
        if str(EXAMPLE_DIR) not in sys.path:
            sys.path.insert(0, str(EXAMPLE_DIR))
        cmeel_site = _cmeel_site_packages()
        if cmeel_site.exists() and str(cmeel_site) not in sys.path:
            sys.path.insert(0, str(cmeel_site))
        from common import get_model_data

        return tuple(get_model_data().keys())
    except Exception:
        return ("ur5", "franka", "dual", "real_robot", "kinova", "stretch", "so101")


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    label: str
    kind: str
    default: Any
    minimum: float | int | None = None
    maximum: float | int | None = None
    step: float | int | None = None
    choices: tuple[Any, ...] = ()
    help_text: str = ""


@dataclass(frozen=True)
class ExampleSpec:
    module: str
    title: str
    category: str
    description: str
    interactive: bool
    parameters: tuple[ParameterSpec, ...] = field(default_factory=tuple)


MODEL_CHOICES = _available_models()
SUPPORTED_CONSTRAINED_MODELS = tuple(
    model for model in ("ur5", "franka", "kinova") if model in MODEL_CHOICES
) or ("ur5",)


COMMON_VISER_PARAMS = (
    ParameterSpec("host", "可视化主机", "str", "localhost"),
    ParameterSpec("port", "可视化端口", "str", "8000"),
)


EXAMPLES: tuple[ExampleSpec, ...] = (
    ExampleSpec(
        "example_scene",
        "场景基础",
        "Core",
        "创建关节配置、载入模型场景，并输出场景信息。",
        False,
    ),
    ExampleSpec(
        "example_ik",
        "基础逆解",
        "IK",
        "支持拖拽末端目标的交互式逆运动学示例。",
        True,
        (
            ParameterSpec("model", "机器人型号", "choice", "ur5", choices=MODEL_CHOICES),
            ParameterSpec("max_iters", "最大迭代", "int", 100, 1, 10000, 10),
            ParameterSpec("step_size", "步长", "float", 1.0, 0.001, 10.0, 0.05),
            ParameterSpec(
                "max_linear_error_norm",
                "线性误差阈值",
                "float",
                0.001,
                0.0,
                1.0,
                0.001,
            ),
            ParameterSpec(
                "max_angular_error_norm",
                "角度误差阈值",
                "float",
                0.001,
                0.0,
                1.0,
                0.001,
            ),
            ParameterSpec("check_collisions", "检查碰撞", "bool", True),
            *COMMON_VISER_PARAMS,
        ),
    ),
    ExampleSpec(
        "example_oink",
        "OInK 跟踪",
        "IK",
        "带可拖拽 Viser 目标和可选碰撞约束的 OInK 伺服示例。",
        True,
        (
            ParameterSpec("model", "机器人型号", "choice", "ur5", choices=MODEL_CHOICES),
            ParameterSpec("task_gain", "任务增益", "float", 1.0, 0.0, 1.0, 0.05),
            ParameterSpec("lm_damping", "LM 阻尼", "float", 0.01, 0.0, 1.0, 0.001),
            ParameterSpec("regularization", "正则化", "float", 1e-3, 0.0, 1.0, 0.0001),
            ParameterSpec("control_freq", "控制频率", "float", 100.0, 1.0, 1000.0, 5.0),
            ParameterSpec(
                "reference_filter_tau",
                "参考滤波时间常数",
                "float",
                0.1,
                0.0,
                5.0,
                0.01,
            ),
            ParameterSpec("self_collision_num_pairs", "自碰撞对数", "int", 0, 0, 200, 1),
            ParameterSpec("self_collision_d_min", "自碰撞最小距离", "float", 0.02, 0.0, 1.0, 0.005),
            ParameterSpec("self_collision_d_max", "自碰撞最大距离", "float", 0.25, 0.0, 2.0, 0.01),
            ParameterSpec("self_collision_gain", "自碰撞增益", "float", 1.0, 0.0, 100.0, 0.5),
            ParameterSpec("limit_acceleration", "限制加速度", "bool", False),
            *COMMON_VISER_PARAMS,
        ),
    ),
    ExampleSpec(
        "example_oink_position_barriers",
        "OInK 位置约束",
        "IK",
        "带控制屏障函数位置安全约束的 OInK 伺服示例。",
        True,
        (
            ParameterSpec("model", "机器人型号", "choice", "ur5", choices=MODEL_CHOICES),
            ParameterSpec("task_gain", "任务增益", "float", 1.0, 0.0, 1.0, 0.05),
            ParameterSpec("lm_damping", "LM 阻尼", "float", 0.01, 0.0, 1.0, 0.001),
            ParameterSpec("regularization", "正则化", "float", 1e-3, 0.0, 1.0, 0.0001),
            ParameterSpec("control_freq", "控制频率", "float", 400.0, 1.0, 1000.0, 10.0),
            ParameterSpec("barrier_gain", "屏障增益", "float", 10.0, 0.0, 100.0, 0.5),
            ParameterSpec("barrier_size", "屏障尺寸", "float", 0.5, 0.05, 5.0, 0.05),
            ParameterSpec("safety_margin", "安全裕度", "float", 0.05, 0.0, 1.0, 0.005),
            ParameterSpec("max_position_error", "最大位置误差", "float", 0.15, 0.0, 2.0, 0.01),
            ParameterSpec("max_rotation_error", "最大姿态误差", "float", 0.5, 0.0, 3.14, 0.05),
            ParameterSpec("reference_filter_tau", "参考滤波时间常数", "float", 0.1, 0.0, 5.0, 0.01),
            *COMMON_VISER_PARAMS,
        ),
    ),
    ExampleSpec(
        "example_rrt",
        "RRT 规划器",
        "Planning",
        "拖拽末端目标后使用 RRT 规划，再进行时间参数化和动画回放。",
        True,
        (
            ParameterSpec("model", "机器人型号", "choice", "ur5", choices=MODEL_CHOICES),
            ParameterSpec("max_connection_distance", "连接距离", "float", 3.0, 0.01, 20.0, 0.1),
            ParameterSpec("collision_check_step_size", "碰撞检查步长", "float", 0.05, 0.001, 1.0, 0.005),
            ParameterSpec("collision_check_use_bisection", "二分碰撞检查", "bool", True),
            ParameterSpec("goal_biasing_probability", "目标偏置概率", "float", 0.15, 0.0, 1.0, 0.01),
            ParameterSpec("max_nodes", "最大节点数", "int", 1000, 1, 100000, 100),
            ParameterSpec("max_planning_time", "规划时间", "float", 2.0, 0.01, 120.0, 0.25),
            ParameterSpec("rrt_connect", "RRT-Connect", "bool", False),
            ParameterSpec("rrt_star", "RRT*", "bool", False),
            ParameterSpec("rewire_distance", "重连距离", "float", 5.0, 0.01, 20.0, 0.1),
            ParameterSpec("fast_return", "快速返回", "bool", True),
            ParameterSpec("include_shortcutting", "包含捷径优化", "bool", False),
            ParameterSpec("max_shortcutting_iters", "捷径迭代", "int", 100, 1, 10000, 10),
            ParameterSpec("interactive_goal", "交互目标", "bool", True),
            ParameterSpec(
                "toppra_mode",
                "TOPPRA 模式",
                "choice",
                "Adaptive",
                choices=("Adaptive", "Hermite", "Cubic"),
            ),
            ParameterSpec("rng_seed", "随机种子", "optional_int", None, 0, 2**31 - 1, 1),
            ParameterSpec("include_obstacles", "包含障碍物", "bool", False),
            ParameterSpec("include_octrees", "包含八叉树", "bool", False),
            *COMMON_VISER_PARAMS,
        ),
    ),
    ExampleSpec(
        "example_toppra_joint_planning",
        "关节空间 TOPPRA 预览",
        "Planning",
        "生成平滑的关节空间路点路径，并用 TOPPRA 做时间参数化；支持预览与执行。",
        True,
        (
            ParameterSpec("model", "机器人型号", "choice", "ur5", choices=MODEL_CHOICES),
            ParameterSpec(
                "toppra_mode",
                "TOPPRA 模式",
                "choice",
                "Adaptive",
                choices=("Hermite", "Cubic", "Adaptive", "LinearBlend"),
            ),
            ParameterSpec("waypoint_count", "路点数量", "int", 6, 2, 50, 1),
            ParameterSpec("path_span", "路径跨度", "float", 0.45, 0.01, 5.0, 0.05),
            ParameterSpec("curvature_scale", "曲率系数", "float", 0.25, 0.0, 1.0, 0.05),
            ParameterSpec("dt", "采样步长", "float", 0.01, 0.001, 1.0, 0.001),
            ParameterSpec("velocity_scale", "速度缩放", "float", 1.0, 0.01, 1.0, 0.05),
            ParameterSpec("acceleration_scale", "加速度缩放", "float", 1.0, 0.01, 1.0, 0.05),
            ParameterSpec(
                "max_adaptive_iterations",
                "自适应迭代",
                "int",
                10,
                1,
                100,
                1,
            ),
            ParameterSpec(
                "max_adaptive_step_size",
                "自适应步长",
                "float",
                0.05,
                0.001,
                1.0,
                0.005,
            ),
            ParameterSpec(
                "max_blend_deviation",
                "融合偏差",
                "float",
                0.01,
                0.0,
                1.0,
                0.001,
            ),
            *COMMON_VISER_PARAMS,
        ),
    ),
    ExampleSpec(
        "example_constrained_rrt",
        "约束 RRT",
        "Planning",
        "拖拽末端目标后规划保持夹爪竖直并位于安全区域内的路径。",
        True,
        (
            ParameterSpec("model", "机器人型号", "choice", "ur5", choices=SUPPORTED_CONSTRAINED_MODELS),
            ParameterSpec("max_tilt_degrees", "最大倾角", "float", 5.0, 0.1, 45.0, 0.5),
            ParameterSpec("path_step_size", "路径步长", "float", 0.1, 0.001, 1.0, 0.01),
            ParameterSpec("max_connection_distance", "连接距离", "float", 0.5, 0.01, 10.0, 0.05),
            ParameterSpec("max_nodes", "最大节点数", "int", 40000, 1, 200000, 1000),
            ParameterSpec("max_planning_time", "规划时间", "float", 1.0, 0.01, 120.0, 0.25),
            ParameterSpec("collision_check_step_size", "碰撞检查步长", "float", 0.05, 0.001, 1.0, 0.005),
            ParameterSpec("rrt_star", "RRT*", "bool", True),
            ParameterSpec("rewire_distance", "重连距离", "float", 1.0, 0.01, 20.0, 0.1),
            ParameterSpec("rng_seed", "随机种子", "int", 1234, 0, 2**31 - 1, 1),
            ParameterSpec("interactive_goal", "交互目标", "bool", True),
            *COMMON_VISER_PARAMS,
        ),
    ),
    ExampleSpec(
        "example_cartesian_planning",
        "笛卡尔规划器",
        "Planning",
        "通过可拖拽的 Viser 目标规划锯齿形笛卡尔路径，并绘制生成的关节轨迹。",
        True,
        (
            ParameterSpec("model", "机器人型号", "choice", "ur5", choices=MODEL_CHOICES),
            ParameterSpec(
                "speed_mode",
                "速度模式",
                "choice",
                "TimeOptimal",
                choices=("TimeOptimal", "Bounded"),
            ),
            ParameterSpec("max_linear_speed", "最大线速度", "float", 0.1, 0.001, 5.0, 0.01),
            ParameterSpec("max_angular_speed", "最大角速度", "float", 0.5, 0.001, 10.0, 0.05),
            ParameterSpec("max_linear_acceleration", "最大线加速度", "float", 0.5, 0.001, 20.0, 0.05),
            ParameterSpec("max_angular_acceleration", "最大角加速度", "float", 2.5, 0.001, 50.0, 0.1),
            ParameterSpec("max_position_error", "最大位置误差", "float", 0.01, 0.0, 1.0, 0.001),
            ParameterSpec("max_orientation_error", "最大姿态误差", "float", 0.1, 0.0, 3.14, 0.01),
            ParameterSpec("velocity_scale", "速度缩放", "float", 1.0, 0.01, 1.0, 0.05),
            ParameterSpec("acceleration_scale", "加速度缩放", "float", 1.0, 0.01, 1.0, 0.05),
            ParameterSpec("dt", "采样步长", "float", 0.01, 0.001, 1.0, 0.001),
            ParameterSpec("path_size", "路径尺寸", "float", 0.15, 0.01, 2.0, 0.01),
            ParameterSpec("path_num_passes", "来回次数", "int", 5, 1, 100, 1),
            ParameterSpec("path_corner_radius", "拐角半径", "float", 0.0, 0.0, 1.0, 0.005),
            ParameterSpec("path_corner_arc_step_deg", "拐角弧步长", "float", 1.0, 0.1, 45.0, 0.5),
            ParameterSpec("ik_max_iters", "最大迭代", "int", 100, 1, 10000, 10),
            ParameterSpec("ik_step_size", "步长", "float", 1.0, 0.001, 10.0, 0.05),
            ParameterSpec(
                "ik_max_linear_error_norm",
                "线性误差阈值",
                "float",
                0.001,
                0.0,
                1.0,
                0.001,
            ),
            ParameterSpec(
                "ik_max_angular_error_norm",
                "角度误差阈值",
                "float",
                0.001,
                0.0,
                1.0,
                0.001,
            ),
            ParameterSpec("ik_check_collisions", "检查碰撞", "bool", True),
            ParameterSpec("interactive_goal", "交互目标", "bool", True),
            *COMMON_VISER_PARAMS,
        ),
    ),
    ExampleSpec(
        "example_action_chunk_tracking",
        "动作片段跟踪",
        "Tracking",
        "通过 OInK 跟踪模拟学习策略生成的笛卡尔或关节动作片段。",
        True,
        (
            ParameterSpec("model", "机器人型号", "choice", "ur5", choices=MODEL_CHOICES),
            ParameterSpec("action_space", "动作空间", "choice", "cartesian", choices=("cartesian", "joint")),
            ParameterSpec("chunk_horizon", "片段时域", "int", 6, 1, 100, 1),
            ParameterSpec("action_scale", "动作缩放", "float", 1.0, 0.0, 10.0, 0.1),
            ParameterSpec("segment_time", "片段时长", "float", 0.5, 0.001, 10.0, 0.05),
            ParameterSpec("control_freq", "控制频率", "float", 100.0, 1.0, 1000.0, 5.0),
            ParameterSpec("task_gain", "任务增益", "float", 1.0, 0.0, 1.0, 0.05),
            ParameterSpec("lm_damping", "LM 阻尼", "float", 0.01, 0.0, 1.0, 0.001),
            ParameterSpec("regularization", "正则化", "float", 1e-6, 0.0, 1.0, 0.000001),
            ParameterSpec("limit_acceleration", "限制加速度", "bool", False),
            ParameterSpec("sleep", "生成时休眠", "bool", False),
            ParameterSpec("playback_speed", "回放速度", "float", 1.0, 0.01, 10.0, 0.1),
            *COMMON_VISER_PARAMS,
        ),
    ),
    ExampleSpec(
        "example_teleop",
        "键盘遥操作",
        "Tracking",
        "使用键盘指令在终端中遥操作一个或多个末端执行器。",
        True,
        (
            ParameterSpec("model", "机器人型号", "choice", "ur5", choices=MODEL_CHOICES),
            ParameterSpec("device", "设备", "choice", "keyboard", choices=("keyboard",)),
            ParameterSpec("ee_mode", "末端模式", "choice", "first", choices=("first", "all")),
            ParameterSpec("control_freq", "控制频率", "float", 50.0, 1.0, 1000.0, 5.0),
            ParameterSpec("linear_sensitivity", "线速度灵敏度", "float", 0.3, 0.001, 5.0, 0.05),
            ParameterSpec("angular_sensitivity", "角速度灵敏度", "float", 0.5, 0.001, 10.0, 0.05),
            ParameterSpec("reference_filter_tau", "参考滤波时间常数", "float", 0.1, 0.0, 5.0, 0.01),
            ParameterSpec("task_gain", "任务增益", "float", 1.0, 0.0, 1.0, 0.05),
            ParameterSpec("lm_damping", "LM 阻尼", "float", 0.01, 0.0, 1.0, 0.001),
            ParameterSpec("regularization", "正则化", "float", 1e-6, 0.0, 1.0, 0.000001),
            ParameterSpec("config_task_gain", "构型任务增益", "float", 1e-4, 0.0, 1.0, 0.0001),
            ParameterSpec("target_axes_length", "目标坐标轴长度", "float", 0.1, 0.001, 1.0, 0.01),
            ParameterSpec("target_axes_radius", "目标坐标轴半径", "float", 0.005, 0.0001, 0.1, 0.001),
            *COMMON_VISER_PARAMS,
        ),
    ),
)
class ParameterEditor(QWidget):
    def __init__(self, spec: ExampleSpec, port: int, parent: QWidget | None = None):
        super().__init__(parent)
        self.spec = spec
        self.widgets: dict[str, QWidget] = {}
        self._build(port)

    def _build(self, port: int) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(12)

        sections = self._group_parameters()
        for title, parameters in sections:
            section = self._build_section(title, parameters, port)
            layout.addWidget(section)
        layout.addStretch(1)

    def _group_parameters(self) -> list[tuple[str, list[ParameterSpec]]]:
        common: list[ParameterSpec] = []
        connection: list[ParameterSpec] = []
        advanced: list[ParameterSpec] = []

        for parameter in self.spec.parameters:
            if parameter.name in {"host", "port"}:
                connection.append(parameter)
            elif parameter.name in ADVANCED_PARAMETER_NAMES:
                advanced.append(parameter)
            else:
                common.append(parameter)

        sections: list[tuple[str, list[ParameterSpec]]] = []
        if common:
            sections.append(("常用参数", common))
        if connection:
            sections.append(("可视化连接", connection))
        if advanced:
            sections.append(("高级参数", advanced))
        return sections

    def _build_section(
        self,
        title: str,
        parameters: list[ParameterSpec],
        port: int,
    ) -> QWidget:
        section = QFrame()
        section.setObjectName("ParameterSection")
        section_layout = QVBoxLayout(section)
        section_layout.setContentsMargins(14, 12, 14, 14)
        section_layout.setSpacing(8)

        title_label = QLabel(title)
        title_label.setObjectName("ParameterSectionTitle")
        section_layout.addWidget(title_label)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        form.setFormAlignment(Qt.AlignTop)
        form.setRowWrapPolicy(QFormLayout.DontWrapRows)
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(9)

        for parameter in parameters:
            widget = self._make_widget(parameter, port)
            self.widgets[parameter.name] = widget
            label = QLabel(_zh_parameter_label(parameter))
            label.setObjectName("ParamLabel")
            label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            label.setMinimumWidth(150)
            label.setWordWrap(False)
            form.addRow(label, widget)

        section_layout.addLayout(form)
        return section

    def _make_widget(self, parameter: ParameterSpec, port: int) -> QWidget:
        if parameter.kind == "bool":
            checkbox = QCheckBox()
            checkbox.setChecked(bool(parameter.default))
            checkbox.setMinimumHeight(28)
            return checkbox
        if parameter.kind == "choice":
            combo = QComboBox()
            for choice in parameter.choices:
                combo.addItem(_zh_choice(parameter, choice), choice)
            default_index = combo.findData(parameter.default)
            if default_index >= 0:
                combo.setCurrentIndex(default_index)
            combo.setMinimumWidth(240)
            return combo
        if parameter.kind in {"int", "optional_int"}:
            spin = QSpinBox()
            minimum = -1 if parameter.kind == "optional_int" else int(parameter.minimum or 0)
            spin.setRange(minimum, int(parameter.maximum or 2**31 - 1))
            spin.setSingleStep(int(parameter.step or 1))
            spin.setSpecialValueText("无" if parameter.kind == "optional_int" else "")
            spin.setValue(spin.minimum() if parameter.default is None else int(parameter.default))
            spin.setMinimumWidth(240)
            return spin
        if parameter.kind == "float":
            spin = QDoubleSpinBox()
            spin.setRange(float(parameter.minimum if parameter.minimum is not None else -1e9), float(parameter.maximum if parameter.maximum is not None else 1e9))
            spin.setSingleStep(float(parameter.step or 0.1))
            spin.setDecimals(6)
            spin.setValue(float(parameter.default))
            spin.setMinimumWidth(240)
            return spin

        line = QLineEdit(str(port) if parameter.name == "port" else str(parameter.default))
        line.setMinimumWidth(240)
        return line

    def values(self) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for parameter in self.spec.parameters:
            widget = self.widgets[parameter.name]
            if isinstance(widget, QCheckBox):
                value = widget.isChecked()
            elif isinstance(widget, QComboBox):
                value = widget.currentData()
            elif isinstance(widget, QSpinBox):
                if parameter.kind == "optional_int" and widget.value() == widget.minimum():
                    value = None
                else:
                    value = widget.value()
            elif isinstance(widget, QDoubleSpinBox):
                value = widget.value()
            elif isinstance(widget, QLineEdit):
                value = widget.text()
            else:
                continue
            values[parameter.name] = value
        return values

    def set_values(self, values: dict[str, Any]) -> None:
        for parameter in self.spec.parameters:
            if parameter.name not in values:
                continue
            widget = self.widgets[parameter.name]
            value = values[parameter.name]
            if isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            elif isinstance(widget, QComboBox):
                index = widget.findData(value)
                if index >= 0:
                    widget.setCurrentIndex(index)
            elif isinstance(widget, QSpinBox):
                if value is None and parameter.kind == "optional_int":
                    widget.setValue(widget.minimum())
                elif value is not None:
                    widget.setValue(int(value))
            elif isinstance(widget, QDoubleSpinBox):
                if value is not None:
                    widget.setValue(float(value))
            elif isinstance(widget, QLineEdit):
                widget.setText(str(value))

    def set_running(self, running: bool) -> None:
        for widget in self.widgets.values():
            widget.setEnabled(not running)


class ParameterDialog(QDialog):
    def __init__(self, spec: ExampleSpec, port: int, values: dict[str, Any], parent: QWidget | None = None):
        super().__init__(parent)
        self.spec = spec
        self.setWindowTitle(f"{_zh_example_title(spec)} - 参数设置")
        self.setMinimumSize(960, 760)
        self.resize(1020, 820)
        self.editor = ParameterEditor(spec, port)
        self.editor.set_values(values)
        self._build()

    def _build(self) -> None:
        self.setObjectName("ParameterDialog")
        self.setModal(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 22, 22, 22)
        layout.setSpacing(16)

        header = QFrame()
        header.setObjectName("ParameterDialogHeader")
        _add_shadow(header, color="#8ea7cf", blur_radius=20.0, y_offset=8.0)
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(18, 16, 18, 16)
        header_layout.setSpacing(6)

        title = QLabel("参数设置")
        title.setObjectName("DialogTitle")
        subtitle = QLabel(f"{_zh_example_title(self.spec)} · 调整后点击“保存并关闭”。")
        subtitle.setObjectName("DialogSubtitle")
        subtitle.setWordWrap(True)
        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        layout.addWidget(header)

        scroll = QScrollArea()
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(self.editor)
        layout.addWidget(scroll, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Save).setText("保存并关闭")
        buttons.button(QDialogButtonBox.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def values(self) -> dict[str, Any]:
        return self.editor.values()


class ExamplePage(QWidget):
    def __init__(self, spec: ExampleSpec, index: int, parent: QWidget | None = None):
        super().__init__(parent)
        self.spec = spec
        self.process: QProcess | None = None
        self.port = 8000 + index
        self.param_values = {
            parameter.name: _default_parameter_value(parameter) for parameter in spec.parameters
        }
        self._build()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)

        page_splitter = QSplitter(Qt.Horizontal)
        page_splitter.setObjectName("PageSplitter")
        page_splitter.setChildrenCollapsible(False)
        page_splitter.setHandleWidth(10)

        left_panel = QFrame()
        left_panel.setObjectName("PageLeftPanel")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)

        hero = QFrame()
        hero.setObjectName("PageHero")
        hero_layout = QGridLayout(hero)
        hero_layout.setContentsMargins(22, 18, 22, 18)
        hero_layout.setHorizontalSpacing(16)
        hero_layout.setVerticalSpacing(8)

        category = QLabel(self.spec.category.upper())
        category.setObjectName("Eyebrow")
        title = QLabel(_zh_example_title(self.spec))
        title.setObjectName("PageTitle")
        description = QLabel(_zh_example_description(self.spec))
        description.setWordWrap(True)
        description.setObjectName("Description")
        mode_badge = QLabel("交互式" if self.spec.interactive else "批处理")
        mode_badge.setObjectName("ModeBadge")
        mode_badge.setAlignment(Qt.AlignCenter)

        hero_layout.addWidget(category, 0, 0)
        hero_layout.addWidget(mode_badge, 0, 1, 2, 1, Qt.AlignRight | Qt.AlignTop)
        hero_layout.addWidget(title, 1, 0)
        hero_layout.addWidget(description, 2, 0, 1, 2)
        _add_shadow(hero, color="#9ab0c8", blur_radius=18.0, y_offset=6.0)
        left_layout.addWidget(hero)

        if self.spec.parameters:
            params_bar = QFrame()
            params_bar.setObjectName("ParameterStrip")
            params_layout = QHBoxLayout(params_bar)
            params_layout.setContentsMargins(18, 14, 18, 14)
            params_layout.setSpacing(12)

            info_layout = QVBoxLayout()
            info_layout.setSpacing(2)
            params_title = QLabel("参数设置")
            params_title.setObjectName("SectionTitle")
            self.param_summary = QLabel(f"已载入 {len(self.spec.parameters)} 项参数，点击打开弹窗修改。")
            self.param_summary.setObjectName("ParamSummary")
            self.param_summary.setWordWrap(True)
            info_layout.addWidget(params_title)
            info_layout.addWidget(self.param_summary)
            params_layout.addLayout(info_layout, 1)

            self.params_button = QPushButton("打开参数设置")
            self.params_button.setObjectName("SecondaryButton")
            self.params_button.clicked.connect(self.open_parameter_dialog)
            params_layout.addWidget(self.params_button)
            _add_shadow(params_bar, color="#9ab0c8", blur_radius=16.0, y_offset=6.0)
            left_layout.addWidget(params_bar)
        else:
            self.param_summary = None
            self.params_button = None

        action_bar = QFrame()
        action_bar.setObjectName("ActionBar")
        button_row = QHBoxLayout(action_bar)
        button_row.setContentsMargins(14, 12, 14, 12)
        button_row.setSpacing(10)
        self.preview_button: QPushButton | None = None
        if self.spec.module == "example_toppra_joint_planning":
            self.preview_button = QPushButton("预览")
            self.preview_button.setObjectName("PreviewButton")
            self.preview_button.clicked.connect(self.preview_example)
            button_row.addWidget(self.preview_button)

        run_label = "执行" if self.spec.module == "example_toppra_joint_planning" else "运行"
        self.run_button = QPushButton(run_label)
        self.run_button.setObjectName("PrimaryButton")
        self.run_button.clicked.connect(self.start_example)
        self.stop_button = QPushButton("停止")
        self.stop_button.setObjectName("DangerButton")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_example)
        self.open_button = QPushButton("浏览器打开")
        self.open_button.setObjectName("SecondaryButton")
        self.open_button.clicked.connect(self.open_viser)
        self.clear_button = QPushButton("清空日志")
        self.clear_button.setObjectName("GhostButton")
        self.clear_button.clicked.connect(lambda: self.log.clear())
        button_row.addWidget(self.run_button)
        button_row.addWidget(self.stop_button)
        button_row.addWidget(self.open_button)
        button_row.addStretch(1)
        button_row.addWidget(self.clear_button)
        _add_shadow(action_bar, color="#9ab0c8", blur_radius=16.0, y_offset=6.0)
        left_layout.addWidget(action_bar)

        self.status = QLabel("准备就绪。")
        self.status.setObjectName("Status")
        left_layout.addWidget(self.status)

        self.solve_stats = QLabel("求解统计：-")
        self.solve_stats.setObjectName("SolveStats")
        self.solve_stats.setWordWrap(True)
        left_layout.addWidget(self.solve_stats)

        self.plan_stats = QLabel("规划统计：-")
        self.plan_stats.setObjectName("PlanStats")
        self.plan_stats.setWordWrap(True)
        left_layout.addWidget(self.plan_stats)

        log_panel = QFrame()
        log_panel.setObjectName("LogPanel")
        log_layout = QVBoxLayout(log_panel)
        log_layout.setContentsMargins(14, 12, 14, 14)
        log_layout.setSpacing(10)
        log_title = QLabel("运行日志")
        log_title.setObjectName("SectionTitle")
        log_layout.addWidget(log_title)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setLineWrapMode(QTextEdit.NoWrap)
        fixed_font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        fixed_font.setPointSize(10)
        self.log.setFont(fixed_font)
        log_layout.addWidget(self.log, 1)
        _add_shadow(log_panel, color="#9ab0c8", blur_radius=16.0, y_offset=6.0)
        left_layout.addWidget(log_panel, 1)

        right_panel = QFrame()
        right_panel.setObjectName("ViserPanel")
        right_panel.setMinimumWidth(560)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(14, 12, 14, 14)
        right_layout.setSpacing(10)

        viewer_header = QHBoxLayout()
        viewer_title_box = QVBoxLayout()
        viewer_title_box.setSpacing(2)
        viewer_title = QLabel("Viser 预览")
        viewer_title.setObjectName("SectionTitle")
        self.viewer_hint = QLabel("右侧显示内嵌预览。点击“预览”或“执行”后，轨迹会出现在这里。")
        self.viewer_hint.setObjectName("ViewerHint")
        self.viewer_hint.setWordWrap(True)
        viewer_title_box.addWidget(viewer_title)
        viewer_title_box.addWidget(self.viewer_hint)
        viewer_header.addLayout(viewer_title_box, 1)

        self.refresh_view_button = QPushButton("刷新预览")
        self.refresh_view_button.setObjectName("SecondaryButton")
        self.refresh_view_button.clicked.connect(self.refresh_viser_view)
        viewer_header.addWidget(self.refresh_view_button)
        right_layout.addLayout(viewer_header)

        self.viser_stack = QStackedWidget()
        self.viser_stack.setObjectName("ViserStack")
        self.viser_placeholder = QLabel("内嵌预览将在示例启动后连接到本地 Viser。")
        self.viser_placeholder.setObjectName("ViserPlaceholder")
        self.viser_placeholder.setAlignment(Qt.AlignCenter)
        self.viser_placeholder.setWordWrap(True)
        self.viser_placeholder.setMinimumHeight(320)
        self.viser_placeholder.setMargin(20)
        self.viser_stack.addWidget(self.viser_placeholder)

        self.viser_view: QWidget | None = None
        if QWebEngineView is not None:
            viewer = QWebEngineView()
            if QuietWebEnginePage is not None:
                viewer.setPage(QuietWebEnginePage(viewer))
            viewer.setObjectName("ViserWebView")
            viewer.setUrl(QUrl("about:blank"))
            self.viser_view = viewer
            self.viser_stack.addWidget(viewer)
        else:
            fallback = QLabel("当前环境缺少 Qt WebEngine，无法在 GUI 内嵌显示。")
            fallback.setObjectName("ViserPlaceholder")
            fallback.setAlignment(Qt.AlignCenter)
            fallback.setWordWrap(True)
            fallback.setMinimumHeight(320)
            fallback.setMargin(20)
            self.viser_stack.addWidget(fallback)

        self.viser_stack.setCurrentWidget(self.viser_placeholder)
        right_layout.addWidget(self.viser_stack, 1)
        _add_shadow(right_panel, color="#9ab0c8", blur_radius=16.0, y_offset=6.0)

        page_splitter.addWidget(left_panel)
        page_splitter.addWidget(right_panel)
        page_splitter.setStretchFactor(0, 1)
        page_splitter.setStretchFactor(1, 1)
        page_splitter.setSizes([680, 640])
        layout.addWidget(page_splitter, 1)

    def open_parameter_dialog(self) -> None:
        if not self.spec.parameters:
            return
        dialog = ParameterDialog(self.spec, self.port, self.param_values, self)
        if dialog.exec() == QDialog.Accepted:
            self.param_values = dialog.values()
            if self.param_summary is not None:
                self.param_summary.setText("参数已保存，可直接预览或执行。")
            self.status.setText("参数已更新。")

    def _launch_example(
        self,
        *,
        extra_params: dict[str, Any] | None = None,
        action_label: str,
        confirm: str | None = None,
    ) -> None:
        if self.process is not None:
            return
        if not RUNNER.exists():
            QMessageBox.critical(self, "缺少启动器", f"找不到 {RUNNER}")
            return

        if confirm is not None:
            reply = QMessageBox.question(
                self,
                f"{action_label}轨迹",
                confirm,
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        params = dict(self.param_values)
        if extra_params:
            params.update(extra_params)
        self.log.append(f"示例：{_zh_example_title(self.spec)}")
        self.log.append(f"命令：$ {sys.executable} {RUNNER.name} {self.spec.module}")
        self.log.append(f"{action_label}：{json.dumps(params, ensure_ascii=False)}\n")

        process = QProcess(self)
        process.setProgram(sys.executable)
        process.setArguments(
            [str(RUNNER), self.spec.module, "--params-json", json.dumps(params)]
        )
        process.setWorkingDirectory(str(EXAMPLE_DIR))
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        current_pythonpath = env.value("PYTHONPATH")
        paths = [str(EXAMPLE_DIR)]
        cmeel_site = _cmeel_site_packages()
        if cmeel_site.exists():
            paths.append(str(cmeel_site))
        if current_pythonpath:
            paths.append(current_pythonpath)
        env.insert("PYTHONPATH", os.pathsep.join(paths))
        current_ld_library_path = env.value("LD_LIBRARY_PATH")
        lib_paths = _shared_library_paths()
        if current_ld_library_path:
            lib_paths.append(current_ld_library_path)
        if lib_paths:
            env.insert("LD_LIBRARY_PATH", os.pathsep.join(lib_paths))
        process.setProcessEnvironment(env)
        process.setProcessChannelMode(QProcess.SeparateChannels)
        process.readyReadStandardOutput.connect(self._read_stdout)
        process.readyReadStandardError.connect(self._read_stderr)
        process.finished.connect(self._finished)
        process.errorOccurred.connect(self._error)
        self.process = process
        self._set_running(True)
        self.status.setText(f"{action_label}运行中。")
        process.start()

    def start_example(self) -> None:
        action_label = "运行"
        confirm = None
        if self.spec.module == "example_toppra_joint_planning":
            action_label = "执行"
            confirm = (
                "执行将播放已规划的轨迹。\n\n"
                "请先使用预览检查轨迹，再将动作送到真实机器人。"
            )
        self._launch_example(action_label=action_label, confirm=confirm)

    def preview_example(self) -> None:
        self._launch_example(extra_params={"preview_only": True}, action_label="预览")

    def stop_example(self) -> None:
        if self.process is None:
            return
        self.status.setText("正在停止...")
        if sys.platform != "win32":
            pid = self.process.processId()
            if pid:
                os.kill(pid, signal.SIGINT)
        else:
            self.process.terminate()
        if not self.process.waitForFinished(2500):
            self.process.kill()

    def open_viser(self) -> None:
        params = dict(self.param_values)
        host = params.get("host", "localhost")
        port = params.get("port", "8000")
        QDesktopServices.openUrl(QUrl(f"http://{host}:{port}"))

    def refresh_viser_view(self) -> None:
        if self.viser_view is None:
            return
        params = dict(self.param_values)
        host = params.get("host", "localhost")
        port = params.get("port", "8000")
        url = QUrl(f"http://{host}:{port}")
        self.viser_view.setUrl(url)
        self.viser_stack.setCurrentWidget(self.viser_view)
        self.viewer_hint.setText(f"已刷新到 {url.toString()}")

    def _sync_viser_view(self) -> None:
        if self.viser_view is None:
            self.viser_stack.setCurrentWidget(self.viser_placeholder)
            return
        if self.process is not None:
            self.refresh_viser_view()
        else:
            self.viser_stack.setCurrentWidget(self.viser_placeholder)
            self.viewer_hint.setText("点击“预览”或“执行”后，轨迹会显示在这里。")

    def close(self) -> bool:
        self.stop_example()
        return super().close()

    def _read_stdout(self) -> None:
        if self.process is None:
            return
        text = bytes(self.process.readAllStandardOutput()).decode(errors="replace")
        self._append_log(text)

    def _read_stderr(self) -> None:
        if self.process is None:
            return
        text = bytes(self.process.readAllStandardError()).decode(errors="replace")
        self._append_log(text)

    def _append_log(self, text: str) -> None:
        cursor = self.log.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.log.setTextCursor(cursor)
        self.log.ensureCursorVisible()
        for line in text.splitlines():
            if line.startswith("Solve stats:"):
                self.solve_stats.setText(
                    f"求解统计：{line.removeprefix('Solve stats:').strip()}"
                )
            elif line.startswith("Plan stats:"):
                self.plan_stats.setText(
                    f"规划统计：{line.removeprefix('Plan stats:').strip()}"
                )

    def _finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        status = "崩溃" if exit_status == QProcess.CrashExit else "完成"
        self.log.append(f"\n[{_zh_example_title(self.spec)} {status}，退出码 {exit_code}]")
        self.process = None
        self._set_running(False)
        self._sync_viser_view()

    def _error(self, error: QProcess.ProcessError) -> None:
        self.status.setText(f"进程错误：{_zh_process_error(error)}")

    def _set_running(self, running: bool) -> None:
        if self.params_button is not None:
            self.params_button.setEnabled(not running)
        self.run_button.setEnabled(not running)
        if self.preview_button is not None:
            self.preview_button.setEnabled(not running)
        self.stop_button.setEnabled(running)
        self.status.setText("运行中。" if running else "准备就绪。")
        if running:
            QTimer.singleShot(1200, self._sync_viser_view)
        else:
            self._sync_viser_view()


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("RoboPlan 控制台")
        self.resize(1460, 900)
        self.setMinimumSize(1180, 760)
        self.pages: list[ExamplePage] = []
        self.page_transition: QPropertyAnimation | None = None
        self._build()

    def _build(self) -> None:
        self.setUnifiedTitleAndToolBarOnMac(True)

        self._build_actions()
        self._build_menus()

        status_bar = QStatusBar(self)
        status_bar.setObjectName("MainStatusBar")
        self.setStatusBar(status_bar)
        self.statusBar().showMessage("准备就绪。")

        shell = QWidget()
        shell.setObjectName("AppShell")
        shell_layout = QVBoxLayout(shell)
        shell_layout.setContentsMargins(18, 14, 18, 18)
        shell_layout.setSpacing(14)

        header = QFrame()
        header.setObjectName("AppHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(18, 14, 18, 14)
        header_layout.setSpacing(12)

        brand_block = QVBoxLayout()
        brand = QLabel("RoboPlan")
        brand.setObjectName("BrandTitle")
        subtitle = QLabel("规划、预览与执行工作台")
        subtitle.setObjectName("BrandSubtitle")
        self.metrics_label = QLabel(f"{len(EXAMPLES)} 个示例 · {sum(1 for spec in EXAMPLES if spec.interactive)} 个交互式 · 规划 / 预览 / 执行")
        self.metrics_label.setObjectName("MetricsLabel")
        brand_block.addWidget(brand)
        brand_block.addWidget(subtitle)
        brand_block.addWidget(self.metrics_label)

        right_block = QVBoxLayout()
        right_block.setSpacing(8)
        self.live_indicator = QLabel("空闲")
        self.live_indicator.setObjectName("LiveIndicator")
        self.live_indicator.setAlignment(Qt.AlignCenter)
        self.live_indicator_effect = QGraphicsOpacityEffect(self.live_indicator)
        self.live_indicator_effect.setOpacity(1.0)
        self.live_indicator.setGraphicsEffect(self.live_indicator_effect)
        self.live_pulse = QPropertyAnimation(self.live_indicator_effect, b"opacity", self)
        self.live_pulse.setDuration(900)
        self.live_pulse.setStartValue(0.62)
        self.live_pulse.setEndValue(1.0)
        self.live_pulse.setEasingCurve(QEasingCurve.InOutSine)
        self.live_pulse.setLoopCount(-1)
        self.context_label = QLabel("当前页面")
        self.context_label.setObjectName("ContextPill")
        self.context_label.setAlignment(Qt.AlignCenter)
        self.context_label.setMinimumWidth(180)
        right_block.addWidget(self.live_indicator, alignment=Qt.AlignRight)
        right_block.addWidget(self.context_label, alignment=Qt.AlignRight)

        header_layout.addLayout(brand_block)
        header_layout.addStretch(1)
        header_layout.addLayout(right_block)
        _add_shadow(header, blur_radius=30.0, y_offset=12.0)
        shell_layout.addWidget(header)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setObjectName("MainSplitter")

        sidebar = QFrame()
        sidebar.setObjectName("SidebarPanel")
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(16, 16, 16, 16)
        sidebar_layout.setSpacing(12)

        sidebar_title = QLabel("示例")
        sidebar_title.setObjectName("SectionTitle")
        sidebar_layout.addWidget(sidebar_title)

        self.search_box = QLineEdit()
        self.search_box.setObjectName("SidebarSearch")
        self.search_box.setPlaceholderText("搜索示例、参数或分类")
        self.search_box.textChanged.connect(self._filter_examples)
        sidebar_layout.addWidget(self.search_box)

        self.list_widget = QListWidget()
        self.list_widget.setObjectName("ExampleList")
        self.list_widget.setMaximumWidth(320)
        self.list_widget.currentRowChanged.connect(self._select_page)
        sidebar_layout.addWidget(self.list_widget, 1)

        self.sidebar_hint = QLabel("先选择示例，再调参数；关节规划建议先预览，再执行。")
        self.sidebar_hint.setWordWrap(True)
        self.sidebar_hint.setObjectName("SidebarHint")
        sidebar_layout.addWidget(self.sidebar_hint)
        _add_shadow(sidebar, blur_radius=26.0, y_offset=10.0)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.tabBar().hide()

        for index, spec in enumerate(EXAMPLES):
            item = QListWidgetItem(f"{_zh_category(spec.category)} / {_zh_example_title(spec)}")
            item.setData(Qt.UserRole, index)
            item.setData(Qt.UserRole + 1, f"{spec.module} {spec.category} {spec.title} {spec.description} {_zh_category(spec.category)} {_zh_example_title(spec)} {_zh_example_description(spec)}".lower())
            self.list_widget.addItem(item)
            page = ExamplePage(spec, index)
            self.pages.append(page)
            self.tabs.addTab(page, _zh_example_title(spec))

        splitter.addWidget(sidebar)
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([320, 1120])
        shell_layout.addWidget(splitter, 1)

        self.setCentralWidget(shell)
        self.list_widget.setCurrentRow(0)
        self._sync_current_page_actions()
        self._start_status_timer()

    def _build_actions(self) -> None:
        style = QApplication.style()
        self.preview_action = QAction(style.standardIcon(QStyle.SP_MediaPlay), "预览", self)
        self.preview_action.setToolTip("预览当前示例")
        self.preview_action.setShortcut(QKeySequence("Ctrl+P"))
        self.preview_action.triggered.connect(self._preview_current_page)

        self.run_action = QAction(style.standardIcon(QStyle.SP_DialogApplyButton), "执行", self)
        self.run_action.setToolTip("运行当前示例")
        self.run_action.setShortcut(QKeySequence("Ctrl+R"))
        self.run_action.triggered.connect(self._start_current_page)

        self.stop_action = QAction(style.standardIcon(QStyle.SP_MediaStop), "停止", self)
        self.stop_action.setToolTip("停止当前示例")
        self.stop_action.setShortcut(QKeySequence("Esc"))
        self.stop_action.triggered.connect(self._stop_current_page)

        self.open_action = QAction(style.standardIcon(QStyle.SP_DirOpenIcon), "打开预览窗口", self)
        self.open_action.setToolTip("在浏览器中打开预览窗口")
        self.open_action.setShortcut(QKeySequence("Ctrl+O"))
        self.open_action.triggered.connect(self._open_current_page)

        self.clear_action = QAction(style.standardIcon(QStyle.SP_DialogResetButton), "清空日志", self)
        self.clear_action.setToolTip("清空当前日志")
        self.clear_action.setShortcut(QKeySequence("Ctrl+L"))
        self.clear_action.triggered.connect(self._clear_current_page)

        self.search_action = QAction("聚焦搜索", self)
        self.search_action.setToolTip("聚焦到示例搜索框")
        self.search_action.setShortcut(QKeySequence("Ctrl+K"))
        self.search_action.triggered.connect(self._focus_search)

        self.usage_action = QAction("使用说明", self)
        self.usage_action.setToolTip("打开使用说明")
        self.usage_action.triggered.connect(self._show_usage)

        self.about_action = QAction("关于", self)
        self.about_action.setToolTip("关于 RoboPlan 控制台")
        self.about_action.triggered.connect(self._show_about)

        self.quit_action = QAction("退出", self)
        self.quit_action.setToolTip("退出程序")
        self.quit_action.setShortcut(QKeySequence("Ctrl+Q"))
        self.quit_action.triggered.connect(self.close)

        self.toggle_sidebar_action = QAction("切换侧栏", self, checkable=True)
        self.toggle_sidebar_action.setToolTip("显示或隐藏侧栏")
        self.toggle_sidebar_action.setChecked(True)
        self.toggle_sidebar_action.triggered.connect(self._toggle_sidebar)

        self.maximize_action = QAction(style.standardIcon(QStyle.SP_TitleBarMaxButton), "最大化/还原", self)
        self.maximize_action.setToolTip("最大化或还原窗口")
        self.maximize_action.setShortcut(QKeySequence("F11"))
        self.maximize_action.triggered.connect(self._toggle_maximize)

    def _build_menus(self) -> None:
        menu_bar = self.menuBar()
        menu_bar.setObjectName("MainMenuBar")
        file_menu = menu_bar.addMenu("文件")
        file_menu.addAction(self.quit_action)

        view_menu = menu_bar.addMenu("视图")
        view_menu.addAction(self.search_action)
        view_menu.addSeparator()
        view_menu.addAction(self.toggle_sidebar_action)
        view_menu.addAction(self.maximize_action)

        help_menu = menu_bar.addMenu("帮助")
        help_menu.addAction(self.usage_action)
        help_menu.addAction(self.about_action)

    def _build_toolbar(self) -> None:
        toolbar = QToolBar()
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonIconOnly)
        toolbar.setIconSize(QSize(18, 18))
        toolbar.setObjectName("MainToolBar")
        self.addToolBar(toolbar)
        toolbar.addAction(self.preview_action)
        toolbar.addAction(self.run_action)
        toolbar.addAction(self.stop_action)
        toolbar.addSeparator()
        toolbar.addAction(self.search_action)
        toolbar.addAction(self.usage_action)
        toolbar.addAction(self.maximize_action)
        toolbar.addSeparator()
        toolbar.addAction(self.open_action)
        toolbar.addAction(self.clear_action)

    def _current_page(self) -> ExamplePage:
        page = self.tabs.currentWidget()
        assert isinstance(page, ExamplePage)
        return page

    def _select_page(self, index: int) -> None:
        if index < 0 or index >= len(self.pages):
            return
        self.tabs.setCurrentIndex(index)
        self._sync_current_page_actions()
        self._animate_page_transition(self._current_page())
        self.statusBar().showMessage(f"已选择 {_zh_example_title(self.pages[index].spec)}。")

    def _sync_current_page_actions(self) -> None:
        page = self._current_page()
        is_toppra = page.spec.module == "example_toppra_joint_planning"
        self.preview_action.setEnabled(is_toppra)
        self.run_action.setText("执行" if is_toppra else "运行")
        self.run_action.setToolTip("执行当前示例" if is_toppra else "运行当前示例")
        self.context_label.setText(_zh_example_title(page.spec))
        self.toggle_sidebar_action.setChecked(self.list_widget.isVisible())
        self._update_surface_badge(page)

    def _update_surface_badge(self, page: ExamplePage) -> None:
        mode = "交互式" if page.spec.interactive else "批处理"
        self.metrics_label.setText(
            f"{len(EXAMPLES)} 个示例 · {sum(1 for spec in EXAMPLES if spec.interactive)} 个交互式 · 当前分类：{_zh_category(page.spec.category)}"
        )
        self.context_label.setToolTip(f"{_zh_example_title(page.spec)} · {mode}")
        self.live_indicator.setToolTip(f"{_zh_example_title(page.spec)} · {mode}")

    def _animate_page_transition(self, page: ExamplePage) -> None:
        effect = QGraphicsOpacityEffect(page)
        effect.setOpacity(0.0)
        page.setGraphicsEffect(effect)
        animation = QPropertyAnimation(effect, b"opacity", self)
        animation.setDuration(240)
        animation.setStartValue(0.0)
        animation.setEndValue(1.0)
        animation.setEasingCurve(QEasingCurve.OutCubic)
        animation.finished.connect(lambda: page.setGraphicsEffect(None))
        self.page_transition = animation
        animation.start(QAbstractAnimation.DeleteWhenStopped)

    def _start_current_page(self) -> None:
        self._current_page().start_example()

    def _preview_current_page(self) -> None:
        page = self._current_page()
        if page.preview_button is not None:
            page.preview_example()
        else:
            page.start_example()

    def _stop_current_page(self) -> None:
        self._current_page().stop_example()

    def _open_current_page(self) -> None:
        self._current_page().open_viser()

    def _clear_current_page(self) -> None:
        self._current_page().log.clear()

    def _toggle_sidebar(self, checked: bool) -> None:
        self.list_widget.setVisible(checked)
        self.search_box.setVisible(checked)
        self.sidebar_hint.setVisible(checked)

    def _toggle_maximize(self) -> None:
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    def _filter_examples(self, text: str) -> None:
        query = text.strip().lower()
        visible_count = 0
        first_visible = -1
        current_item = self.list_widget.currentItem()
        current_visible = current_item is not None and not current_item.isHidden()
        for row in range(self.list_widget.count()):
            item = self.list_widget.item(row)
            haystack = str(item.data(Qt.UserRole + 1) or "")
            visible = not query or query in haystack
            item.setHidden(not visible)
            if visible:
                visible_count += 1
                if first_visible < 0:
                    first_visible = row
        self.sidebar_hint.setText(
            f"正在显示 {visible_count} / {self.list_widget.count()} 个示例。"
            if query
            else "先选择示例，再调参数；关节规划建议先预览，再执行。"
        )
        if not current_visible and first_visible >= 0:
            self.list_widget.setCurrentRow(first_visible)

    def _focus_search(self) -> None:
        self.search_box.setFocus(Qt.ShortcutFocusReason)
        self.search_box.selectAll()

    def _page_is_running(self) -> bool:
        page = self._current_page()
        return page.process is not None

    def _update_live_indicator(self) -> None:
        running = self._page_is_running()
        page = self._current_page()
        if running:
            self.live_indicator.setText("运行中")
            self.live_indicator.setProperty("active", True)
            if self.live_pulse.state() != QAbstractAnimation.Running:
                self.live_pulse.start()
            self.statusBar().showMessage(f"{_zh_example_title(page.spec)} 运行中。")
        else:
            self.live_indicator.setText("空闲")
            self.live_indicator.setProperty("active", False)
            if self.live_pulse.state() == QAbstractAnimation.Running:
                self.live_pulse.stop()
            self.live_indicator_effect.setOpacity(1.0)
            self.statusBar().showMessage(f"已选择 {_zh_example_title(page.spec)}。")
        self.live_indicator.style().unpolish(self.live_indicator)
        self.live_indicator.style().polish(self.live_indicator)

    def _start_status_timer(self) -> None:
        self.activity_timer = QTimer(self)
        self.activity_timer.setInterval(350)
        self.activity_timer.timeout.connect(self._update_live_indicator)
        self.activity_timer.start()

    def _show_usage(self) -> None:
        QMessageBox.information(self, "使用说明", USAGE_TEXT_ZH)

    def _show_about(self) -> None:
        QMessageBox.information(
            self,
            "关于 RoboPlan 控制台",
            "用于 RoboPlan 示例的中文控制台，支持可视化预览、执行和日志查看。",
        )

    def closeEvent(self, event) -> None:
        for page in self.pages:
            page.stop_example()
        event.accept()


def main() -> int:
    if QWebEngineView is not None:
        QApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setStyleSheet(
        """
        QWidget {
            background: transparent;
            color: #172033;
            font-family: "Inter", "Noto Sans CJK SC", "Microsoft YaHei", "Segoe UI", sans-serif;
            font-size: 14px;
        }
        QMainWindow, #AppShell {
            background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1,
                                        stop: 0 #e9f3f7, stop: 0.46 #f4f7fb, stop: 1 #e8edf6);
        }
        QToolTip {
            background: #ffffff;
            color: #172033;
            border: 1px solid #cbd7e4;
            border-radius: 6px;
            padding: 6px 8px;
        }
        QMenuBar {
            background: rgba(255, 255, 255, 0.84);
            color: #243247;
            border-bottom: 1px solid rgba(196, 211, 226, 0.66);
            padding: 6px 10px;
        }
        QMenuBar::item {
            background: transparent;
            padding: 6px 10px;
            border-radius: 7px;
        }
        QMenuBar::item:selected {
            background: rgba(232, 240, 255, 0.95);
        }
        QMenu {
            background: rgba(255, 255, 255, 0.97);
            color: #243247;
            border: 1px solid rgba(203, 215, 228, 0.92);
            padding: 6px;
            border-radius: 10px;
        }
        QMenu::item {
            padding: 8px 18px;
            border-radius: 6px;
        }
        QMenu::item:selected {
            background: #e8f0ff;
        }
        QToolBar {
            background: rgba(255, 255, 255, 0.82);
            border: none;
            spacing: 8px;
            padding: 8px 12px;
            border-bottom: 1px solid rgba(196, 211, 226, 0.58);
        }
        QToolBar QToolButton {
            background: transparent;
            border: none;
            border-radius: 10px;
            padding: 7px;
        }
        QToolBar QToolButton:hover {
            background: rgba(232, 240, 255, 0.92);
        }
        QToolBar QToolButton:pressed {
            background: rgba(215, 228, 248, 0.95);
        }
        QStatusBar {
            background: rgba(248, 251, 255, 0.92);
            color: #5f6f83;
            border-top: 1px solid rgba(196, 211, 226, 0.58);
        }
        QFrame#AppHeader,
        QFrame#SidebarPanel,
        QFrame#PageHero,
        QFrame#PageLeftPanel,
        QFrame#ParameterStrip,
        QFrame#ParameterDialogHeader,
        QFrame#ParameterSection,
        QFrame#ActionBar,
        QFrame#ViserPanel,
        QFrame#LogPanel {
            background: rgba(255, 255, 255, 0.82);
            border: 1px solid rgba(203, 215, 228, 0.82);
            border-radius: 14px;
        }
        QFrame#PageHero {
            background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1,
                                        stop: 0 rgba(255, 255, 255, 0.98),
                                        stop: 0.52 rgba(245, 249, 255, 0.96),
                                        stop: 1 rgba(232, 241, 255, 0.96));
            border: 1px solid rgba(195, 209, 225, 0.88);
        }
        QFrame#SidebarPanel {
            background: rgba(255, 255, 255, 0.86);
        }
        QFrame#PageLeftPanel {
            background: transparent;
        }
        QFrame#ParameterStrip,
        QFrame#ParameterDialogHeader,
        QFrame#ActionBar {
            background: rgba(255, 255, 255, 0.88);
        }
        QFrame#ParameterSection {
            background: rgba(255, 255, 255, 0.74);
        }
        QFrame#ViserPanel {
            background: rgba(255, 255, 255, 0.84);
        }
        QFrame#LogPanel {
            background: rgba(255, 255, 255, 0.84);
        }
        QLabel#BrandTitle {
            font-size: 24px;
            font-weight: 700;
            color: #101827;
        }
        QLabel#BrandSubtitle {
            color: #647084;
        }
        QLabel#MetricsLabel {
            color: #75869a;
            font-size: 12px;
        }
        QLabel#LiveIndicator {
            min-width: 72px;
            padding: 7px 12px;
            border-radius: 999px;
            border: 1px solid rgba(159, 177, 199, 0.76);
            background: rgba(245, 249, 255, 0.94);
            color: #41678f;
            font-weight: 700;
        }
        QLabel#LiveIndicator[active="true"] {
            background: rgba(233, 255, 246, 0.96);
            border: 1px solid rgba(43, 182, 115, 0.42);
            color: #1e8f59;
        }
        QLabel#ContextPill,
        QLabel#ModeBadge {
            background: rgba(232, 240, 255, 0.94);
            color: #215c9a;
            border: 1px solid rgba(196, 211, 226, 0.84);
            border-radius: 999px;
            padding: 8px 14px;
            font-weight: 600;
        }
        QLabel#Eyebrow {
            color: #55789d;
            font-size: 11px;
            font-weight: 700;
        }
        QLabel#PageTitle {
            font-size: 26px;
            font-weight: 700;
            color: #101827;
        }
        QLabel#Description {
            color: #5d6d7f;
        }
        QLabel#SectionTitle {
            color: #18324d;
            font-size: 14px;
            font-weight: 700;
        }
        QLabel#DialogTitle {
            color: #101827;
            font-size: 18px;
            font-weight: 700;
        }
        QLabel#DialogSubtitle {
            color: #5d6d7f;
        }
        QLabel#ParamSummary {
            color: #627387;
        }
        QLabel#ParameterSectionTitle {
            color: #2b3d52;
            font-size: 13px;
            font-weight: 700;
            padding-bottom: 2px;
        }
        QLabel#ParamLabel {
            color: #304155;
            font-weight: 600;
            padding-right: 4px;
        }
        QLabel#Status {
            color: #637488;
            padding-left: 4px;
        }
        QLabel#SolveStats {
            color: #22506f;
            background: rgba(235, 244, 251, 0.92);
            border: 1px solid rgba(197, 215, 231, 0.95);
            border-radius: 8px;
            padding: 8px 10px;
        }
        QLabel#PlanStats {
            color: #5a3d1f;
            background: rgba(250, 241, 229, 0.96);
            border: 1px solid rgba(230, 210, 188, 0.96);
            border-radius: 8px;
            padding: 8px 10px;
        }
        QLabel#SidebarHint {
            color: #627387;
            padding-top: 4px;
        }
        QLabel#ViewerHint {
            color: #627387;
        }
        QLabel#ViserPlaceholder {
            color: #7b8797;
            background: rgba(248, 251, 255, 0.88);
            border: 1px dashed rgba(196, 211, 226, 0.95);
            border-radius: 10px;
        }
        QLineEdit#SidebarSearch {
            background: rgba(255, 255, 255, 0.94);
            border: 1px solid rgba(203, 215, 228, 0.92);
            border-radius: 12px;
            padding: 10px 12px;
            color: #1f2a37;
        }
        QLineEdit#SidebarSearch:focus {
            border: 1px solid #2f6fed;
        }
        QListWidget#ExampleList {
            background: rgba(255, 255, 255, 0.90);
            border: 1px solid rgba(203, 215, 228, 0.92);
            border-radius: 12px;
            padding: 8px;
            outline: none;
        }
        QListWidget#ExampleList::item {
            padding: 12px 12px;
            margin: 4px 0;
            border-radius: 8px;
            color: #233243;
        }
        QListWidget#ExampleList::item:selected {
            background: rgba(232, 240, 255, 0.96);
            color: #113a66;
            border: 1px solid rgba(139, 180, 234, 0.88);
        }
        QListWidget#ExampleList::item:hover {
            background: #f3f7fc;
        }
        QFrame#ParameterStrip QLabel,
        QFrame#ParameterDialogHeader QLabel,
        QFrame#ParameterSection QLabel,
        QFrame#LogPanel QLabel {
            background: transparent;
        }
        QLineEdit,
        QComboBox,
        QSpinBox,
        QDoubleSpinBox {
            background: #ffffff;
            color: #1f2a37;
            border: 1px solid #cbd7e4;
            border-radius: 8px;
            padding: 8px 10px;
            selection-background-color: #2f6fed;
            selection-color: #ffffff;
            min-height: 34px;
        }
        QLineEdit:hover,
        QComboBox:hover,
        QSpinBox:hover,
        QDoubleSpinBox:hover {
            border-color: #9fb1c7;
        }
        QLineEdit:focus,
        QComboBox:focus,
        QSpinBox:focus,
        QDoubleSpinBox:focus {
            border: 1px solid #2f6fed;
            background: #ffffff;
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
        QScrollArea {
            border: none;
            background: transparent;
        }
        QSplitter#PageSplitter {
            background: transparent;
        }
        QSplitter#PageSplitter::handle {
            background: rgba(170, 186, 203, 0.28);
            width: 8px;
            border-radius: 4px;
        }
        QStackedWidget#ViserStack {
            background: transparent;
        }
        QDialog#ParameterDialog {
            background: rgba(247, 249, 252, 0.98);
        }
        QDialogButtonBox QPushButton {
            min-width: 116px;
        }
        QCheckBox {
            spacing: 8px;
            color: #223243;
            min-height: 28px;
        }
        QCheckBox::indicator {
            width: 16px;
            height: 16px;
            border-radius: 4px;
            border: 1px solid #b9c8d9;
            background: #ffffff;
        }
        QCheckBox::indicator:checked {
            background: #2ebd85;
            border: 1px solid #2ebd85;
        }
        QPushButton {
            min-height: 34px;
            padding: 6px 14px;
            border-radius: 8px;
            border: 1px solid rgba(199, 211, 226, 0.92);
            background: rgba(255, 255, 255, 0.96);
            color: #223243;
        }
        QPushButton:hover {
            background: rgba(243, 248, 255, 0.98);
            border-color: #9fb1c7;
        }
        QPushButton:pressed {
            background: #eaf2ff;
        }
        QPushButton#PrimaryButton {
            background: #2f6fed;
            border-color: #2f6fed;
            color: #ffffff;
            font-weight: 700;
        }
        QPushButton#PrimaryButton:hover {
            background: #255ed0;
            border-color: #255ed0;
        }
        QPushButton#PreviewButton {
            background: rgba(231, 241, 255, 0.96);
            border-color: rgba(158, 195, 242, 0.90);
            color: #215c9a;
        }
        QPushButton#PreviewButton:hover {
            background: #dcebff;
        }
        QPushButton#SecondaryButton {
            background: rgba(246, 249, 252, 0.95);
            border-color: rgba(203, 216, 231, 0.92);
        }
        QPushButton#SecondaryButton:hover {
            background: #edf4fb;
        }
        QPushButton#DangerButton {
            background: rgba(255, 240, 242, 0.96);
            border-color: rgba(240, 184, 192, 0.96);
            color: #a53a4c;
        }
        QPushButton#DangerButton:hover {
            background: #ffe5ea;
        }
        QPushButton#GhostButton {
            background: transparent;
            border-color: rgba(203, 216, 231, 0.92);
            color: #5e6f83;
        }
        QPushButton:disabled {
            background: rgba(243, 246, 250, 0.80);
            color: #a1afbf;
            border-color: rgba(215, 224, 234, 0.80);
        }
        QTextEdit,
        QPlainTextEdit {
            background: rgba(255, 255, 255, 0.92);
            color: #1f2a37;
            border: 1px solid rgba(203, 215, 228, 0.92);
            border-radius: 8px;
            selection-background-color: #8bb4ea;
        }
        QTabWidget::pane {
            border: 1px solid rgba(203, 215, 228, 0.82);
            border-radius: 12px;
            background: rgba(255, 255, 255, 0.84);
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
        QGroupBox {
            background: rgba(255, 255, 255, 0.86);
            border: 1px solid rgba(203, 215, 228, 0.88);
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
            background: rgba(255, 255, 255, 0.92);
        }
        """
    )
    window = MainWindow()
    window.showMaximized()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
