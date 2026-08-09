# Wheeled Arm + PICO Teleoperation Handoff Notes

本文档记录 `wheeled_arm` 机器人和 `wheeled_arm_pico` 遥操作集成过程中已经遇到的问题、根因、处理方案和常用验证命令。后续使用其他智能体继续分析时，可先阅读本文件，避免重复排查。

## 当前目标

- 在 LeRobot 中支持自定义机器人 `wheeled_arm`。
- 使用机器人 SDK/LCM 接口控制和记录左右臂、左右夹爪。
- 相机使用 `lerobot_camera_ros2`。
- 使用 PICO 控制器作为 `wheeled_arm_pico` teleoperator。
- 支持 `viser` 可视化环境，便于不连接实物机器人时先验证 PICO 到 IK/action 的链路。

## 主要文件

- Robot:
  - `src/lerobot/robots/wheeled_arm/config_wheeled_arm.py`
  - `src/lerobot/robots/wheeled_arm/wheeled_arm.py`
  - `src/lerobot/robots/wheeled_arm/hardware_interface/lcm_handler.py`
- Teleoperator:
  - `src/lerobot/teleoperators/wheeled_arm_pico/config_wheeled_arm_pico.py`
  - `src/lerobot/teleoperators/wheeled_arm_pico/wheeled_arm_pico.py`
  - `src/lerobot/teleoperators/wheeled_arm_pico/ik_utils.py`
  - `src/lerobot/teleoperators/wheeled_arm_pico/visualization.py`
  - `src/lerobot/teleoperators/wheeled_arm_pico/wheel_arm_teleop.py`
- Tests:
  - `tests/robots/test_wheeled_arm.py`
  - `tests/teleoperators/test_wheeled_arm_pico.py`

## 已实现行为

### `wheeled_arm` robot

- 默认记录和控制 16 个标量：
  - `left_arm_0.pos` 到 `left_arm_6.pos`
  - `right_arm_0.pos` 到 `right_arm_6.pos`
  - `left_gripper.pos`
  - `right_gripper.pos`
- LCM 23 维 package 映射：
  - `[0:7]`: 左臂
  - `[7:14]`: 右臂
  - `[14]`: 左夹爪
  - `[15]`: 右夹爪
  - 其余头、腰、腿暂不作为 LeRobot feature 暴露。
- 默认 `controlled_parts` 包含：
  - `left_arm`
  - `right_arm`
  - `left_gripper`
  - `right_gripper`
- `has_valid_feedback` 用于判断是否已经收到新鲜的左右臂 LCM 状态。

### `wheeled_arm_pico` teleoperator

- IK 仍然只解左右 7 轴机械臂，共 14 维。
- 输出 action 时在 14 维臂关节后追加左右夹爪，共 16 维。
- 默认 PICO 控制方式：
  - `left_grip` / `right_grip`: 激活左右机械臂跟随。
  - `left_trigger` / `right_trigger`: 控制左右夹爪。
  - `Y`: 重置 PICO 相对位姿基准。
- 夹爪范围可配置：
  - `--teleop.gripper_open_pos=0.0`
  - `--teleop.gripper_closed_pos=1.0`
- 如果实物夹爪单位不是 0..1，可在 CLI 中调整这两个值。

### 可视化

- `wheel_arm_teleop.py` 是独立可视化验证脚本。
- 支持 `--mock-xr`，不连接 PICO 也能看 IK/URDF/action 链路。
- 可视化依赖 `viser`、`yourdfpy` 只在开启可视化时导入，不应影响普通 teleop import。
- 数据采集使用 `--display_data=true --display_mode=rerun` 时，Rerun 窗口会额外显示 `robot`
  3D view，包含 IK 机器人骨架、左右 TCP 坐标轴、PICO target、target error 连线和碰撞状态。
- `--teleop.visualize=true` 的 viser 完整 URDF 可视化仍然保留，可与 Rerun 同时打开。

## 已遇到的问题和处理方案

### 1. Python 3.10 解析失败

现象：

```text
ImportError: cannot import name 'Self' from 'typing'
SyntaxError: def deserialize_json_into_object[T: JsonLike](...)
ImportError: cannot import name 'Unpack' from 'typing'
```

根因：

- 当前 `xr` conda 环境是 Python 3.10。
- 仓库中部分代码使用了 Python 3.11/3.12 的 typing 语法。

处理：

- `Self` / `Unpack` / `NotRequired` 等改为从 `typing_extensions` 兜底。
- PEP 695 新语法改成 Python 3.10 可解析的 `TypeVar` / `Generic` / 普通类型别名。

建议：

- 长期更推荐创建 Python 3.12 环境。
- 如果继续使用 Python 3.10，注意不要再引入 PEP 695 语法。

### 2. ROS2 camera 插件重复注册

现象：

```text
Could not import third-party plugin: lerobot_camera_ros2
ValueError: Cannot register ... as ros2 because ... is already registered as ros2
```

根因：

- 本地包已经通过 `lerobot.cameras.lerobot_camera_ros2` 注册了 `ros2`。
- 第三方插件扫描又导入一次 `lerobot_camera_ros2`，导致 draccus 重复注册。

处理：

- `register_third_party_plugins()` 中对 `"is already registered as"` 的 `ValueError` 降级为跳过，不再打印整段 traceback。

### 3. `cv_bridge` 与 NumPy 2 不兼容

现象：

```text
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6
AttributeError: _ARRAY_API not found
```

根因：

- ROS Humble 的 `cv_bridge` 通常按 NumPy 1.x 编译。
- 当前 `xr` 环境使用 NumPy 2.2.6。

处理：

- ROS2 RGB 相机在 NumPy 2 环境下默认跳过 `cv_bridge`，使用手动 RGB/BGR/mono 转换。
- 如果启用 depth topic，仍然要求可用的 `cv_bridge`。

建议：

- RGB 采集可继续用当前 fallback。
- 深度图采集建议使用 NumPy 1.x 环境，或重编译 `cv_bridge`。

### 4. `ROS2Camera.read_latest()` FutureWarning

现象：

```text
FutureWarning: ROS2Camera.read_latest() is not implemented.
```

根因：

- `WheeledArm.get_observation()` 调用 `cam.read_latest()`。
- ROS2 camera 原先没有覆盖基类方法。

处理：

- ROS2 camera 增加：
  - `latest_image_timestamp`
  - `latest_depth_timestamp`
  - `read_latest()`
  - `read_latest_depth()`

### 5. 不连接实物时 teleop 被拉回零位

现象：

- `python wheel_arm_teleop.py` 可视化正常。
- `lerobot-record --robot.type=wheeled_arm --teleop.type=wheeled_arm_pico ...` 中遥操作总回零位。

根因：

- `lerobot-record` 每帧先读 `robot.get_observation()`，再调用 `teleop.send_feedback(obs)`。
- 不连接实物时，`LCMHandler.joint_current_pos` 初始是全 0。
- 于是 teleop 内部 IK 状态每帧被 0 observation 覆盖。

处理：

- `LCMHandler` 记录左右臂状态时间。
- `WheeledArm.has_valid_feedback` 只有收到左右臂 LCM 状态且未超时时才为 True。
- `lerobot_record.py` 和 `lerobot_teleoperate.py` 只有 `has_valid_feedback=True` 时才调用 `teleop.send_feedback(obs)`。

注意：

- 没连接实物时，数据集里的 `observation.state` 仍然可能是 0，这是正常的。
- 此模式适合验证 PICO/action/相机链路，不适合作为真实状态训练数据。

### 6. `LockedJointsTask` 不能实例化

现象：

```text
TypeError: Can't instantiate abstract class LockedJointsTask with abstract method __repr__
```

根因：

- bundled Pink 的 `Task` 抽象基类要求实现 `__repr__`。

处理：

- `make_locked_joints_task_class()` 中给 `LockedJointsTask` 补了 `__repr__`。

### 7. 夹爪控制与记录

需求：

- 遥操作采集时控制并记录左右夹爪。

处理：

- `WHEELED_ARM_JOINT_NAMES` 从 14 维扩到 16 维。
- `WheeledArm.get_observation()` 记录 LCM package `[14]`、`[15]` 为左右夹爪。
- `WheeledArm.send_action()` 支持发送左右夹爪，并设置 `left_gripper_moving` / `right_gripper_moving`。
- `WheeledArmPico.get_action()` 从 PICO trigger 生成 `left_gripper.pos` / `right_gripper.pos`。

### 8. PICO IK 调参接口

`wheeled_arm_pico` 当前暴露的 IK 相关 CLI/config 参数包括：

- Frame task:
  - `--teleop.position_cost=5.0`
  - `--teleop.orientation_cost=1.0`
  - `--teleop.frame_lm_damping=0.0`
  - `--teleop.task_gain=0.5`
- Posture task:
  - `--teleop.posture_cost=0.0001`
  - `--teleop.posture_gain=1.0`
  - `--teleop.posture_lm_damping=0.0`
- Locked-joints hard constraint:
  - `--teleop.locked_joints_gain=1.0`
  - `--teleop.locked_joints_lm_damping=0.0`
- QP / solve_ik:
  - `--teleop.solver=daqp`
  - `--teleop.ik_damping=1e-12`
  - `--teleop.ik_safety_break=false`
  - `--teleop.enforce_limits=true`
  - `--teleop.solver_kwargs='{\"verbose\": false}'`
- Self-collision barrier:
  - `--teleop.use_self_collision=true`
  - `--teleop.d_min=0.03`
  - `--teleop.initial_ignore_distance=0.03`
  - `--teleop.self_collision_gain=10.0`
  - `--teleop.self_collision_safe_displacement_gain=5.0`
  - `--teleop.collision_warning_distance=0.01`
- Rerun robot 3D visualization:
  - `--teleop.rerun_visualize_robot=true`
  - `--teleop.rerun_robot_update_hz=10.0`
  - `--teleop.rerun_robot_prefix=robot`
  - `--teleop.rerun_robot_axis_length=0.12`

`position_cost` / `orientation_cost` 可传单个标量，也可传 3 维列表做各轴 anisotropic cost。

### 9. `wheeled_arm_pico` IK 依赖缺失

现象：

```text
ModuleNotFoundError: No module named 'hppfcl'
ImportError: wheeled_arm_pico requires Pinocchio/Pink IK dependencies plus the PICO SDK.
```

根因：

- `wheeled_arm_pico` 运行 IK 时必须有 Pinocchio；默认开启 self-collision 时还需要 FCL 碰撞库。
- 当前 `xr` 环境里缺 `pinocchio`，也缺 `hppfcl` / `coal`。

处理：

- `import_runtime_dependencies()` 会一次性列出缺失依赖，不再只停在第一个 import error。
- FCL 后端兼容 `hppfcl` 和新版 `coal` 模块名。
- 如果临时不需要 self-collision，可加：
  - `--teleop.use_self_collision=false`

注意：

- 即使关闭 self-collision，仍然需要安装 `pinocchio`。
- 当前 LeRobot 依赖集更推荐 Python 3.12+；Python 3.10 环境可能需要走 conda-forge 安装 Pinocchio/FCL。

### 10. 默认 URDF 文件位置

- `real_robot.urdf` 已放在：
  - `src/lerobot/teleoperators/wheeled_arm_pico/assets/wheeled_robot_sim/urdf/real_robot.urdf`
- `default_urdf_path()` 默认使用这个位置。
- 日常运行采集或独立可视化时不需要再传 `--teleop.urdf_path` / `--urdf-path`。

### 11. PICO/IK `viser` 可视化依赖缺失

现象：

```text
ModuleNotFoundError: No module named 'viser'
ImportError: WheeledArmPico visualization requires `viser` and `yourdfpy`.
```

根因：

- `--teleop.visualize=true` 打开的是 PICO/IK 的 `viser` 界面，需要 `viser` 和 `yourdfpy`。
- `--display_data=true --display_mode=rerun` 打开的是采集数据流的 Rerun 界面，不依赖 `viser`。

处理：

- 当前 `xr` 环境已通过 conda-forge 安装 `viser`。
- 程序会在启动 PICO SDK 前先检查 `viser` / `yourdfpy`，避免可视化依赖缺失后设备侧异常退出。
- 如果只需要 Rerun 数据界面，可关闭 PICO/IK 可视化：
  - `--teleop.visualize=false`

### 12. `wheeled_arm` LCM 依赖和 multicast route

现象：

```text
ModuleNotFoundError: No module named 'lcm'
ImportError: 'lcm' is required to control wheeled_arm.
```

处理：

- 当前 `xr` 环境已通过 pip 安装 Python LCM binding：
  - `/home/kuanli/miniconda3/envs/xr/bin/python -m pip install lcm`
- `WheeledArmConfig` 增加 `lcm_url`，默认：
  - `udpm://239.255.76.67:8880?ttl=1`
- 如需切换 LCM 地址，可在 CLI 覆盖：
  - `--robot.lcm_url='udpm://239.255.76.67:8880?ttl=1'`

如果安装后出现：

```text
RuntimeError: Couldn't create LCM
LCM requires a valid multicast route.
```

说明 Linux 当前网络没有可用 multicast route。仅本机/回环测试时可临时执行：

```bash
sudo ip link set lo multicast on
sudo ip route add 224.0.0.0/4 dev lo
```

连接真实机器人时，应确保连接到机器人所在网络，并且该网络接口有 multicast route。

### 13. 相机图像尺寸与 dataset feature 不一致

现象：

```text
ValueError: The feature 'observation.images.front' of shape '(480, 640, 3)'
does not have the expected shape '(720, 1280, 3)' or '(1280, 3, 720)'.
```

根因：

- `wheeled_arm` 默认相机配置登记的是 1280x720。
- 当前 ROS2 前置相机实际发布的是 640x480 图像。
- 数据集 feature 在 `robot.connect()` 前根据 config 创建，因此 ROS2 camera 连接后读取到的实际尺寸不会反向更新本次 dataset metadata。

处理：

- `wheeled_arm_cameras_config()` 的默认 `front` 相机尺寸改为：
  - `width=640`
  - `height=480`

注意：

- 如果后续相机 topic 改成其他分辨率，需要同步修改 robot camera config，或在 CLI 中覆盖 `--robot.cameras=...`。

## 常用命令

### 安装 Python 运行依赖

在已有 conda 环境中安装当前 wheeled_arm/PICO 采集所需依赖：

```bash
bash scripts/setup_wheeled_arm_pico_conda.bash --env xr
```

如果要创建新环境：

```bash
bash scripts/setup_wheeled_arm_pico_conda.bash --create-env --env xr --python 3.12
```

该脚本会安装：

- 本地 editable LeRobot：`pip install -e ".[core_scripts,gui]"`
- IK/碰撞/QP/可视化依赖：`pinocchio`、`hpp-fcl`、`qpsolvers`、`daqp`、`viser`、`yourdfpy`
- LCM Python binding：`lcm`
- PICO SDK：`XRoboToolkit-PC-Service-Pybind`
- 可选数据格式转换依赖：使用 `--with-conversion-deps` 时安装 `tensorflow`、`tensorflow-datasets`、
  `h5py`、`ray[default]`、`datatrove[ray]`、`apache-beam`

### PySide6 图形界面

面向不熟悉命令行的用户，可直接启动桌面 GUI：

```bash
conda activate xr
lerobot-wheeled-arm-gui
```

GUI 入口文件：

- `src/lerobot/scripts/wheeled_arm_gui.py`

GUI 当前封装了这些流程：

- 数据采集：默认生成 `wheeled_arm` + `wheeled_arm_pico` 的 `lerobot-record` 命令，支持
  Rerun 数据窗口、Rerun 机器人 3D、viser URDF 窗口、mock PICO、采集参数和高级 CLI 参数追加。
- 数据集查看：生成 `lerobot-dataset-viz` 命令查看本地 episode；采集结束后会尝试从
  `$HF_LEROBOT_HOME` 中自动识别最近生成的带时间戳数据集并填入查看页。
- 数据集编辑：生成 `lerobot-edit-dataset` 命令，支持查看信息、删除 episode、拆分、合并、
  删除 feature、修改任务文本、图片转视频、重算统计和重编码视频。GUI 会对修改原数据集的操作给出确认提示。
  该页借鉴 Unitree 数据编辑器思路并在当前 GUI 中实现了 LeRobot 数据集预览器：
  可加载本地 episode、播放/暂停、切换 episode、查看最多 4 路相机画面、拖动帧进度条，并可一键把当前
  episode 填入“删除 Episode”。区间选择目前仅用于预览和记录范围，GUI 不直接裁剪帧；LeRobot v3 数据集
  需要通过正式 dataset operation 同步更新 parquet/video/meta，不能按参考程序的方式直接删除图片和 JSON。
- 格式转换：基于保留的 `GUI_reference/Any4LeRobotGUI/backend` 增加独立“格式转换”页，支持 OpenX/AgiBot/RoboMIND/LIBERO
  转 LeRobot、LeRobot 转 RLDS，以及 LeRobot v1.6/v2.0/v2.1/v3.0 之间的常见版本转换。该页默认调用
  `GUI_reference/Any4LeRobotGUI/backend` 下的转换脚本；`LeRobot v2.1 -> v3.0` 例外，优先调用当前项目维护的
  `src/lerobot/scripts/convert_dataset_v21_to_v30.py`，避免使用参考目录里的旧拷贝。可在界面中改 Python 路径和 backend 路径。
  `GUI_reference` 目录已经精简，只保留转换 backend；`LeRobot v1.6 -> v2.0` 仍依赖旧版 `lerobot.common.*` 环境，
  当前环境中 GUI 会显示命令，但实际转换请切换到匹配旧版 LeRobot 的环境。
  默认安装脚本不会安装 TensorFlow/Ray/Datatrove/Beam 等重依赖，如需运行这些转换，可执行：
  `bash scripts/setup_wheeled_arm_pico_conda.bash --env xr --with-conversion-deps`。
  如果 GUI 被安装到 conda site-packages 后误判项目根目录，可设置 `LEROBOT_PROJECT_ROOT=/home/kuanli/Documents/lerobot`；
  GUI 也会在已保存的 Backend 路径不存在时自动回退到当前仓库的 `GUI_reference/Any4LeRobotGUI/backend`。
- 常用命令：生成 `src/lerobot/scripts` 下常用脚本命令，当前包含 `lerobot-info`、`lerobot-find-cameras`、
  `lerobot-find-port`、`lerobot-teleoperate`、`lerobot-replay`、`lerobot-calibrate`、
  `lerobot-setup-motors`、`lerobot-find-joint-limits`、`lerobot-setup-can`、`lerobot-train`、
  `lerobot-eval`、`lerobot-rollout`、`lerobot-annotate`、`lerobot-imgtransform-viz`、
  `augment_dataset_quantile_stats`、`lerobot-convert-dcp` 和 `lerobot-train-tokenizer`。
  其中 `find-port` 这类需要交互的脚本可通过 GUI 的“发送 Enter”按钮继续。
  `lerobot-record`、`lerobot-dataset-viz`、`lerobot-edit-dataset` 和 `convert_dataset_v21_to_v30`
  分别由“采集”“查看数据集”“编辑数据集”“格式转换”专页覆盖。

GUI 仍然以子进程方式运行原始 CLI，并在右侧显示完整命令预览和运行日志；因此现场排查时，
可以直接复制命令到终端复现。

如果启动 GUI 时出现 Qt xcb 插件错误：

```text
qt.qpa.plugin: From 6.5.0, xcb-cursor0 or libxcb-cursor0 is needed
Could not load the Qt platform plugin "xcb"
```

说明系统缺少 PySide6/Qt 的桌面运行库。安装：

```bash
sudo apt-get update && sudo apt-get install -y \
  libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
  libxcb-render-util0 libxcb-xinerama0 libxcb-xkb1 libxkbcommon-x11-0 \
  libegl1 libgl1
```

`scripts/setup_wheeled_arm_pico_conda.bash` 已默认检查并安装这些 Ubuntu 系统包；若不想安装系统包，
可传 `--skip-system-packages`。

如果当前用户没有 sudo 权限，也可在 conda 环境里补运行库：

```bash
conda install -n xr -c conda-forge -y xcb-util-cursor
```

### 独立可视化验证，不连接机器人

```bash
python /home/kuanli/Documents/lerobot/src/lerobot/teleoperators/wheeled_arm_pico/wheel_arm_teleop.py --mock-xr
```

真实 PICO：

```bash
python /home/kuanli/Documents/lerobot/src/lerobot/teleoperators/wheeled_arm_pico/wheel_arm_teleop.py
```

### 数据采集

```bash
lerobot-record \
  --robot.type=wheeled_arm \
  --teleop.type=wheeled_arm_pico \
  --teleop.visualize=true \
  --display_data=true \
  --display_mode=rerun \
  --dataset.repo_id=kuanli/wheeled_arm_pico_test \
  --dataset.single_task="PICO teleoperate wheeled arm" \
  --dataset.num_episodes=1 \
  --dataset.episode_time_s=30 \
  --dataset.reset_time_s=5 \
  --dataset.push_to_hub=false
```

如果要调整夹爪范围：

```bash
lerobot-record \
  --robot.type=wheeled_arm \
  --teleop.type=wheeled_arm_pico \
  --teleop.visualize=true \
  --teleop.gripper_open_pos=0.0 \
  --teleop.gripper_closed_pos=100.0 \
  --display_data=true \
  --display_mode=rerun \
  --dataset.repo_id=kuanli/wheeled_arm_pico_test \
  --dataset.single_task="PICO teleoperate wheeled arm" \
  --dataset.num_episodes=1 \
  --dataset.episode_time_s=30 \
  --dataset.reset_time_s=5 \
  --dataset.push_to_hub=false
```

`--teleop.visualize=true` 打开 PICO/IK 的 viser 界面；`--display_data=true --display_mode=rerun`
打开数据采集流的 Rerun 界面，用于实时查看 observation、action、相机图像和 recording metadata。

### 本地数据集可视化

省略 `--root`，让 LeRobot 使用默认 cache：

```bash
lerobot-dataset-viz \
  --repo-id kuanli/wheeled_arm_pico_test_20260809_110636 \
  --episode-index 0 \
  --mode local
```

或显式传完整数据集目录：

```bash
lerobot-dataset-viz \
  --repo-id kuanli/wheeled_arm_pico_test_20260809_110636 \
  --root /home/kuanli/.cache/huggingface/lerobot/kuanli/wheeled_arm_pico_test_20260809_110636 \
  --episode-index 0 \
  --mode local
```

不要把 `--root` 只写成 `/home/kuanli/.cache/huggingface/lerobot`，当前版本会把 root 当成数据集根目录本身，导致查找 `/meta/info.json` 失败。

如果 `lerobot-dataset-viz` 输出：

```text
Some episodes in the provided episodes list are out of range for this dataset (0)
Ignoring episode indices outside the dataset range [0, 0): [0]
```

说明该数据集目录存在，但 `meta/info.json` 里的 `total_episodes` 是 0，没有任何可查看 episode。
这通常是采集初始化后中途失败或停止，尚未执行成功的 `dataset.save_episode()`。

PySide6 GUI 已在打开数据集前检查 `total_episodes`，空数据集会在界面内提示；“使用最近采集”会跳过空目录，
自动选择最近一个已保存 episode 的数据集。

## 验证记录

已通过：

```bash
/home/kuanli/miniconda3/envs/xr/bin/python -m py_compile \
  src/lerobot/robots/wheeled_arm/config_wheeled_arm.py \
  src/lerobot/robots/wheeled_arm/wheeled_arm.py \
  src/lerobot/robots/wheeled_arm/hardware_interface/lcm_handler.py \
  src/lerobot/teleoperators/wheeled_arm_pico/config_wheeled_arm_pico.py \
  src/lerobot/teleoperators/wheeled_arm_pico/ik_utils.py \
  src/lerobot/teleoperators/wheeled_arm_pico/wheeled_arm_pico.py \
  src/lerobot/teleoperators/wheeled_arm_pico/wheel_arm_teleop.py
```

本次相机尺寸修复已通过：

```bash
/home/kuanli/miniconda3/envs/xr/bin/python -m py_compile \
  src/lerobot/robots/wheeled_arm/config_wheeled_arm.py \
  tests/robots/test_wheeled_arm.py

git -c filter.lfs.process= -c filter.lfs.clean=cat -c filter.lfs.required=false diff --check -- \
  src/lerobot/robots/wheeled_arm/config_wheeled_arm.py \
  tests/robots/test_wheeled_arm.py \
  src/lerobot/robots/wheeled_arm/WHEELED_ARM_PICO_HANDOFF.md
```

已通过 mock 可视化短跑：

```bash
PATH=/home/kuanli/miniconda3/envs/xr/bin:$PATH PYTHONPATH=src \
python src/lerobot/teleoperators/wheeled_arm_pico/wheel_arm_teleop.py \
  --mock-xr --duration-s=1 --no-browser --visualization-update-hz=0
```

输出中应包含：

```text
grippers=[...]
```

未完成：

- `pytest` 未运行成功，因为当前 `xr` 环境没有安装 `pytest`。

## 后续排查建议

- 如果实物机器人动作异常，优先检查 LCM topic 是否发布：
  - `MANIP_LEFT_ARM_STATE`
  - `MANIP_RIGHT_ARM_STATE`
  - `MANIP_LEFT_GRIPPER_STATE`
  - `MANIP_RIGHT_GRIPPER_STATE`
- 如果 teleop 又回零，检查 `WheeledArm.has_valid_feedback` 和 LCM 状态时间戳。
- 如果夹爪方向反了，交换 `gripper_open_pos` 和 `gripper_closed_pos`，或调整实物 SDK 的夹爪单位。
- 如果相机没有图像，确认 ROS2 topic 默认是 `/camera/color/image_raw`，并检查 encoding 是否为 `rgb8`、`bgr8`、`rgba8`、`bgra8` 或 `mono8`。
- 如果需要深度图，先修复 `cv_bridge` 与 NumPy 的兼容问题。
