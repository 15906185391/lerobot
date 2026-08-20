# Wheeled Arm + PICO Teleoperation Handoff Notes

本文档记录 `wheeled_arm` 机器人和 `wheeled_arm_pico` 遥操作集成过程中已经遇到的问题、根因、处理方案和常用验证命令。后续使用其他智能体继续分析时，可先阅读本文件，避免重复排查。

## 当前目标

- 在 LeRobot 中支持自定义机器人 `wheeled_arm`。
- 使用机器人 SDK/LCM 接口控制和记录左右臂、左右夹爪。
- 相机默认使用 Orbbec（`type: orbbec`），默认前置相机名为 `Orbbec Gemini 335`。
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
- 支持显式 mock 机器人本体：
  - `--robot.mock=true`: 不连接 LCM，不等待左右臂状态，关节 observation 使用软件内部状态并随 action 更新。
  - 默认仍会连接真实相机，适合“已连接相机、未连接实物机器人”的遥操作采集/数据写入测试。
  - `--robot.mock_cameras=true`: 仅完全离线测试时使用，会跳过真实相机并生成合成图像。
  - 实物测试不要开启 `--robot.mock=true`；默认 `require_fresh_feedback=true` 会继续保护机器人，避免从旧状态或零位跳变。

### `wheeled_arm_pico` teleoperator

- IK 仍然只解左右 7 轴机械臂，共 14 维。
- 输出 action 时在 14 维臂关节后追加左右夹爪，共 16 维。
- 默认 PICO 控制方式：
  - `left_grip` / `right_grip`: 激活左右机械臂跟随。
  - `left_trigger` / `right_trigger`: 控制左右夹爪。
  - `Y`: 重置 PICO 相对位姿基准。
- 数据采集控制按钮默认开启：
  - `A`: 开始等待中的 episode，或提前结束当前采集/reset 阶段。
  - `B`: 丢弃当前 episode 并重录。
  - `X`: 停止整个采集流程。
- 如果希望启动命令后先遥操作到准备姿态，再由 PICO 开始写入 episode，可加：
  - `--wait_for_episode_start=true`
- 采集控制按钮可配置：
  - `--teleop.recording_advance_button=A`
  - `--teleop.recording_rerecord_button=B`
  - `--teleop.recording_stop_button=X`
  - `--teleop.recording_control=false`
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

更新：

- 如果未连接实物机器人但已经连接真实相机，推荐使用：

```bash
--robot.mock=true --robot.mock_cameras=false --teleop.mock_xr=false
```

- 如果 PICO 和机器人都未连接、只想完全离线验证采集/写盘/Rerun，可使用：

```bash
--robot.mock=true --robot.mock_cameras=true --teleop.mock_xr=true
```

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
- PICO 输入滤波:
  - `--teleop.pico_position_smoothing_alpha=0.5`
  - `--teleop.pico_orientation_smoothing_alpha=0.5`
  - `--teleop.pico_position_deadband_m=0.0015`
  - `--teleop.pico_orientation_deadband_rad=0.01`
- Posture task:
  - `--teleop.posture_cost=0.0001`
  - `--teleop.posture_gain=1.0`
  - `--teleop.posture_lm_damping=0.0`
- Joint damping task:
  - `--teleop.damping_task_cost=0.05`
- Locked-joints hard constraint:
  - `--teleop.locked_joints_gain=1.0`
  - `--teleop.locked_joints_lm_damping=0.0`
- QP / solve_ik:
  - `--teleop.solver=daqp`
  - `--teleop.ik_damping=1e-12`
  - `--teleop.ik_safety_break=false`
  - `--teleop.enforce_limits=true`
  - `--teleop.solver_kwargs='{\"verbose\": false}'`
- IK 输出后关节平滑/软限幅:
  - `--teleop.arm_action_smoothing_alpha=0.6`
  - `--teleop.max_joint_velocity_rad_s=1.5`
  - `--teleop.max_joint_acceleration_rad_s2=8.0`
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

#### 30Hz 实物首测推荐 IK 参数

当数据采集频率为 30Hz 时，建议 `solve_frequency_hz` 与采集/控制循环保持一致，也设为 30。
不要为了“更稳”随意把 IK 频率设成 10/20Hz，因为当前 teleop 中 `solve_frequency_hz`
会决定 IK 积分步长 `dt`；如果实际外层循环仍以 30Hz 调用，较低的 solve frequency 可能导致单步积分更大。

首次上实物建议先使用偏保守、偏稳的参数，优先验证没有跳变和明显抖动：

```bash
--teleop.scale=0.5 \
--teleop.position_only=true \
--teleop.pico_position_smoothing_alpha=0.5 \
--teleop.pico_orientation_smoothing_alpha=0.5 \
--teleop.pico_position_deadband_m=0.0015 \
--teleop.pico_orientation_deadband_rad=0.01 \
--teleop.task_gain=0.25 \
--teleop.position_cost=3.0 \
--teleop.orientation_cost=0.2 \
--teleop.frame_lm_damping=1e-3 \
--teleop.ik_damping=1e-6 \
--teleop.arm_action_smoothing_alpha=0.6 \
--teleop.max_joint_velocity_rad_s=1.0 \
--teleop.max_joint_acceleration_rad_s2=6.0 \
--teleop.posture_cost=1e-3 \
--teleop.posture_gain=0.5 \
--teleop.use_self_collision=true \
--teleop.d_min=0.04 \
--teleop.self_collision_gain=5.0 \
--teleop.self_collision_safe_displacement_gain=2.0 \
--teleop.enforce_limits=true \
--teleop.solve_frequency_hz=30
```

参数含义和首测理由：

- `position_only=true`：首测先只跟踪末端位置。姿态跟踪更容易把 PICO 手柄姿态抖动放大到腕部/肘部。
- `scale=0.5`：降低 PICO 位移到末端位移的比例，先牺牲灵敏度换稳定性。
- `pico_position_smoothing_alpha=0.5`：PICO 位置输入 EMA 滤波；越小越稳但手感延迟越明显。
- `pico_orientation_smoothing_alpha=0.5`：PICO 姿态输入使用 quaternion slerp 滤波，减少手柄微小姿态抖动。
- `pico_position_deadband_m=0.0015`：忽略 1.5 mm 内的位置微抖。
- `pico_orientation_deadband_rad=0.01`：忽略约 0.57 度内的姿态微抖。
- `task_gain=0.25`：降低 FrameTask 跟踪增益，让目标跟踪更柔和。
- `position_cost=3.0` / `orientation_cost=0.2`：位置为主，弱化姿态约束，减少腕部为了追姿态快速摆动。
- `frame_lm_damping=1e-3`：给 FrameTask 加 LM damping，靠近奇异位形或目标快速变化时更稳。
- `ik_damping=1e-6`：比默认 `1e-12` 更保守，改善数值稳定性。
- `arm_action_smoothing_alpha=0.6`：对 IK 输出后的关节目标做 EMA 平滑；1.0 表示不平滑，越小越稳但越慢。
- `max_joint_velocity_rad_s=1.0`：限制每个控制周期的最大关节目标变化，30Hz 下约等于每帧 0.033 rad。
- `max_joint_acceleration_rad_s2=6.0`：限制每帧目标增量变化，减少速度突变导致的抖动。
- `damping_task_cost=0.03`：给关节速度加全局阻尼，优先压掉高频小抖动。若动作还是“硬”，可升到 `0.05` 到 `0.1`；若明显变钝，再降回 `0.01` 左右。
- `posture_cost=1e-3` / `posture_gain=0.5`：让冗余关节更愿意贴近参考姿态，减少肘部自由漂移。
- `use_self_collision=true` / `enforce_limits=true`：首次实物测试建议保留自碰撞和关节限位。
- `self_collision_gain=5.0` / `self_collision_safe_displacement_gain=2.0`：比默认更温和，避免接近 barrier 时动作突然弹开。

实物首测顺序：

1. 不按 grip，只连接，确认机器人保持当前姿态。
2. 只按单侧 grip，手柄只做 1 到 2 cm 小幅平移。
3. 先测试单臂，再测试双臂。
4. 先保持 `position_only=true`，稳定后再尝试 `position_only=false`。
5. 若稳定，再逐步提高 `scale`、提高 `task_gain`、降低 `frame_lm_damping`。

如果观察到机器人抖动，优先按以下顺序调整：

1. 降低 `--teleop.scale`。
2. 降低 `--teleop.pico_position_smoothing_alpha` / `--teleop.pico_orientation_smoothing_alpha`。
3. 适当提高 `--teleop.pico_position_deadband_m` / `--teleop.pico_orientation_deadband_rad`。
4. 降低 `--teleop.task_gain`。
5. 保持或切回 `--teleop.position_only=true`。
6. 降低 `--teleop.max_joint_velocity_rad_s`，例如 `1.5 -> 1.0 -> 0.6`。
7. 降低 `--teleop.max_joint_acceleration_rad_s2`，例如 `8.0 -> 6.0 -> 4.0`。
8. 降低 `--teleop.arm_action_smoothing_alpha`，例如 `0.6 -> 0.45 -> 0.3`。
9. 提高 `--teleop.frame_lm_damping`，例如 `1e-3 -> 3e-3 -> 1e-2`。
10. 提高 `--teleop.ik_damping`，例如 `1e-6 -> 1e-5`。
11. 降低 `--teleop.orientation_cost`。
12. 略微提高 `--teleop.posture_cost`，例如 `1e-3 -> 3e-3`。

不建议首次实物测试关闭 `enforce_limits` 或 `use_self_collision`。这两个选项更适合短时排查问题，
不适合作为首测默认配置。

#### 实物首测总建议

实物首测建议按“安全、可观测、可回退”三条线准备。第一次上实物不要急着追求“跟手”，
先追求“慢但完全可预测”：只要不跳、不抖、不乱跑，后续手感可以逐步调回来。

安全策略：

- 第一次测试只做空载、小幅、低速动作，工作空间内不要放物体。
- 硬件急停一定先验证有效，再启动 teleop。
- 开始时不要按 grip，只连接并观察机器人是否保持当前位置。
- 每次改 IK 参数只改一两个，不要一次改很多。
- 保留 `enforce_limits=true` 和 `use_self_collision=true`。

推荐启动顺序：

1. 先确认 LCM 左右臂状态持续发布。
2. 打开 GUI，但先不采集。
3. 用“常用命令 > 遥操作”先跑短时间测试。
4. 只测试单臂 grip，小幅平移 1 到 2 cm。
5. 再测试双臂。
6. 最后再进入正式数据采集。

重点观察：

- 不按 grip 时机器人是否完全静止。
- 按 grip 瞬间是否跳变。
- 松开 grip 后是否保持当前位置。
- PICO 小幅移动时，关节是否连续、无突跳。
- Rerun/viser 中机器人状态是否和实物一致。
- LCM feedback 是否持续新鲜，没有间歇超时。

如果出现抖动，优先按以下方向处理：

1. 降低 `scale`。
2. 降低 PICO 输入滤波 alpha，并适当增大 PICO 输入死区。
3. 降低 `task_gain`。
4. 确认或切回 `position_only=true`。
5. 降低 `max_joint_velocity_rad_s` 和 `max_joint_acceleration_rad_s2`。
6. 降低 `arm_action_smoothing_alpha`。
7. 提高 `frame_lm_damping`。
8. 提高 `ik_damping`。
9. 降低 `orientation_cost`。
10. 小幅提高 `posture_cost`。

现场建议额外准备：

- 开一个终端实时看 GUI 日志，或复制 GUI 里的命令单独运行，方便看到完整 traceback。
- 记录每次测试使用的参数组合，不要靠记忆。
- 正式采集前，用关节点动控制台确认能从异常位置安全回到默认姿态。
- 第一轮数据采集建议每集时间短一点，例如 10 到 15 秒；确认保存和复位流程稳定后再加长。

实物测试中已发现并修复的 grip 松开逻辑：

- 现象：实物遥操作时，松开 grip 后手臂有回到初始/参考关节姿态的趋势，mock 下不明显。
- 根因：旧逻辑中 grip 松开只把 TCP target 设为当前末端位姿，但仍然继续运行 FrameTask + PostureTask。
  PostureTask 的参考姿态是启动时的初始姿态，因此在“末端位置尽量不动”的同时，会持续把冗余关节拉回初始关节。
- 修复：未激活的手臂不再参与 IK 求解，最终 action 中该侧关节保持最新 LCM feedback；同时清零该侧关节输出平滑的历史 step。
- 结论：松开 grip 应表示“停止控制并保持当前实物反馈姿态”，不是“以当前 TCP 为目标继续做 IK”。

Action 抖动观察：

- `lerobot-record` 和 `lerobot-teleoperate` 都支持把最终下发给 robot 的 action 发布为 ROS2 `sensor_msgs/msg/JointState`：

```bash
--publish_action_ros2=true \
--action_ros2_topic=/lerobot/action
```

- GUI 的采集页和“常用命令 > 遥操作”中也有“发布 action 到 ROS2”开关。
- 查看消息：

```bash
ros2 topic echo /lerobot/action
```

- 观察某个关节是否抖动，可用 PlotJuggler 或 rqt_plot 订阅 `/lerobot/action`。消息中：
  - `name`: 关节名，例如 `left_arm_0`
  - `position`: 对应 action 目标值，单位通常为 rad 或夹爪配置单位
- 该 topic 发布的是 `robot.send_action()` 返回后的 action，因此包含 robot 侧 `max_relative_target` 等限幅后的结果。

一个实用的首测最小参数组合：

```bash
--teleop.scale=0.5 \
--teleop.position_only=true \
--teleop.pico_position_smoothing_alpha=0.5 \
--teleop.pico_orientation_smoothing_alpha=0.5 \
--teleop.pico_position_deadband_m=0.0015 \
--teleop.pico_orientation_deadband_rad=0.01 \
--teleop.task_gain=0.25 \
--teleop.frame_lm_damping=1e-3 \
--teleop.ik_damping=1e-6 \
--teleop.arm_action_smoothing_alpha=0.6 \
--teleop.max_joint_velocity_rad_s=1.0 \
--teleop.max_joint_acceleration_rad_s2=6.0 \
--teleop.posture_cost=1e-3 \
--teleop.use_self_collision=true \
--teleop.enforce_limits=true
```

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

### 13. Orbbec 相机图像尺寸与 dataset feature 不一致

现象：

```text
ValueError: The feature 'observation.images.front' of shape '(480, 640, 3)'
does not have the expected shape configured in dataset metadata.
```

根因：

- `wheeled_arm` 默认 Orbbec 前置相机配置登记的是 640x480@30。
- 数据集 feature 在 `robot.connect()` 前根据 config 创建，因此 Orbbec camera 连接后读取到的实际尺寸不会反向更新本次 dataset metadata。
- 如果实际 Orbbec profile 与 config 不一致，采集时会出现 shape mismatch。

处理：

- `wheeled_arm_cameras_config()` 默认 `front` 相机改为：
  - `type=orbbec`
  - `serial_number_or_name="Orbbec Gemini 335"`
  - `width=640`
  - `height=480`
  - `fps=30`

注意：

- 如果现场 Orbbec 的序列号/名称或分辨率不同，需要同步修改 robot camera config，或在 CLI 中覆盖 `--robot.cameras=...`。

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

先确认 Orbbec 设备 ID / 名称：

```bash
lerobot-find-cameras orbbec
```

如果只接了一台默认型号相机，下面命令可直接使用；否则把 `serial_number_or_name` 改成上一步输出中的 `Id`。

```bash
lerobot-record \
  --robot.type=wheeled_arm \
  --robot.cameras='{front: {type: orbbec, serial_number_or_name: "Orbbec Gemini 335", width: 640, height: 480, fps: 30}}' \
  --teleop.type=wheeled_arm_pico \
  --teleop.visualize=true \
  --wait_for_episode_start=true \
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
  --robot.cameras='{front: {type: orbbec, serial_number_or_name: "Orbbec Gemini 335", width: 640, height: 480, fps: 30}}' \
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
- 如果相机没有图像，先运行 `lerobot-find-cameras orbbec`，确认 `pyorbbecsdk2` 可导入、设备能被枚举，并把输出中的 `Id` 或唯一 `Name` 填入 `serial_number_or_name`。
- 如果 `pyorbbecsdk` 报 native symbol / shared library 错误，检查 Orbbec SDK native libraries 与 Python wheel 版本是否匹配。
- 当前 Orbbec 接入默认只采集 RGB；如需深度图，需要另行扩展 Orbbec depth stream。

## 2026-08-10 近期变更记录

### 1. `setup_wheeled_arm_pico_conda.bash` 安装脚本

- 默认 Python 版本从 `3.12` 改为 `3.10`，`--help` 文案和示例同步更新。
- 安装逻辑改为尽量幂等：
  - Ubuntu Qt/xcb 系统包仍按缺失项安装。
  - conda-forge 运行依赖会先检查 conda 包或可导入模块，已存在则跳过。
  - pip-only 包会先检查 Python 模块，已存在则跳过。
  - `pip` 不再每次强制 upgrade，只在当前环境没有 pip 时执行 `ensurepip`。
  - LeRobot editable 安装会先确认当前 repo 的 `lerobot`、`datasets`、`rerun`、`PySide6` 是否可用。
  - XRoboToolkit SDK 模块 `xrobotoolkit_sdk` 已可导入时，跳过 clone/install。
  - SDK checkout 已存在时复用本地目录，不再每次自动 `git pull`。
- 已验证：
  - `bash -n scripts/setup_wheeled_arm_pico_conda.bash`
  - `bash scripts/setup_wheeled_arm_pico_conda.bash --help`

### 2. `hardware_interface/robot_model.py` 只保留 movej

- `robot_model.py` 已精简为只服务 movej 关节控制，保留：
  - `LCMHandler`
  - `Collision_Detection`
  - `MOVEJ`
  - `movej_plan_target_position_list`
  - `trajectory_segment_index`
  - `robot_movej_to_target_position()`
- 已移除 moveL、moveC、力控、运动学/动力学模型、CSV 轨迹读取与重采样等无关逻辑。
- 新增轻量 `dynamics_related_functions/collision_detection.py`，提供 MOVEJ 所需接口：
  - `collision_detection_level`
  - `collision_detection_index`
  - `start_collision_detection()`
  - `stop_collision_detection()`
- 当前轻量碰撞检测默认不做实际碰撞检测，和示例脚本中 `collision_detection_level = 0` 的用法一致。
- `trajectory_plan/moveJ.py` 已兼容两种导入方式：
  - 直接在 `hardware_interface` 下运行脚本。
  - 作为 `lerobot.robots.wheeled_arm.hardware_interface` 包内模块导入。
- 已验证：
  - `PYTHONPATH=src/lerobot/robots/wheeled_arm/hardware_interface python -c "import robot_model"`
  - `python -m py_compile robot_model.py movej_to_target_position.py trajectory_plan/moveJ.py`

### 3. PICO 数据采集初始关节状态与 movej 复位

需求：

- 使用 PICO 遥操作采集时，开始遥操作要从 LCM 获取当前机器人关节状态，作为 PICO IK/遥操作初始关节位置。
- episode 间复位使用 movej。
- 复位目标关节角度为：
  - 右臂：`[-20, 70, 75, 100.0, 25, 0, 0]` 度
  - 左臂：`[20, 70, -75, 100.0, -25, 0, 0]` 度
- 程序中需要转为弧度。

处理：

- `WheeledArmConfig` 默认 `require_fresh_feedback=True`，实物安全模式下必须读到新鲜左右臂 LCM 状态才允许继续。
- `WheeledArmConfig.feedback_wait_timeout_s=5.0` 控制安全门等待 LCM feedback 的最大时间。
- `WheeledArm.connect()` 会等待新鲜 `MANIP_LEFT_ARM_STATE` / `MANIP_RIGHT_ARM_STATE`；默认未收到则直接报错中止连接，而不是继续使用零位。
- `lerobot_record.py` 新增 `_sync_teleop_feedback_from_robot(robot, teleop)`：
  - robot 为 `wheeled_arm` 时，默认先调用 `robot.require_valid_feedback("initial teleop feedback sync")`。
  - 通过后读取 `robot.get_observation()`；左右臂关节来自 LCM listener 写入的 `joint_current_pos`。
  - 调用 `teleop.send_feedback(obs)` 将当前 LCM 关节状态同步给 PICO teleop。
  - 同步后调用 `teleop.reset_baseline()`，避免 PICO 相对位姿沿用旧基线导致跳变。
- `record()` 中在 `robot.connect()` 后立即同步一次 PICO 初始关节状态。
- episode 间 reset 阶段前，如果 robot 暴露 `reset_to_rest_pose()`，则先调用该方法进行 movej 复位，再同步一次 PICO 初始状态。
- `WheeledArm.reset_to_rest_pose()` 复用当前 robot 的 `_handler`，不新建第二个 LCM 连接。
- `WheeledArm.reset_to_rest_pose()` 行为：
  - reset 前先要求新鲜左右臂 LCM feedback。
  - 从 `self._handler.joint_current_pos` 读取 23 维当前状态；该缓存只由 LCM 状态 listener 更新，不再由 movej 插补本地写入。
  - 只替换 `[0:7]` 左臂和 `[7:14]` 右臂目标。
  - 左臂目标：`np.deg2rad([20.0, 70.0, -75.0, 100.0, -25.0, 0.0, 0.0])`
  - 右臂目标：`np.deg2rad([-20.0, 70.0, 75.0, 100.0, 25.0, 0.0, 0.0])`
  - movej 复位时只打开左右臂 moving flag，关闭夹爪、头、腰、腿 moving flag。
  - movej 完成后再次等待 reset 开始之后的新鲜 LCM feedback，再允许后续同步给 PICO teleop。
  - movej 插补会通过 `progress_callback` 刷新 viser/Rerun 可视化，但不会写入 `joint_current_pos` 状态缓存。
- 新增测试：
  - `tests/robots/test_wheeled_arm.py::test_reset_to_rest_pose_uses_movej_with_arm_targets_in_radians`
  - 校验左右臂目标角度已转弧度。
  - 校验 23 维 package 中非左右臂部分保持原值。
  - 校验复位时 only left/right arm moving flags 为 True。
- 已通过：
  - `python -m py_compile src/lerobot/robots/wheeled_arm/wheeled_arm.py src/lerobot/scripts/lerobot_record.py src/lerobot/robots/wheeled_arm/hardware_interface/trajectory_plan/moveJ.py tests/robots/test_wheeled_arm.py`

pytest 状态：

- 系统默认 Python 环境中 pytest 被缺少 `draccus` 阻塞。
- `gmr` conda 环境中 `draccus` 存在，但 `pytest` 未安装。

### 4. PySide6 GUI 使用说明、菜单栏、侧边栏和视觉升级

文件：

- `src/lerobot/scripts/wheeled_arm_gui.py`

新增帮助功能：

- 顶部右侧新增 `使用说明` 按钮。
- 菜单栏新增 `帮助 -> 打开使用说明`。
- 快捷键：`F1` 打开使用说明。
- `HelpDialog` 使用 `QDialog + QTabWidget + QTextBrowser` 实现。
- 帮助页签包括：
  - 采集
  - 遥操作流程
  - 数据集
  - 常用命令
  - 关节点动
  - 故障排查
- 帮助窗口支持 `复制说明`，会把纯文本版说明复制到剪贴板。
- 修复帮助内容背景发黑问题：
  - `QTextBrowser#HelpText` 强制白底深色字。
  - `viewport()` 强制白底深色字。
  - HTML `body` 默认样式强制白底深色字。
  - offscreen 验证中 palette base color 为 `#ffffff`。

菜单栏：

- 新增菜单：`文件`、`视图`、`运行`、`帮助`。
- 新增快捷键：
  - `Ctrl+Shift+C`：复制当前命令
  - `Ctrl+L`：清空运行日志
  - `Ctrl+Q`：退出
  - `Ctrl+R`：运行当前页命令
  - `Ctrl+.`：停止当前页任务
  - `F1`：打开使用说明

侧边栏：

- 新增左侧导航栏：`采集`、`查看`、`编辑`、`转换`、`常用`、`点动`。
- 侧边栏按钮与主 `QTabWidget` 双向同步。
- 侧边栏底部保留 `F1 帮助` 快捷按钮。

视觉升级：

- 主窗口最小尺寸从 `1180x760` 调整为 `1280x800`。
- 主背景改为浅色渐变。
- 侧边栏和右侧状态/预览/日志面板使用半透明白色面板，模拟毛玻璃质感。
- 侧边栏和右侧面板增加 `QGraphicsDropShadowEffect` 柔和阴影。
- `QGroupBox`、`QTabWidget`、`QMenuBar`、侧边栏按钮、帮助按钮均增加更现代的圆角/半透明/hover/选中态样式。

### 5. `gmr` conda 环境验证记录

当前用户说明 Python 环境为 conda 的 `gmr` 环境。直接调用：

```bash
/home/kuanli/miniconda3/envs/gmr/bin/python
```

验证结果：

```text
Python 3.10.20
PySide6 6.10.3
draccus: ok
pytest: missing
PySide6: ok
```

说明：

- 在只读沙箱中 `conda run -n gmr ...` 会失败，因为 conda 需要创建临时脚本，但当前沙箱没有可用临时目录。
- 直接调用 `/home/kuanli/miniconda3/envs/gmr/bin/python` 可以正常验证。
- 导入 GUI 时，torch/dill 也需要可用临时目录；offscreen GUI 深度验证在提升权限后通过。

已通过：

```text
compile ok
```

offscreen GUI 验证已通过：

```text
dialog title: Wheeled Arm PICO 使用说明
dialog size: 920 700
help pages: 6
first page object: HelpText
first page stylesheet contains white: True
window title: LeRobot Wheeled Arm 控制台
tabs: ['采集', '查看数据集', '编辑数据集', '格式转换', '常用命令', '关节点动']
sidebar buttons: ['采集', '查看', '编辑', '转换', '常用', '点动']
menus: ['文件', '视图', '运行', '帮助']
help button: 使用说明
help base color: #ffffff
viewport has white: True
document css has white: True
```

还通过：

```bash
git diff --check -- src/lerobot/scripts/wheeled_arm_gui.py
```

### 6. 当前未完成事项

- `gmr` 环境尚未安装 `pytest`，因此未运行完整 pytest。
- 尚未在实物机器人上验证 movej 复位下发。
- 尚未在真实桌面会话中打开 GUI 做截图/人工视觉验收；当前只完成 offscreen 构造和样式检查。

使用说明内容同步：

- “采集”页补充说明：reset 过程中按住 `X` 会中断 movej 并结束本次采集流程。
- “采集/常用命令”页补充 `发布 action 到 ROS2` 用法，默认 topic 为 `/lerobot/action`，用于观察最终下发关节命令。
- “数据集”页补充本地预览流程：加载预览、切换 episode、播放、拖动帧、Shift+拖动选择区间、填入删除当前 Episode。
- “数据集”页明确 `裁剪区间` 目前只弹出提示，不会直接修改 parquet/video；帧级裁剪需要新增安全 dataset operation 后再接入 GUI。
- “常用命令”页补充 `关节点动` 作为异常姿态救援入口。
- “关节点动”页补充打开控制台前的安全确认框：主 GUI 只启动独立窗口，真正连接 LCM 和 viser 必须在新窗口手动点击 `启动连接`。

### 7. 复位过程中的 PICO 急停逻辑

背景：

- 普通 PICO 遥操作时，左右 grip/trigger 需要按下才会驱动机器人，相对安全。
- 但 episode 间复位是 `record()` 阻塞调用 `WheeledArm.reset_to_rest_pose()`，如果 movej 复位期间不继续轮询 PICO，操作者无法用 PICO 及时停止异常动作。

处理：

- `WheeledArmPicoConfig` 新增：
  - `emergency_stop_button: str = "X"`
- `WheeledArmPico` 新增：
  - `emergency_stop_requested()`
  - 该方法是 level-triggered：只要配置的 PICO 按钮处于按下状态就返回 `True`。
  - 默认按钮为 `X`，与整次采集停止按钮一致。
- `lerobot_record.py` 新增：
  - `_teleop_emergency_stop_requested(teleop)`
  - reset 阶段调用 `reset_to_rest_pose(stop_requested=lambda: _teleop_emergency_stop_requested(teleop))`。
- `MOVEJ` 新增可选 `stop_requested` 回调：
  - movej 插补循环开始和每次发布后都会检查急停。
  - 检测到急停后停止碰撞检测线程，关闭所有 moving flags，并返回 `False`。
- `WheeledArm.reset_to_rest_pose()` 现在返回 `bool`：
  - `True`：复位完成。
  - `False`：复位被 PICO 急停中断。
- `record()` 在复位返回 `False` 时：
  - 设置 `events["stop_recording"] = True`。
  - 设置 `events["exit_early"] = True`。
  - 输出 `Emergency stop requested during robot reset`。
  - 跳出后续 reset loop，结束本次采集流程。
- GUI 使用说明同步强调：普通录制阶段 `X` 是停止整次采集；reset 阶段按住 `X` 会中断 movej 复位轨迹，并让 `record()` 结束当前采集流程。

注意：

- 这是软件层急停：停止继续发布 movej 复位轨迹，并关闭 LCM moving flags。
- 它不能替代硬件急停按钮、断电保护或底层控制器安全机制。
- 实物调试时仍应保留硬件急停，并先在低速/空载/安全空间内验证。

验证：

- gmr 环境下直接运行新增测试函数已通过：
  - `test_reset_to_rest_pose_uses_movej_with_arm_targets_in_radians`
  - `test_reset_to_rest_pose_stops_when_movej_is_interrupted`
  - `test_pico_emergency_stop_button_is_level_triggered`
- `git diff --check` 通过。

### 8. 停止采集后的 Rerun 清理与 reset 可视化同步

问题：

- GUI 点击停止采集后，如果 Rerun 窗口仍占用默认端口，再次开始采集可能失败，用户需要手动关闭 Rerun。
- episode 结束后执行 movej reset 时，机器人实际会复位，但 Rerun/viser 里的机器人模型可能停留在 reset 前一帧。

原因：

- Rerun SDK `rr.spawn()` 默认 `detach_process=True`，viewer 会脱离 `lerobot-record` 进程组；GUI 停止采集只中断 record 进程，旧 Rerun viewer 可能继续占用端口。
- movej reset 在 `record_loop()` 外部阻塞执行，原来只下发 LCM 轨迹，不调用 teleop 的可视化刷新，也不向 Rerun 记录机器人 3D 状态。

处理：

- `src/lerobot/utils/rerun_visualization.py`
  - 本地 `rr.spawn()` 现在默认 `detach_process=False`，使 GUI 停止采集时 Rerun viewer 跟随采集进程组清理。
  - `--display_port` 对本地 Rerun spawn 也生效。
  - 记录本次本地 Rerun viewer PID，`shutdown_rerun()` 时会先断开 SDK，再主动 SIGTERM/SIGKILL 关闭该 viewer。
  - 可通过环境变量 `LEROBOT_RERUN_DETACH_PROCESS=true` 恢复旧的 detached 行为。
  - `shutdown_rerun()` 会先 `rr.disconnect()`，再 `rr.rerun_shutdown()`。
- `src/lerobot/scripts/wheeled_arm_gui.py`
  - GUI 本地 Rerun 不手动填端口时，会自动挑一个空闲高位端口传入 `--display_port`，避免旧窗口占用 9876 阻塞新采集。
- `src/lerobot/robots/wheeled_arm/hardware_interface/trajectory_plan/moveJ.py`
  - movej 插补发布后会更新本地 `joint_current_pos` 缓存。
  - 新增 `progress_callback`，用于 reset 过程中同步可视化。
- `src/lerobot/robots/wheeled_arm/wheeled_arm.py`
  - `reset_to_rest_pose()` 新增 `progress_callback` 参数。
  - 新增 `joint_observation_from_package()`，可只用关节 package 构造 16 维关节/夹爪 observation，不读取相机。
- `src/lerobot/teleoperators/wheeled_arm_pico/wheeled_arm_pico.py`
  - 新增 `refresh_visualization_from_feedback()`，可在 IK 主循环外强制刷新 viser 和 Rerun robot 状态。
  - `log_rerun_robot_visualization(..., force=True)` 可绕过刷新频率限制，立即写一帧。
- `src/lerobot/scripts/lerobot_record.py`
  - 初始同步、movej reset 过程中、movej reset 完成后，都会将机器人反馈同步到 teleop 可视化。
  - reset 过程中约 10Hz 刷新 Rerun/viser，reset 完成后强制写最后一帧复位姿态。

验证：

- `python -m py_compile` 已通过相关文件。
- `git diff --check` 已通过相关文件。
- `xr` 环境没有安装 `pytest`，因此未跑 pytest runner；已直接调用关键测试函数验证 reset/PICO 控制逻辑。

补充：

- 2026-08-10 已在 conda `xr` 环境安装：
  - `pytest==8.4.2`
  - `lark==1.3.1`
  - `packaging==25.0`
- 选择 pytest 8.x 是为了兼容 ROS Humble 的 `launch_testing` pytest 插件；pytest 9 会触发旧 hook 签名不兼容。
- 在当前沙箱里直接运行 pytest 会被 `DISPLAY=:0` / pynput 权限限制影响；使用
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` 已通过：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/kuanli/miniconda3/envs/xr/bin/python -m pytest \
  tests/robots/test_wheeled_arm.py \
  tests/teleoperators/test_wheeled_arm_pico.py -q
```

结果：

```text
17 passed
```

### 9. 最新版本再分析与排查

最新代码里，`wheeled_arm_pico` 已经补了几层稳定化处理：

- `XrPoseFilter` 对 PICO 手柄位姿做了 EMA + deadband 过滤。
- `smooth_joint_positions()` 对 IK 输出做了 EMA + 速度/加速度软限幅。
- `lerobot-record` 会在开始录制和 reset 后把机器人反馈同步给 teleop。
- `wheeled_arm` 默认要求新鲜左右臂反馈，避免旧状态直接进入闭环。

所以如果现在真机仍然抖，优先级要和旧版本不一样，不要再把“没有平滑”当第一嫌疑。

更可能的原因按优先级排序：

1. 机器人反馈本身在抖，或者左右臂反馈不同步。
2. `activation_threshold` 附近来回波动，导致 active / inactive 频繁切换。
3. 自碰撞 barrier 在边界附近反复触发，目标被拒绝后又重新求解。
4. IK 输出虽然平滑了，但最终还是直接发位置目标，没有真正的底层 rate limiter。
5. 实机控制器增益、回读延迟、机械背隙或共振在放大小幅命令变化。

建议按这个顺序排查：

1. 先看 `has_valid_feedback` 是否稳定为真，左右臂状态时间戳是否持续刷新。
2. 再看日志里是否频繁出现：
   - `Waiting for valid PICO XR data`
   - `PICO IK solver failed`
   - `PICO IK target rejected by self-collision barrier`
3. 把 `activation_threshold` 临时调低一点，观察是否还会“抖一下就重置”。
4. 临时关掉 `use_self_collision`，排除碰撞边界抖动。
5. 临时设 `position_only=true`，排除姿态跟踪带来的高频修正。
6. 如果 `mock_xr` 平稳、真机抖，重点放在 LCM 反馈质量和底层控制器，不要继续盯 PICO 的上层平滑。

### 10. 2026-08-12 实机抖动治理：控制节拍、反馈、active arm 与插值

本轮问题现象：

- PICO/viser/Rerun 里看到的机器人关节数据比较平滑。
- 但实物机器人手臂运行时明显震动、卡顿。
- 这说明上层 IK 输出已经不是唯一嫌疑，问题更可能出现在“低频目标如何下发给实机控制器”和“实机反馈如何重新进入 IK 状态”之间。

参考 `src/lerobot/teleoperators/unitree_g1` 的思路后，当前判断是：

- Unitree G1 路径更强调机器人侧稳定控制节拍，例如 `control_dt=1/250`。
- wheeled arm 之前主要由 30Hz 的 record/teleoperate loop 直接触发 LCM 下发。
- 如果 30Hz IK 目标直接变成实机位置命令，底层会看到阶梯状目标；即使可视化曲线看起来平滑，实机控制器仍可能因为目标保持、反馈延迟或零速度命令语义出现抖动。

本轮已落实的代码改动：

1. 机器人侧增加 250Hz 控制节拍。
   - 文件：`src/lerobot/robots/wheeled_arm/config_wheeled_arm.py`
   - 文件：`src/lerobot/robots/wheeled_arm/wheeled_arm.py`
   - 新增默认配置：
     - `use_control_loop: bool = True`
     - `control_dt: float = 1.0 / 250.0`
   - `send_action()` 不再直接依赖外层 30Hz loop 的调用时刻发布所有命令，而是保存最新目标，由机器人侧后台 control loop 按 `control_dt` 稳定发布。
   - `reset_to_rest_pose()` 会暂停 control loop，避免 movej 复位轨迹和后台控制线程抢发命令。

2. 30Hz IK/记录频率与 250Hz 发布频率之间增加关节数据插值。
   - 文件：`src/lerobot/robots/wheeled_arm/config_wheeled_arm.py`
   - 文件：`src/lerobot/robots/wheeled_arm/wheeled_arm.py`
   - 新增默认配置：
     - `interpolate_control_loop_actions: bool = True`
     - `action_interpolation_duration_s: float = 1.0 / 30.0`
   - control loop 收到新的低频 action target 后，会在约一个 30Hz 周期内从上一条发布目标插值到新目标。
   - 目标是把 30Hz 阶梯命令变成 250Hz 小步进命令，降低实机位置控制器看到的瞬时跳变。
   - 插值只作用在当前 moving/active 的 action keys 上，不会强行驱动未激活手臂。

3. 未激活手臂不再发布 arm action 命令。
   - 文件：`src/lerobot/robots/wheeled_arm/config_wheeled_arm.py`
   - 文件：`src/lerobot/robots/wheeled_arm/wheeled_arm.py`
   - 文件：`src/lerobot/teleoperators/wheeled_arm_pico/wheeled_arm_pico.py`
   - 新增内部 action metadata key：
     - `WHEELED_ARM_ACTIVE_ARMS_ACTION_KEY = "__wheeled_arm_active_arms"`
   - `WheeledArmPico._make_action()` 仍会生成完整 action，便于数据集和可视化保持完整关节字段。
   - 但 robot 侧会读取该 metadata，只对真正按下 grip/trigger 的 active arm 发布 moving flags 和 LCM arm commands。
   - 当左右手都未激活，metadata 为空，robot 侧不会因为“保持姿态”而持续给未操作手臂下发 arm 命令。

4. 默认不再把每帧实机 LCM 反馈送回 PICO IK。
   - 文件：`src/lerobot/teleoperators/wheeled_arm_pico/config_wheeled_arm_pico.py`
   - 文件：`src/lerobot/scripts/lerobot_record.py`
   - 文件：`src/lerobot/scripts/lerobot_teleoperate.py`
   - 新增默认配置：
     - `use_continuous_robot_feedback: bool = False`
   - record/teleoperate 的主循环中，wheeled arm PICO 默认不再每帧执行 `teleop.send_feedback(obs)`。
   - 初始 connect 后同步、episode reset 后同步仍然保留，让 IK 初始状态和复位状态对齐。
   - 这样可以避免实机 LCM 延迟反馈在每帧覆盖 IK 内部平滑状态，引入“旧反馈把新目标拉回去”的抖动。

5. 命令行显示和 mock 路径适配内部 metadata。
   - 文件：`src/lerobot/scripts/lerobot_teleoperate.py`
   - 文件：`src/lerobot/robots/wheeled_arm/wheeled_arm.py`
   - terminal display 会跳过内部 metadata，避免 tuple 类型被当 float 打印。
   - mock publish path 会剥离 `__wheeled_arm_active_arms`，保持 mock action 和真实硬件 action 语义一致。

建议实机基线命令：

```bash
lerobot-teleoperate \
  --robot.type=wheeled_arm \
  --teleop.type=wheeled_arm_pico \
  --fps=30 \
  --display_data=false \
  --teleop.visualize=false \
  --teleop.position_only=true \
  --teleop.scale=0.2 \
  --teleop.max_joint_velocity_rad_s=0.2 \
  --teleop.max_joint_acceleration_rad_s2=1.0
```

这条命令会使用当前默认值：

- `--robot.use_control_loop=true`
- `--robot.control_dt=0.004`
- `--robot.interpolate_control_loop_actions=true`
- `--robot.action_interpolation_duration_s=0.0333333333`
- `--teleop.use_continuous_robot_feedback=false`

建议逐项对比开关：

1. 对比旧式直接发布路径：

```bash
--robot.use_control_loop=false
```

如果关闭后明显更抖，说明机器人侧 250Hz 节拍有效。

2. 只关闭插值，保留 250Hz loop：

```bash
--robot.interpolate_control_loop_actions=false
```

如果关闭插值后出现更明显的阶梯感或抖动，说明 30Hz 到 250Hz 的插值有效。

3. 恢复每帧实机反馈进入 PICO IK：

```bash
--teleop.use_continuous_robot_feedback=true
```

如果恢复后抖动加重，说明 LCM 反馈延迟确实在污染 IK 状态。

4. 如果仍有轻微卡顿，可以把插值周期略微加长：

```bash
--robot.action_interpolation_duration_s=0.04
```

或：

```bash
--robot.action_interpolation_duration_s=0.05
```

注意不要盲目加太大，否则遥操作会变得明显拖手。

实机验证建议：

1. 先只激活左手，监控右臂 LCM command channel 是否没有持续新命令。
2. 再只激活右手，监控左臂 LCM command channel 是否没有持续新命令。
3. 保持 PICO 不动但 grip/trigger 按下，观察实机是否仍高频微抖。
4. 缓慢移动单手，比较开启/关闭插值时实机动作是否从“阶梯跳动”变成“小步连续跟随”。
5. 如果可视化仍平滑但实机继续抖，下一步重点查 LCM command package 里 `vel=0.0` 的底层语义。

仍未完全确认的问题：

- 当前 arm command 中的 velocity 字段仍需要实机侧确认。
- 如果底层把 `vel=0.0` 理解为“目标速度必须为 0”，而不是“速度前馈为空/忽略”，那么持续发送变化的位置目标加零速度目标可能会让控制器在每个点上急停式跟踪。
- 若上述四项改动后实机仍抖，应优先尝试记录/打印实际发布的 position、velocity、moving flag，并确认底层控制器期望的速度字段、增益和模式。

本轮测试结果：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/kuanli/miniconda3/envs/xr/bin/python -m pytest \
  tests/scripts/test_wheeled_arm_feedback_loop.py \
  tests/teleoperators/test_wheeled_arm_pico.py \
  tests/robots/test_wheeled_arm.py -q
```

结果：

```text
34 passed in 0.44s
```

此外，相关 Python 文件已通过 `py_compile`，相关 diff 已通过 `git diff --check`。

### 11. 2026-08-12 采集链路进一步加固：真实 sent action、锁外发布、watchdog 与自适应插值

背景：

- 上一轮已经把 PICO IK/record 的 30Hz action 通过 robot 侧 control loop 转成 250Hz LCM 发布。
- 继续排查后发现，采集数据可信度和实机安全性还需要补三类保护：
  - 数据集应记录 robot 侧真正接受/限幅后的 action，而不是只记录 teleop 原始 action。
  - 250Hz control loop 不应在内部锁中执行 LCM publish，避免 publish 慢时阻塞下一帧 action 更新。
  - 外层 record/teleop loop 如果被相机、Rerun 或 CPU 抖动拖慢，robot 侧不能一直保持旧 action 发布。
- 另外，固定 `1/30s` 插值周期不适合真实 loop 抖动，需要按实际 action 到达间隔自适应。

本轮已落实的代码改动：

1. wheeled_arm 数据集 action 记录 robot 侧 sent action。
   - 文件：`src/lerobot/scripts/lerobot_record.py`
   - 新增 `_action_values_for_dataset(robot, requested_action, sent_action)`。
   - 对普通机器人保持旧行为，仍记录 processor 后的 requested action。
   - 对 `robot.name == "wheeled_arm"`：
     - 先用 requested action 中属于 `robot.action_features` 的字段构造完整 action。
     - 再用 `robot.send_action()` 返回的 sent action 覆盖对应字段。
     - 这样 `max_relative_target` 等 robot 侧限幅后的真实发送值会进入 dataset。
     - 内部 metadata，例如 `__wheeled_arm_active_arms`，不会进入 dataset。

2. control loop 缩短锁作用域，LCM publish 移到锁外。
   - 文件：`src/lerobot/robots/wheeled_arm/wheeled_arm.py`
   - 新增 `_control_loop_publish_snapshot(now)`。
   - control loop 现在在锁内只读取/更新 target、插值状态和 watchdog 状态。
   - `_set_moving_flags(...)` 和 `upper_body_data_publisher(package)` 在锁外执行。
   - 这样即使 LCM encode/publish 偶发变慢，外层 `send_action()` 仍能及时更新最新目标。

3. 增加 action watchdog。
   - 文件：`src/lerobot/robots/wheeled_arm/config_wheeled_arm.py`
   - 文件：`src/lerobot/robots/wheeled_arm/wheeled_arm.py`
   - 新增默认配置：
     - `action_watchdog_timeout_s: float | None = 0.1`
   - 如果 robot 侧超过该时间没有收到新的 action，control loop 会：
     - 清空 control target。
     - 停止所有 moving flags。
     - 打印一次 watchdog timeout warning。
   - 可通过下面参数关闭 watchdog：

```bash
--robot.action_watchdog_timeout_s=null
```

4. PICO 松开后 active action 为空时，立即停止 moving flags。
   - 文件：`src/lerobot/robots/wheeled_arm/wheeled_arm.py`
   - control loop 模式下，如果本帧 action keys 为空，会立即 `_set_moving_flags(set())`。
   - 这避免上一帧 grip/trigger 激活状态残留，导致松手后继续保持旧 arm moving flag。

5. 插值周期改为按实际 action 到达间隔自适应。
   - 文件：`src/lerobot/robots/wheeled_arm/config_wheeled_arm.py`
   - 文件：`src/lerobot/robots/wheeled_arm/wheeled_arm.py`
   - 新增默认配置：
     - `adaptive_action_interpolation_duration: bool = True`
     - `action_interpolation_min_duration_s: float = 0.02`
     - `action_interpolation_max_duration_s: float = 0.06`
   - `action_interpolation_duration_s` 仍保留，作为首帧/回退值；关闭自适应时也使用它作为固定插值周期。
   - 第二条 action 开始，robot 会用相邻 action 的实际到达间隔作为插值周期，并 clamp 到 `[0.02, 0.06]`。
   - 这样当外层 record/teleop loop 从标称 30Hz 掉到 20Hz 或出现轻微抖动时，250Hz 发布侧不会继续按固定 33ms 追目标。

建议新增对比开关：

```bash
--robot.adaptive_action_interpolation_duration=false
```

用于恢复固定 `--robot.action_interpolation_duration_s`，方便和自适应插值做 A/B 对比。

如果实机仍有轻微跟随卡顿，可尝试：

```bash
--robot.action_interpolation_max_duration_s=0.08
```

但不建议一开始把 max duration 调太大，否则会明显增加遥操作延迟。

本轮测试结果：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/kuanli/miniconda3/envs/xr/bin/python -m pytest \
  tests/scripts/test_wheeled_arm_feedback_loop.py \
  tests/teleoperators/test_wheeled_arm_pico.py \
  tests/robots/test_wheeled_arm.py -q
```

结果：

```text
40 passed in 0.54s
```

此外，相关 Python 文件已通过 `py_compile`，相关 diff 已通过 `git diff --check`。
