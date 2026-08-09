# GUI Reference Assets

本目录已经精简为当前 PySide6 GUI 仍会调用的外部转换后端。

保留内容：

- `Any4LeRobotGUI/backend/openx2lerobot`
- `Any4LeRobotGUI/backend/agibot2lerobot`
- `Any4LeRobotGUI/backend/robomind2lerobot`
- `Any4LeRobotGUI/backend/libero2lerobot`
- `Any4LeRobotGUI/backend/lerobot2rlds`
- `Any4LeRobotGUI/backend/ds_version_convert`

已清理内容：

- Any4LeRobotGUI 的 Flutter 前端、平台工程、图片资源和 Git 元数据。
- 仅用于参考设计思路的 leLab 与 unitree_lerobot 完整外部工程。

注意：

- `src/lerobot/scripts/wheeled_arm_gui.py` 的“格式转换”页默认调用
  `GUI_reference/Any4LeRobotGUI/backend` 下的脚本。
- `LeRobot v2.1 → v3.0` 优先调用当前项目维护的
  `src/lerobot/scripts/convert_dataset_v21_to_v30.py`，不再使用参考目录中的旧拷贝。
- `LeRobot v3.0 → v2.1` 保留在
  `Any4LeRobotGUI/backend/ds_version_convert/v30_to_v21`，已按当前 LeRobot API 做兼容。
- `LeRobot v1.6 → v2.0` 依赖旧版 `lerobot.common.*` 模块；当前环境可查看命令参数，
  实际转换请使用匹配旧版 LeRobot 的环境。
- 如果移动该目录，请在 GUI 的 `Backend` 输入框中同步修改路径。
