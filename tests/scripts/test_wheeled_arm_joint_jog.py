#!/usr/bin/env python

from pathlib import Path


def test_wheeled_arm_joint_jog_supports_suction_controls():
    source = Path("src/lerobot/scripts/wheeled_arm_joint_jog.py").read_text()

    assert "SUCTION_POSITION_UPPER = 1.0" in source
    assert "self.target.setSingleStep(1.0 if self.is_suction else (5.0 if self.is_gripper else 1.0))" in source
    assert 'self.target.setSuffix("" if self.is_gripper or self.is_suction else " deg")' in source
    assert 'self.current_label.setText(f"{self.current_value:.2f} kPa")' in source
    assert "self._is_suction_joint(index)" in source
    assert "left_suction_moving" in source
    assert "right_suction_moving" in source
    assert "--left-end-effector" in source
    assert "--right-end-effector" in source


def test_wheeled_arm_gui_plumbs_joint_jog_end_effector_settings():
    source = Path("src/lerobot/scripts/wheeled_arm_gui.py").read_text()

    assert "self.joint_jog_left_end_effector = self._end_effector_combo" in source
    assert 'form.addRow("左臂末端", self.joint_jog_left_end_effector)' in source
    assert 'self._load_end_effector_setting(self.joint_jog_left_end_effector, "joint_jog/left_end_effector")' in source
    assert 'self.settings.setValue("joint_jog/left_end_effector", self.joint_jog_left_end_effector.currentText())' in source
    assert 'command.extend(["--left-end-effector", self.joint_jog_left_end_effector.currentText()])' in source
    assert 'command.extend(["--right-end-effector", self.joint_jog_right_end_effector.currentText()])' in source


def test_wheeled_arm_gui_plumbs_roboplan_joint_planning_page():
    source = Path("src/lerobot/scripts/wheeled_arm_gui.py").read_text()

    assert (
        'TOPPRA_JOINT_PLANNER_MODULE = "lerobot.scripts.wheeled_arm_toppra_joint_planning"'
        in source
    )
    assert 'self.joint_plan_runner = ProcessRunner("关节规划")' in source
    assert 'self.tabs.addTab(self.joint_plan_tab, "关节规划")' in source
    assert 'self.preview_tabs.addTab(self.joint_plan_command_preview, "规划")' in source
    assert '"model": self.joint_plan_model.currentText()' in source
    assert '"preview_only": preview_only' in source
    assert '"lcm_url": self.joint_plan_lcm_url.text().strip()' in source
    assert '"connect_robot": self.joint_plan_connect_robot.isChecked()' in source
    assert '"initial_joint_position": self.joint_plan_initial_joint_position.text().strip()' in source
    assert '"execute_command_hz": self.joint_plan_execute_command_hz.value()' in source
    assert '"execution_duration_s": self.joint_plan_execution_duration.value()' in source
    assert "self.joint_plan_execute_command_hz = self._double_spin(1.0, 500.0, 250.0, 1)" in source
    assert "self.joint_plan_execution_duration = self._double_spin(0.0, 600.0, 0.0, 2)" in source
    assert 'form.addRow("LCM URL", self.joint_plan_lcm_url)' in source
    assert 'form.addRow("执行时长", self.joint_plan_execution_duration)' in source
    assert 'form.addRow("指定初始关节", self.joint_plan_initial_joint_position)' in source
    assert 'self.settings.setValue("joint_plan/connect_robot", self.joint_plan_connect_robot.isChecked())' in source
    assert '_module_command(TOPPRA_JOINT_PLANNER_MODULE)' in source
    assert "main_split = QSplitter(Qt.Orientation.Horizontal)" in source
    assert "main_split.setStretchFactor(1, 3)" in source
    assert "left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)" in source
    assert "left_panel.setMinimumWidth(430)" in source
    assert "self.joint_plan_viser_stack.setMinimumSize(QSize(640, 620))" in source
    assert "JOINT_PLAN_VISER_REFRESH_ATTEMPTS = 12" in source
    assert "def _http_endpoint_ready(" in source
    assert 'connection.request("GET", "/")' in source
    assert "self._schedule_joint_plan_viser_refresh()" in source
    assert 'self.joint_plan_runner.start(' in source
    assert 'env_overrides=_toppra_env_overrides()' in source
    assert 'self.settings.setValue("joint_plan/model", self.joint_plan_model.currentText())' in source


def test_wheeled_arm_toppra_joint_planning_script_registered():
    pyproject = Path("pyproject.toml").read_text()
    source = Path("src/lerobot/scripts/wheeled_arm_toppra_joint_planning.py").read_text()

    assert (
        'lerobot-wheeled-arm-toppra-joint-planning="'
        'lerobot.scripts.wheeled_arm_toppra_joint_planning:main"'
    ) in pyproject
    assert "def available_models() -> tuple[str, ...]:" in source
    assert 'return (DEFAULT_MODEL_NAME,)' in source
    assert (
        'parser.add_argument("--interactive-goal", action=argparse.BooleanOptionalAction, default=True)'
        in source
    )
    assert "DEFAULT_EXECUTE_COMMAND_HZ = 250.0" in source
    assert 'parser.add_argument("--connect-robot", action=argparse.BooleanOptionalAction, default=False)' in source
    assert '"--initial-joint-position"' in source
    assert '"--execution-duration-s"' in source
    assert "execution_duration_s=float(args.execution_duration_s)" in source
    assert "def _planned_execution_duration_s(" in source
    assert 'parser.add_argument("--lcm-url", default=DEFAULT_LCM_URL)' in source
    assert "Robot connected: using current LCM left/right arm state as the TOPPRA start configuration." in source
    assert "Robot connection disabled: using the specified/model start configuration." in source
    assert "def _resample_trajectory_positions(" in source
    assert "def _save_joint_command_plot(" in source
    assert "handler.upper_body_data_publisher(package)" in source
    assert "Preview-only mode: not sending LCM control commands." in source
