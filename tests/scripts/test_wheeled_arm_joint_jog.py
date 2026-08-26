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

    assert 'ROBOPLAN_JOINT_PLANNER_MODULE = "example_toppra_joint_planning"' in source
    assert 'self.joint_plan_runner = ProcessRunner("关节规划")' in source
    assert 'self.tabs.addTab(self.joint_plan_tab, "关节规划")' in source
    assert 'self.preview_tabs.addTab(self.joint_plan_command_preview, "规划")' in source
    assert '"model": self.joint_plan_model.currentText()' in source
    assert '"preview_only": preview_only' in source
    assert 'ROBOPLAN_EXAMPLE_RUNNER' in source
    assert 'self.joint_plan_runner.start(' in source
    assert 'env_overrides=_roboplan_env_overrides()' in source
    assert 'self.settings.setValue("joint_plan/model", self.joint_plan_model.currentText())' in source
