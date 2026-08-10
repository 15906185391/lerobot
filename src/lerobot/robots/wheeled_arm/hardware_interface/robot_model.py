from lcm_handler import LCMHandler
from trajectory_plan.moveJ import MOVEJ
from dynamics_related_functions.collision_detection import Collision_Detection


class robot_model:
    def __init__(self):
        self.lcm_handler = LCMHandler()
        self.Collision_Detection = Collision_Detection(self.lcm_handler)

        self.movej_plan_target_position_list = None
        self.trajectory_segment_index = 0
        self.MOVEJ = MOVEJ(self.lcm_handler, self.Collision_Detection)

    # 执行该函数之前需要先对 movej_plan_target_position_list 赋值
    def robot_movej_to_target_position(self):
        if not self.movej_plan_target_position_list:
            raise ValueError("movej_plan_target_position_list is empty; set target positions before running movej.")

        while self.trajectory_segment_index < len(self.movej_plan_target_position_list):
            with self.lcm_handler.data_lock:
                if self.trajectory_segment_index == 0:
                    current_joint_position = self.lcm_handler.joint_current_pos.copy()
                    print(f"current_joint_position = {current_joint_position}")
                else:
                    current_joint_position = self.MOVEJ.interpolation_result

                target_joint_position = self.movej_plan_target_position_list[self.trajectory_segment_index]
                self.MOVEJ.moveJ2target(current_joint_position, target_joint_position)
                self.trajectory_segment_index += 1
