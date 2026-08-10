from collections.abc import Callable

import numpy as np
import time
import sys

try:
    from trajectory_plan.seven_segment_speed_plan import seven_segment_speed_plan
except ModuleNotFoundError:
    from .seven_segment_speed_plan import seven_segment_speed_plan

try:
    from dynamics_related_functions.collision_detection import Collision_Detection
except ModuleNotFoundError:
    from ..dynamics_related_functions.collision_detection import Collision_Detection


class MOVEJ():
    def __init__(
        self,
        LCMHandler,
        Collision_Detection,
        stop_requested: Callable[[], bool] | None = None,
    ):

        # lcm
        self.lcm_handler = LCMHandler
        self.Collision_Detection = Collision_Detection
        self.stop_requested = stop_requested
        self.interrupted = False


        # MOVEJ变量
        self.movej_plan_jerk_max = np.pi * 0.75*1
        self.movej_plan_acc_max = np.pi * 0.5*1
        self.movej_plan_speed_max = np.pi *1/ 6
        self.interpolation_period = 2 # 2
        self.joint_position_dim = 23
        self.interpolation_result = np.zeros(self.joint_position_dim)

        self.movej_plan_current_joint_position = None
        self.movej_plan_target_joint_position = None

        self.joint_delta_angle = None
        self.joint_delta_angle_max = None
        self.joint_delta_angle_index = None
        self.joint_movement_direction = None

        self.speed_plan = None
        
        self.MIN_VAL = 0.0000001  

        

    def moveJ2target(self, current_position, target_position):
        current_position = np.array(current_position)
        target_position = np.array(target_position)

        
        self.movej_plan_current_joint_position = current_position
        # print("self.movej_plan_current_joint_position  = {} ".format(self.movej_plan_current_joint_position ))
        self.movej_plan_target_joint_position = target_position
        self.joint_delta_angle = np.zeros(current_position.shape)
        self.joint_movement_direction = np.zeros(current_position.shape)
        for i in range(self.joint_position_dim):
            self.joint_delta_angle[i] = target_position[i] - current_position[i]
            if self.joint_delta_angle[i] > self.MIN_VAL:
                self.joint_movement_direction[i] = 1
            else:
                self.joint_movement_direction[i] = -1
                
            self.joint_delta_angle[i] = np.fabs(self.joint_delta_angle[i])
        
        self.joint_delta_angle_max = np.max(self.joint_delta_angle)
        self.joint_delta_angle_index = np.argmax(self.joint_delta_angle)


        self.speed_plan = seven_segment_speed_plan(self.movej_plan_jerk_max, self.movej_plan_acc_max,
                                                    self.movej_plan_speed_max, self.joint_delta_angle_max)
        return self.movej_speed_plan_interpolation()




    def movej_speed_plan_interpolation(self):
        self.Collision_Detection.start_collision_detection()
        for interpolation_time in np.arange(0, self.speed_plan.time_length, self.interpolation_period / 1000):
            if self._stop_requested():
                print("收到 PICO 急停请求，停止 movej 复位插补！！！！")
                self.interrupted = True
                self._stop_all_moving_flags()
                self.Collision_Detection.stop_collision_detection()
                return False

            start_time = time.time()  # 记录循环开始的时间
            if 0 <= interpolation_time <= self.speed_plan.accacc_time:
                self.speed_plan.cal_accacc_segment_data(interpolation_time)
            elif self.speed_plan.accacc_time < interpolation_time <= self.speed_plan.uniacc_time + self.speed_plan.accacc_time:
                interpolation_time = interpolation_time - self.speed_plan.accacc_time
                self.speed_plan.cal_uniacc_segment_data(interpolation_time)
            elif self.speed_plan.uniacc_time + self.speed_plan.accacc_time < interpolation_time <= self.speed_plan.acceleration_segment_time:
                interpolation_time = interpolation_time - (self.speed_plan.uniacc_time + self.speed_plan.accacc_time)
                self.speed_plan.cal_decacc_segment_data(interpolation_time)
            elif self.speed_plan.acceleration_segment_time < interpolation_time <= self.speed_plan.acceleration_segment_time + self.speed_plan.unispeed_time:
                interpolation_time = interpolation_time - self.speed_plan.acceleration_segment_time
                self.speed_plan.cal_unispeed_segment_data(interpolation_time)
            elif self.speed_plan.acceleration_segment_time + self.speed_plan.unispeed_time < interpolation_time <= self.speed_plan.acceleration_segment_time + self.speed_plan.unispeed_time + self.speed_plan.accdec_time:
                interpolation_time = interpolation_time - (self.speed_plan.acceleration_segment_time + self.speed_plan.unispeed_time)
                self.speed_plan.cal_accdec_segment_data(interpolation_time)
            elif self.speed_plan.acceleration_segment_time + self.speed_plan.unispeed_time + self.speed_plan.accdec_time < interpolation_time <= self.speed_plan.time_length - self.speed_plan.decdec_time:
                interpolation_time = interpolation_time - (self.speed_plan.acceleration_segment_time + self.speed_plan.unispeed_time + self.speed_plan.accdec_time)
                self.speed_plan.cal_unidec_segment_data(interpolation_time)
            else:
                interpolation_time = interpolation_time - (self.speed_plan.time_length - self.speed_plan.decdec_time)
                self.speed_plan.cal_decdec_segment_data(interpolation_time)

            for i in range(self.joint_position_dim):
                self.interpolation_result[i] = self.movej_plan_current_joint_position[i] + self.speed_plan.cur_disp_normalization_ratio * self.joint_movement_direction[i] * self.joint_delta_angle[i]
            

            if(self.Collision_Detection.collision_detection_index):
                print("发生了碰撞，结束碰撞检测线程，退出当前插补函数！！！！")
                self.Collision_Detection.stop_collision_detection()
                sys.exit()    # 退出程序循环，机械臂停止运动

            self.lcm_handler.upper_body_data_publisher(self.interpolation_result)

            if self._stop_requested():
                print("收到 PICO 急停请求，停止 movej 复位插补！！！！")
                self.interrupted = True
                self._stop_all_moving_flags()
                self.Collision_Detection.stop_collision_detection()
                return False

            # 用于保证下发周期是2ms
            elapsed_time = (time.time() - start_time)  # 已经过的时间，单位是秒
            delay = max(0, self.interpolation_period / 1000 - elapsed_time)  # 4毫秒减去已经过的时间
            time.sleep(delay)  # 延迟剩余的时间
        
        print("运行结束，到达目标点位！！！")
        self.Collision_Detection.stop_collision_detection()
        return True

    def _stop_requested(self):
        if self.stop_requested is None:
            return False
        try:
            return bool(self.stop_requested())
        except Exception as exc:
            print(f"读取 PICO 急停状态失败，继续 movej：{exc}")
            return False

    def _stop_all_moving_flags(self):
        for flag_name in (
            "left_arm_moving",
            "right_arm_moving",
            "left_gripper_moving",
            "right_gripper_moving",
            "head_moving",
            "waist_moving",
            "leg_moving",
        ):
            if hasattr(self.lcm_handler, flag_name):
                setattr(self.lcm_handler, flag_name, False)
