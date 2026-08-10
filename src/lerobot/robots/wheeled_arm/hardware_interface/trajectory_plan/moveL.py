from lcm_handler import LCMHandler
import numpy as np
import time
from trajectory_plan.seven_segment_speed_plan import seven_segment_speed_plan
from copy import deepcopy
import quaternion  # 这将 numpy 的 ndarray 类型扩展以支持四元数
import pinocchio as pin
from robot_kinematics_and_dynamics_models.Kinematic_Model import Kinematic_Model
from dynamics_related_functions.collision_detection import Collision_Detection
import csv
import sys



class MOVEL():
    def __init__(self, LCMHandler, Collision_Detection, Kinematic_Model, Force_Control):
        # lcm
        self.lcm_handler = LCMHandler
        self.Collision_Detection = Collision_Detection
        self.Force_Control = Force_Control

        # MOVEL的变量
        self.movel_plan_jerk_max = 0.75
        self.movel_plan_acc_max = 0.5
        self.movel_plan_speed_max = 0.2
        
        self.movel_plan_current_cart_position = None
        self.movel_plan_current_cart_pose = None
        self.movel_plan_current_cart_quat = None
        self.movel_plan_target_cart_position = None
        self.movel_plan_target_cart_pose = None
        self.movel_plan_target_cart_quat = None
        self.movel_plan_position_delta_disp = np.zeros(3)
        self.movel_plan_displacement = 0       
        self.movel_plan_target_position_list = None
        self.movel_plan_current_joint_position = None
        self.movel_plan_target_joint_position = None

        # 右臂movel运动规划路径数据以及规划时的相关参数设置
        self.right_arm_movel_plan_current_cart_position = None
        self.right_arm_movel_plan_current_cart_pose = None
        self.right_arm_movel_plan_current_cart_quat = None
        self.right_arm_movel_plan_target_cart_position = None
        self.right_arm_movel_plan_target_cart_pose = None
        self.right_arm_movel_plan_target_cart_quat = None
        self.right_arm_movel_plan_position_delta_disp = np.zeros(3)
        self.right_arm_movel_plan_displacement = 0       
        self.right_arm_movel_plan_target_position_list = None
        self.right_arm_movel_plan_current_joint_position = None
        self.right_arm_movel_plan_target_joint_position = None
        self.cart_interpolation_position = np.zeros(3)
        self.cart_interpolation_pose = np.zeros((3, 3))
        self.right_arm_cart_interpolation_position = np.zeros(3)
        self.right_arm_cart_interpolation_pose = np.zeros((3, 3))

        self.whether_save_movel_position = 0
        self.left_arm_inverse_kinematics_solution_success_flag = None
        self.right_arm_inverse_kinematics_solution_success_flag = None

        self.speed_plan = None
        self.MIN_VAL = 0.0000001 
        self.interpolation_period = 2
        self.interpolation_result = None

        self.left_arm_interp = pin.SE3.Identity()
        self.right_arm_interp = pin.SE3.Identity()

        self.right_arm_movel_plan_current_cart = pin.SE3.Identity()
        self.right_arm_movel_plan_target_cart = pin.SE3.Identity()
        self.movel_plan_current_cart = pin.SE3.Identity()
        self.movel_plan_target_cart = pin.SE3.Identity()

        self.Kinematic_Model = Kinematic_Model

        # self.arm_FT = None
        # self.target_force_z_FT = None
        self.constant_force_track = False

    def cal_movel_plan_data(self, left_arm_current_position, left_arm_target_position, right_arm_current_position, right_arm_target_position):
            # 左臂笛卡尔空间下当前点与期望点之间路径参数的计算
            self.movel_plan_current_cart_position = deepcopy(left_arm_current_position.translation)
            self.movel_plan_current_cart_pose = deepcopy(left_arm_current_position.rotation)
            self.movel_plan_current_cart_quat = quaternion.from_rotation_matrix(self.movel_plan_current_cart_pose)
            if self.whether_save_movel_position:
                print("self.movel_plan_current_cart_position  = {} ".format(self.movel_plan_current_cart_position))
                print("self.movel_plan_current_cart_pose  = {} ".format(self.movel_plan_current_cart_pose))

            self.movel_plan_target_cart_position = deepcopy(left_arm_target_position.translation)
            self.movel_plan_target_cart_pose = deepcopy(left_arm_target_position.rotation)
            self.movel_plan_target_cart_quat = quaternion.from_rotation_matrix(self.movel_plan_target_cart_pose)   
            if self.whether_save_movel_position:
                print("self.movel_plan_target_cart_position  = {} ".format(self.movel_plan_target_cart_position))
                print("self.movel_plan_target_cart_pose  = {} ".format(self.movel_plan_target_cart_pose)) 

            self.movel_plan_displacement = np.linalg.norm(self.movel_plan_target_cart_position - self.movel_plan_current_cart_position)
            self.movel_plan_position_delta_disp[0] = self.movel_plan_target_cart_position[0] - self.movel_plan_current_cart_position[0] 
            self.movel_plan_position_delta_disp[1] = self.movel_plan_target_cart_position[1] - self.movel_plan_current_cart_position[1]   
            self.movel_plan_position_delta_disp[2] = self.movel_plan_target_cart_position[2] - self.movel_plan_current_cart_position[2]   

            if np.dot(self.movel_plan_current_cart_quat.components, self.movel_plan_target_cart_quat.components) < 0:
                self.movel_plan_target_cart_quat = - self.movel_plan_target_cart_quat   


            # 右臂笛卡尔空间下当前点与期望点之间路径参数的计算
            self.right_arm_movel_plan_current_cart_position = deepcopy(right_arm_current_position.translation)
            self.right_arm_movel_plan_current_cart_pose = deepcopy(right_arm_current_position.rotation)
            self.right_arm_movel_plan_current_cart_quat = quaternion.from_rotation_matrix(self.right_arm_movel_plan_current_cart_pose)
            if self.whether_save_movel_position:
                print("self.right_arm_movel_plan_current_cart_position  = {} ".format(self.right_arm_movel_plan_current_cart_position))
                print("self.right_arm_movel_plan_current_cart_pose  = {} ".format(self.right_arm_movel_plan_current_cart_pose))

            self.right_arm_movel_plan_target_cart_position = deepcopy(right_arm_target_position.translation)
            self.right_arm_movel_plan_target_cart_pose = deepcopy(right_arm_target_position.rotation)
            self.right_arm_movel_plan_target_cart_quat = quaternion.from_rotation_matrix(self.right_arm_movel_plan_target_cart_pose)   
            if self.whether_save_movel_position: 
                print("self.right_arm_movel_plan_target_cart_position  = {} ".format(self.right_arm_movel_plan_target_cart_position))
                print("self.right_arm_movel_plan_target_cart_pose  = {} ".format(self.right_arm_movel_plan_target_cart_pose))

            self.right_arm_movel_plan_displacement = np.linalg.norm(self.right_arm_movel_plan_target_cart_position - self.right_arm_movel_plan_current_cart_position)
            self.right_arm_movel_plan_position_delta_disp[0] = self.right_arm_movel_plan_target_cart_position[0] - self.right_arm_movel_plan_current_cart_position[0] 
            self.right_arm_movel_plan_position_delta_disp[1] = self.right_arm_movel_plan_target_cart_position[1] - self.right_arm_movel_plan_current_cart_position[1]   
            self.right_arm_movel_plan_position_delta_disp[2] = self.right_arm_movel_plan_target_cart_position[2] - self.right_arm_movel_plan_current_cart_position[2]   

            if np.dot(self.right_arm_movel_plan_current_cart_quat.components, self.right_arm_movel_plan_target_cart_quat.components) < 0:
                self.right_arm_movel_plan_target_cart_quat = - self.right_arm_movel_plan_target_cart_quat        


    def movel_speed_plan_interpolation(self):
        self.Collision_Detection.start_collision_detection()

        for interpolation_time in np.arange(0, self.speed_plan.time_length, self.interpolation_period / 1000):
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

            # 计算左臂插补位置与姿态
            self.cart_interpolation_position[0] = self.movel_plan_current_cart_position[0] + self.movel_plan_position_delta_disp[0] * self.speed_plan.cur_disp_normalization_ratio
            self.cart_interpolation_position[1] = self.movel_plan_current_cart_position[1] + self.movel_plan_position_delta_disp[1] * self.speed_plan.cur_disp_normalization_ratio
            self.cart_interpolation_position[2] = self.movel_plan_current_cart_position[2] + self.movel_plan_position_delta_disp[2] * self.speed_plan.cur_disp_normalization_ratio

            slerped_quaternions = quaternion.slerp(self.movel_plan_current_cart_quat, self.movel_plan_target_cart_quat, 0, 1, self.speed_plan.cur_disp_normalization_ratio)
            self.cart_interpolation_pose = quaternion.as_rotation_matrix(slerped_quaternions)

            if self.constant_force_track :
                self.Force_Control.dxyz_cal()

            self.left_arm_interp.translation = self.cart_interpolation_position + self.Force_Control.dxyz_l
            self.left_arm_interp.rotation = self.cart_interpolation_pose

            # # 左臂逆解 逆解 逆解 
            self.Kinematic_Model.left_arm_inverse_kinematics(self.left_arm_interp.rotation, self.left_arm_interp.translation, self.movel_plan_current_joint_position[:7])


            # 计算右臂插补位置与姿态
            self.right_arm_cart_interpolation_position[0] = self.right_arm_movel_plan_current_cart_position[0] + self.right_arm_movel_plan_position_delta_disp[0] * self.speed_plan.cur_disp_normalization_ratio
            self.right_arm_cart_interpolation_position[1] = self.right_arm_movel_plan_current_cart_position[1] + self.right_arm_movel_plan_position_delta_disp[1] * self.speed_plan.cur_disp_normalization_ratio
            self.right_arm_cart_interpolation_position[2] = self.right_arm_movel_plan_current_cart_position[2] + self.right_arm_movel_plan_position_delta_disp[2] * self.speed_plan.cur_disp_normalization_ratio

            slerped_quaternions = quaternion.slerp(self.right_arm_movel_plan_current_cart_quat, self.right_arm_movel_plan_target_cart_quat, 0, 1, self.speed_plan.cur_disp_normalization_ratio)
            self.right_arm_cart_interpolation_pose = quaternion.as_rotation_matrix(slerped_quaternions)

            self.right_arm_interp.translation = self.right_arm_cart_interpolation_position + self.Force_Control.dxyz_r
            self.right_arm_interp.rotation = self.right_arm_cart_interpolation_pose

            # # 右臂逆解 逆解 逆解
            self.Kinematic_Model.right_arm_inverse_kinematics(self.right_arm_interp.rotation, self.right_arm_interp.translation, self.movel_plan_current_joint_position[7:14])

            # 双臂的逆解结果校验一下 
            if (self.Kinematic_Model.left_arm_inverse_kinematics_solution_success_flag and self.Kinematic_Model.right_arm_inverse_kinematics_solution_success_flag) == False:
                print("逆解失败咯，机器人应该停止运行了，请调整合理的位置呀！！！！")
                self.interpolation_result = self.movel_plan_current_joint_position
                break
            else:
                self.interpolation_result = self.movel_plan_current_joint_position
                self.interpolation_result[7:14] = self.Kinematic_Model.right_arm_interpolation_result
                self.interpolation_result[:7] = self.Kinematic_Model.left_arm_interpolation_result

            if self.whether_save_movel_position:
                self.interpolation_result_cart = deepcopy(self.Kinematic_Model.left_arm_forward_kinematics(self.interpolation_result[:7]))
                self.interpolation_result_cart_position = self.interpolation_result_cart.translation
                self.interpolation_result_cart_pose = self.interpolation_result_cart.rotation
                self.interpolation_result_cart_quat = quaternion.from_rotation_matrix(self.interpolation_result_cart_pose)

                with open("movel_interpolate_trajectory.csv", 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(self.interpolation_result)

                with open("movel_left_arm_interpolation_result_cart_position.csv", 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(self.interpolation_result_cart_position)

                with open("movel_left_arm_interpolation_result_cart_pose.csv", 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(self.interpolation_result_cart_pose)

                self.right_arm_interpolation_result_cart = deepcopy(self.Kinematic_Model.right_arm_forward_kinematics(self.interpolation_result[7:14]))
                self.right_arm_interpolation_result_cart_position = self.right_arm_interpolation_result_cart.translation
                self.right_arm_interpolation_result_cart_pose = self.right_arm_interpolation_result_cart.rotation
                self.right_arm_interpolation_result_cart_quat = quaternion.from_rotation_matrix(self.right_arm_interpolation_result_cart_pose)


                with open("movel_right_arm_interpolation_result_cart_position.csv", 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(self.right_arm_interpolation_result_cart_position)

                with open("movel_right_arm_interpolation_result_cart_pose.csv", 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(self.right_arm_interpolation_result_cart_pose)

            if(self.Collision_Detection.collision_detection_index):
                print("发生了碰撞，结束碰撞检测线程，退出当前插补函数！！！！")
                self.Collision_Detection.stop_collision_detection()
                sys.exit()    # 退出程序循环，机械臂停止运动

                
            self.lcm_handler.upper_body_data_publisher(self.interpolation_result)            
            self.movel_plan_current_joint_position = self.interpolation_result



            # with open("interpolate_trajectory1.csv", 'a', newline='', encoding='utf-8') as csvfile:
            #     writer = csv.writer(csvfile)
            #     writer.writerow(self.interpolation_result)


            # 用于保证下发周期是2ms
            elapsed_time = (time.time() - start_time)  # 已经过的时间，单位是秒
            delay = max(0, self.interpolation_period / 1000 - elapsed_time)  # 4毫秒减去已经过的时间
            time.sleep(delay)  # 延迟剩余的时间


        print("运行结束，到达目标点位！！！")
        self.Collision_Detection.stop_collision_detection()

    # 通过传入目标的笛卡尔位置进行MOVEL
    def moveL2target(self, left_arm_current_position, left_arm_target_position, right_arm_current_position, right_arm_target_position, robot_current_qpos):

        self.movel_plan_current_joint_position = robot_current_qpos
        self.cal_movel_plan_data(left_arm_current_position, left_arm_target_position, right_arm_current_position, right_arm_target_position)
        self.speed_plan = seven_segment_speed_plan(self.movel_plan_jerk_max, self.movel_plan_acc_max, self.movel_plan_speed_max, max(self.movel_plan_displacement, self.right_arm_movel_plan_displacement))
        self.movel_speed_plan_interpolation()


    def cal_movel_plan_data_by_joint_position(self, current_joint_position, target_joint_position):
        current_position = np.array(current_joint_position)
        target_position = np.array(target_joint_position)

        self.movel_plan_current_joint_position = current_position
        # print("self.movej_plan_current_joint_position  = {} ".format(self.movej_plan_current_joint_position ))
        self.movel_plan_target_joint_position = target_position


        # --------------- 左臂相关数据计算-------------------------------------------------
        # 获取左臂当前位置对应的末端笛卡尔位置姿态和四元数
        self.movel_plan_current_cart = deepcopy(self.Kinematic_Model.left_arm_forward_kinematics(self.movel_plan_current_joint_position[:7]))
        self.movel_plan_current_cart_position = self.movel_plan_current_cart.translation
        self.movel_plan_current_cart_pose = self.movel_plan_current_cart.rotation
        self.movel_plan_current_cart_quat = quaternion.from_rotation_matrix(self.movel_plan_current_cart_pose)
        if self.whether_save_movel_position:
            print("self.movel_plan_current_cart_position  = {} ".format(self.movel_plan_current_cart_position))
            print("self.movel_plan_current_cart_pose  = {} ".format(self.movel_plan_current_cart_pose))
        print("self.movel_plan_current_cart_position  = {} ".format(self.movel_plan_current_cart_position))
        print("self.movel_plan_current_cart_pose  = {} ".format(self.movel_plan_current_cart_pose))

        # 获取左臂期望位置对应的末端笛卡尔位置姿态和四元数
        self.movel_plan_target_cart = deepcopy(self.Kinematic_Model.left_arm_forward_kinematics(self.movel_plan_target_joint_position[:7]))
        self.movel_plan_target_cart_position = self.movel_plan_target_cart.translation
        self.movel_plan_target_cart_pose = self.movel_plan_target_cart.rotation
        self.movel_plan_target_cart_quat = quaternion.from_rotation_matrix(self.movel_plan_target_cart_pose)
        if self.whether_save_movel_position:
            print("self.movel_plan_target_cart_position  = {} ".format(self.movel_plan_target_cart_position))
            print("self.movel_plan_target_cart_pose  = {} ".format(self.movel_plan_target_cart_pose))
        print("self.movel_plan_target_cart_position  = {} ".format(self.movel_plan_target_cart_position))

        # 计算左臂规划需要总长度以及插补需要的各个变量
        self.movel_plan_displacement = np.linalg.norm(self.movel_plan_target_cart_position - self.movel_plan_current_cart_position)
        self.movel_plan_position_delta_disp[0] = self.movel_plan_target_cart_position[0] - self.movel_plan_current_cart_position[0] 
        self.movel_plan_position_delta_disp[1] = self.movel_plan_target_cart_position[1] - self.movel_plan_current_cart_position[1]   
        self.movel_plan_position_delta_disp[2] = self.movel_plan_target_cart_position[2] - self.movel_plan_current_cart_position[2]   

        # 左臂目标姿态的四元数是否需要更改
        if np.dot(self.movel_plan_current_cart_quat.components, self.movel_plan_target_cart_quat.components) < 0:
            self.movel_plan_target_cart_quat = - self.movel_plan_target_cart_quat

        # --------------- 右臂相关数据计算-------------------------------------------------
        # 获取右臂当前位置对应的末端笛卡尔位置姿态和四元数
        self.right_arm_movel_plan_current_cart = deepcopy(self.Kinematic_Model.right_arm_forward_kinematics(self.movel_plan_current_joint_position[7:14]))
        self.right_arm_movel_plan_current_cart_position = self.right_arm_movel_plan_current_cart.translation
        self.right_arm_movel_plan_current_cart_pose = self.right_arm_movel_plan_current_cart.rotation
        self.right_arm_movel_plan_current_cart_quat = quaternion.from_rotation_matrix(self.right_arm_movel_plan_current_cart_pose)
        if self.whether_save_movel_position:
            print("self.right_arm_movel_plan_current_cart_position  = {} ".format(self.right_arm_movel_plan_current_cart_position))
            print("self.right_arm_movel_plan_current_cart_pose  = {} ".format(self.right_arm_movel_plan_current_cart_pose))


        # 获取右臂期望位置对应的末端笛卡尔位置姿态和四元数
        self.right_arm_movel_plan_target_cart = deepcopy(self.Kinematic_Model.right_arm_forward_kinematics(self.movel_plan_target_joint_position[7:14]))
        self.right_arm_movel_plan_target_cart_position = self.right_arm_movel_plan_target_cart.translation
        self.right_arm_movel_plan_target_cart_pose = self.right_arm_movel_plan_target_cart.rotation
        self.right_arm_movel_plan_target_cart_quat = quaternion.from_rotation_matrix(self.right_arm_movel_plan_target_cart_pose)
        if self.whether_save_movel_position:
            print("self.right_arm_movel_plan_target_cart_position  = {} ".format(self.right_arm_movel_plan_target_cart_position))
            print("self.right_arm_movel_plan_target_cart_pose  = {} ".format(self.right_arm_movel_plan_target_cart_pose))


        # 计算左臂规划需要总长度以及插补需要的各个变量
        self.right_arm_movel_plan_displacement = np.linalg.norm(self.right_arm_movel_plan_target_cart_position - self.right_arm_movel_plan_current_cart_position)
        self.right_arm_movel_plan_position_delta_disp[0] = self.right_arm_movel_plan_target_cart_position[0] - self.right_arm_movel_plan_current_cart_position[0] 
        self.right_arm_movel_plan_position_delta_disp[1] = self.right_arm_movel_plan_target_cart_position[1] - self.right_arm_movel_plan_current_cart_position[1]   
        self.right_arm_movel_plan_position_delta_disp[2] = self.right_arm_movel_plan_target_cart_position[2] - self.right_arm_movel_plan_current_cart_position[2]   

        # 右臂目标姿态的四元数是否需要更改
        if np.dot(self.right_arm_movel_plan_current_cart_quat.components, self.right_arm_movel_plan_target_cart_quat.components) < 0:
            self.right_arm_movel_plan_target_cart_quat = - self.right_arm_movel_plan_target_cart_quat



    # 通过传入的关节角度正解进行MOVEL
    def moveL2targetjointposition(self, current_joint_position, target_joint_position):
        self.cal_movel_plan_data_by_joint_position(current_joint_position, target_joint_position)
        self.speed_plan = seven_segment_speed_plan(self.movel_plan_jerk_max, self.movel_plan_acc_max, self.movel_plan_speed_max, max(self.movel_plan_displacement, self.right_arm_movel_plan_displacement))
        self.movel_speed_plan_interpolation()


    def moveL2targetjointposition_FT(self, current_joint_position, target_joint_position):
        self.constant_force_track = True
        self.cal_movel_plan_data_by_joint_position(current_joint_position, target_joint_position)
        self.speed_plan = seven_segment_speed_plan(self.movel_plan_jerk_max, self.movel_plan_acc_max, self.movel_plan_speed_max, max(self.movel_plan_displacement, self.right_arm_movel_plan_displacement))
        self.movel_speed_plan_interpolation()
        self.constant_force_track = False





        




