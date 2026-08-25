import time 
import numpy as np 
import math 
from robot_model import robot_model 
 
if __name__ == "__main__": 
    robot = robot_model() 
    # 该参数设置为0 是不进行碰撞检测 
    robot.Collision_Detection.collision_detection_level = 0 
    # 设置机器人部位移动状态: 1: 移动, 0: 不移动
    robot.lcm_handler.left_arm_moving = 1
    robot.lcm_handler.right_arm_moving = 1
    robot.lcm_handler.head_moving = 1
    robot.lcm_handler.waist_moving = 1
    robot.lcm_handler.leg_moving = 1
    robot.lcm_handler.left_gripper_moving = 0
    robot.lcm_handler.right_gripper_moving = 0
    if hasattr(robot.lcm_handler, "left_suction_moving"):
        robot.lcm_handler.left_suction_moving = 0
    if hasattr(robot.lcm_handler, "right_suction_moving"):
        robot.lcm_handler.right_suction_moving = 0
    # movej_plan_target_position_list: movej模式下的目标位置列表
    # 左臂(7) + 右臂(7) + 左手(1) + 右手(1) + 头(2) + 腰(3) + 腿(2) = 23
    robot.movej_plan_target_position_list  =  [
                                                [0.0, 0.0, 0, 0, 0, 0, 0,   # 左臂
                                                 0.0, 0.0, 0, 0, 0, 0, 0,   # 右臂
                                                 0.0, 0,                             # 左手(1) + 右手(1)
                                                 0.0, -0.0,                             # 头(2)
                                                 0.0, 0.0, -0.3,                          # 腰(3)
                                                 -0.4, -0.1],                            # 腿(2)
                                                [-0.0, 1.2, 0, 0, 0, 0, 0,
                                                 -0.0, 1.2, 0, 0, 0, 0, 0] +[0, 0] + [0.2, 0.3] + [-0.0, 0.0, -0.3] + [-0.4, -0.1],
                                                [0.0, 0.0, 0, 1.5, 0, 0, 0,
                                                 0.0, 0.0, 0, 1.5, 0, 0, 0] +[0, 0] + [0.0, -0.0] + [0.5, 0.0, -1.0] + [-1.8, -0.8],
                                                # [-1.0, -1.2, 0, 0, 0, 0, 0,
                                                #  -1.0, -1.2, 0, 0, 0, 0, 0] +[0, 0] + [-0.5, 0.5] + [-0.8, 0.0, -0.5] + [0.0, 0.3],
                                                # [1.0, 1.2, 0, 0, 0, 0, 0,
                                                #  1.0, 1.2, 0, 0, 0, 0, 0] +[0, 0] + [0.5, -0.5] + [0.8, 0.0, 0.5] + [-0.5, -0.3],
                                                # [-1.0, -1.2, 0, 0, 0, 0, 0,
                                                #  -1.0, -1.2, 0, 0, 0, 0, 0] +[0, 0] + [-0.5, 0.5] + [-0.8, 0.0, -0.5] + [0.0, 0.3],
                                            ] 
    time.sleep(0.5)
 
    while True:
        robot.trajectory_segment_index = 0
        robot.robot_movej_to_target_position() 
