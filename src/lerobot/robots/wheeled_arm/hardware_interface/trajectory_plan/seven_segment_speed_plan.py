import math
import numpy as np
class seven_segment_speed_plan:
    # 初始条件
    jerk_max = 0
    acc_max = 0
    speed_max = 0
    displacement = 0
    MIN_VAL = 0.0000001

    # 七段规划所需的变量
    time_length = 0
    plan_disp_length = 0

    # 当前时刻对应的速度相关数据的计算结果
    cur_jerk = 0
    cur_acc = 0
    cur_speed = 0
    cur_disp = 0
    cur_disp_normalization_ratio = 0    # 当前时刻位移长度对应总长度的比例 进行归一化 用于各个关节的插补计算
    
    # 加加速段
    accacc_time = 0
    accacc_Jerk = 0
    accacc_Acc = 0
    accacc_Speed = 0
    accacc_Disp= 0
    
    # 匀加速段
    uniacc_time = 0
    uniacc_Jerk = 0
    uniacc_Acc = 0
    uniacc_Speed = 0
    uniacc_Disp = 0

    # 减加加速段
    decacc_time = 0
    decacc_Jerk = 0
    decacc_Acc = 0
    decacc_Speed = 0
    decacc_Disp = 0

    # 匀速段
    unispeed_time = 0
    unispeed_Jerk = 0
    unispeed_Acc = 0
    unispeed_Speed = 0
    unispeed_Disp = 0

    # 加减速段
    accdec_time = 0
    accdec_Jerk = 0
    accdec_Acc = 0
    accdec_Speed = 0
    accdec_Disp = 0

    # 匀减速段
    unidec_time = 0
    unidec_Jerk = 0
    unidec_Acc = 0
    unidec_Speed = 0
    unidec_Disp = 0

    # 减减速段
    decdec_time = 0
    decdec_Jerk = 0
    decdec_Acc = 0
    decdec_Speed = 0
    decdec_Disp = 0

    # 加速段时间长度
    acceleration_segment_time = 0
    # 加速段位移长度
    acceleration_segment_disp = 0
    # 减速段时间长度
    deceleration_segment_time = 0
    # 减速段位移长度
    deceleration_segment_disp = 0


    def __init__(self, jerk_max, acc_max, speed_max, displacement):
        self.jerk_max = jerk_max
        self.acc_max = acc_max
        self.speed_max = speed_max
        self.displacement = displacement
        self.MIN_VAL = 1e-7
        self.unispeed_time = 0
        self.seven_segment_speed_plan_func()
        self.cal_accacc_segment_end_data()
        self.cal_uniacc_segment_end_data()
        self.cal_decacc_segment_end_data()
        self.cal_unispeed_segment_end_data()
        self.cal_accdec_segment_end_data()
        self.cal_unidec_segment_end_data()
        self.cal_decdec_segment_end_data()
        self.cal_total_segment_end_data()
        self.cal_max_Acc_Speed()


    def seven_segment_speed_plan_func(self):
        # 首先假设最大速度可达（包含最大加速度可以达到和最大加速度不可以达到） 计算有没有匀速段
            # 计算当前的最大速度是否满足最大加速度需求
        if self.speed_max * self.jerk_max - self.acc_max * self.acc_max > self.MIN_VAL:
            # 当前最大速度可以满足最大加速度 最大加速度可达 存在匀加速段
            self.accacc_time = self.acc_max / self.jerk_max
            self.decacc_time = self.accacc_time
            self.accdec_time = self.accacc_time
            self.decdec_time = self.accacc_time

            self.uniacc_time = self.speed_max / self.acc_max - self.acc_max / self.jerk_max
            self.unidec_time = self.uniacc_time
        else:
            # 当前最大速度不满足最大加速度 最大加速度不可达 不存在匀加速段 需要重新计算最大加速度值
            self.accacc_time = math.sqrt(self.speed_max / self.jerk_max)
            self.decacc_time = self.accacc_time
            self.accdec_time = self.accacc_time
            self.decdec_time = self.accacc_time

            self.uniacc_time = 0
            self.unidec_time = self.uniacc_time

            self.acc_max = self.accacc_time * self.jerk_max

        # 判断不论是上方任何一种情况是否存在匀速段
        self.acceleration_segment_disp = self.jerk_max * self.accacc_time *  (self.accacc_time + self.uniacc_time) * (0.5 * self.uniacc_time + self.accacc_time)
        self.deceleration_segment_disp = self.acceleration_segment_disp
        self.unispeed_Disp = self.displacement - self.acceleration_segment_disp - self.deceleration_segment_disp

        if self.unispeed_Disp > self.MIN_VAL:
            # 存在匀速段 如果之前的计算中满足了最大加速度约束 则此时是正常的七段 
            # 如果之前的计算中不满足最大加速度约束 则此时是缺少匀加速段和匀减速段的五段
            self.unispeed_time = self.unispeed_Disp / self.speed_max
        else:
            # 不存在匀速段 需要重新计算最大速度
            # 需要根据最大位移约束进一步判断是否满足最大加速度约束
            self.unispeed_Disp = 0
            if (self.displacement - 2 * (self.acc_max ** 3) / (self.jerk_max ** 2)) > self.MIN_VAL:
                # 当前位移长度 在没有匀速段的时候 满足最大加速度约束 此时会计算成不含匀速段的6段
                # 只需要重新计算最大速度
                self.accacc_time = self.acc_max / self.jerk_max
                self.decacc_time = self.accacc_time
                self.accdec_time = self.accacc_time
                self.decdec_time = self.accacc_time

                self.uniacc_time = -1.5 * self.accacc_time + math.sqrt((self.accacc_time / 2) ** 2 + self.displacement / self.acc_max)
                self.unidec_time = self.uniacc_time
                self.unispeed_time = 0
            else:
                # 当前位移长度 在没有匀速段的时候 不满足最大加速度约束 
                # 此时会计算成不含匀速段 匀加速段 匀减速段的4段
                # 需要重新计算最大速度 最大加速度
                self.accacc_time = (self.displacement / (2 * self.jerk_max)) ** (1/3)
                self.decacc_time = self.accacc_time
                self.accdec_time = self.accacc_time
                self.decdec_time = self.accacc_time

                self.uniacc_time = 0
                self.unidec_time = 0
                self.unispeed_time = 0
                self.acc_max = self.jerk_max * self.accacc_time

        self.time_length = self.accacc_time + self.uniacc_time + self.decacc_time + self.unispeed_time + self.accdec_time + self.unidec_time + self.decdec_time
        # print(f"self.time_length =  {self.time_length}")


    def cal_accacc_segment_end_data(self):
        self.accacc_Jerk = self.jerk_max
        self.accacc_Acc = self.jerk_max * self.accacc_time
        self.accacc_Speed = 0.5 * self.jerk_max * self.accacc_time * self.accacc_time
        self.accacc_Disp = self.jerk_max * self.accacc_time * self.accacc_time * self.accacc_time / 6

    def cal_accacc_segment_data(self, time):
        self.cur_jerk = self.jerk_max
        self.cur_acc = self.jerk_max * time
        self.cur_speed = 0.5 * self.jerk_max * time * time
        self.cur_disp = self.jerk_max * time * time * time / 6
        self.cur_disp_normalization_ratio = self.cur_disp / self.displacement


    def cal_uniacc_segment_end_data(self):
        self.uniacc_Jerk = 0
        self.uniacc_Acc = self.accacc_Acc
        self.uniacc_Speed = self.accacc_Speed + self.accacc_Acc * self.uniacc_time
        self.uniacc_Disp = self.accacc_Speed * self.uniacc_time + 0.5 * self.accacc_Acc * self.uniacc_time * self.uniacc_time

    def cal_uniacc_segment_data(self, time):
        self.cur_jerk = 0
        self.cur_acc = self.accacc_Acc
        self.cur_speed = self.accacc_Speed + self.accacc_Acc * time
        self.cur_disp = self.accacc_Disp + self.accacc_Speed * time + 0.5 * self.accacc_Acc * time * time
        self.cur_disp_normalization_ratio = self.cur_disp / self.displacement


    def cal_decacc_segment_end_data(self):
        self.decacc_Jerk = - self.jerk_max
        self.decacc_Acc = self.uniacc_Acc - self.jerk_max * self.decacc_time
        self.decacc_Speed = self.uniacc_Speed + self.uniacc_Acc * self.decacc_time - 0.5 * self.jerk_max * self.decacc_time * self.decacc_time
        self.decacc_Disp = self.uniacc_Speed * self.decacc_time + 0.5 * self.uniacc_Acc * self.decacc_time * self.decacc_time - self.jerk_max * self.decacc_time * self.decacc_time * self.decacc_time / 6
        self.acceleration_segment_disp = self.decacc_Disp + self.uniacc_Disp + self.accacc_Disp
        self.acceleration_segment_time = self.decacc_time + self.uniacc_time + self.accacc_time

    def cal_decacc_segment_data(self, time):
        self.cur_jerk = - self.jerk_max
        self.cur_acc = self.uniacc_Acc - self.jerk_max * time
        self.cur_speed = self.uniacc_Speed + self.uniacc_Acc * time - 0.5 * self.jerk_max * time * time
        self.cur_disp = self.accacc_Disp + self.uniacc_Disp + self.uniacc_Speed * time + 0.5 * self.uniacc_Acc * time * time - self.jerk_max * time * time * time / 6
        self.cur_disp_normalization_ratio = self.cur_disp / self.displacement


    def cal_unispeed_segment_end_data(self):
        self.unispeed_Jerk = 0
        self.unispeed_Acc = 0
        self.unispeed_Speed = self.decacc_Speed
        self.unispeed_Disp = self.unispeed_Speed * self.unispeed_time

    def cal_unispeed_segment_data(self, time):
        self.cur_jerk = 0
        self.cur_acc = 0
        self.cur_speed = self.decacc_Speed
        self.cur_disp = self.acceleration_segment_disp + self.unispeed_Speed * time
        self.cur_disp_normalization_ratio = self.cur_disp / self.displacement


    def cal_accdec_segment_end_data(self):
        self.accdec_Jerk = - self.jerk_max
        self.accdec_Acc = - self.jerk_max * self.accdec_time
        self.accdec_Speed = self.unispeed_Speed - 0.5 * self.jerk_max * self.accdec_time * self.accdec_time
        self.accdec_Disp = self.unispeed_Speed * self.accdec_time - self.jerk_max * self.accdec_time * self.accdec_time * self.accdec_time / 6

    def cal_accdec_segment_data(self, time):
        self.cur_jerk = - self.jerk_max
        self.cur_acc = - self.jerk_max * time
        self.cur_speed = self.unispeed_Speed - 0.5 * self.jerk_max * time * time
        self.cur_disp = self.acceleration_segment_disp + self.unispeed_Disp + self.unispeed_Speed * time - self.jerk_max * time * time * time / 6
        self.cur_disp_normalization_ratio = self.cur_disp / self.displacement


    def cal_unidec_segment_end_data(self):
        self.unidec_Jerk = 0
        self.unidec_Acc = self.accdec_Acc
        self.unidec_Speed = self.accdec_Speed + self.accdec_Acc * self.unidec_time
        self.unidec_Disp = self.accdec_Speed * self.unidec_time + 0.5 * self.accdec_Acc * self.unidec_time * self.unidec_time

    def cal_unidec_segment_data(self, time):
        self.cur_jerk = 0
        self.cur_acc = self.accdec_Acc
        self.cur_speed = self.accdec_Speed + self.accdec_Acc * time
        self.cur_disp = self.acceleration_segment_disp + self.unispeed_Disp + self.accdec_Disp + self.accdec_Speed * time + 0.5 * self.accdec_Acc * time * time
        self.cur_disp_normalization_ratio = self.cur_disp / self.displacement


    def cal_decdec_segment_end_data(self):
        self.decdec_Jerk = self.jerk_max
        self.decdec_Acc = self.unidec_Acc + self.jerk_max * self.decdec_time
        self.decdec_Speed = self.unidec_Speed + self.unidec_Acc * self.decdec_time + 0.5 * self.jerk_max * self.decdec_time * self.decdec_time
        self.decdec_Disp = self.unidec_Speed * self.decdec_time + 0.5 * self.unidec_Acc * self.decdec_time * self.decdec_time + self.jerk_max * self.decdec_time * self.decdec_time * self.decdec_time / 6
        self.deceleration_segment_disp = self.accdec_Disp + self.unidec_Disp + self.decdec_Disp
        self.deceleration_segment_time = self.accdec_time + self.unidec_time + self.decdec_time
        self.plan_disp_length = self.acceleration_segment_disp + self.unispeed_Disp + self.deceleration_segment_disp

    def cal_decdec_segment_data(self, time):
        self.cur_jerk = self.jerk_max
        self.cur_acc = self.unidec_Acc + self.jerk_max * time
        self.cur_speed = self.unidec_Speed + self.unidec_Acc * time + 0.5 * self.jerk_max * time * time
        self.cur_disp = self.acceleration_segment_disp + self.unispeed_Disp + self.accdec_Disp + self.unidec_Disp + self.unidec_Speed * time + 0.5 * self.unidec_Acc * time * time + self.jerk_max * time * time * time / 6
        self.cur_disp_normalization_ratio = self.cur_disp / self.displacement

    def cal_total_segment_end_data(self):
        self.total_Disp = self.acceleration_segment_disp + self.unispeed_Disp + self.decdec_Disp
    
    def cal_max_Acc_Speed(self):
        self.actual_max_acc = self.acc_max
        self.actual_max_speed = self.speed_max

        # print(f"self.actual_max_acc =  {self.actual_max_acc}")
        # print(f"self.actual_max_speed =  {self.actual_max_speed}")


    def cal_seven_segment_speed_data(self):
        # 计算各段终点时刻对应的加加速度 加速度 速度 位置
        self.cal_accacc_segment_end_data()
        self.cal_uniacc_segment_end_data()
        self.cal_decacc_segment_end_data()
        self.cal_unispeed_segment_end_data()
        self.cal_accdec_segment_end_data()
        self.cal_unidec_segment_end_data()
        self.cal_decdec_segment_end_data()

