

import math
import torch
from itertools import chain
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import patches as patches
import pandas as pd
import numpy as np
# import cupy as np
import time

import modules.operations_base as opr_base
from modules.modules_channel import Channel
from modules.modules_subsystems import Subsystem
from modules.modules_machine import PS, Device
from modules.modules_networks import GlobalNetwork, LocalNetwork
from modules.modules_opt import Optimizer

import optlib.operations as op
import optlib.common_func as cf
from browser.options_base import DEVICETYPE

class Environments:
    """控制全局仿真的进行"""
    def __init__(self, args):
        self.args = args
        self.subsystems_list = []
        self.nmse_tofile = []
        self.optimizer = None
        self.corr_curve = None
    
    def Environments_Init(self, channel, dataset):
        # 根据任务分配子系统，部署训练网络，实现并调用Environments_TaskDivision方法 
        self.Environments_TaskDivision(dataset)

        # subsystem初始化
        for subsystems in self.subsystems_list:
            subsystems.Subsystem_Init()
        
        # 优化器初始化
        self.optimizer = Optimizer(self.args)
        # self.optimizer.Optimizer_Init(channel)

        # 导入相关曲线
        # corr_curve = pd.read_csv('./input/corr.csv',index_col=0)
        # self.corr_curve = {
        #     q: corr_curve.iloc[:,q] for q in range(self.args.Q)
        # }
         
    def Environments_TaskDivision(self, dataset):
        for i,task in enumerate(self.args.task_list):
            subsystem_temp = Subsystem(self.args)
            # 全局网络建立并绑定至PS
            subsystem_temp.PS_list.append(
                PS(self.args, 
                    GlobalNetwork(self.args, dataset[task + "_test_origin"])
                )
            )
            # 局部网络建立并绑定至每个Device
            for j in range(self.args.users_num[i]):
                subsystem_temp.Devices_list.append(
                    Device(self.args,
                        LocalNetwork(self.args, i, j, dataset[task + "_train_users"][j])
                    )
                )
            self.subsystems_list.append(subsystem_temp)

    # def Environments_flyingchannels(self,channel):
    #     Nfly=self.args.N



    
    def Environments_Opt(self, channel, epoch,opt=False):
        self.optimizer.Optimizer_Collector(self.subsystems_list)
        if opt == True:
            self.optimizer.Optimizer_Init(channel)
            self.optimizer.Optimizer_Optimize(self.args.method,epoch)
        self.optimizer.Optimizer_ResultBroadcast(self.subsystems_list)
        # self.optimizer.Optimizer_Ref(self.subsystems_list)
    
    def Environments_Assess(self, channel):  ### wrong!
        for i,subsystem in enumerate(self.subsystems_list):
            power_hist = np.zeros(self.args.users_num[i])
            
            r_hat = np.array((subsystem.PS_list[0].GlobalNetwork.grad_update).cpu())
            gradlen = len(r_hat)
            r = np.zeros(gradlen)

            for j,device in enumerate(subsystem.Devices_list):
                local_grad = (device.LocalNetwork.grad_basic[0,:]).cpu()
                # r += self.args.datasets_num_dvc[i][j] * subsystem.PS_list[0].x_slct[j] * \
                #     np.pad(
                #         local_grad, (0,gradlen-local_grad.shape[0]),'constant',constant_values=(0,0)
                #     )
                r += self.args.datasets_num_dvc[i][j] * 1 * \
                     np.pad(
                         local_grad, (0, gradlen - local_grad.shape[0]), 'constant', constant_values=(0, 0)
                     )
                try:
                    power_hist[j] = np.linalg.norm(self.optimizer.bf_tx[i][j])**2 
                except KeyError:
                    power_hist[j] = 0
            # r = r/subsystem.PS_list[0].K
            r = r / self.args.datasets_num[i]

            # 评估
            MSE_array = np.linalg.norm(r-r_hat)** 2 / (np.linalg.norm(r)**2)
            COS = np.dot(r, r_hat) / (np.linalg.norm(r) * np.linalg.norm(r_hat) )
            # print(np.mean(r), np.mean(r_hat))

            uavloc=np.concatenate((np.array(subsystem.PS_list[0].uavloc[str(i) + 'x']),np.array(subsystem.PS_list[0].uavloc[str(i) + 'y']))).reshape(2,-1).T

            subsystem.mse_list.append(MSE_array)
            subsystem.cos_list.append(COS)
            subsystem.r_list.append(np.linalg.norm(r)**2)
            subsystem.r_hat_list.append(np.linalg.norm(r_hat)**2)
            subsystem.power_list.append(power_hist)
            subsystem.uavloc_list.append(uavloc)

            epoch = len(subsystem.mse_list)-1
            if epoch % self.args.save_freq == 0:
                plt.subplot(211)
                plt.title('r')
                plt.plot(r.reshape(-1,))
                plt.subplot(212)
                plt.title('r_hat')
                plt.plot(r_hat.reshape(-1,))
                plt.savefig('./output/fig/task_{}_MC_{}_r_r_hat_{}.png'.format(i, self.args.MC, epoch))
                plt.clf()
        # print('obj = {}'.format(self.optimizer.obj_temp))

    def Environments_Iter(self, channel, epoch):
        # 子系统给每一个客户端创建可迭代数据集
        for i,subsystem in enumerate(self.subsystems_list):
            subsystem.Subsystem_DeviceDatasetIter(i)

        # 根据预先设定好的迭代轮次进行迭代
        ### update device location
        # if epoch == 0:
        #     channel.Channel_Init(loc_update=True)
        # else:
        #     channel.Channel_Init(loc_update=False)  #就改一下试试;True的含义是，每个epoch无人机的轨迹都是从头开始优化
        # if epoch == 0:  # 继承之前的轨迹
        if epoch % self.args.J == 0:  # 但是也其实可以每个下次优化继承之前的轨迹
            channel.Channel_Init(loc_update=True)
        # else:
        #     channel.Channel_Init(loc_update=False)  #就改一下试试;True的含义是，每个epoch无人机的轨迹都是从头开始优化
        ### update UAV-PS location

        # 每个子系统， 在设备端，训练-获取梯度
        for i,subsystem in enumerate(self.subsystems_list):
            subsystem.Subsystem_DeviceTrain()
        
        # 测试相关性
        for i,subsystem in enumerate(self.subsystems_list):
            # subsystem.Subsystem_Corr(self.corr_curve[i][epoch])
            subsystem.Subsystem_Corr()

        # # random choice，对用户选择变量的初始化
        # if self.args.userselection == 0:
        #     self.optimizer = opr_base.all_choice(self.args, self.optimizer)
        # elif self.args.userselection == 1:
        #     self.optimizer = opr_base.random_choice(self.args, self.optimizer)
        # elif self.args.userselection == 2:
        #     self.optimizer = opr_base.dis_choice(self.args, self.optimizer, channel)    # yes，这里改变的是x_slct，这个就是用户选择变量，实在后面优化中优化了的

        # # 更新selected sum
        # for i,subsystem in enumerate(self.subsystems_list):
        #     subsystem.PS_list[0].K = np.dot(self.optimizer.x_slct[i], self.args.datasets_num_dvc[i])

        # 无噪声
        if self.args.method == 0:
            for i,subsystem in enumerate(self.subsystems_list):
                subsystem.Subsystem_PSTrain(channel, i, epoch, self.optimizer)
        else:
            if epoch % self.args.J == 0:  # 每J个epoch才有一次优化
                if epoch % self.args.optfreq == 0:# 每两次优化一次
                    self.Environments_Opt(channel, epoch, opt=True)  #### 优化因J改变
                channel.Channel_Init(loc_update=False)
            for i, subsystem in enumerate(self.subsystems_list):  # 本地模型怎么训还是怎么训
                subsystem.Subsystem_DeviceMod(channel)

            # for n in range(self.args.N):
            # 每个子系统， 在设备端，获取功率控制值-发送至信道
            # 初始化信道矩阵，信道接收到子系统信号，进行运算，加入噪声
            channel.Channel_Accumulation_Uplink(self.subsystems_list, self.optimizer, epoch)  #### 这里是Aggregation，需要因J改变

            # 每个子系统，在PS端，解压缩估计梯度和-稀疏化-获取压缩后梯度-获取功率控制值-发送至信道
            for i,subsystem in enumerate(self.subsystems_list):
                subsystem.Subsystem_PSTrain(channel, i, epoch, self.optimizer, self.args.J)

        # 评估系统
        self.Environments_Assess(channel)

        # 每个子系统更新
        for i,subsystem in enumerate(self.subsystems_list):
            subsystem.Subsystem_DeviceRefresh(subsystem.PS_list[0].GlobalNetwork.args.lr[i])

    def Environments_Test(self, epoch):
        # 每轮迭代后评价结果
        if not epoch % self.args.tp:
            for i, subsystem in enumerate(self.subsystems_list):
                subsystem.PS_list[0].GlobalNetwork.start_test(subsystem.PS_list[0].GlobalNetwork.data)
                print("     network%d" % i + "-accuracy:%f" % \
                    subsystem.PS_list[0].GlobalNetwork.accuracy[int(epoch / self.args.tp)])
                # 保存学习率
                subsystem.acc_list.append(subsystem.PS_list[0].GlobalNetwork.accuracy[int(epoch / self.args.tp)])

