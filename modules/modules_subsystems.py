
import math
import torch
from itertools import chain
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import patches as patches
import pandas as pd
import numpy as np
import time


import optlib.operations as op
import optlib.common_func as cf
from browser.options_base import DEVICETYPE

class Subsystem:
    """组织其中的PS和Devices"""
    def __init__(self, args = None):
        self.args = args
        self.PS_list = []
        self.Devices_list = []
        # output data
        self.acc_list = []
        self.corr_list = []
        self.corr_matrix = None
        self.corr_matrix_list = []
        self.mse_list = []
        self.cos_list = []
        self.r_list= []
        self.r_hat_list = []
        self.power_list = []
        self.sin_list = []
        self.uavloc_list=[]
    
    def Subsystem_Init(self):
        # 初始梯度状态字典同步
        for device in self.Devices_list:
            device.LocalNetwork.load_state_dict(self.PS_list[0].GlobalNetwork.state_dict())
    
    def Subsystem_DeviceDatasetIter(self, subsystemNumber):
        for device in self.Devices_list:
            device.LocalNetwork.iter_data = \
                DataLoader(device.LocalNetwork.data, batch_size=device.LocalNetwork.bs,shuffle=True, num_workers=0) # Dataloader就是把数据集按batch切割分，然后形成一个datalist
    
    def Subsystem_DeviceTrain(self):
        # 针对每一个网络，训练-获取梯度-获取压缩后梯度-获取功率控制值
        for i,device in enumerate(self.Devices_list):
            device.LocalNetwork.start_IterTrain()
            device.LocalNetwork.start_GetGrad()

    def Subsystem_DeviceMod(self, channel):
        # 针对每一个网络，训练-获取梯度-获取压缩后梯度-获取功率控制值
        for i,device in enumerate(self.Devices_list):
            device.Modulation()   ### OModulation

    def Subsystem_PSTrain(self, channel, task_id, epoch, optimizer,J):
        # 服务器端AMP算法解压梯度数据从而获得每一个全局网络下的梯度更新量
        for i,ps in enumerate(self.PS_list):
            #TODO: 加用户选择
            # 无噪声
            ps.r0 = self.args.r0[task_id]
            if self.args.method == 0: # error free的情况
                # 解调
                ps.GlobalNetwork.grad_update = 0
                for j,device in enumerate(self.Devices_list):
                    ps.GlobalNetwork.grad_update += optimizer.x_slct[task_id][j]\
                        * torch.tensor(self.args.datasets_num_dvc[task_id][j]).to(DEVICETYPE)\
                        * device.LocalNetwork.grad_basic
                # ps.GlobalNetwork.grad_update /= self.args.datasets_num[i]
                # ps.GlobalNetwork.grad_update /= self.PS_list[0].K
                ps.GlobalNetwork.grad_update = ps.GlobalNetwork.grad_update.view(-1,)


                # scaling
                if self.args.scaling == True: # Scaling gradients at the PS
                    mean = torch.mean(ps.GlobalNetwork.grad_update)
                    grad_temp = ps.GlobalNetwork.grad_update - mean
                    norm = torch.norm(grad_temp)
                    ps.GlobalNetwork.grad_update = grad_temp/norm + mean
                    ps.GlobalNetwork.grad_update *= ps.r0
            else:
                j=epoch%J
                ps.DeModulation(j)  # 是对应machine里PS的demodulation
            # 更新学习率
            ps.GlobalNetwork.RefreshLearningRate(task_id, epoch)
    
    def Subsystem_DeviceRefresh(self, lr):
        """"设备端更新梯度字典"""
        # 点对点下行链路
        for device in self.Devices_list:
            device.LocalNetwork.load_state_dict(self.PS_list[0].GlobalNetwork.state_dict(), strict=True)
    
    def Subsystem_Corr(self, rho_sta = 0):
        m = len(self.Devices_list)
        E_hat = np.zeros((m, m), dtype = complex)
        gradlen = (self.Devices_list[0].LocalNetwork.grad_normal).shape[0]*2

        for i,device_i in enumerate(self.Devices_list):
            for j,device_j in enumerate(self.Devices_list):
                E_temp = np.dot(np.conj(device_i.LocalNetwork.grad_normal), device_j.LocalNetwork.grad_normal)
                E_hat[i,j] = E_temp / (gradlen)
        if self.args.correlation == 0:
            E = E_hat
        elif self.args.correlation == 1:
            E = np.ones((m,m)) * rho_sta
        elif self.args.correlation == 2:
            E = np.identity(m)
        elif self.args.correlation == 3:
            E = np.ones((m,m))
        elif self.args.correlation == 4:
            E = np.identity(m) + (np.ones((m,m)) - np.identity(m)) * self.args.rho
        self.corr_matrix = np.real(E)
        rho = np.sum(E_hat - np.identity(m))/(m**2-m)
        self.corr_list.append([np.real(rho), np.imag(rho)])
        self.corr_matrix_list.append(np.reshape(np.real(E_hat), -1))
   