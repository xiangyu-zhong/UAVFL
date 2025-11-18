
import math
from tkinter.tix import Tree
import torch
from itertools import chain
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

import modules.operations_base as opr_base
from browser.options_base import DEVICETYPE

class PS:
    """服务器"""
    def __init__(self, args, global_network):
        # ------------------------------------#
        self.args = args
        self.ps_tx_signal = None
        self.ps_rx_signal = None
        self.AMP_A = torch.zeros(1, 1)  # A
        self.GlobalNetwork = global_network
        # 通信
        # self.bf_rx = 0
        self.c_q = 0
        self.c_qJ = 0
        self.x_slct = 0
        self.x_slctJ = 0
        self.x_slctMN = 0  ###
        self.x_slctJMN = 0  ####
        self.g_bar = []
        self.scale = []
        self.K = 0
        self.r0 = 0
        self.uavloc = {}
    
    def GenerateAMatrix(self, rows, cols):
        # A ~ (0,1/s~)
        compress_code = math.sqrt(1 / rows) * torch.randn(rows, cols)
        # A = A/||A||
        compress_code = torch.div(compress_code, torch.norm(compress_code, 2, 0).repeat(rows, 1))
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        self.AMP_A = compress_code.to(DEVICETYPE)
    
    def DeModulation(self, j):
        # 解套放在Channel上
        self.GlobalNetwork.grad_update = opr_base.np2tensorcplx(self.ps_rx_signal).to(DEVICETYPE)
        aaaa = self.GlobalNetwork.grad_update.detach().cpu().numpy()

        # self.GlobalNetwork.grad_update /= torch.tensor(self.scale[j]).to(DEVICETYPE)  # 这就对应上面的r0

        # scaling
        if self.args.scaling == True:
            norm = torch.norm(self.GlobalNetwork.grad_update)
            # self.GlobalNetwork.grad_update /= norm
            # self.GlobalNetwork.grad_update *= self.r0   # 是5e-2；不对，应该是关了
            # std = torch.std(self.GlobalNetwork.grad_update.cpu())
            # self.GlobalNetwork.grad_update /= std

        # 加均值：
        self.GlobalNetwork.grad_update += torch.tensor(self.g_bar[j]).to(DEVICETYPE)
        self.GlobalNetwork.grad_update /= torch.tensor(self.scale[j]).to(DEVICETYPE) # 这就对应上面的r0

        # self.GlobalNetwork.grad_update += torch.tensor(self.g_bar[j]/self.scale[j]).to(DEVICETYPE)

        norm = torch.norm(self.GlobalNetwork.grad_update)
        # self.GlobalNetwork.grad_update /= norm
        # self.GlobalNetwork.grad_update *= self.r0 # 草 怎么感觉发挥lr的作用呢因为后面PS更新参数直接更新了没乘lr……都是0.05

        aaaaa = self.GlobalNetwork.grad_update.detach().cpu().numpy()
        # 限幅
        self.GlobalNetwork.grad_update[self.GlobalNetwork.grad_update > 1e2] = 1e2
        self.GlobalNetwork.grad_update[self.GlobalNetwork.grad_update < -1e2] = -1e2

        # mean
        # g_bar_hat = np.array(torch.mean(self.GlobalNetwork.grad_update).cpu())
        # print(g_bar_hat, self.g_bar)

    def Modulation(self, subsystemNum, csi):
        signal = []
        for i in range(len(csi)):
            signal.append(self.GlobalNetwork.grad_compress.unsqueeze(dim=0).to(DEVICETYPE) \
                 * math.sqrt(self.GlobalNetwork.alpha)/csi[i]) 
        self.ps_tx_signal = torch.cat(signal, 0).to(DEVICETYPE)
    
    def OModulation(self, subsystemNum, csi_cplx):
        signal = []
        for i in range(len(csi_cplx)):
            signal_to_per_dvc = opr_base.tensor2npcplx(
                self.GlobalNetwork.grad_compress.to(DEVICETYPE)
            )
            signal.append( np.divide(signal_to_per_dvc, csi_cplx[i]) * math.sqrt(self.GlobalNetwork.alpha)) 
        self.ps_tx_signal = np.array(signal)

class Device:
    """单个设备"""
    def __init__(self, args, local_network):
        # ------------------------------------#
        self.args = args
        self.dvc_tx_signal = None       # 
        self.dvc_rx_signal = None       # 
        self.AMP_A = torch.zeros(1, 1)  # A
        self.LocalNetwork = local_network
        # 通信
        self.bf_tx = 0

    def Modulation(self, csi=None):
        self.dvc_tx_signal = self.LocalNetwork.grad_normal

    def OModulation(self, csi_cplx=None):
        self.dvc_tx_signal = self.LocalNetwork.grad_normal * self.bf_tx

    def ReceiveSignal(self, signal):
        if self.args.ch_data_type == "real":
            self.dvc_rx_signal = signal.squeeze(dim = 0)
        elif self.args.ch_data_type == "cplx":
            self.dvc_rx_signal = opr_base.np2tensorcplx(np.squeeze(signal))

    def DeModulation(self):
        self.LocalNetwork.grad_update = opr_base.AMP_cuda(
                self.AMP_A, self.dvc_rx_signal.to(DEVICETYPE), self.args.it 
            )
        # 梯度归一化
        self.LocalNetwork.grad_update /= torch.norm(self.LocalNetwork.grad_update, 2)


    

    