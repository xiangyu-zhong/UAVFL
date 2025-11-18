

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
import modules.modules_networks as mm_net
import modules.modules_opt as mm_opt

import optlib.operations as op
import optlib.common_func as cf
from browser.options_base import DEVICETYPE

class Channel:
    """信道"""
    def __init__(self, args):
        self.args = args
        self.csi = None
        self.csi_cplx = None
        self.type = None
        self.signal = None
        self.signalN= None
        self.noise = None
        self.indexRange = opr_base.GetIndexRange(args.users_num, [1 for i in range(args.Q)])
        # receive signal
        self.signal = torch.zeros(1,1)  # 接收信号矩阵
        self.loc = {}                   # 保存用户位置
        self.uavloc={}
        self.distance=None   ### distance matrix M*N
        self.csi_cplxMN=None

    def Channel_Init(self, loc_update=True):       
        self.csi_cplx = {}
        self.csi_cplxMN = {}
        loc = {}
        Q = self.args.Q
        for q in range(Q):
            for k in range(Q):
                self.csi_cplxMN[str(q) + str(k)] = op.CompH(self.args.N, self.args.users_num[k], self.args.n_tx[k], self.args.n_rx[q])   ### H?
        
        if loc_update == True:
            # 考虑路损
            for q in range(Q):
                self.uavloc[str(q) + 'x'], self.uavloc[str(q) + 'y'], self.loc[str(q) + 'x'], self.loc[str(q) + 'y'], self.csi_cplxMN, self.distance = \
                    op.CsiGenerator(q, self.args, self.csi_cplxMN, np.array([0,0,0]))
        else:
            for q in range(Q):
                _, _, _, _, self.csi_cplxMN,self.distance = \
                    op.CsiGenerator(q, self.args, self.csi_cplxMN, np.array([0,0,0]), self.uavloc[str(q) + 'x'], self.uavloc[str(q) + 'y'], \
                        self.loc[str(q) + 'x'], self.loc[str(q) + 'y'])

        # # 图示
        # fig,ax = plt.subplots(1)
        # ax.set_aspect('equal')
        # ax.add_patch(patches.Circle((0,0,0), self.args.R, fill=False))
        # for q in range(Q):
        #     # x = -self.args.R * np.sign(self.args.BS_loc[q][0]) - (self.args.rect[0] / 2)
        #     # y = - (self.args.rect[1] / 2)
        #     # ax.add_patch(patches.Rectangle((x, y), self.args.rect[0], self.args.rect[1], fill = False))

        #     plt.scatter(loc[str(q) + 'x'], loc[str(q) + 'y'])
        #     plt.scatter(self.args.BS_loc[q][0], self.args.BS_loc[q][1], c='r',marker='v')
        # # plt.scatter(Sec_loc[0],Sec_loc[1],c='k',marker='+')
        # # plt.show()

    def Channel_Accumulation_Uplink(self, subsystems_list, optimizer, epoch):
        """Simulate the behavior of channel"""
        Q = self.args.Q
        M = self.args.users_num
        N = self.args.N
        sigma = self.args.sigma
        n_rx = self.args.n_rx
        grad = {}
        gradmean = {}
        x_slct = optimizer.x_slct  ### 分为矩阵和合计两个
        x_slctJ = optimizer.x_slctJ
        x_slctMN = optimizer.x_slctMN  ### 分为矩阵和合计两个
        x_slctJMN = optimizer.x_slctJMN
        bf_tx = optimizer.bf_tx
        bf_rx = optimizer.bf_rx  ### 全设为1向量（维度根据接收天线数）
        # gradmean = optimizer.gradmean
        # gradstd = optimizer.gradstd
        c_q = optimizer.c_q  ### 换成向量
        c_qN = optimizer.c_qN  ### 换成向量
        c_qJN = optimizer.c_qJN
        # Km = optimizer.Km
        # rho = optimizer.rho
        self.uavloc = optimizer.uavloc  ### 在优化后更新UAV位置
        J = self.args.J

        # 聚合成tensor
        for i, subsystem in enumerate(subsystems_list):
            grad[i] = []
            gradmean[i] = np.ones((self.args.users_num[i])) ####
            for j, device in enumerate(subsystem.Devices_list):
                grad[i].append(device.dvc_tx_signal)
                gradmean[i][j] = device.LocalNetwork.grad_mean
            grad[i] = np.array(grad[i])


            x_slctMN[i] = subsystem.PS_list[0].x_slctMN
            # x_slct[i] = subsystem.PS_list[0].x_slct
        gradlen = grad[0].shape[-1]
        self.signal = {}

        # 解调在Channel上进行
        for q, subsystem in enumerate(subsystems_list):
            # self.signal[q] = np.zeros((n_rx[q], gradlen), dtype=complex)
            self.signal[q] = np.zeros(gradlen, dtype=complex)
            self.signalN = {}
            # bf_tx1=bf_tx[q][]

            #### J
            j = epoch % J  # 比paper -1
            iota = epoch // J
            c_qN[q] = c_qJN[q][j]
            x_slctMN[q] = x_slctJMN[q][j]

            for n in range(N // J):  # 这里因J改变
                c_q[q] = c_qN[q][n]  ###  #### 修改zeta_j
                x_slct[q] = x_slctMN[q][:, n]  ###

                inter = 0
                signal = 0
                # self.signalN[n] = np.zeros((n_rx[q],gradlen), dtype=complex)
                self.signalN[n] = np.zeros(gradlen, dtype=complex)
                noise0 = cf.CplxNoiseGenerator(sigma ** 2, gradlen, n_rx[q])
                for k in range(Q):
                    self.csi_cplx[str(q) + str(k)] = self.csi_cplxMN[str(q) + str(k)][:, N // J * j + n]  ###

                    for jj in range(M[k]):
                        if k == q:
                            signal += np.squeeze(
                                c_q[q] * x_slct[k][jj] * bf_tx[q][jj][N // J * j + n] *
                                np.conj(np.transpose(bf_rx[q])) * self.csi_cplx[str(q) + str(k)][jj] * grad[k][jj],
                                axis=0
                            )
                        else:
                            inter += np.squeeze(
                                c_q[q] * x_slct[k][jj] * np.matmul(
                                    np.conj(np.transpose(bf_rx[q])), \
                                    np.matmul(self.csi_cplx[str(q) + str(k)][jj], grad[k][jj])), axis=0
                            )
                    # if np.sum(x_slct[k]) < 1:
                    #     noise=0    ### 飞行中间时若不覆盖，则不通信，也就没有噪声
                    # else:
                    #     noise = np.squeeze(
                    #             c_q[q] * np.matmul(
                    #             np.conj(np.transpose(bf_rx[q])), noise0), axis=0
                    #         )
                    noise = np.squeeze(
                        c_q[q] * np.matmul(
                            np.conj(np.transpose(bf_rx[q])), noise0), axis=0
                    )
                if self.args.noiseless == 0:
                    self.signalN[n] = signal + noise
                elif self.args.noiseless == 1:
                    self.signalN[n] = signal
                self.signal[q] = self.signal[q] + self.signalN[n]

            self.signal[q] = self.signal[q]  # / subsystem.PS_list[0].K

            # pwr_s = np.linalg.norm(signal)**2
            # pwr_i = np.linalg.norm(inter)**2
            # pwr_n = np.linalg.norm(noise)**2
            # print("     Task%d: SINR:%.2fdB,SIR:%.2fdB,SNR:%.2fdB"\
            #    % (q, 10*np.log10(pwr_s/(pwr_i+pwr_n)), 10*np.log10(pwr_s/pwr_i), 10*np.log10(pwr_s/pwr_n) ))
            subsystem.PS_list[0].ps_rx_signal = self.signal[q]
            subsystem.PS_list[0].ps_rx_signal = self.signal[q]  # 聚合在这里，Aggregation
            # subsystem.PS_list[0].r0 = opr_base.scaling(Km[q], gradmean[q], gradstd[q], x_slct[q], rho[q], gradlen*2)
            # (subsystem.sin_list).append([pwr_s, pwr_i, pwr_n])
            # (subsystem.sin_list).append([
            #     10*np.log10(pwr_s/(pwr_i+pwr_n)),
            #     10*np.log10(pwr_s/pwr_i),
            #     10*np.log10(pwr_s/pwr_n)]
            # )
            subsystem.PS_list[0].g_bar = np.zeros(J)
            subsystem.PS_list[0].scale = np.zeros(J)
            for j in range(J):
                for m in range(M[0]):
                    subsystem.PS_list[0].scale[j] = subsystem.PS_list[0].scale[j] + x_slctJ[q][j][m] * 1 / M[0]
                    for n in range(N // J):
                        subsystem.PS_list[0].g_bar[j] = subsystem.PS_list[0].g_bar[j] + x_slctJMN[q][j][m, n] * 1 / M[0] * gradmean[q][m]

            # subsystem.PS_list[0].g_bar = np.zeros(J)
            # subsystem.PS_list[0].scale = np.zeros(J)
            # for j in range(J):
            #     for m in range(M[0]):
            #         subsystem.PS_list[0].scale[j] = subsystem.PS_list[0].scale[j] +  1 / M[0]
            #         subsystem.PS_list[0].g_bar[j] = subsystem.PS_list[0].g_bar[j] +  1 / M[0] * gradmean[q][m]



    
    



    