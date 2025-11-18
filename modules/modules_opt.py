import cvxpy as cp
import numpy as np
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS
from matplotlib import pyplot as plt 
import pandas as pd

import modules.operations_base as opr_base
# import optlib.FP_Operations_Liu as FPopLiu
import optlib.operations as op
# import optlib.FP_Operations as FPop
import optlib.common_func as cf
import optlib.opt as opt
import copy
import torch


class Optimizer:
    """波束赋形优化器"""
    def __init__(self, args):
        self.args = args
        self.x_slct = {}
        self.x_slctJ = {}
        self.x_slctMN = {}  ###
        self.x_slctJMN = {}  ###
        self.selected_sum = {}
        self.x_slct_1d = None
        self.bf_tx = {}
        self.bf_rx = {}
        self.Km = {}
        self.c_q = {}
        self.c_qN = {}  ###
        self.c_qJN = {}  ###
        self.rho = {}

        self.gradstd = {}
        self.gradmean = {}
        self.g_bar = {}

        self.uavloc = {}
        self.loc = {}
        self.C = 0

    def Optimizer_Init(self, channel):
        args = self.args
        Q = args.Q
        # 变量初始化
        # 把用户选择量传进来
        # self.x_slct = {q:np.ones(args.users_num[q]) for q in range(Q)}
        # self.selected_sum = {q:np.dot(args.datasets_num_dvc[q], self.x_slct[q]) for q in range(Q)}   ### 暂时不鸟他看看
        # self.x_slct_1d = np.concatenate(([self.x_slct[q] for q in range(Q)]))  ### 用不到
        # self.bf_tx = {
        #     q: [np.sqrt(1/2) * (np.random.randn(args.n_tx[q],1) + \
        #         np.random.randn(args.n_tx[q],1) * 1j) for i in range(args.users_num[q])] \
        #         for q in range(Q)
        # }
        # self.bf_tx = {
        #     q: [[np.sqrt(args.P_0 / 2) * ( np.conj(channel.csi_cplxMN[str(q) + str(q)][i][j])
        #                                   / np.linalg.norm(channel.csi_cplxMN[str(q) + str(q)][i][j])
        #                                   ) for j in range(args.N) ] for i in range(args.users_num[q])] for q in range(Q)  # 相位抵消！！！
        self.bf_tx = {
            q: [[np.sqrt(args.P_0 / 2) for j in range(args.N)] for i in range(args.users_num[q])] for q in range(Q)
        }  ### SISO
        # self.bf_rx = {
        #     q: np.sqrt(1/2) * (np.random.randn(args.n_rx[q],1) + \
        #         np.random.randn(args.n_rx[q],1) * 1j) for q in range(Q)
        # }
        self.bf_rx = {
            q: np.ones((args.n_rx[q], 1)) for q in range(Q)
        }
        # self.bf_rx = np.sqrt(1/2) * (np.random.randn(args.n_rx[0],1) + \
            # np.random.randn(args.n_rx[0],1) * 1j)
        # 参数赋值
        self.H = channel.csi_cplx   ### 用不到
        self.loc=channel.loc
        self.uavloc =channel.uavloc

    
    def Optimizer_Collector(self, subsystems_list):
        args = self.args
        Q = args.Q
        self.C=0
        for i,subsystems in enumerate(subsystems_list):
            self.Km[i] = args.datasets_num_dvc[i]
            # self.gradmean[i] = np.ones((self.args.users_num[i]), dtype=complex)
            self.gradmean[i] = np.ones((self.args.users_num[i]))
            self.gradstd[i] = np.ones((args.users_num[i]))
            for j,device in enumerate(subsystems.Devices_list):
                self.gradmean[i][j] = device.LocalNetwork.grad_mean
                self.gradstd[i][j] = device.LocalNetwork.grad_std
                self.C=np.maximum(self.C, len(device.LocalNetwork.grad_normal))
            # self.g_bar[i] = np.sum(self.Km[i]*self.gradmean[i])*np.array(1+1j)

            # 计算相关性
            self.rho[i] = subsystems.corr_matrix
        
    def Optimizer_Optimize(self, method,epoch):
        args = self.args
        Q = args.Q
        J = args.J
        N = args.N
        x_slct = self.x_slct
        x_slctMN=self.x_slctMN  ###
        x_slctJMN = self.x_slctJMN  ###
        bf_tx = self.bf_tx
        bf_rx = self.bf_rx
        H = self.H
        Km = self.Km
        rho = self.rho
        gradstd = self.gradstd
        gradmean = self.gradmean
        selected_sum = self.selected_sum
        uavloc=self.uavloc

        if epoch == 0:
            x_slctJMN = {q: [np.random.choice([0, 1], size=(args.users_num[q], N//J)) for j in range(J)]  for q in range(Q)}

 
        if method == 1:
            # method == proposed
            [obj_new, _, self.bf_tx, self.bf_rx, self.c_q] = opt.WeightedSumOpt(args, x_slct, bf_tx, bf_rx, H, Km, rho, gradstd, selected_sum)
        elif method == 2:
            # method == SNRLiu
            [obj_new, _, self.bf_tx, self.bf_rx, self.c_q] = opt.SNRLiuOpt(args, x_slct, bf_tx, bf_rx, H, Km, rho, gradstd, selected_sum)
        elif method == 3:
            # method == Gibbspro
            [obj_new, self.x_slct, self.bf_tx, self.bf_rx, self.c_q] = opt.Gibbs(args, H, Km, rho, gradstd)
        elif method == 4:
            # method == GibbsLiu
            [obj_new, self.x_slct, self.bf_tx, self.bf_rx, self.c_q] = opt.LiuGibbs(args, bf_tx, bf_rx, H, Km, rho, gradstd)
            # [obj_new, self.x_slct, self.bf_tx, self.bf_rx, self.c_q] = opt.LiuGibbs(args, H, Km, rho, gradstd)
        elif method == 5:
            # method == MIMOOpt
            [obj_new, _, self.bf_tx, self.bf_rx, self.c_q] = opt.MIMOOpt(args, x_slct, bf_tx, bf_rx, H, Km, rho, gradstd, selected_sum)
        elif method == 6:
            # method == DCDSOpt
            [obj_new, self.x_slct, self.bf_tx, self.bf_rx, self.c_q] = opt.DCDSOpt(args, x_slct, bf_tx, bf_rx, H, Km, rho, gradstd, selected_sum)
        elif method == 7:
            # method == CaoOpt
            [obj_new, _, self.bf_tx, self.bf_rx, self.c_q] = opt.CaoOpt(args, x_slct, bf_tx, bf_rx, H, Km, rho, gradstd, selected_sum)
        elif method == 8:
            # method == NoOpt
            [obj_new, _, _, _, self.c_q] = opt.NoOpt(args, x_slct, bf_tx, bf_rx, H, Km, rho, gradstd, selected_sum)
        elif method == 9:
            # method == NoOpt
            [obj_new, self.x_slct, self.bf_tx, self.bf_rx, self.c_q] = opt.SMC(args, H, Km, rho, gradstd)
        elif method == 10:
            # method == LiuOpt
            [obj_new, _, self.bf_tx, self.bf_rx, self.c_q] = opt.SCALiu(args, x_slct, bf_tx, bf_rx, H, Km, rho, gradstd, selected_sum)
        elif method == 11:
            [obj_new, self.x_slctMN, self.x_slct, self.uavloc, self.c_qN] = opt.XYZwithoutL(self.C, args, Km, rho, gradstd, uavloc, self.loc,epoch)
        elif method == 12:
            [obj_new, self.x_slctMN, self.x_slct, self.uavloc, self.c_qN] = opt.XYZwithoutLrho(self.C, args, Km, rho, gradstd, uavloc, self.loc,epoch)
        elif method == 13:
            [obj_new, self.x_slctMN, self.x_slct, self.uavloc, self.c_qN] = opt.XYZ(self.C, args, Km, rho, gradmean, gradstd, uavloc, self.loc,epoch)
        elif method == 14:
            [obj_new, self.x_slctJMN, self.x_slctJ, self.uavloc, self.c_qJN] = opt.XYZ_J(self.C, args, x_slctJMN, Km, rho,
                                                                                         gradmean, gradstd, uavloc,
                                                                                         self.loc, epoch)

        print(obj_new)

    def Optimizer_ResultBroadcast(self, subsystems_list):
        M = self.args.users_num[0]
        N = self.args.N
        J = self.args.J
        # args = self.args
        # Q = args.Q
        for i, subsystems in enumerate(subsystems_list):
            # subsystems.PS_list[0].x_slct = self.x_slct[i]
            subsystems.PS_list[0].x_slctJ = self.x_slctJ[i]
            # subsystems.PS_list[0].x_slctMN = self.x_slctMN[i]    ###
            subsystems.PS_list[0].x_slctJMN = self.x_slctJMN[i]
            # # selected_sum
            # subsystems.PS_list[0].K = np.sum(self.args.datasets_num_dvc[i] * self.x_slctJMN[i][m,:])  # 没用

            # 此处需要将选择结果：注意此处是三个向量，而不是N*1维的矩阵
            # 这里对应公式18的g_bar
            # subsystems.PS_list[0].g_bar = \
            #     np.real(np.sum(self.Km[i] * self.gradmean[i] * 1)) / self.args.K[0]

            # 挪到channel后面更新
            # subsystems.PS_list[0].g_bar = np.zeros(J)
            # subsystems.PS_list[0].scale = np.zeros(J)
            # for j in range(J):
            #     for m in range(M):
            #         subsystems.PS_list[0].scale[j] = subsystems.PS_list[0].scale[j] + self.x_slctJ[i][j][m] * 1/M
            #         for n in range(N // J):
            #             subsystems.PS_list[0].g_bar[j] = subsystems.PS_list[0].g_bar[j] + self.x_slctJMN[i][j][m,n] * 1/M * self.gradmean[i][m]



            # self.c_q[i]=self.c_qN[i][0]   ## 随便返回一个   #### 也没用呗
            # subsystems.PS_list[0].c_q = self.c_q[i]

            # subsystems.PS_list[0].c_qN = self.c_qN[i]   ###  你这个都没定义，所以也没用呗
            subsystems.PS_list[0].c_qJN = self.c_qJN[i]
            subsystems.PS_list[0].uavloc[str(i) + 'x'] = self.uavloc[str(i) + 'x']  ###
            subsystems.PS_list[0].uavloc[str(i) + 'y'] = self.uavloc[str(i) + 'y']  ###

            # subsystems.PS_list[0].bf_rx = self.bf_rx[i]
            for j, device in enumerate(subsystems.Devices_list):
                # device.bf_tx = self.bf_tx[i][j]
                device.bf_tx = self.bf_tx[i][j][0]  ### 随便取一个点送回去，要不然就是N个功率点   // 满功率发送下
    
    def Optimizer_Ref(self, subsystems_list):  ### 不用管
        args = self.args
        Q = args.Q
        c_q = self.c_q
        x_slct = self.x_slct
        bf_tx = self.bf_tx
        bf_rx = self.bf_rx
        H = self.H
        Km = self.Km
        gradstd = self.gradstd
        gradmean = self.gradmean
        g_bar = self.g_bar
        m = args.users_num

        grad = {}
        grad_c = {}
        for i,subsystem in enumerate(subsystems_list):
            grad[i] = []
            grad_c[i] = []
            for j in range(m[i]):       
                grad[i].append(subsystem.Devices_list[j].LocalNetwork.grad_basic)
                grad_c[i].append(opr_base.tensor2npcplx(torch.squeeze(grad[i][j], dim=0).cpu()))
            grad_c[i] = np.array(grad_c[i]).reshape((-1,m[i]))
        gradlen = np.max(grad_c[0].shape)

        for q in range(Q):
            cf.MSETest(q, Q, c_q, x_slct, bf_tx, bf_rx, H, Km, args.users_num, \
                grad_c, gradmean, gradstd, g_bar, gradlen, args.sigma, args.n_rx)


        
    
