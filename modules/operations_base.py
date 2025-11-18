
import copy
import math
import torch
import heapq
import numpy as np
# import cupy as np
import scipy.io as sio
import pandas as pd
from browser.options_base import DEVICETYPE
from functools import wraps
import time


OUTPUT_PATH = "./output/"

# TODO 整个都可以优化
def sparsity_cuda(g, k):
    # k-Sparsify local gradients
    vector = copy.deepcopy(g).to(DEVICETYPE)
    zero_index = abs(vector).argsort()[0, 0:vector.numel() - int(k)]
    for i in zero_index:
        vector[0, i] = 0
    return vector


# def sparsity_heap(g, k):
#     # k-Sparsify local gradients
#     vector = g.squeeze()
#     zero_index = map(list(abs(vector)).index, heapq.nsmallest(int(k), abs(vector)))
#     for i in zero_index:
#         vector[i] = 0
#     return vector.unsqueeze()


def AMP_cuda(A, y, iteration):
    M = A.size()[0]
    N = A.size()[1]
    OLS = sio.loadmat("./datasets/OptimumLambdaSigned.mat")
    lamb = torch.tensor(np.interp(M / N, OLS['delta_vec'][0], OLS['lambda_opt'][0]))

    z_t = copy.deepcopy(y).to(DEVICETYPE)
    x_t = torch.zeros(N).to(DEVICETYPE)

    # mse = []
    for i in range(iteration):
        x_hat = torch.matmul(A.t(), z_t).to(DEVICETYPE) + x_t
        sigma_hat = math.sqrt(1 / M * (torch.norm(z_t, 2) ** 2))
        x_t = (abs(x_hat) > lamb * sigma_hat) * (abs(x_hat) - lamb * sigma_hat) * torch.sign(x_hat).to(DEVICETYPE)
        # mse.append((np.linalg.norm(x - x_t) ** 2) / N)
        # mse.append((np.linalg.norm(x - x_t) ** 2) / (np.linalg.norm(x) ** 2))
        z_t = y - torch.matmul(A, x_t).to(DEVICETYPE) + 1 / M * z_t * len(x_t[abs(x_t) > 0])
    # data = pd.DataFrame(data=mse)
    # data.to_csv(OUTPUT_PATH + 'mse.csv',index = False)
    return x_t

# def AMP_other(A, y, iteration, x):
#     At = np.transpose(A)
#     M = A.shape[0]
#     N = A.shape[1]
#     delta = M / N
#     OLS = sio.loadmat("./datasets/OptimumLambdaSigned.mat")
#     lamb = np.interp(delta, OLS['delta_vec'][0], OLS['lambda_opt'][0])
#
#     z_t = copy.deepcopy(y)
#     x_t = np.zeros([N, 1])
#
#     mse = []
#     for iter in range(iteration):
#         x_hat = At.dot(z_t) + x_t
#         sigma_hat = math.sqrt(1 / M * (np.linalg.norm(z_t) ** 2))
#         x_t = (np.abs(x_hat) > lamb * sigma_hat) * (np.abs(x_hat) - lamb * sigma_hat) * np.sign(x_hat)
#         # mse.append((np.linalg.norm(x - x_t) ** 2) / N)
#         mse.append((np.linalg.norm(x - x_t) ** 2) / (np.linalg.norm(x) ** 2))
#         z_t = y - A.dot(x_t) + 1 / M * z_t * len(x[np.abs(x_t) > 0])
#     return torch.tensor(x_t).to(DEVICETYPE)


def GetIndexRange(users_num_list, ps_num_list):
    IndexRange = []
    pre = 0
    subsystemNum = len(users_num_list)
    for i in range(subsystemNum):
        for j in range(ps_num_list[i]):
            IndexRange.append((pre, pre+users_num_list[i]-1))
            pre += users_num_list[i]
    return IndexRange

def SINR(signal, noise, interference = 0):
    sinr = (np.linalg.norm(signal) ** 2) / ((np.linalg.norm(noise) ** 2) + (np.linalg.norm(interference) ** 2))
    return 10*math.log(sinr, 10)

def SNR2sigma(signal, snr):
    var =  (10 ** (snr/10))
    return (np.linalg.norm(signal) ** 2)/var

def NMSE(input_signal, ref):
    return (np.linalg.norm(ref - input_signal) ** 2) / (np.linalg.norm(ref) ** 2)

def MSE(input_signal, ref):
    return (np.linalg.norm(ref - input_signal) ** 2) / len(ref)

def tensor2npcplx(a):
    """一维tensor转numpy复数"""
    N_temp = int(a.shape[0]/2)
    N = int(np.ceil(a.shape[0]/2))
    # 循环 3s
    # b = np.zeros((2,N))
    # for i in range(N):
    #     b[0,i] = a[i*2]
    #     b[1,i] = a[i*2+1]
    # c = np.array(b[0,:] + 1j*b[1,:])
    # 加速代码 0.03s
    index_even = torch.arange(0, a.shape[0], 2, dtype=torch.long)
    index_odd = torch.arange(1, a.shape[0], 2, dtype=torch.long)
    b = np.zeros(N, dtype = complex)
    b.real[:N] = a[index_even]
    b.imag[:N_temp] = a[index_odd]
    return b

def np2tensorcplx(a):
    """numpy复数转一维tensor"""
    N = len(a)
    c = torch.zeros(2*N, dtype=torch.double)
    # 循环 0.72s
    # for i in range(N):
    #     c[2*i]   = torch.tensor(np.real(a[i]))
    #     c[2*i+1] = torch.tensor(np.imag(a[i]))
    # 下标 0.003s
    index = torch.arange(0, N, 1, dtype=torch.long)
    c[index*2] = torch.tensor(a.real)
    c[index*2+1] = torch.tensor(a.imag)
    return c.to(DEVICETYPE)

def CplxNormal(mean, var, size):
    """产生复高斯分布"""
    a = np.random.randn(int(size[0]),int(size[1])) * math.sqrt(var/2) + mean
    b = np.random.randn(int(size[0]),int(size[1])) * math.sqrt(var/2) + mean
    return np.array(a + 1j*b)

def col2csv(data, path):
    export_data = np.array(data).T
    export_data = pd.DataFrame(data=export_data) 
    export_data.to_csv(path, index = True)

def random_choice(args, optimizer):
    m = np.array(args.users_num) * np.array(args.C)
    m = m.astype(np.int32)
    for q in range(args.Q):
        optimizer.x_slct[q] = np.zeros(args.users_num[q]) 
        optimizer.x_slct[q][np.random.choice(args.users_num[q], m[q], replace = False)] = 1.0 
    return optimizer

def all_choice(args, optimizer):
    for q in range(args.Q):
        optimizer.x_slct[q] = np.ones(args.users_num[q], dtype=float) 
    return optimizer

def dis_choice(args, optimizer, channel):
    # m = np.array(args.users_num) * np.array(args.C)
    # m = m.astype(np.int32)
    dis_temp=channel.distance
    for q in range(args.Q):
        # dis_temp = np.sqrt((channel.loc[str(q) + 'x']-channel.uavloc[str(q) + 'x'])**2 + channel.loc[str(q) + 'y']**2)
        # index_sort = np.argsort(dis_temp)
        # index_slct = index_sort[:m[q]]
        # optimizer.x_slct[q] = np.zeros(args.users_num[q], dtype=float)
        # optimizer.x_slct[q][index_slct] = 1.0

        optimizer.x_slct[q] = np.zeros(args.users_num[q], dtype=float)
        optimizer.x_slctMN[q]= np.zeros((args.users_num[q],args.N),dtype=float)
        for m in range(args.users_num[q]):
            for n in range(args.N):
                if dis_temp[m,n] <= np.sqrt(1 / args.lama) * args.z0:
                    optimizer.x_slctMN[q][m,n]=1.0
                    optimizer.x_slct[q][m] = optimizer.x_slct[q][m] + 1.0
        # for i, dist in enumerate(dis_temp):
        #     if dist <= np.sqrt(1/args.lama) * args.z0:
        #         optimizer.x_slct[q][i] = 1.0
    return optimizer

def scaling(Km, gradmean, gradstd, x_slct, rho, gradlen):   ### 没用到
    K = np.dot(Km, x_slct)
    M = len(Km)
    r0 = 0
    for i in range(M):
        for j in range(M):
            r0 += 2 * Km[i] * Km[j] * x_slct[i] * x_slct[j] * \
                gradlen * (np.real(rho[i,j]) * gradstd[i] * gradstd[j] \
                    + np.real(gradmean[i]) * np.real(gradmean[j]))
    r0 /= K**2
    r0 = np.sqrt(r0)
    return  r0