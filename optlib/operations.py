import cvxpy as cp
import numpy as np
from matplotlib import pyplot as plt 
import pandas as pd
import copy

def CsiGenerator(q, args, H, Sec_loc, udx=None, udy=None, dx = None, dy = None):
    Q = args.Q
    R = args.R           # Devices distribution radius
    alpha_direct = args.PL  # User-BS Path loss exponent liu:3.76
    fc = 915 * 10 ** 6  # carrier frequency, wavelength lambda_c=3.0*10**8/fc
    BS_Gain = 10 ** (5.0 / 10)  # BS antenna gain
    User_Gain = 10 ** (0.0 / 10)  # User antenna gain
    # BS_loc = np.array(args.BS_loc[q])  # location of the BS/PS
    M = args.users_num[q]
    N_TX = args.n_tx[q]
    N_RX = args.n_rx[q]

    if type(dx) == type(None) or type(dx) == type(None):
        # UAV trajectory initialization and device location setting
        if args.scenario == 'A':
            v0 = pd.read_csv('./input/devicelocationnn4.csv', index_col=0)  # devicelocationn7 for best
        if args.scenario == 'B':
            v0 = pd.read_csv('./input/devicelocationnnn3.csv', index_col=0)  # devicelocationn7 for best
        v = v0.values[:, 0:args.users_num[q]].T  # (x,y)*M
        dx=v[:,0]
        dy=v[:,1]

        ### initialize UAV trajectory according to devices location     circular
        rsum = int(args.N * args.Vmax * args.t_n / 2 / np.pi) + args.Rwave
        rxy = np.mean(v, axis=0)  # cal. the  weight center of devices as the circle center
        # if args.cir:
        rxy[0] = 885 - rsum  ### 保持起始点不变
        rxy[1] = -10
        u = np.zeros((args.N, 2))
        for i in range(0, args.N):
            u[i, 0] = int(rxy[0] + rsum * np.cos(2 * np.pi * i / (args.N)))  # - 3*np.pi/4))
            u[i, 1] = int(rxy[1] + rsum * np.sin(2 * np.pi * i / (args.N)))  # - 3*np.pi/4) )
        u0 = u[0, :]
        uopt = u
        udx = u[:, 0]
        udy = u[:, 1]

        #### empirical circle
        # rxy = np.mean(v, axis=0)
        # initial_point = np.array([885, -10])
        # radius = np.linalg.norm(rxy - initial_point)
        # theta = np.linspace(0, 2 * np.pi, args.N)
        # udx = rxy[0] + radius * np.cos(theta)
        # udy = rxy[1] + radius * np.sin(theta)

        # # half devices have x\in[-20,0]
        # # Type1: one circle
        # r = np.sqrt(np.random.rand(M)) * R
        # theta = np.random.uniform(-np.pi, np.pi, size=M)
        # dx = r*np.cos(theta)
        # dy = r*np.sin(theta)
        
        # Type2: two rectangular
        # dx = np.random.uniform(-0.5, 0.5, size=M) * (args.rect[0]) - R * np.sign(args.BS_loc[q][0])
        # dy = np.random.uniform(-0.5, 0.5, size=M) * (args.rect[1])

        # Type3: two circles
        # r = np.random.rand(int(np.round(M / 2))) * R
        # theta = np.random.uniform(-np.pi, np.pi, size=int(np.round(M / 2)))
        # dx1 = r*np.cos(theta) + BS_loc[0]
        # dy1 = r*np.sin(theta) + BS_loc[1]
        # if args.set == 1:
        #     # Setting 1:
        #     # For M=40, K=750
        #     # the other half devices also have x\in[-20,0]
        #     r = np.random.rand(int(M - np.round(M / 2))) * R
        #     theta = np.random.uniform(-np.pi, np.pi, size=int(M - np.round(M / 2)))
        #     dx2 = r*np.cos(theta) + BS_loc[0]
        #     dy2 = r*np.sin(theta) + BS_loc[1]
        # else:
        #     # Setting 2:
        #     # Half (random selected) devices have Uniform[1000,2000] data, the other half have Uniform[100,200] data
        #     # the other half devices have x\in[100,120]
        #     r = np.random.rand(int(M - np.round(M / 2))) * R
        #     theta = np.random.uniform(-np.pi, np.pi, size=int(M - np.round(M / 2)))
        #     dx2 = r*np.cos(theta) + Sec_loc[0]
        #     dy2 = r*np.sin(theta) + Sec_loc[1]
        # concatenate all the x locations
        # dx = np.concatenate((dx1, dx2))
        # dy = np.concatenate((dy1, dy2))
    d_direct=np.zeros((M,args.N))
    for k in range(Q):
        for n in range(args.N):
            d_direct[:,n] = ( args.z0**2 + (dx - udx[n]) ** 2 + (dy - udy[n]) ** 2 ) ** 0.5
            for m in range(M):
                H[str(k) + str(q)][m, n] *= np.conj(H[str(k) + str(q)][m,n]) ####
                # H[str(k) + str(q)][m, n] *=np.sqrt(args.P_0/2) ####
                H[str(k) + str(q)][m,n] *= (args.fading * d_direct[m,n] ** alpha_direct) ** 0.5



        # BS_loc_temp = args.BS_loc[k]
        # # distance of direct User-BS channel
        # d_direct = ((dx - BS_loc_temp[0]) ** 2 + (dy - BS_loc_temp[1]) ** 2 + BS_loc_temp[2] ** 2) ** 0.5
        # # # Path loss of direct channel
        # PL_direct = BS_Gain * User_Gain * (3 * 10 ** 8 / fc / 4 / np.pi / d_direct) ** alpha_direct
        # # channels coefficents (after scaling)
        # for i in range(M):
        #     H[str(k) + str(q)][i] *= PL_direct[i] ** 0.5
    return udx, udy, dx,dy,H, d_direct

def CompH(N,M_arange, N_TX, N_RX):
    H=np.zeros((M_arange,N),dtype=complex)
    for n in range(N):
        for m in range(M_arange):
            # thet=2*np.pi*np.random.random(N_RX,N_TX)
            thet = 2 * np.pi * np.random.random() ### SISO
            H[m,n]=np.cos(thet) +  np.sin(thet) * 1j

    # H = [np.sqrt(1/2) * (np.random.randn(N_RX,N_TX) + np.random.randn(N_RX,N_TX) * 1j) for i in range(M_arange)]
    return H

def real2cplx(grad_r, grad_c):
    gradlen = np.size(grad_r)
    for i in range(int(gradlen/2)):
        grad_c[i] = grad_r[2*i] + 1j * grad_r[2*i+1]
    return grad_c

def InequalSizeGen(M):
    Km = np.random.randint(1000, high=2001, size=(int(M)))
    Km2 = np.random.randint(100, high=201, size=(int(M / 2)))
    lessuser = np.random.choice(M, size=int(M / 2), replace=False)
    Km[lessuser] = Km2
    return Km.astype(np.int64)

def CompBfTxInv(args, q, x_slct, bf_rx, H, Km, gradstd, c_q):
    bf_tx_inv = []
    bf_tx_inv_norm = np.zeros(args.users_num[q])
    for i in range(args.users_num[q]):
        bf_tx_inv.append(Km[q][i]* gradstd[q][i]/\
            (c_q[q] * np.linalg.norm(\
                np.matmul(np.conj(np.transpose(bf_rx[q])), H[str(q)*2][i]))**2)\
            *(np.matmul(np.conj(np.transpose(H[str(q)*2][i])), bf_rx[q])) )
    # for i in range(args.users_num[q]):
        # bf_tx_inv_norm[i] = np.linalg.norm(bf_tx_inv[i]) * x_slct[q][i]
        # if bf_tx_inv_norm[i] > 1: # P_0
            # bf_tx_inv[i] /= bf_tx_inv_norm[i] 
    return bf_tx_inv

def CompCosSim(a, b):
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a.dtype == 'complex' or b.dtype == 'complex':
        cos = np.dot(np.conj(a.T),b)/(a_norm * b_norm)
        # cos = np.dot(a,b)/(a_norm * b_norm)
        cos = np.abs(cos)
    else:
        cos = np.dot(a,b)/(a_norm * b_norm)
    return cos

def x_dict21d(x_slct, Q):
    x_slct_1d = np.concatenate(([x_slct[q] for q in range(Q)]))
    return x_slct_1d

def x_1d2dict(x_slct_1d, m, Q):
    x_slct = {
        q: x_slct_1d[int(np.sum(m[0: q])) : int(np.sum(m[0: q])+m[q])].astype(float) for q in range(Q)
    }
    return x_slct