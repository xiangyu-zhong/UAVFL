# Import packages.
import cvxpy as cp
import numpy as np
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import time
import copy
import mosek
from scipy.optimize import minimize
import os

from . import common_func as cf
from . import operations as op
from . import FP_Operations as FPop
from . import FP_Operations_Liu as FPopLiu
from . import FP_Operations_DCDS as FPopDCDS
from . import FP_Operations_Cao as FPopCao
# from .smc.exec import smc_pf
print(INSTALLED_SOLVERS)


def XYZ(C, args, Km, rhoQ, gradmean, gradstd, uavloc, loc,epoch):
    Q=args.Q   ### 就直接考虑1的情况
    M1=args.users_num
    N=args.N
    sigma=args.sigma
    I=args.T_max
    P_0=np.sqrt(args.P_0/2)   ### 是根号p
    Epsilon=args.Epsilon
    lama=args.lama
    z=args.z0
    fading=(args.fading)**0.5  ### 1e-3
    Vmax=args.Vmax
    delta=args.t_n
    zetaQ= {}
    x_slct={}
    x_slctMN={}

    big=args.big

    # l=1

    for q in range(Q):
        u=np.concatenate((np.array(uavloc[str(q) + 'x']),np.array(uavloc[str(q) + 'y']))).reshape(2,-1).T
        v = np.concatenate((np.array(loc[str(q) + 'x']), np.array(loc[str(q) + 'y']))).reshape(2,-1).T
        uopt=u
        rho=rhoQ[q]

        M=M1[q]

        P = P_0 * np.ones((M, N))
        d2 = np.zeros((M, N))
        alpha = np.zeros((M, N))
        K = np.zeros((M, N))
        Del = np.zeros((M, N))
        rgradstdvarvec = np.zeros((M, 1))
        b=np.zeros((M,1))
        u0 = u[0, :]
        l=0
        qq=np.zeros((M,1),dtype=complex)
        Omg=np.zeros((M,M))



        # 初始化这些要优化的变量 K和alpha
        for n in range(0, N):
            for m in range(0, M):
                d2[m, n] = z ** 2 + np.linalg.norm(u[n, :] - v[m, :], ord=2) ** 2
                # alpha[m, n] = z ** 2 / (z ** 2 + lama * np.linalg.norm(u[n, :] - v[m, :]) ** 2)
                if np.linalg.norm(u[n, :] - v[m, :]) ** 2 <= z ** 2 / lama:
                    alpha[m, n] = 1
                else:
                    alpha[m, n] = 0
                # alpha[m, n] = 0.5  # 230511修改
                K[m, n] = alpha[m, n] * P[m, n] / np.sqrt(d2[m, n])
                # Del[m, n] = np.sqrt(d2[m, n])
                Del[m, n] = d2[m, n]  # 修改1230
                l=l+ Km[q][m]/args.K[q] * alpha[m, n]

        for m in range(0, M):
            rgradstdvarvec[m] = Km[q][m]/args.K[q] * gradstd[q][m]
            b[m]=Km[q][m]/args.K[q]
            qq[m]=b[m]*gradmean[q][m]
            # qq[m]=b[m]*np.real(gradmean[q][m])+1j*b[m]*np.imag(gradmean[q][m])

        Omg=(np.ones((M,1)) @ b.T - np.ones((M,M))).T @ qq @ qq.T.conjugate() @ (np.ones((M,1)) @ b.T - np.ones((M,M)))

        zeta = fading * l * np.linalg.inv( sigma ** 2 / 2 * np.identity(N) + fading ** 2 * K.T @ rho @ K) @ K.T @ rho @ rgradstdvarvec

        msenew = [2 * C / l ** 2 * (sigma ** 2 / 2 * zeta.T @ zeta
                                    + (l * rgradstdvarvec - fading * K @ zeta).T @ rho @ ( l * rgradstdvarvec - fading * K @ zeta)
                                    + np.ones((N,1)).T @ alpha.T @ Omg @ alpha @ np.ones((N,1)))]
        #
        # # 画图
        if args.show == 1:
            print("MSE initially:{}".format(msenew[0]))
        #
        # plt.scatter(u[:, 0], u[:, 1], s= 1 / lama * z ** 2 * np.ones((N)), c='r')
        # plt.scatter(u[:, 0], u[:, 1], s=5, c='w')
        # plt.plot(u[:, 0], u[:, 1], c='g')
        # plt.title("UAV initialization trajectory")
        # plt.scatter(v[:, 0], v[:, 1])
        # plt.savefig('../UAV_figures/UAV_trajectory_in_0_initialization.png')
        # plt.show()
        # plt.close()

        for i in range(0, I):

            Kcvx = cp.Variable((M, N))
            ucvx = cp.Variable((N, 2))
            Delcvx = cp.Variable((M, N))
            # scvx=cp.Variable((M,1))
            Picvx = cp.Variable((M, N))

            ui = u
            Deli = Del
            alphai = alpha
            # zetai=zeta

            constraints = [cp.norm(ucvx[0, :] - ucvx[N - 1, :]) ** 2 <= (int(Vmax * delta)) ** 2]  # Original
            constraints.append(ucvx[0, :] == u0)  # Original
            # constraints.append(Kcvx @ zeta @ zeta.T<= l * fading * rgradstdvarvec @ zeta.T)
            for n in range(0, N):
                if n != N - 1:
                    constraints.append(
                        cp.norm(ucvx[n + 1, :] - ucvx[n, :]) ** 2 <= (int(Vmax * delta)) ** 2)  # Original
                for m in range(0, M):
                    # -------------------------------- For (31b) < --------------------------------------------------------------------------------
                    # constraints.append(
                    #     Kcvx[m, n] <= alphai[m, n] * P[m, n]  * (np.sqrt(Deli[m, n]) ** (-1))
                    #     - alphai[m, n] * P[m, n]  / 2 / (Deli[m, n]) ** 1.5 * (Delcvx[m, n] - Deli[m, n])  )                # Original：④I的Del版，本质完全一样，Delcvx=(z ** 2 + cp.norm(ucvx[n, :] - v[m, :]) ** 2)
                    # constraints.append(z ** 2 + cp.norm(ucvx[n, :] - v[m, :]) ** 2
                    #                    <= Deli[m, n] ** 2 + 2 * Deli[m, n] * ( Delcvx[m, n] - Deli[m, n]))                  # Original：根据④I，没必要再加Del，就直接norm就是凸的
                    # constraints.append(
                    #     Kcvx[m, n] <=  P[m, n] * (np.sqrt(Deli[m, n]) ** (-1))
                    #     -  P[m, n] / 2 / (Deli[m, n]) ** 1.5 * (Delcvx[m, n] - Deli[m, n]))  # 对比Original只是没有了alphai
                    # constraints.append(Delcvx[m, n] >= (z ** 2 + cp.norm(ucvx[n, :] - v[m, :]) ** 2))
                    # constraints.append(Delcvx[m, n] >= np.sqrt(z ** 2 + np.linalg.norm(ui[n, :] - v[m, :]) ** 2)
                    #                    + 1 / (2 * np.sqrt(z ** 2 + np.linalg.norm(ui[n, :] - v[m, :]) ** 2))
                    #                    * (cp.norm(ucvx[n, :] - v[m, :]) ** 2 - np.linalg.norm(ui[n, :] - v[m, :]) ** 2))

                    # ④I 看看是不是等价
                    constraints.append(
                        Kcvx[m, n] <= alphai[m, n] * P[m, n] * (
                                    np.sqrt(z ** 2 + np.linalg.norm(ui[n, :] - v[m, :]) ** 2) ** (-1))
                        - alphai[m, n] * P[m, n] / 2 / (z ** 2 + np.linalg.norm(ui[n, :] - v[m, :]) ** 2) ** 1.5 * (
                                    cp.norm(ucvx[n, :] - v[m, :]) ** 2 - np.linalg.norm(
                                ui[n, :] - v[m, :]) ** 2))  # (31b)'

                    # -------------------------------- For (31c) > --------------------------------------------------------------------------------
                    # constraints.append(Kcvx[m, n] >= 0)                                                                     # Original
                    # 正确处理中……
                    constraints.append(
                        Kcvx[m, n] >= alphai[m, n] * P[m, n] * cp.inv_pos(Picvx[
                                                                              m, n]) - 4e-4)  # 提交稿的，尝试统一，注意Picvx= sqrt( (z ** 2 + cp.norm(ucvx[n, :] - v[m, :]) ** 2) )
                    constraints.append(
                        Picvx[m, n] <= np.sqrt(z ** 2 + np.linalg.norm(ui[n, :] - v[m, :]) ** 2)
                        + (ui[n, :] - v[m, :]).T @ (ucvx[n, :] - v[m, :]) / np.sqrt(
                            z ** 2 + np.linalg.norm(ui[n, :] - v[m, :]) ** 2))  # 提交稿的

                    # constraints.append(
                    #     Kcvx[m, n] >= alphai[m, n] * P[m, n] * cp.inv_pos(Picvx[m, n])  ) # 不对
                    # constraints.append(
                    #     Picvx[m, n] >= 0)
                    # constraints.append(
                    #     Picvx[m, n] <=  z ** 2 + np.linalg.norm(ui[n, :] - v[m, :]) ** 2
                    #     + 2* (ui[n, :] - v[m, :]).T @ (ucvx[n, :] - v[m, :])  )  # 不对



            obj = cp.Minimize( 2*C * (fading ** 2 * cp.quad_form(Kcvx @ zeta,
                                                                 rho) - 2 * l * fading * rgradstdvarvec.T @ rho @ Kcvx @ zeta))
            prob = cp.Problem(obj, constraints)
            mydict = {"MSK_IPAR_INFEAS_REPORT_AUTO": 1}
            # prob.solve(solver='MOSEK', verbose=True, mosek_params=mydict)
            prob.solve(solver='MOSEK', verbose=False) #, mosek_params=mydict)

            if args.show == 1:
                if (i + 1) % 1 == 0:
                    print("optimization iteration {}-th status: {}".format(i + 1, prob.status))


            # # test
            # K = Kcvx.value
            # u = ucvx.value
            # Del = Delcvx.value
            # right=np.zeros((M, N))
            # rightdirect = np.zeros((M, N))
            # for n in range(0, N):
            #     for m in range(0, M):
            #         right[m, n] = alpha[m, n] * P[m, n] / (z ** 2 + np.linalg.norm(u[n, :] - v[m, :], ord=2) ** 2) **0.5
            #         rightdirect[m, n] = alphai[m, n] * P[m, n]  * ((Deli[m, n]) ** (-1)) - alphai[m, n] * P[m, n]  / (Deli[m, n]) ** 2 * (Del[m, n] - Deli[m, n])
            #
            # test=right-K
            # testdirect= rightdirect - K



            K = Kcvx.value
            u = ucvx.value
            Del = Delcvx.value

            if args.show == 1:
                msetemp = 2 * C / l ** 2 * (
                            sigma ** 2 / 2 * zeta.T @ zeta + (l * rgradstdvarvec - fading * K @ zeta).T @ rho @ (
                            l * rgradstdvarvec - fading * K @ zeta) + np.ones((N,1)).T @ alpha.T @ Omg @ alpha @ np.ones((N,1)) )
                if (i + 1) % 1 == 0:
                    print("MSE_Ku: {}".format(msetemp))
                    # print("MINIMUtestdirect: {}".format(np.max(testdirect)))







            # 优化alpha
            alphacvx = cp.Variable((M, N))
            # Kcvx=cp.Variable((M,N))

            Ki = K
            alphai = alpha
            # ui=u
            gami = (2 * C / l ** 2 * (sigma ** 2 / 2 * zeta.T @ zeta
                                      + (l * rgradstdvarvec - fading * K @ zeta).T @ rho @ (
                                              l * rgradstdvarvec - fading * K @ zeta)
                                      + np.ones((N, 1)).T @ alpha.T @ Omg @ alpha @ np.ones((N, 1)))).real

            constraints = []
            for n in range(0, N):
                for m in range(0, M):
                    # constraints.append(
                    #     Kcvx[m, n] == alphacvx[m,n] * P[m, n]  / np.sqrt(z ** 2 + np.linalg.norm(ui[n, :] - v[m, :]) ** 2) )
                    constraints.append(alphacvx[m, n] >= 0)
                    constraints.append(alphacvx[m, n] <= 1)

            obj = cp.Minimize(
                # 2*C * fading ** 2 * cp.quad_form(Kcvx @ zeta, rho)
                -2 * 2 * C * b.T @ alphacvx @ np.ones((N, 1)) @ rgradstdvarvec.T @ rho @ Ki @ zeta
                # + cp.quad_form(alphacvx @ np.ones((N, 1)), 2 * C * Omg + 2 * C * rgradstdvarvec.T @ rho @ rgradstdvarvec * b @ b.T - gami * b @ b.T)
                + 2 * C * cp.quad_form(alphacvx @ np.ones((N, 1)), Omg)
                - 2 * gami * b.T @ alphai @ np.ones((N, 1)) @ b.T @ alphacvx @ np.ones((N, 1))
                # + 2 * C * rgradstdvarvec.T @ rho @ rgradstdvarvec @ cp.quad_form(alphacvx @ np.ones((N,1)), b @ b.T)
            )
            prob = cp.Problem(obj, constraints)
            mydict = {"MSK_IPAR_INFEAS_REPORT_AUTO": 1}
            prob.solve(solver='MOSEK', verbose=False)  # , mosek_params=mydict)

            if args.show == 1:
                if (i + 1) % 1 == 0:
                    print("optimization iteration {}-th status: {}".format(i + 1, prob.status))

            alpha = alphacvx.value

            l = 0
            l = b.T @ alpha @ np.ones((N, 1))
            for m in range(0, M):
                # l=l+ b[m] * alpha[m, n]
                for n in range(0, N):
                    d2[m, n] = z ** 2 + np.linalg.norm(u[n, :] - v[m, :], ord=2) ** 2
                    # # alpha[m, n] = z ** 2 / (z ** 2 + lama * np.linalg.norm(u[n, :] - v[m, :]) ** 2)
                    # alpha[m, n] = K[m, n] * np.sqrt(d2[m, n]) / P[m, n]
                    # if alpha[m, n] < 0.5:
                    #     alpha[m, n] = 0
                    # else:
                    #     alpha[m, n] = 1
                    # if np.linalg.norm(u[n, :] - v[m, :]) ** 2 <= z ** 2 / lama:
                    #     alpha[m, n] = 1
                    # else:
                    #     alpha[m, n] = 0
                    K[m, n] = alpha[m, n] * P[m, n] / np.sqrt(d2[m, n])
                    # Del[m, n] = d2[m, n]  # 20231230 新加

            if args.show == 1:
                msetemp = 2 * C / l ** 2 * (
                        sigma ** 2 / 2 * zeta.T @ zeta + (l * rgradstdvarvec - fading * K @ zeta).T @ rho @ (
                        l * rgradstdvarvec - fading * K @ zeta) + np.ones(
                    (N, 1)).T @ alpha.T @ Omg @ alpha @ np.ones(
                    (N, 1)))
                if (i + 1) % 1 == 0:
                    print("MSE_alpha: {}".format(msetemp))
                    print("MINIMUMalpha: {}".format(np.min(alpha)))










            # zeta without optimization
            zeta = fading * l * np.linalg.inv( sigma ** 2 / 2 * np.identity(N) + fading ** 2 * K.T @ rho @ K) @ K.T @ rho @ rgradstdvarvec

            # calculate the current MSE
            # M1=
            msecvx = 2 * C / l ** 2 * (sigma ** 2 / 2 * zeta.T @ zeta + (l * rgradstdvarvec - fading * K @ zeta).T @ rho @ (
                    l * rgradstdvarvec - fading * K @ zeta) + np.ones((N,1)).T @ alpha.T @ Omg @ alpha @ np.ones((N,1)))
            msenew.append(msecvx)

            if args.show == 1:
                if (i + 1) % 1 == 0:
                    print("MSE: {}".format(msecvx))

            # receive the UAV optimization trajectory
            uopt = np.append(uopt, u, axis=0)

            # if (i + 1) % 1 == 0:
            #     # plt.scatter(uopt[i*N:i*N+N,0],uopt[i*N:i*N+N,1],s=lam*z**2*np.ones((N)),c='r')
            #     plt.scatter(u[:, 0], u[:, 1], s=(1 - lam) / lam / lama * z ** 2 * np.ones((N)), c='r')
            #     plt.scatter(u[:, 0], u[:, 1], s=5, c='w')
            #     plt.plot(u[:, 0], u[:, 1], c='g')
            #     plt.title("UAV trajectory in {} iteration".format(i + 1))
            #     # plt.xlim(xmin=-350,xmax=350)
            #     # plt.ylim(ymin=-350, ymax=350)
            #     plt.scatter(v[:, 0], v[:, 1])
            #     plt.savefig('../UAV_figures/UAV_trajectory_in_{}_iteration.png'.format(i + 1))
            #     plt.show()
            #     plt.close()
            if np.abs((msecvx - msenew[i]) / msenew[i]) < Epsilon or i==I-1:
                plt.scatter(u[:, 0], u[:, 1], s=np.sqrt( 120 / lama * z ** 2) * np.ones((N)), c='r')
                plt.scatter(u[:, 0], u[:, 1], s=5, c='w')
                plt.plot(u[:, 0], u[:, 1], c='g')
                plt.title("UAV trajectory in {} iteration".format(i + 1))
                # plt.xlim(xmin=-350,xmax=350)
                # plt.ylim(ymin=-350, ymax=350)
                plt.scatter(v[:, 0], v[:, 1])
                plt.savefig('./output/opti/UAV ({})/Epoch{}_UAV_trajectory.png'.format(args.seed, epoch))
                # plt.show()
                plt.clf()
                # plt.close()
                plt.plot(np.array(msenew).reshape(-1))
                plt.title("MSE")
                # plt.ylim(ymin=0,ymax=6)
                plt.savefig('./output/opti/UAV ({})/Epoch{}_optMSEresult.png'.format(args.seed, epoch))
                # plt.show()
                plt.clf()
                break

        x_slct[q] = np.zeros(M, dtype=float)
        x_slctMN[q] = np.zeros((M, N), dtype=float)
        for m in range(M):
            for n in range(N):
                alpha[m, n] = K[m, n] * np.sqrt(d2[m, n]) / P[m, n]
                if alpha[m, n] >= 0.5:
                    # if np.linalg.norm(u[n, :] - v[m, :]) ** 2 <= z ** 2 / lama:
                    x_slctMN[q][m, n] = 1.0
                    x_slct[q][m] = x_slct[q][m] + 1.0
        zetaQ[q]=zeta
        uavloc[str(q) + 'x'] = u[:, 0].tolist()
        uavloc[str(q) + 'y'] = u[:, 1].tolist()

        # plt.plot(np.array(msenew).reshape(i + 2))
        # plt.title("MSE")
        # # plt.ylim(ymin=0,ymax=6)
        # plt.savefig('optMSEresult.png')
        # plt.show()

    return msecvx, x_slctMN, x_slct, uavloc, zetaQ


def XYZ_J(C, args, x_slctJMN, Km, rhoQ, gradmean, gradstd, uavloc, loc,epoch):   
    Q=args.Q   ### 就直接考虑1的情况
    M1=args.users_num
    N=args.N
    sigma=args.sigma
    I=args.T_max
    P_0=np.sqrt(args.P_0/2)   ### 是根号p
    Epsilon=args.Epsilon
    lama=args.lama
    z=args.z0
    fading=(args.fading)**0.5  ### 1e-3
    Vmax=args.Vmax
    delta=args.t_n
    zetaQ= {}
    zetaQJ = {}
    x_slct={}
    x_slctJ = {}
    x_slctMN={}
    # x_slctJMN = {}
    J=args.J
    iota = epoch // J

    big=args.big

    # l=1

    for q in range(Q):
        u=np.concatenate((np.array(uavloc[str(q) + 'x']),np.array(uavloc[str(q) + 'y']))).reshape(2,-1).T
        v = np.concatenate((np.array(loc[str(q) + 'x']), np.array(loc[str(q) + 'y']))).reshape(2,-1).T
        uopt=u
        rho=rhoQ[q]

        M=M1[q]

        P = P_0 * np.ones((M, N))
        d2 = np.zeros((M, N))
        alpha = [np.zeros((M, N//J)) for _ in range(J)]
        K = [np.zeros((M, N//J)) for _ in range(J)]
        Del = np.zeros((M, N))
        rgradstdvarvec = np.zeros((M, 1))  # 先不加，在于只能用j=0时的初始值
        b=np.zeros((M,1))
        u0 = u[0, :]
        l=np.zeros(J)
        qq=np.zeros((M,1))  # 先不加，同上
        Omg=np.zeros((M,M))  # 先不加，同上
        zeta = [np.zeros((N//J, 1)) for _ in range(J)]
        weight = np.zeros(J)

        # alpha = x_slctJMN[q]

        # 初始化这些要优化的变量 K和alpha
        for j in range(0,J):
            for n in range(0,N//J):  # 注意range和闭区间和[]的区别
                if args.userselection ==1:
                    distances = np.linalg.norm(v - u[N//J * j + n, :], axis=1)
                    top_k_indices = np.argsort(distances)[:int(args.users_num[q] * args.C[q])]
                for m in range(0, M):
                    d2[m, N//J * j + n] = z ** 2 + np.linalg.norm(u[N//J * j + n, :] - v[m, :], ord=2) ** 2
                    # alpha[m, n] = z ** 2 / (z ** 2 + lama * np.linalg.norm(u[N//J * j + n, :] - v[m, :]) ** 2)
                    if args.ComConstraint:
                        if np.linalg.norm(u[N//J * j + n, :] - v[m, :]) ** 2 <= z ** 2 / lama:
                            alpha[j][m, n] = 1
                        else:
                            alpha[j][m, n] = 0 + 1e-10
                        if args.userselection == 1:
                            if m in top_k_indices:
                                alpha[j][m, n] = 1
                            else:
                                alpha[j][m, n] = 0 + 1e-10
                    else:
                        # alpha[m, n] = 0.5  # 230511修改
                        alpha[j][m, n] = np.random.choice([0, 1])
                    K[j][m, n] = alpha[j][m, n] * P[m, N//J * j + n] / np.sqrt(d2[m, N//J * j + n])
                    # Del[m, n] = np.sqrt(d2[m, n])
                    Del[m, N//J * j + n] = d2[m, N//J * j + n]  # 修改1230
                    l[j]=l[j]+ Km[q][m]/args.datasets_num[q] * alpha[j][m, n]
            # 以防初始化alpha全是0的初始化
            if l[j] == 0:
                alpha[j][0,0] = 1
                K[j][0,0] = alpha[j][0,0] * P[0, N // J * j + 0] / np.sqrt(d2[0, N // J * j + 0])
                l[j] = l[j]+ Km[q][0]/args.datasets_num[q] * alpha[j][0,0]

        # for j in range(0,J):
        for m in range(0, M):
            rgradstdvarvec[m] = Km[q][m]/args.datasets_num[q] * gradstd[q][m]
            b[m]=Km[q][m]/args.datasets_num[q]
            qq[m]=b[m]*gradmean[q][m]
            # qq[m]=b[m]*np.real(gradmean[q][m])+1j*b[m]*np.imag(gradmean[q][m])

        # Omg=(np.ones((M,1)) @ b.T - np.ones((M,M))).T @ qq @ qq.T.conjugate() @ (np.ones((M,1)) @ b.T - np.ones((M,M)))
        Omg = (np.ones((M, 1)) @ b.T - np.eye(M, M)).T @ qq @ qq.T @ (
                    np.ones((M, 1)) @ b.T - np.eye(M, M))

        for j in range(0,J):
            if args.zeta_opt:
                zeta[j] = fading * l[j] * np.linalg.inv( sigma ** 2 / 2 * np.identity(N//J) + fading ** 2 * K[j].T @ rho @ K[j]) @ K[j].T @ rho @ rgradstdvarvec
            else:
                zeta[j] = 166.53171748532267 * np.ones((N//J,1))
                zeta[j] = np.ones((N // J, 1))
            # weight[j] = (1-args.mu*args.lr) ** (-(iota * J + j))   # 这里paper因为iota=1而体现不出来，哎呀其实是一样的
            weight[j] = (1 - args.mu * args.lr[q]) ** (-(j+1))

        msenew = [ sum( weight[j] * (2 * C / l[j] ** 2 * (sigma ** 2 / 2 * zeta[j].T @ zeta[j]
                                    + (l[j] * rgradstdvarvec - fading * K[j] @ zeta[j]).T @ rho @ ( l[j] * rgradstdvarvec - fading * K[j] @ zeta[j])
                                    + np.ones((N//J,1)).T @ alpha[j].T @ Omg @ alpha[j] @ np.ones((N//J,1)))) for j in range(0,J)) ]   # 向量化的式子


        #
        # # 画图
        if args.show == 1:
            print("MSE initially:{}".format(msenew[0]))
        #
        # plt.scatter(u[:, 0], u[:, 1], s= 1 / lama * z ** 2 * np.ones((N)), c='r')
        # plt.scatter(u[:, 0], u[:, 1], s=5, c='w')
        # plt.plot(u[:, 0], u[:, 1], c='g')
        # plt.title("UAV initialization trajectory")
        # plt.scatter(v[:, 0], v[:, 1])
        # plt.savefig('../UAV_figures/UAV_trajectory_in_0_initialization.png')
        # plt.show()
        # plt.close()

        for i in range(0, I):
            if not args.cir:
                Kcvx = [cp.Variable((M, N // J)) for _ in range(J)]
                # Kcvx = cp.Variable((J, M, N//J))
                ucvx = cp.Variable((N, 2))
                # Delcvx = cp.Variable((M, N))
                # scvx=cp.Variable((M,1))
                Picvx = cp.Variable((M, N))

                ui = u
                # Deli = Del
                alphai = alpha
                # zetai=zeta

                constraints = [cp.norm(ucvx[0, :] - ucvx[N - 1, :]) ** 2 <= (int(Vmax * delta)) ** 2]  # Original
                constraints.append(ucvx[0, :] == u0)  # Original
                # constraints.append(Kcvx @ zeta @ zeta.T<= l * fading * rgradstdvarvec @ zeta.T)
                for j in range(0, J):
                    for n in range(0,N//J):
                        if N//J * j + n != N - 1:
                            constraints.append(
                                cp.norm(ucvx[N//J * j + n + 1, :] - ucvx[N//J * j + n, :]) ** 2 <= (int(Vmax * delta)) ** 2)  # Original
                        for m in range(0, M):
                            # -------------------------------- For (31b) < --------------------------------------------------------------------------------
                            # ④I 看看是不是等价
                            constraints.append(
                                Kcvx[j][m, n] <= alphai[j][m, n] * P[m, N//J * j + n] * (
                                            np.sqrt(z ** 2 + np.linalg.norm(ui[N//J * j + n, :] - v[m, :]) ** 2) ** (-1))
                                - alphai[j][m, n] * P[m, N//J * j + n] / 2 / (z ** 2 + np.linalg.norm(ui[N//J * j + n, :] - v[m, :]) ** 2) ** 1.5 * (
                                            cp.norm(ucvx[N//J * j + n, :] - v[m, :]) ** 2 - np.linalg.norm(
                                        ui[N//J * j + n, :] - v[m, :]) ** 2))  # (31b)'

                            # -------------------------------- For (31c) > --------------------------------------------------------------------------------
                            # 正确处理中……
                            constraints.append(
                                Kcvx[j][m, n] >= alphai[j][m, n] * P[m, N//J * j + n] * cp.inv_pos(Picvx[
                                                                                      m, N//J * j + n]) - 4e-4)  # 提交稿的，尝试统一，注意Picvx= sqrt( (z ** 2 + cp.norm(ucvx[N//J * j + n, :] - v[m, :]) ** 2) )
                            # constraints.append(Kcvx[j][m,n]>=0)
                            constraints.append(
                                Picvx[m, N//J * j + n] <= np.sqrt(z ** 2 + np.linalg.norm(ui[N//J * j + n, :] - v[m, :]) ** 2)
                                + (ui[N//J * j + n, :] - v[m, :]).T @ (ucvx[N//J * j + n, :] - ui[N//J * j + n, :]) / np.sqrt(
                                    z ** 2 + np.linalg.norm(ui[N//J * j + n, :] - v[m, :]) ** 2))  # 提交稿的

                # # 加速度约束
                # constraints.append(cp.norm(ucvx[0, :] - 2 * ucvx[N - 1, :] + ucvx[N - 2, :]) ** 2 <= (
                #     int(15 * delta ** 2)) ** 2)
                # constraints.append(
                #     cp.norm(ucvx[1, :] - 2 * ucvx[0, :] + ucvx[N - 1, :]) ** 2 <= (int(15 * delta ** 2)) ** 2)
                # for j in range(0, J):
                #     for n in range(0, N // J):
                #         if N // J * j + n < N - 2:
                #             constraints.append(
                #                 cp.norm(ucvx[N // J * j + n + 2, :] - 2 * ucvx[N // J * j + n + 1, :] + ucvx[
                #                                                                                         N // J * j + n,
                #                                                                                         :]) ** 2 <= (
                #                     int(15 * delta ** 2)) ** 2)  # Original



                obj = cp.Minimize( cp.sum([weight[j] * (2*C * (fading ** 2 * cp.quad_form(Kcvx[j] @ zeta[j],
                                                                     rho) - 2 * l[j] * fading * rgradstdvarvec.T @ rho @ Kcvx[j] @ zeta[j])) for j in range(0, J) ]) )
                prob = cp.Problem(obj, constraints)
                # mydict = {"MSK_IPAR_INFEAS_REPORT_AUTO": 1}
                # mydict = {
                #     "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-6,  # 增加允许的相对误差
                #     "MSK_DPAR_INTPNT_CO_TOL_DFEAS": 1e-7,  # 放宽对 primal 可行性的容差
                #     "MSK_DPAR_INTPNT_CO_TOL_PFEAS": 1e-7  # 放宽对 dual 可行性的容差
                # }
                mydict = {
                    "MSK_IPAR_INTPNT_MAX_ITERATIONS": 8000,  # 增加最大迭代次数
                    # "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-4,  # 跑J40的时候开的下面三个
                    # "MSK_DPAR_INTPNT_CO_TOL_DFEAS": 1e-6,  # 放宽对 primal 可行性的容差
                    #     "MSK_DPAR_INTPNT_CO_TOL_PFEAS": 1e-6
                }

                # prob.solve(solver='MOSEK', verbose=True, mosek_params=mydict)
                prob.solve(solver='MOSEK', verbose=False, mosek_params=mydict)
                # prob.solve(solver='MOSEK', verbose=False) #, mosek_params=mydict)

                if args.show == 1:
                    if (i + 1) % 1 == 0:
                        print("optimization iteration {}-th status: {}".format(i + 1, prob.status))



                for j in range(J):
                    K[j] = Kcvx[j].value
                u = ucvx.value
                # Del = Delcvx.value

                if args.show == 1:
                    msetemp = sum( weight[j] * (2 * C / l[j] ** 2 * (sigma ** 2 / 2 * zeta[j].T @ zeta[j]
                                        + (l[j] * rgradstdvarvec - fading * K[j] @ zeta[j]).T @ rho @ ( l[j] * rgradstdvarvec - fading * K[j] @ zeta[j])
                                        + np.ones((N//J,1)).T @ alpha[j].T @ Omg @ alpha[j] @ np.ones((N//J,1)))) for j in range(0,J))
                    if (i + 1) % 1 == 0:
                        print("MSE_Ku: {}".format(msetemp))
                        # print("MINIMUtestdirect: {}".format(np.max(testdirect)))






            if (i+1) % args.Iter_alpha == 0:
                # 优化alpha
                alphacvx = [cp.Variable((M, N // J)) for _ in range(J)]
                # Kcvx=cp.Variable((M,N))

                Ki = K
                alphai = alpha
                # ui=u
                gami = np.zeros((J,1))
                for j in range(J):
                    gami[j] = (2 * C / l[j] ** 2 * (sigma ** 2 / 2 * zeta[j].T @ zeta[j]
                                          + (l[j] * rgradstdvarvec - fading * K[j] @ zeta[j]).T @ rho @ (
                                                  l[j] * rgradstdvarvec - fading * K[j] @ zeta[j])
                                          + np.ones((N//J, 1)).T @ alpha[j].T @ Omg @ alpha[j] @ np.ones((N//J, 1)))).real

                constraints = []
                for j in range(0, J):
                    for n in range(0,N//J):
                        for m in range(0, M):
                            # constraints.append(
                            #     Kcvx[m, n] == alphacvx[m,n] * P[m, n]  / np.sqrt(z ** 2 + np.linalg.norm(ui[N//J * j + n, :] - v[m, :]) ** 2) )
                            constraints.append(alphacvx[j][m, n] >= 0)
                            constraints.append(alphacvx[j][m, n] <= 1)

                obj = cp.Minimize(
                    cp.sum([weight[j] * (# 2*C * fading ** 2 * cp.quad_form(Kcvx @ zeta, rho)
                    -2 * 2 * C * b.T @ alphacvx[j] @ np.ones((N//J, 1)) @ rgradstdvarvec.T @ rho @ Ki[j] @ zeta[j]
                    # + cp.quad_form(alphacvx @ np.ones((N, 1)), 2 * C * Omg + 2 * C * rgradstdvarvec.T @ rho @ rgradstdvarvec * b @ b.T - gami * b @ b.T)
                    + 2 * C * cp.quad_form(alphacvx[j] @ np.ones((N//J, 1)), Omg)
                    - 2 * gami[j] * b.T @ alphai[j] @ np.ones((N//J, 1)) @ b.T @ alphacvx[j] @ np.ones((N//J, 1))
                    # + 2 * C * rgradstdvarvec.T @ rho @ rgradstdvarvec @ cp.quad_form(alphacvx @ np.ones((N,1)), b @ b.T)
                        ) for j in range(0,J)] )
                )
                prob = cp.Problem(obj, constraints)
                mydict = {"MSK_IPAR_INFEAS_REPORT_AUTO": 1}
                prob.solve(solver='MOSEK', verbose=False)  # , mosek_params=mydict)

                if args.show == 1:
                    if (i + 1) % 1 == 0:
                        print("optimization iteration {}-th status: {}".format(i + 1, prob.status))

                for j in range(J):
                    alpha[j] = alphacvx[j].value

                # l = 0
                min_alpha = float('inf')
                max_alpha = float('-inf')
                for j in range(0,J):
                    l[j] = b.T @ alpha[j] @ np.ones((N//J, 1))
                    for m in range(0, M):
                        # l=l+ b[m] * alpha[m, n]
                        for n in range(0,N//J):
                            d2[m, N//J * j + n] = z ** 2 + np.linalg.norm(u[N//J * j + n, :] - v[m, :], ord=2) ** 2
                            # # alpha[m, n] = z ** 2 / (z ** 2 + lama * np.linalg.norm(u[N//J * j + n, :] - v[m, :]) ** 2)
                            # alpha[m, n] = K[m, n] * np.sqrt(d2[m, n]) / P[m, n]
                            # if alpha[m, n] < 0.5:
                            #     alpha[m, n] = 0
                            # else:
                            #     alpha[m, n] = 1
                            if np.linalg.norm(u[N // J * j + n, :] - v[m, :]) ** 2 <= z ** 2 / lama:
                                min_alpha = min(min_alpha, alpha[j][m, n])
                                max_alpha = max(max_alpha, alpha[j][m, n])
                            # else:
                            #     alpha[m, n] = 0
                            K[j][m, n] = alpha[j][m, n] * P[m, N//J * j + n] / np.sqrt(d2[m, N//J * j + n])
                            # Del[m, n] = d2[m, n]  # 20231230 新加

                if args.show == 1:
                    msetemp = sum(weight[j] * (2 * C / l[j] ** 2 * (sigma ** 2 / 2 * zeta[j].T @ zeta[j]
                                                          + (l[j] * rgradstdvarvec - fading * K[j] @ zeta[j]).T @ rho @ (
                                                                      l[j] * rgradstdvarvec - fading * K[j] @ zeta[j])
                                                          + np.ones((N//J, 1)).T @ alpha[j].T @ Omg @ alpha[j] @ np.ones(
                                (N//J, 1)))) for j in range(0, J))
                    if (i + 1) % 1 == 0:
                        print("MSE_alpha: {}".format(msetemp))
                        print("MINIMUMalpha: {}， MAXalpha: {}".format(np.min(alpha), np.max(alpha)))



                # # 优化alpha 和 K 联合但解耦
                # alphacvx = [cp.Variable((M, N // J), nonneg=True) for _ in range(J)]
                # Kcvx = [cp.Variable((M, N // J)) for _ in range(J)]
                #
                # # for j in range(0, J):
                # #     for n in range(0, N // J):
                # #         for m in range(0, M):
                # #             alpha[j]=
                #
                # Ki = K
                # alphai = alpha
                # # ui=u
                # gami = np.zeros((J, 1))
                # for j in range(J):
                #     gami[j] = (2 * C / l[j] ** 2 * (sigma ** 2 / 2 * zeta[j].T @ zeta[j]
                #                                     + (l[j] * rgradstdvarvec - fading * K[j] @ zeta[
                #                 j]).T @ rho @ (
                #                                             l[j] * rgradstdvarvec - fading * K[j] @ zeta[j])
                #                                     + np.ones((N // J, 1)).T @ alpha[j].T @ Omg @ alpha[
                #                                         j] @ np.ones((N // J, 1)))).real
                #
                # constraints = []
                # for j in range(0, J):
                #     for n in range(0, N // J):
                #         for m in range(0, M):
                #             constraints.append(
                #                 Kcvx[j][m, n] == alphacvx[j][m,n] * P[m, N // J * j + n]  / np.sqrt(z ** 2 + np.linalg.norm(ui[N // J * j + n, :] - v[m, :]) ** 2) )
                #             constraints.append(alphacvx[j][m, n] >= 0)
                #             constraints.append(alphacvx[j][m, n] <= 1)
                #
                # obj = cp.Minimize(
                #     cp.sum([weight[j] * (   2*C * fading ** 2 * cp.quad_form(Kcvx[j] @ zeta[j], rho)
                #             - 2 * C * b.T @ alphacvx[j] @ np.ones((N // J, 1)) @ rgradstdvarvec.T @ rho @ Ki[
                #         j] @ zeta[j]
                #             - 2 * C * b.T @ alphai[j] @ np.ones((N // J, 1)) @ rgradstdvarvec.T @ rho @ Kcvx[
                #                                 j] @ zeta[j]
                #             # + cp.quad_form(alphacvx @ np.ones((N, 1)), 2 * C * Omg + 2 * C * rgradstdvarvec.T @ rho @ rgradstdvarvec * b @ b.T - gami * b @ b.T)
                #             + 2 * C * cp.quad_form(alphacvx[j] @ np.ones((N // J, 1)), Omg)
                #             - 2 * gami[j] * b.T @ alphai[j] @ np.ones((N // J, 1)) @ b.T @ alphacvx[
                #                 j] @ np.ones((N // J, 1))
                #         # + 2 * C * rgradstdvarvec.T @ rho @ rgradstdvarvec @ cp.quad_form(alphacvx @ np.ones((N,1)), b @ b.T)
                #     ) for j in range(0, J)])
                # )
                # prob = cp.Problem(obj, constraints)
                # mydict = {"MSK_IPAR_INFEAS_REPORT_AUTO": 1}
                # prob.solve(solver='MOSEK', verbose=False)  # , mosek_params=mydict)
                #
                # if args.show == 1:
                #     if (i + 1) % 1 == 0:
                #         print("optimization iteration {}-th status: {}".format(i + 1, prob.status))
                #
                # for j in range(J):
                #     alpha[j] = alphacvx[j].value
                #     K[j] = Kcvx[j].value
                #     alpha[j] = np.clip(alpha[j], 0, 1)
                #
                #
                # # l = 0
                # for j in range(0, J):
                #     l[j] = b.T @ alpha[j] @ np.ones((N // J, 1))
                #     for m in range(0, M):
                #         # l=l+ b[m] * alpha[m, n]
                #         for n in range(0, N // J):
                #             d2[m, N // J * j + n] = z ** 2 + np.linalg.norm(u[N // J * j + n, :] - v[m, :],
                #                                                             ord=2) ** 2
                #             # # alpha[m, n] = z ** 2 / (z ** 2 + lama * np.linalg.norm(u[N//J * j + n, :] - v[m, :]) ** 2)
                #             # alpha[m, n] = K[m, n] * np.sqrt(d2[m, n]) / P[m, n]
                #             # if alpha[m, n] < 0.5:
                #             #     alpha[m, n] = 0
                #             # else:
                #             #     alpha[m, n] = 1
                #             # if np.linalg.norm(u[N//J * j + n, :] - v[m, :]) ** 2 <= z ** 2 / lama:
                #             #     alpha[m, n] = 1
                #             # else:
                #             #     alpha[m, n] = 0
                #             # K[j][m, n] = alpha[j][m, n] * P[m, N // J * j + n] / np.sqrt(d2[m, N // J * j + n])
                #             # Del[m, n] = d2[m, n]  # 20231230 新加
                #
                # if args.show == 1:
                #     msetemp = sum(weight[j] * (2 * C / l[j] ** 2 * (sigma ** 2 / 2 * zeta[j].T @ zeta[j]
                #                                                     + (l[j] * rgradstdvarvec - fading * K[j] @
                #                                                        zeta[j]).T @ rho @ (
                #                                                             l[j] * rgradstdvarvec - fading * K[
                #                                                         j] @ zeta[j])
                #                                                     + np.ones((N // J, 1)).T @ alpha[
                #                                                         j].T @ Omg @ alpha[j] @ np.ones(
                #                 (N // J, 1)))) for j in range(0, J))
                #     if (i + 1) % 1 == 0:
                #         print("MSE_alpha: {}".format(msetemp))
                #         print("MINIMUMalpha: {}， MAXalpha: {}".format(np.min(alpha), np.max(alpha)))





            if args.zeta_opt:
                # zeta without optimization
                for j in range(0, J):
                    zeta[j] = fading * l[j] * np.linalg.inv(
                        sigma ** 2 / 2 * np.identity(N//J) + fading ** 2 * K[j].T @ rho @ K[j]) @ K[
                                  j].T @ rho @ rgradstdvarvec

            # calculate the current MSE
            # M1=
            msecvx = sum(weight[j] * (2 * C / l[j] ** 2 * (sigma ** 2 / 2 * zeta[j].T @ zeta[j]
                                                      + (l[j] * rgradstdvarvec - fading * K[j] @ zeta[j]).T @ rho @ (
                                                                  l[j] * rgradstdvarvec - fading * K[j] @ zeta[j])
                                                      + np.ones((N//J, 1)).T @ alpha[j].T @ Omg @ alpha[j] @ np.ones(
                            (N//J, 1)))) for j in range(0, J))
            msenew.append(msecvx)

            if args.show == 1:
                if (i + 1) % 1 == 0:
                    print("MSE: {}".format(msecvx))

            # receive the UAV optimization trajectory
            uopt = np.append(uopt, u, axis=0)

            # if (i + 1) % 1 == 0:
            #     # plt.scatter(uopt[i*N:i*N+N,0],uopt[i*N:i*N+N,1],s=lam*z**2*np.ones((N)),c='r')
            #     plt.scatter(u[:, 0], u[:, 1], s=(1 - lam) / lam / lama * z ** 2 * np.ones((N)), c='r')
            #     plt.scatter(u[:, 0], u[:, 1], s=5, c='w')
            #     plt.plot(u[:, 0], u[:, 1], c='g')
            #     plt.title("UAV trajectory in {} iteration".format(i + 1))
            #     # plt.xlim(xmin=-350,xmax=350)
            #     # plt.ylim(ymin=-350, ymax=350)
            #     plt.scatter(v[:, 0], v[:, 1])
            #     plt.savefig('../UAV_figures/UAV_trajectory_in_{}_iteration.png'.format(i + 1))
            #     plt.show()
            #     plt.close()
            if np.abs((msecvx - msenew[i]) / msenew[i]) < Epsilon or i==I-1:
                plt.scatter(u[:, 0], u[:, 1], s= 32*np.sqrt( z ** 2 / lama ) * np.ones((N)), c='r')
                plt.scatter(u[:, 0], u[:, 1], s=5, c='w')
                plt.plot(u[:, 0], u[:, 1], c='g')
                plt.title("UAV trajectory in {} iteration".format(i + 1))
                # plt.xlim(xmin=-350,xmax=350)
                # plt.ylim(ymin=-350, ymax=350)
                plt.scatter(v[:, 0], v[:, 1])

                ### 自己创建
                folder_path = './output/opti/UAV ({})'.format(args.seed)
                os.makedirs(folder_path, exist_ok=True)

                plt.savefig('./output/opti/UAV ({})/Epoch{}_UAV_trajectory.png'.format(args.seed, epoch))
                # plt.show()
                plt.clf()
                # plt.close()
                plt.plot(np.array(msenew).reshape(-1))
                plt.title("MSE")
                # plt.ylim(ymin=0,ymax=6)
                plt.savefig('./output/opti/UAV ({})/Epoch{}_optMSEresult.png'.format(args.seed, epoch))
                # plt.show()
                plt.clf()
                break

        #### 赋值部分有变化
        x_slctJ[q] = [np.zeros(M) for _ in range(J)]
        x_slctJMN[q] = [np.zeros((M, N // J)) for _ in range(J)]
        zetaQJ[q] = [np.zeros(N // J) for _ in range(J)]
        for j in range(0, J):
            for m in range(M):
                for n in range(N // J):
                    # alpha[j][m, n] = K[j][m, n] * np.sqrt(d2[m, N // J * j + n]) / P[m, N // J * j + n]  # 不影响
                    if alpha[j][m, n] >= 0.5:
                        # if np.linalg.norm(u[N//J * j + n, :] - v[m, :]) ** 2 <= z ** 2 / lama:
                        x_slctJMN[q][j][m, n] = 1.0
                        x_slctJ[q][j][m] = x_slctJ[q][j][m] + 1.0
            zetaQJ[q][j] = zeta[j].flatten()


        # x_slct[q] = np.zeros(M, dtype=float)
        # x_slctMN[q] = np.zeros((M, N), dtype=float)
        # for m in range(M):
        #     for n in range(N):
        #         alpha[m, n] = K[m, n] * np.sqrt(d2[m, n]) / P[m, n]
        #         if alpha[m, n] >= 0.5:
        #             # if np.linalg.norm(u[N//J * j + n, :] - v[m, :]) ** 2 <= z ** 2 / lama:
        #             x_slctMN[q][m, n] = 1.0
        #             x_slct[q][m] = x_slct[q][m] + 1.0
        # zetaQ[q]=zeta
        uavloc[str(q) + 'x'] = u[:, 0].tolist()
        uavloc[str(q) + 'y'] = u[:, 1].tolist()

        # plt.plot(np.array(msenew).reshape(i + 2))
        # plt.title("MSE")
        # # plt.ylim(ymin=0,ymax=6)
        # plt.savefig('optMSEresult.png')
        # plt.show()

    return msecvx, x_slctJMN, x_slctJ, uavloc, zetaQJ






