import cvxpy as cp
import numpy as np
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS
from matplotlib import pyplot as plt 
import pandas as pd

def CplxNoiseGenerator(sigma_square, gradlen, arange):
    noise = np.zeros((arange,gradlen), dtype=complex)
    for i in range(arange):
        noise[i,:] = np.transpose(np.random.normal(loc=0, scale=np.sqrt(sigma_square/2), \
            size=(gradlen,2)).view(np.complex128))
        # noise[i,:] = np.sqrt(sigma_square/2) * (np.random.randn(1,gradlen) + np.random.randn(1,gradlen) * 1j)
    return noise

def CompRho(grad, M_arange, gradlen, gradmean, gradstd):
    rho = np.zeros((M_arange, M_arange))
    for i in range(M_arange):
        for j in range(M_arange):
            temp = 0
            for g in range(gradlen):
                temp += (grad[g,i]-gradmean[i])/gradstd[i] * (grad[g,j]-gradmean[j])/gradstd[j]
            rho[i,j] = temp/gradlen
    return rho

def CompMSE(q, Q, c_q, x_slct, bf_tx, bf_rx, H, Km, selected_sum, m, rho, gradstd, sigma):
    """
    :param q    : selected subsystem index
    :param Q    : number of subsystems
    :param c_q  : aggregation scalar
    :param x_slct: users selection vector 
    :param bf_tx: transmission beamforming vector
    :param bf_rx: reception beamforming vector
    :param H    : channel csi matrix
    :param Km   : datasets num on each devices
    :param selected_sum: selected sum datasets
    :param m    : numbers of devices
    :param rho  : correlation matrix
    :param gradstd: gradient std
    :param sigma: noise std
    """
    MSE = 0
    for k in range(Q):
        for i in range(m[k]):
            for j in range(m[k]):
                if q == k:
                    MSE += x_slct[q][i] * x_slct[q][j] * rho[q][i,j] \
                        *(Km[q][j]*gradstd[q][j] - \
                            c_q[q]*np.matmul(np.conj(np.transpose(bf_tx[q][j])),\
                            np.matmul(np.conj(np.transpose(H[str(q)*2][j])) , bf_rx[q]))
                        )\
                        *(Km[q][i]*gradstd[q][i] - \
                            c_q[q]*np.matmul(np.conj(np.transpose(bf_rx[q])),\
                            np.matmul(H[str(q)*2][i], bf_tx[q][i]))
                        )
                else:
                    MSE += x_slct[k][i] * x_slct[k][j] * c_q[q]**2 * rho[k][i,j] \
                        * np.matmul(np.conj(np.transpose(bf_tx[k][j])),\
                            np.matmul(np.conj(np.transpose(H[str(q)+str(k)][j])) , bf_rx[q]))\
                        * np.matmul(np.conj(np.transpose(bf_rx[q])),\
                            np.matmul(H[str(q)+str(k)][i], bf_tx[k][i]))
    MSE += c_q[q]**2 * (sigma**2/2) * np.linalg.norm(bf_rx[q])**2
    MSE = np.real( MSE/(selected_sum[q]**2) )
    return MSE.reshape(-1,)

def ADMM_QCQP(Variable_len, H, b, P_0):
    x = np.random.randn(Variable_len,1) + np.random.randn(Variable_len,1) * 1j
    z = np.random.randn(Variable_len,1) + np.random.randn(Variable_len,1) * 1j
    u = np.random.randn(Variable_len,1) + np.random.randn(Variable_len,1) * 1j
    # H = 1e5*H
    # b = 1e5*b
    rho_ADMM = 1e-3
    obj_value = 1000
    while True:
        x = np.matmul(
            np.linalg.inv(H+rho_ADMM*np.identity(Variable_len)),\
            b + rho_ADMM * (z+u)
        )
        zeta = x - u
        # 高效更新方式
        mu = np.max([0, np.linalg.norm(zeta)/np.sqrt(P_0)-1])
        z = 1/(1+mu)*zeta

        # 求解器
        # z_temp = cp.Variable((Variable_len,1), complex=True)
        # obj = cp.Problem(
        # cp.Minimize(
        #     cp.norm(z_temp-zeta)**2),
        #     [cp.norm(z_temp)**2<=P_0]
        # )
        # prob = obj.solve(solver = cp.MOSEK)
        # z = z_temp.value

        u = u+z-x
        obj_temp = np.real(
                np.matmul(np.conj(np.transpose(x)), np.matmul(H,x)) - \
                2*np.real(np.matmul(np.conj(np.transpose(b)), x))
            )
        # print(obj_temp)
        if np.abs(obj_value - obj_temp)  <= 1e-6:
            break
        else:
            obj_value = obj_temp
    return x

def CVX_QCQP(Variable_len, H, b, P_0):
    x_temp = cp.Variable((Variable_len,1), complex=True)
    H_RI = np.block([[H.real, -H.imag],[H.imag, H.real]])
    x_RI = cp.vstack((cp.real(x_temp), cp.imag(x_temp)))
    obj = cp.Minimize(
        cp.quad_form(x_RI, H_RI) - 2*cp.real(np.conj(np.transpose(b)) @ x_temp)
    )
    prob = cp.Problem(obj,[cp.norm(x_temp)**2 <= P_0])
    prob.solve(solver = cp.MOSEK, verbose = False)
    return x_temp.value


def MSETest(q, Q, c_q, x_slct, bf_tx, bf_rx, H, Km, m, grad, gradmean, gradstd, g_bar, gradlen, sigma, n_rx):
    """
    :param q    : selected subsystem index
    :param Q    : number of subsystems
    :param c_q  : aggregation scalar 
    :param bf_tx: transmission beamforming vector
    :param bf_rx: reception beamforming vector
    :param H    : channel csi matrix
    :param Km   : datasets num on each devices 
    :param K    : total numbers of datasets on devices 
    :param m    : numbers of devices
    :param grad : gradient 
    :param gradmean: gradient mean
    :param gradstd: gradient std
    :param g_bar: mean of gradient mean
    :param gradlen: gradient length
    :param sigma: noise std
    :param n_rx : reception antenna number
    """
    K = {q: np.dot(Km[q], x_slct[q] ) for q in range(Q)}
    signal0 = np.zeros((1,gradlen), dtype=complex)
    noise0 = CplxNoiseGenerator(sigma**2, gradlen, n_rx[q])
    r0 = np.zeros(gradlen, dtype=complex)
    r_hat0 = np.zeros(gradlen, dtype=complex)

    for i in range(gradlen):
        # 干扰
        for k in range(Q):
            for j in range(m[q]):
                signal0[:,i]  += np.squeeze(
                    x_slct[k][j] * np.matmul(np.conj(np.transpose(bf_rx[q])), \
                        np.matmul(H[str(q) + str(k)][j], bf_tx[k][j])) \
                    *(grad[k][i,j]-gradmean[k][j])/gradstd[k][j], axis=0)
    signal0 += np.matmul(np.conj(np.transpose(bf_rx[q])), noise0)

    # 估计
    for i in range(gradlen):
        r_hat0[i] = signal0[:,i]*c_q[q] + g_bar[q] 
    r_hat0 /= K[q]
    # 计算MSE
    for j in range(m[q]):
        r0 += x_slct[k][j] * Km[q][j]*grad[q][:,j]
    r0 /= K[q]

    MSE0_array = np.linalg.norm(r0-r_hat0)**2 /(gradlen*2)
    print('MSE{} = {}dB,{}'.format(q, 10*np.log10(np.mean(MSE0_array)), np.mean(MSE0_array)))
    print('r_hat0 = {}, r0 = {}'.format(r_hat0[0], r0[0]))