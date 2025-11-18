
import argparse
import torch
import numpy as np

device = torch.device("cuda:0")
DEVICETYPE = torch.device("cuda:0")

def args_parser():
    # 先mnist再fmnist
    parser = argparse.ArgumentParser()
    # renew
    # parser.add_argument('--Sionna', type=bool, default=False, help='use Sionna lib')
    parser.add_argument('--network', type=str, default='Resnet18', help='[Resnet18, CNN]')
    parser.add_argument('--N', type=int, default=120, help="UAV trajectory slots")
    parser.add_argument('--lama', type=int, default=0.06, help="UAV serving range coefficient")
    parser.add_argument('--z0', type=int, default=50, help="UAV flying height")
    parser.add_argument('--t_n', type=float, default=1.0, help="UAV flying time slot")  # satisfy T_fly = N * t_n    delta
    parser.add_argument('--Vmax', type=int, default=50, help="UAV flying max speed")
    parser.add_argument('--Rwave', type=int, default= -250, help="UAV initialization trajectory radius wave") # cir250
    parser.add_argument('--fading', type=float, default=1e-6, help="Channel fading")  # paper 1e-6
    parser.add_argument('--sigma', type=float, default=1e-4/3.162, help="sqrt of noise power")  # paper 1e-4/3.162
    parser.add_argument('--big', type=int, default=230, help="optimization objective enlargement")
    parser.add_argument('--show', type=int, default=1, help="print the result (when in lab-server, choose 0)")
    parser.add_argument('--J', type=int, default=1, help='In one circle, model updates J times.')
    parser.add_argument('--mu', type=float, default=0.2, help='\mu in the paper')
    parser.add_argument('--scenario', type=str, default='A', help='A or B')
    parser.add_argument('--MC', type=int, nargs='+', default=[14],  help="Monte Carlo simulation indices (i values for main(i))")
    parser.add_argument('--Iter_alpha', type=int, default=10000, help="optimize alpha after how many times of opt of KU, default=1 meaning no difference of KU")
    parser.add_argument('--ComConstraint', type=bool, default=True, help='hard distance constraint of communication')
    parser.add_argument('--optfreq', type=int, default=10, help='frequency of opt')
    parser.add_argument('--cir', type=bool, default=False, help='-')
    parser.add_argument('--zeta_opt',type=bool, default=True, help='always True')

    # simulation
    parser.add_argument('--Q', type=int, default=1, help="tasks num")
    parser.add_argument('--task_list', type=list, default=["cifar10"], help="tasks list: fmnist, cifar10") # 
    parser.add_argument('--users_num', type=list, default=[20], help="User num of all tasks")
    parser.add_argument('--alpha', type=list, default=[1], help="The priority scalar of tasks")
    parser.add_argument('--T_max', type=int, default=10, help="Maximum times of optimization iterations")
    parser.add_argument('--Epsilon', type=float, default=1e-3, help="The converge threshold")
    parser.add_argument('--set', type=int, default=3, help=\
        r'=1 iid + equal dataset;\
          =2 iid + unequal dataset;\
          =3 noniid + equal dataset + 2classes')
    parser.add_argument('--Jmax', type=int, default=20, help="Gibbs sampling")
    parser.add_argument('--seed', type=int, default=80, help='random seed')
    parser.add_argument('--ref', type=float, default=1e6, help='scalar for numerical stability ')
    parser.add_argument('--save_freq', type=int, default=1, help='the freq of save figs')
    # Network
    parser.add_argument('--epochs', type=int, default=101, help="Rounds of training")
    parser.add_argument('--tp', type=int, default=1, help="Test period")
    parser.add_argument('--channel_use', type=float, default=1.0, help="Times of channel usage") # 无
    parser.add_argument('--sgd_times', type=float, default=5.0, help="sgd times per epoch")
    parser.add_argument('--lr', type=list, default=[0.01], help="Learning rates") #SGD: [1e-2, 1e-2, 1e-2]; momentum: [2e-3, 2e-3, 2e-3]; Adam: [1e-3, 1e-3, 1e-3]
    parser.add_argument('--target', type=list, default=[0.93], help="Target of accuracy") #   [0.93, 0.85, 0.85]
    parser.add_argument('--K', type=list, default=[60000], help="The maximum of per dataset")
    parser.add_argument('--N_class', type=int, default=5, help="The number of classes included by per dataset")  
    parser.add_argument('--datasets_num', type=list, default=[60000], help="The total num of per dataset") # 重复
    parser.add_argument('--datasets_num_dvc', type=list, default=[], help="The number of dataset on each device")
    # comm
    parser.add_argument('--n_tx', type=list, default=[1], help="The number of antennae on per device")
    parser.add_argument('--n_rx', type=list, default=[1], help="The number of antennae on per PS")
    parser.add_argument('--SNR', type=float, default=100.0, help="SNR")
    parser.add_argument('--PL', type=float, default=-2, help="Path loss exponent")   ### channel factor
    parser.add_argument('--noiseless', type=int, default=0, help="0: AWGN; 1: noiseless")
    parser.add_argument('--P_0', type=float, default=0.64, help="Transmission power")   
    parser.add_argument('--Diss', type=float, default=50.0, help="BS distance")
    # parser.add_argument('--R', type=float, default=100.0, help="Devices distribution radius")  
    # parser.add_argument('--rect', type=list, default=[80.0], help="Rectangular height and width")  
    parser.add_argument('--C', type=list, default=[0.5], help="The fraction of client selected")
    parser.add_argument('--scaling', type=bool, default=True, help="Scaling gradients at the PS")
    parser.add_argument('--r0', type=list, default=[5e-2], help="The fraction of client selected") 
    parser.add_argument('--correlation', type=int,default=0, help=\
        r"=0 Ideal;\
          =1 Statistics;\
          =2 Independent;\
          =3 Correlation;\
          =4 Constants")
    parser.add_argument('--rho', type=float, default=0.5, help="Empirical rho")   ### 1- rho0  eg. 0.9    #### noniid 0.5
    parser.add_argument('--method', type=int, default=14, help= \
        r"=0 Error Free;\
          =13 XYZ;\
          =14 XYZ_J")
    parser.add_argument('--optimizer', type=int, default=1, help=\
        r"=0 SGD;\
          =1 Momentum;\
          =2 Adam")
    parser.add_argument('--userselection', type=int, default=1, help=\
        r"=0 deterministc;\
          =1 stochastic;\
          =2 distance") # if 1 as well as open ComConstraint, meaning selecting top closest

    args = parser.parse_args()
   
    return args
