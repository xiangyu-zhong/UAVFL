
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy as np
import torch
import time
import gc

import modules.modules_environments as mm_env
import modules.modules_machine as mm_mch
import modules.modules_networks as mm_net
import datasets.dataset_base as data_base
import browser.options_base as op_base
import browser.visual_base as vs_base

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
OUTPUT_PATH = "./output/"

def main(MC):
    export_time = []
    export_acc = []
    export_corr = []
    export_mse = []
    export_cos = []
    export_r = []
    export_r_hat = []
    export_power = []
    export_sin = []
    export_corr_matrix = []
    export_uavloc=[]
    # --------create args & dataset & environment-------- #
    # 参数
    args = op_base.args_parser()
    args.seed = MC
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    for task_num, task in enumerate(args.task_list):
        if task == 'cifar10':
            args.datasets_num[task_num] = 50000
            args.K[task_num] = 50000

    # 仿真环境
    environment = mm_env.Environments(args)
    # 数据集来源
    dataset,args = data_base.DatasetGet(args)

    # # ----------------create channel subsystems----------- #
    channel = mm_env.Channel(args)

    # # ----------------initialize the env------------------ #
    environment.Environments_Init(channel, dataset)

    # # -----------------start epochs----------------------- #
    print('Users num:{}; Batch size:{}; J:{}'.format(args.users_num, args.datasets_num_dvc[0][0] // args.sgd_times, args.J))
    print('Datanum list:.{}'.format(args.datasets_num_dvc))
    for epoch in range(args.epochs):
        print("MC:%d, iota:%d, epochs:%d:" % (MC, epoch // args.J + 1, epoch))
        t0 = time.time()
        environment.Environments_Iter(channel, epoch)
        t1 = time.time()
        print("     Duration: %f, Estimated left time: %d h %d min" % ((t1 - t0), int((args.epochs - epoch - 1) * (t1 - t0) / args.J // 3600),int((args.epochs - epoch - 1) * (t1 - t0) / args.J % 3600 // 60)))
        environment.Environments_Test(epoch)
        # Export time
        export_time.append(t1 - t0)
        data = pd.DataFrame(data = np.array(export_time))
        data.to_csv(OUTPUT_PATH + 'time/time' + str(MC) + '.csv',index = True)

        # -----------------Export data------------------------ #
        export_acc = vs_base.pd_one_epoch_to_csv(args, environment, export_acc, 'acc', epoch, 
            1, np.ones(args.Q), PATH = OUTPUT_PATH + 'accuracy/accuracy' + str(MC) + '.csv')
        export_corr = vs_base.pd_one_epoch_to_csv(args, environment, export_corr, 'corr', epoch, 
            int(args.channel_use), np.ones(args.Q)*2, PATH = OUTPUT_PATH + 'corr/corr' + str(MC) + '.csv')
        export_corr_matrix = vs_base.pd_one_epoch_to_csv(args, environment, export_corr_matrix, 'corr_matrix', epoch, 
            int(args.channel_use), np.ones(args.Q)*2, PATH = OUTPUT_PATH + 'corr_matrix/corr_matrix' + str(MC) + '.csv')
        export_mse = vs_base.pd_one_epoch_to_csv(args, environment, export_mse, 'mse', epoch, 
                int(args.channel_use), np.ones(args.Q), PATH = OUTPUT_PATH + 'mse/mse' + str(MC) + '.csv')
        export_cos = vs_base.pd_one_epoch_to_csv(args, environment, export_cos, 'cos', epoch, 
                int(args.channel_use), np.ones(args.Q), PATH = OUTPUT_PATH + 'cos/cos' + str(MC) + '.csv')
        if args.method != 0:
            export_r = vs_base.pd_one_epoch_to_csv(args, environment, export_r, 'r', epoch, 
                int(args.channel_use), np.ones(args.Q), PATH = OUTPUT_PATH + 'r/r' + str(MC) + '.csv')
            export_r_hat = vs_base.pd_one_epoch_to_csv(args, environment, export_r_hat, 'r_hat', epoch, 
                int(args.channel_use), np.ones(args.Q), PATH = OUTPUT_PATH + 'r_hat/r_hat' + str(MC) + '.csv')
            export_power = vs_base.pd_one_epoch_to_csv(args, environment, export_power, 'power', epoch, 
                int(args.channel_use), args.users_num, PATH = OUTPUT_PATH + 'power/power' + str(MC) + '.csv')
            # export_sin = vs_base.pd_one_epoch_to_csv(args, environment, export_sin, 'sin', epoch,
            #     int(args.channel_use), 3, PATH = OUTPUT_PATH + 'sin/sin' + str(MC) + '.csv')
        if args.method>=11:
            export_uavloc = vs_base.pd_one_epoch_to_csv(args, environment, export_uavloc, 'uavloc', epoch,
                                                       int(args.channel_use), args.users_num,
                                                       PATH=OUTPUT_PATH + 'uavloc/uavloc' + str(MC) + '.csv')

# for i in range(2, 5):
#     main(i)
#     gc.collect()  # 清理缓存

args = op_base.args_parser()
for i in args.MC:
    main(i)
    gc.collect()  # 清理缓存