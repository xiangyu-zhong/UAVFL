

import torch
from itertools import chain
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn
from collections import OrderedDict
import numpy as np
import copy
import re

import modules.operations_base as opr_base
from browser.options_base import DEVICETYPE


class BasicBlock(nn.Module):  # 完全正确0726 ## for Resnet
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BasicNetwork(torch.nn.Module):
    # def __init__(self, args, data):
    def __init__(self, args, data, block=BasicBlock, num_blocks=[2, 2, 2, 2], feature_dim=128, input_channel=3,
                 num_classes=10):
        super(BasicNetwork, self).__init__()
        self.args = args
        self.num_classes = num_classes
        self.data = data
        self.loss = []
        self.accuracy = []
        self.bs = 0

        # SGD
        self.lr = 0
        # Momentum
        self.v = 0
        # Adam
        self.mt = 0
        self.vt = 0

        if self.args.network == 'CNN':  # for MNIST series
            # MLP
            # self.network1_linear = torch.nn.Linear(784, 30).to(DEVICETYPE)
            # self.network2_relu = torch.nn.ReLU().to(DEVICETYPE)
            # self.network3_linear = torch.nn.Linear(30, 10).to(DEVICETYPE)
            # self.network4_softmax = torch.nn.Softmax(dim=1).to(DEVICETYPE)
            # self.criterion = torch.nn.MSELoss().to(DEVICETYPE)

            # LeNet
            # self.conv = nn.Sequential(OrderedDict([
            #     ('conv1', nn.Conv2d(1, 6, 5)), # in_channels, out_channels, kernel_size
            #     ('maxpool1', nn.MaxPool2d(2, 2)), # kernel_size, stride
            #     ('relu1', nn.ReLU()),
            #     ('conv2', nn.Conv2d(6, 16, 5)),
            #     ('maxpool2', nn.MaxPool2d(2, 2)),
            #     # ('dropout1', nn.Dropout2d()),
            #     ('batch1', nn.BatchNorm2d(16)),
            #     ('relu2', nn.ReLU()),
            # ])).to(DEVICETYPE)
            # self.fc = nn.Sequential(OrderedDict([
            #     ('fc1',nn.Linear(16*4*4, 120)),
            #     ('relu3',nn.ReLU()),
            #     ('fc2',nn.Linear(120, 84)),
            #     ('relu4',nn.ReLU()),
            #     ('fc3',nn.Linear(84, 10))
            # ])).to(DEVICETYPE)

            # liu
            self.conv = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(1, 16, 5)),  # in_channels, out_channels, kernel_size  # 对由多个输入平面组成的输入信号进行二维卷积
                ('maxpool1', nn.MaxPool2d(2, 2)),  # kernel_size, stride  # 最大值池化
                ('relu1', nn.ReLU()),  # 激活函数max（0，x）
                ('conv2', nn.Conv2d(16, 32, 5)),
                ('drop1', nn.Dropout2d()),  # 每通道按概率置零
                ('maxpool2', nn.MaxPool2d(2, 2)),
                ('relu2', nn.ReLU()),
            ])).to(DEVICETYPE)
            self.fc = nn.Sequential(OrderedDict([
                ('fc1', nn.Linear(32 * 4 * 4, 50)),
                ('relu3', nn.ReLU()),
                ('drop2', nn.Dropout()),
                ('fc2', nn.Linear(50, 10)),
            ])).to(DEVICETYPE)

            # FedAvg
            # self.conv = nn.Sequential(OrderedDict([
            #     ('conv1', nn.Conv2d(1, 32, 5)), # in_channels, out_channels, kernel_size
            #     ('maxpool1', nn.MaxPool2d(2, 2)), # kernel_size, stride
            #     ('relu1', nn.ReLU()),
            #     ('conv2', nn.Conv2d(32, 64, 5)),
            #     ('maxpool2', nn.MaxPool2d(2, 2)),
            #     ('relu2', nn.ReLU()),
            # ])).to(DEVICETYPE)
            # self.fc = nn.Sequential(OrderedDict([
            #     ('fc1',nn.Linear(64*4*4, 512)),
            #     ('relu3',nn.ReLU()),
            #     ('fc2',nn.Linear(512, 10)),
            # ])).to(DEVICETYPE)

            self.criterion = nn.CrossEntropyLoss().to(DEVICETYPE)

        elif self.args.network == 'Resnet18':
            # Resnet
            self.in_planes = 64
            # self.feature_dim = args.featuredim

            self.conv1 = nn.Conv2d(input_channel, 64, kernel_size=3, stride=1, padding=1, bias=False).to(DEVICETYPE)
            self.bn1 = nn.BatchNorm2d(64).to(DEVICETYPE)
            self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1).to(DEVICETYPE)
            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2).to(DEVICETYPE)
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2).to(DEVICETYPE)
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2).to(DEVICETYPE)

            # self.reduction = nn.Sequential(
            #     nn.Linear(512 * block.expansion, 512, bias=False),
            #     nn.BatchNorm1d(512),
            #     nn.ReLU(inplace=True),
            #     nn.Linear(512, self.feature_dim, bias=True)
            # ).to(DEVICETYPE)

            # self.reduction = nn.Sequential(
            #     nn.Linear(512 * block.expansion, 512, bias=False),  # [0]
            #     nn.BatchNorm1d(512),  # [1]
            #     nn.ReLU(inplace=True),  # [2]
            #     nn.Linear(512, 128, bias=False),  # [3]
            #     nn.BatchNorm1d(128),  # [4]
            #     nn.ReLU(inplace=True),  # [5]
            #     nn.Linear(128, 128, bias=False),  # [6]
            #     nn.BatchNorm1d(128),  # [7]
            #     nn.ReLU(inplace=True),  # [8]
            #     nn.Linear(128, self.feature_dim, bias=True)  # [9]
            # ).to(DEVICETYPE)

            self.avgpool = nn.AdaptiveAvgPool2d((1, 1)).to(DEVICETYPE)
            self.fc = nn.Linear(512 * block.expansion, self.num_classes).to(DEVICETYPE)

            # if not self.args.featurelearning:
            self.criterion = nn.CrossEntropyLoss().to(DEVICETYPE)

    def _make_layer(self, block, planes, num_blocks, stride):  ## for Resnet
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.args.network == 'CNN':
            # MLP
            # x = x.view(x.size()[0], -1)
            # x = self.network1_linear(x)
            # x = self.network2_relu(x)
            # x = self.network3_linear(x)
            # x = self.network4_softmax(x)
            # return F.log_softmax(x, dim=1)

            # LeNet
            feature = self.conv(x)
            output = self.fc(feature.view(x.shape[0], -1))
            return output
        elif self.args.network == 'Resnet18':
            # Feature Extraction
            out = F.relu(self.bn1(self.conv1(x)))
            # out = F.max_pool2d(out, kernel_size=3, stride=2, padding=1) # Mayi无
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)

            # if self.args.featurelearning:
            #     # Mayi
            #     out = F.avg_pool2d(out, 4)
            #     out = out.view(out.size(0), -1)
            #     out = self.reduction(out)
            #     return F.normalize(out)  ## feature
            # else:
            # classifier
            out = self.avgpool(out)  # Mayi替换上面
            out = torch.flatten(out, 1)
            out = self.fc(out)
            return out

    def start_test(self, data):
        # 设置为evaluation mode
        self.eval()
        with torch.no_grad():
            test_data = DataLoader(data, batch_size=32, shuffle=True, num_workers=0)  # 把数据集装载进DataLoaer，然后送入深度学习网络进行训练
            accuracy = 0
            for x_batchs, y_batchs in test_data:
                pred_batchs = self.__call__(x_batchs.to(DEVICETYPE))
                pred_batchs = torch.topk(pred_batchs, 1)[1].squeeze(1).to(DEVICETYPE)
                out = [i == 0 for i in pred_batchs - y_batchs.to(DEVICETYPE)]
                accuracy += float(sum(out))
            self.accuracy.append(accuracy / len(data))
        return accuracy

    # def opt(self, method, lr, epoch=0, mmt=0.9, betas=[0.9, 0.999], eps=1e-8):
    #     start = 0
    #     para = self.state_dict()
    #     if method == 'Momentum':
    #         self.v = mmt * self.v - lr * self.grad_update  # grad_update是从哪里来的？，args又是从哪里来的
    #     elif method == 'Adam':
    #         self.mt = betas[0] * self.mt + (1 - betas[0]) * self.grad_update
    #         self.vt = betas[1] * self.vt + (1 - betas[1]) * self.grad_update * self.grad_update
    #         m_cap = self.mt / (1 - (betas[0] ** (epoch + 1)))
    #         v_cap = self.vt / (1 - (betas[1] ** (epoch + 1)))
    #
    #     for key, value in para.items():
    #         if method == 'SGD':
    #             para[key] += -lr * \
    #                          self.grad_update[start:start + value.numel()].view(value.size())
    #         elif method == 'Momentum':
    #             para[key] += self.v[start:start + value.numel()].view(value.size())
    #         elif method == 'Adam':
    #             para[key] += \
    #                 - (lr * m_cap[start:start + value.numel()].view(value.size())) / \
    #                 (torch.sqrt(v_cap[start:start + value.numel()].view(value.size())) + eps)
    #         # 判断更新后数值稳定性
    #         if torch.any(torch.isnan(para[key])):
    #             for key_temp, value in para.items():
    #                 para[key_temp] = torch.zeros_like(para[key_temp])
    #             break
    #         start += value.numel()
    #     self.load_state_dict(para, strict=True)


class LocalNetwork(BasicNetwork):
    def __init__(self, args, task_id, device_id, data):
        super(LocalNetwork, self).__init__(args, data)
        # ------------------for local network------------------#
        self.iter_data = []
        self.grad_basic = torch.zeros([1, 1])  # 根据数据集计算出的原始梯度数据
        self.grad_normal = torch.zeros([1, 1])  # 归一化梯度
        self.grad_update = 0  # AMP算法估计出来的梯度，相当于是对self.combine_x的估计
        self.grad_mean = 0
        self.grad_std = 0
        self.Qki = self.args.datasets_num_dvc[task_id][device_id]
        self.bs = 128 # int(self.Qki / self.args.sgd_times)  # 这个是每个device的batch_size
        self.old_dict = None
        self.lr = args.lr[task_id]

    def start_IterTrain(self):
        self.train()  # 训练模式
        self.old_dict = copy.deepcopy(self.state_dict())
        # decay:
        # self.lr = 0.995 * self.lr
        if self.args.optimizer == 0:
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        elif self.args.optimizer == 1:
            if self.args.network == 'Resnet18':
                optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)  # 是这个
            else:
                optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.5)
        elif self.args.optimizer == 2:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=[0.9, 0.999])

        # for dataset test
        # for idx, (data, image) in enumerate(self.data):
        #     print(idx)

        for x_batchs, y_batchs in self.iter_data:
            optimizer.zero_grad()
            # self.loss = self.criterion(self.__call__(x_batchs.to(DEVICETYPE)), torch.eye(10)[y_batchs, :].to(DEVICETYPE))
            self.loss = self.criterion(self.__call__(x_batchs.to(DEVICETYPE)), y_batchs.to(DEVICETYPE))
            self.loss.backward()
            optimizer.step()

    def start_GetGrad(self):
        # 将所有参数的梯度排成一行
        reshape_data_old = []
        reshape_data_new = []
        for key, value in self.old_dict.items():
            reshape_data_old.append(value.view(1,value.numel()))
        for key, value in self.state_dict().items():
            reshape_data_new.append(value.view(1,value.numel()))
            # eval(返回传入表达式的结果)
               
        # 只传输原始梯度
        # g
        self.grad_basic = \
            torch.cat(reshape_data_new, 1).to(DEVICETYPE) - torch.cat(reshape_data_old, 1).to(DEVICETYPE)
        self.grad_basic = torch.where(
            torch.isnan(self.grad_basic), torch.full_like(self.grad_basic, 0), self.grad_basic
        )
        if self.grad_std < 1e-5:
            self.grad_std = 1e-5
        # 求均值方差
        self.grad_mean = torch.mean(self.grad_basic.cpu())
        self.grad_std = torch.std(self.grad_basic.cpu())
        grad_normal = (self.grad_basic - self.grad_mean)/self.grad_std
        # r(转复数)
        self.grad_normal = opr_base.tensor2npcplx(grad_normal.squeeze(dim = 0).cpu())
        # self.grad_mean = np.array(self.grad_mean*(1+1j))
        self.grad_mean = np.array(self.grad_mean)
        self.grad_std = np.array(self.grad_std)


class GlobalNetwork(BasicNetwork):
    def __init__(self, args, data):
        super(GlobalNetwork, self).__init__(args, data)
        # ------------------for global network------------------#
        self.grad_update = 0  # AMP算法估计出来的梯度，相当于是对self.combine_x的估计
        # self.compress_code = torch.zeros(1, 1)  # A
        self.grad_basic = torch.zeros([1,1])  # 根据数据集计算出的原始梯度数据
        self.grad_error = torch.zeros([1,1])  # 梯度误差积累数据
        self.grad_sparsity = torch.zeros([1,1])  # 梯度稀疏化后数据
        self.grad_compress = torch.zeros([1,1])  # 梯度被压缩矩阵压缩后的数据
        self.alpha = 0  # 功率控制信号
        
    def RefreshLearningRate(self, subsystemNum, epoch):
        # self.lr = self.args.lr[subsystemNum]/(1+epoch/50.0)
        self.lr = self.args.lr[subsystemNum]

        # if self.args.optimizer == 0:
        #     self.opt('SGD', self.lr)
        # elif self.args.optimizer == 1:
        #     self.opt('Momentum', self.lr)
        # elif self.args.optimizer == 2:
        #     self.opt('Adam', self.lr, epoch,)

        start = 0
        para = self.state_dict()
        for key, value in para.items():
            if re.search("num_batches_tracked", key) != None:
                para[key] += \
                    self.grad_update[start:start + value.numel()].view(value.size()).long()
            else:
                para[key] += \
                    self.grad_update[start:start + value.numel()].view(value.size())
            # 判断更新后数值稳定性
            if torch.any(torch.isnan(para[key])):
                for key_temp, value in para.items():
                    para[key_temp] = torch.zeros_like(para[key_temp])
                break
            start += value.numel()
        self.load_state_dict(para, strict=True)
