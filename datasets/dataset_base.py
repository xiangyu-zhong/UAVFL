
import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.utils.data import ConcatDataset
import numpy as np
import optlib.operations as op

def DatasetGet(args):
    """load dataset and split users"""
    dataset = {}

    # data normalized: data_norm = (x - mean)/std 
    # Normalize()
    transmnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    transfmnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
    transkmnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1918,), (0.3483,))])
    # transmnist = transforms.Compose([transforms.ToTensor()])
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        # transforms.RandAugment(num_ops=2, magnitude=14),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    for task_num,task in enumerate(args.task_list):
        if task == "mnist":
            dataset["mnist_train_origin"] = datasets.MNIST('./datasets/mnist/', train=True, download=True, transform=transmnist)
            dataset["mnist_test_origin"] = datasets.MNIST('./datasets/mnist/', train=False, download=True, transform=transmnist)
        elif task == "fmnist":
            dataset["fmnist_train_origin"] = datasets.FashionMNIST('./datasets/fmnist/', train=True, download=True,
                                                           transform=transfmnist)
            dataset["fmnist_test_origin"] = datasets.FashionMNIST('./datasets/fmnist/', train=False, download=True,
                                                          transform=transfmnist)
        elif task == "kmnist":
            dataset["kmnist_train_origin"] = datasets.KMNIST('./datasets/kmnist/', train=True, download=True,
                                                           transform=transkmnist)
            dataset["kmnist_test_origin"] = datasets.KMNIST('./datasets/kmnist/', train=False, download=True,
                                                          transform=transkmnist)
        elif task == 'cifar10':
            dataset["cifar10_train_origin"] = datasets.CIFAR10('../data/cifar/', train=True, download=True,
                                           transform=transform_train)
            dataset["cifar10_test_origin"] = datasets.CIFAR10('../data/cifar/', train=False, download=True,
                                          transform=transform_test)
            # num_classes = 10

        if args.set == 1:
            # 均分数据集
            Km = np.round(args.datasets_num[task_num]/args.users_num[task_num])\
                *np.ones(args.users_num[task_num])
            Km = Km.astype(np.int64)
            dataset[task + "_train_users"] = dataset_iid_split(task, Km, args.K[task_num], dataset)
        elif args.set == 2:
            # 非均分数据集
            Km = op.InequalSizeGen(args.users_num[task_num])
            args.datasets_num[task_num] = np.sum(Km)
            Km = Km.astype(np.int64)
            dataset[task + "_train_users"] = dataset_iid_split(task, Km, args.K[task_num], dataset)
        elif args.set == 3:
            # 均分数据集 + non-iid
            Km = np.round(args.datasets_num[task_num]/args.users_num[task_num])\
                *np.ones(args.users_num[task_num])
            Km = Km.astype(np.int64)
            dataset[task + "_train_users"], Km = dataset_non_iid_split(task, Km, args.K[task_num], dataset, args.N_class)

        args.datasets_num_dvc.append(Km)

    return dataset,args

def Normalize():
    batch_size=60000 #这里是为了后面一次取出所有的数据
    transform=transforms.Compose([transforms.ToTensor()]) #不对数据进行标准化

    #加载数据
    # train_dataset=datasets.MNIST(root='./datasets/mnist',train=True,download=False,
    #                             transform=transform)
    # train_dataset=datasets.FashionMNIST(root='./datasets/fmnist',train=True,download=False,
    #                             transform=transform)
    train_dataset=datasets.KMNIST(root='./datasets/kmnist',train=True,download=False,
                                transform=transform)
    train_loader=DataLoader(train_dataset,shuffle=True,batch_size=batch_size)

    #取出加载在DataLoader中的数据，因为batch_size就是训练集的样本数目，所以一次就取完了所有训练数据
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, targets = data #inpus为所有训练样本
        x=inputs.view(-1,28*28) #将（60000，1，28，28）大小的inputs转换为（60000，28*28）的张量
        x_std=x.std().item() #计算所有训练样本的标准差
        x_mean=x.mean().item() #计算所有训练样本的均值

    print("mean:{}".format(x_mean))
    print("std:{}".format(x_std))

def dataset_iid_split(task, Km, K, dataset):
    dataset_temp, _ = random_split(dataset[task + '_train_origin'], \
                [np.sum(Km), K-np.sum(Km)])
    return random_split(dataset_temp, Km)

def dataset_non_iid_split(task, Km, K, dataset, N):
    Q = 10      # 10个类，如果更换复杂的数据集，需要修改
    M = len(Km) # 用户数
    # 每个用户抽N类 
    Km_sum = np.sum(Km)
    dataset_temp, _ = random_split(dataset[task + '_train_origin'], \
                [Km_sum, K-Km_sum])
    # labels = dataset_temp.dataset.train_labels # 这样就一直是60000
    # labels = dataset_temp.dataset.train_labels[dataset_temp.indices]
    labels = np.array(getattr(dataset_temp.dataset, 'targets',
                              getattr(dataset_temp.dataset, 'train_labels',
                                      getattr(dataset_temp.dataset, 'labels', None))))[np.asarray(dataset_temp.indices)]

    # 按类别分类
    # class_indices = {q: np.array((labels == q).nonzero()).reshape(-1,) for q in range(Q)}  ##### 检查
    indices = np.asarray(dataset_temp.indices)
    class_indices = {
        q: indices[np.where(labels == q)[0]] for q in range(Q)
    }

    class_num = np.array([ class_indices[q].shape[0] for q in range(Q)])
    class_num_user = class_num // (N * M // Q ) # N是每个用户抽去的class数量（实验设的5？），M是用户数量，Q是class数量，这里是把每一class的数据，分成需要的分数，比如5000个数据的一个class，分成10份，20个用户中有一半的用户得到这个类别的数据，这个输出值就是500
    # 随机抽取N类数据的量，作为每个用户的类别和数据量
    dataset_user = []
    for i in range(M):
        Km_temp_index_i = np.random.choice(a=Q, size=N, replace=False) # 从Q个class中，随机选择N个class，返回的是N个类的编号
        Km_temp_i = class_num_user[Km_temp_index_i] # 这选中的这些类分到这个user的数量，比如上面就是500等，有几类就是几微向量
        subset_temp = [
            np.random.choice(a = class_indices[Km_temp_index_i[j]], \
                size=Km_temp_i[j], replace=False)  for j in range(N) # replace是选择后放不放回
        ]
        # dataset_user_temp = [Subset(dataset[task + '_train_origin'], subset_temp[j]) for j in range(N)]
        # dataset_user_temp = [Subset(dataset_temp, subset_temp[j]) for j in range(N)]  # 本来是这个
        merged_indices = np.concatenate(subset_temp)
        dataset_user_temp = Subset(dataset[task + '_train_origin'], merged_indices)
        dataset_user.append(dataset_user_temp)

        Km[i] = np.sum(Km_temp_i) # 把一个用户所有分到的类的数量加和，即一个用户的所有data数
        # dataset_user.append(ConcatDataset(dataset_user_temp))

    return dataset_user, Km

if __name__ == "__main__":
    Normalize()

