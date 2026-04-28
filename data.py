import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
from config import Config
from pathlib import Path

def get_data_loaders():
    '''数据导入与处理：返回训练集、验证集和测试集'''
    cfg = Config()
    dataset_path = cfg.dataset_path
    batch_size = cfg.batch_size
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # 加载完整训练集
    full_trainset = torchvision.datasets.CIFAR10(root=dataset_path, train=True,
                                            download=True, transform=transform)

    # 从训练集中分割出验证集（90% 训练，10% 验证）
    train_size = int(0.9 * len(full_trainset))
    val_size = len(full_trainset) - train_size
    trainset, valset = random_split(full_trainset, [train_size, val_size])

    # 创建训练数据加载器
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=0)
    # 创建验证数据加载器
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                            shuffle=False, num_workers=0)

    # 加载测试集（用于最终评估）
    testset = torchvision.datasets.CIFAR10(root=dataset_path, train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)
    return trainloader, valloader, testloader

def get_path():
    '''创建存储数据的文件夹并返回路径'''
    data_folder = Path.home()/'Documents'/'training_data' # 适应不同用户名，在Documents文件夹里创建tranining_data文件夹
    data_folder.mkdir(parents=True, exist_ok=True)
    str_path = str(data_folder)

    return str_path
