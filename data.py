import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
from pathlib import Path
from config import Config
show = ToPILImage()

def get_data_loaders():
    '''数据导入与处理'''
    # 从Config类获取必要参数（不使用函数本身入口）
    cfg = Config()
    dataset_path = cfg.dataset_path
    batch_size = cfg.batch_size
    # 定义数据预处理流程
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    # 加载CIFAR-10训练集
    trainset = torchvision.datasets.CIFAR10(root=dataset_path, train=True,
                                            download=True, transform=transform)
    # 创建训练数据加载器
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)
    #加载CIFAR-10测试集
    testset = torchvision.datasets.CIFAR10(root=dataset_path, train=False,
                                        download=True, transform=transform)
    # 创建测试数据加载器
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)
    return trainloader, testloader

def get_path():
    '''创建存储数据的文件夹并返回路径'''
    data_folder = Path.home()/'Documents'/'training_data' # 适应不同用户名，在Documents文件夹里创建tranining_data文件夹
    data_folder.mkdir(parents=True, exist_ok=True)
    str_path = str(data_folder)

    return str_path