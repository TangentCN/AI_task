import torch.nn as nn
import torch.nn.functional as F
from config import Config

class LeNet(nn.Module):
    '''LeNet'''
    def __init__(self):
        # nn.Module子类的函数必须在构造函数中执行父类的构造函数
        super(LeNet, self).__init__()
        self.name = 'LeNet'
        # 卷积层 '3'表示输入图片为单通道, '6'表示输出通道数，'5'表示卷积核为5*5
        self.conv1 = nn.Conv2d(3, 6, 5) 
        # 卷积层
        self.conv2 = nn.Conv2d(6, 16, 5) 
        # 仿射层/全连接层，y = Wx + b
        self.fc1   = nn.Linear(16*5*5, 120) 
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x): 
        # 卷积 -> 激活 -> 池化 (relu激活函数不改变输入的形状)
        # [batch size, 3, 32, 32] -- conv1 --> [batch size, 6, 28, 28] -- maxpool --> [batch size, 6, 14, 14]
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # [batch size, 6, 14, 14] -- conv2 --> [batch size, 16, 10, 10] --> maxpool --> [batch size, 16, 5, 5]
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # 把 16 * 5 * 5 的特征图展平，变为 [batch size, 16 * 5 * 5]，以送入全连接层
        x = x.view(x.size()[0], -1) 
        # [batch size, 16 * 5 * 5] -- fc1 --> [batch size, 120]
        x = F.relu(self.fc1(x))
        # [batch size, 120] -- fc2 --> [batch size, 84]
        x = F.relu(self.fc2(x))
        # [batch size, 84] -- fc3 --> [batch size, 10]
        x = self.fc3(x)        
        return x
    
class Dropout_LeNet(nn.Module):
    '''加入Dropout正则化的LeNet'''
    def __init__(self):
        # nn.Module子类的函数必须在构造函数中执行父类的构造函数
        super(Dropout_LeNet, self).__init__()
        cfg = Config()
        self.name = 'Dropout_LeNet'

        # 卷积层 '3'表示输入图片为单通道, '6'表示输出通道数，'5'表示卷积核为5*5
        self.conv1 = nn.Conv2d(3, 6, 5) 
        # 卷积层
        self.conv2 = nn.Conv2d(6, 16, 5) 
        # 仿射层/全连接层，y = Wx + b
        self.fc1   = nn.Linear(16*5*5, 120) 
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)
        # Dropout正则化层
        self.dropout  = nn.Dropout(cfg.dropout_rate)

    def forward(self, x): 
        # 卷积 -> 激活 -> 池化 (relu激活函数不改变输入的形状)
        # [batch size, 3, 32, 32] -- conv1 --> [batch size, 6, 28, 28] -- maxpool --> [batch size, 6, 14, 14]
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # [batch size, 6, 14, 14] -- conv2 --> [batch size, 16, 10, 10] --> maxpool --> [batch size, 16, 5, 5]
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # 把 16 * 5 * 5 的特征图展平，变为 [batch size, 16 * 5 * 5]，以送入全连接层
        x = x.view(x.size()[0], -1) 
        # [batch size, 16 * 5 * 5] -- fc1 --> [batch size, 120]
        x = F.relu(self.fc1(x))
        # 随机dropout
        x = self.dropout(x)
        # [batch size, 120] -- fc2 --> [batch size, 84]
        x = F.relu(self.fc2(x))
        # [batch size, 84] -- fc3 --> [batch size, 10]
        x = self.fc3(x)        
        return x
    
class AlexNet_CIFAR10(nn.Module):
    '''适配CIFAR-10 32*32小图片的AlexNet'''
    def __init__(self):
        super(AlexNet_CIFAR10, self).__init__()
        cfg = Config()
        self.name = 'AlexNet_CIFAR10'
        # 卷积层 (入通道数，出通道数，核大小，步长，*填充)(适配32x32输入，换了小核小步长)
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 192, 3, 1)
            # 前两个各自池化激活，后三个三连
        self.conv3 = nn.Conv2d(192, 384, 3, 1)
        self.conv4 = nn.Conv2d(384, 256, 3, 1)
        self.conv5 = nn.Conv2d(256, 256, 3, 1)
        # 池化层(核大小，步长)(还是单独列出来比较好)
        self.pool = nn.MaxPool2d(2, 2)
        # 全连接层 (输入特征数，输出特征数)(32->16->8->4)
        self.fc1 = nn.Linear(256 * 4 * 4, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)
        # Dropout正则化层
        self.dropout = nn.Dropout(cfg.dropout_rate)
        
    def forward(self, x):
        # Block 1: 32x32 -> 16x16
        x = self.pool(F.relu(self.conv1(x)))
        # Block 2: 16x16 -> 8x8
        x = self.pool(F.relu(self.conv2(x)))
        # Block 3: 8x8 -> 8x8 (无池化)
        x = F.relu(self.conv3(x))
        # Block 4: 8x8 -> 8x8 (无池化)
        x = F.relu(self.conv4(x))
        # Block 5: 8x8 -> 4x4
        x = self.pool(F.relu(self.conv5(x)))
        # 展平
        x = x.view(x.size(0), -1)
        # 全连接层(接dropout)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x