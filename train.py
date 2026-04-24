import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch import optim
from model import Net
from data import get_data_loaders, get_path
from config import Config

def draw(losses):
    '''画图模块'''
    inputs = []
    for i in range(len(losses)):
        inputs.append(i+1)

    plt.style.use('seaborn-v0_8-darkgrid')      # 使用样式
    fig, ax = plt.subplots()                    # 初始化界面fig与图表ax
    ax.plot(inputs, losses, linewidth = 3)      # 指定输入输出，粗细

    ax.set_title('Loss in Training Process', fontsize=24)   # 标题
    ax.set_xlabel('Time / (1000 batches)', fontsize=14)     # x标题
    ax.set_ylabel('Loss', fontsize=14)                      # y标题
    ax.tick_params(axis='both', labelsize=14)               # 刻度

    plt.show()

def train():
    '''训练主程序'''
    cfg = Config()
    net = Net()
    save_path = get_path()
    criterion = nn.CrossEntropyLoss()                                                    # 交叉熵损失函数
    optimizer = optim.SGD(net.parameters(), lr=cfg.learning_rate, momentum=cfg.momentum) # 使用SGD（随机梯度下降）优化
    trainloader, _ = get_data_loaders(cfg.batch_size)
    losses = []

    for epoch in range(cfg.epochs):     
        running_loss = 0.0

        for i, data in enumerate(trainloader, 0):
    
            # 1. 取出数据
            inputs, labels = data
    
            # 梯度清零
            optimizer.zero_grad()
    
            # 2. 前向计算和反向传播
            outputs = net(inputs)             # 送入网络（正向传播）
            loss = criterion(outputs, labels) # 计算损失函数
            
            # 3. 反向传播，更新参数
            loss.backward() # 反向传播
            optimizer.step()

            # 下面的这段代码对于训练无实际作用，仅用于观察训练状态
            running_loss += loss.item()
            if i % 1000 == 0:           # 每1000个batch记录一下训练状态
                if i == 0:
                    pass
                else:
                    avrg_loss = running_loss / 1000
                    losses.append(avrg_loss)
                    print('epoch %d: batch %5d loss: %.3f' \
                        % (epoch+1, i+1, avrg_loss))
                    running_loss = 0.0
                
        torch.save(net.state_dict(), f"{save_path}/epoch_{epoch + 1}_model.pth")
      
    print('Finished Training')
    draw(losses)

if __name__ == '__main__':
    train()