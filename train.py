import torch
import torch.nn as nn
from torch import optim
from model import Net
from data import get_data_loaders, get_path
from config import Config

def train():
    cfg = Config()
    net = Net()
    num_epochs = cfg.epochs
    save_path = get_path()
    criterion = nn.CrossEntropyLoss() # 交叉熵损失函数
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) # 使用SGD（随机梯度下降）优化
    trainloader, _ = get_data_loaders(cfg.batch_size)

    for epoch in range(num_epochs):     
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
    
            # 1. 取出数据
            inputs, labels = data
    
            # 梯度清零
            optimizer.zero_grad()
    
            # 2. 前向计算和反向传播
            outputs = net(inputs) # 送入网络（正向传播）
            loss = criterion(outputs, labels) # 计算损失函数
            
            # 3. 反向传播，更新参数
            loss.backward() # 反向传播
            optimizer.step()

            # 下面的这段代码对于训练无实际作用，仅用于观察训练状态
            running_loss += loss.item()
            if i % 1000 == 999: # 每2000个batch打印一下训练状态
                print('epoch %d: batch %5d loss: %.3f' \
                      % (epoch+1, i+1, running_loss / 2000))
                running_loss = 0.0
                
        torch.save(net.state_dict(), f"{save_path}/epoch_{epoch + 1}_model.pth")
        
    print('Finished Training')

if __name__ == '__main__':
    train()