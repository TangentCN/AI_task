import torch
import datetime
import torch.nn as nn
import matplotlib.pyplot as plt
from torch import optim
from model import LeNet, Dropout_LeNet
from data import get_data_loaders, get_path
from test import run_test
from config import Config

def draw(losses, name):
    '''画图模块'''
    save_path = f"{get_path()}/{name.title()}_in_the_process.png"

    inputs = []
    for i in range(len(losses)):
        inputs.append(i+1)

    plt.style.use('seaborn-v0_8-darkgrid')      # 使用样式
    fig, ax = plt.subplots()                    # 初始化界面fig与图表ax
    ax.plot(inputs, losses, linewidth = 1)      # 指定输入输出，粗细
    ax.set_title(f'{name.title()} in the Training Process', fontsize=24)   # 标题
    ax.set_xlabel('epochs', fontsize=14)     # x标题
    ax.set_ylabel(f'{name.title()}', fontsize=14)                      # y标题
    ax.tick_params(axis='both', labelsize=14)               # 刻度

    plt.savefig(save_path)

def save_training_log(net, losses, val_accuracies, best_epoch, best_accuracy):
    '''保存训练日志到文本文件'''
    cfg = Config()
    save_path = get_path()
    log_path = f"{save_path}/training_log.txt"
    
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("=" * 50 + "\n")
        f.write("训练日志\n")
        f.write(f"训练时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("配置参数:\n")
        f.write(f"{'=' * 30}\n")
        f.write(f"模型 : {net.name}\n")
        f.write(f"批次大小 (batch_size): {cfg.batch_size}\n")
        f.write(f"训练轮数 (epochs): {cfg.epochs}\n")
        f.write(f"学习率 (learning_rate): {cfg.learning_rate}\n")
        f.write(f"动量 (momentum): {cfg.momentum}\n")
        if net.name == 'Dropout_LeNet':
            f.write(f"权重衰减 (weight_decay): {cfg.weight_decay}\n")
        f.write(f"数据集路径: {cfg.dataset_path}\n")
        
        f.write("\n训练结果:\n")
        f.write(f"{'=' * 30}\n")
        f.write(f"训练总轮数: {len(losses)}\n")
        
        final_loss = losses[-1] if losses else 'N/A'
        f.write(f"最终损失值: {final_loss:.4f}\n" if isinstance(final_loss, float) else f"最终损失值: {final_loss}\n")
        
        min_loss = min(losses) if losses else 'N/A'
        f.write(f"最小损失值: {min_loss:.4f}\n" if isinstance(min_loss, float) else f"最小损失值: {min_loss}\n")
        
        if best_epoch is not None:
            f.write(f"\n验证集最佳结果:\n")
            f.write(f"{'=' * 30}\n")
            f.write(f"最佳准确率: {best_accuracy:.4f}\n")
            f.write(f"最佳轮次: 第{best_epoch + 1}轮\n")
            f.write(f"对应损失值: {losses[best_epoch]:.4f}\n")
        
        if val_accuracies:
            f.write(f"\n各轮次准确率:\n")
            f.write(f"{'=' * 30}\n")
            for i, acc in enumerate(val_accuracies, 1):
                f.write(f"第{i:3d}轮: {acc:.4f}\n")
        
        if losses:
            f.write(f"\n损失值记录:\n")
            f.write(f"{'=' * 30}\n")
            for i, loss in enumerate(losses, 1):
                f.write(f"第{i:3d}次记录: {loss:.4f}\n")
    
    print(f"训练日志已保存至: {log_path}")

def evaluate_model(net, dataloader, device):
    '''在验证集上评估模型准确率'''
    net.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    return accuracy

def train():
    '''训练主程序'''
    cfg = Config()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print('device: on gpu')
    else:
        print('device: on cpu')
    net = Dropout_LeNet().to(device)  # 有GPU就用GPU

    save_path = get_path()
    criterion = nn.CrossEntropyLoss()                                                    # 交叉熵损失函数
    optimizer = optim.SGD(net.parameters(), lr=cfg.learning_rate, momentum=cfg.momentum, weight_decay=cfg.weight_decay) # 使用SGD（随机梯度下降）优化
    trainloader, valloader, _ = get_data_loaders()
    
    losses = []
    val_accuracies = []  # 记录测试集（验证集）准确率
    best_accuracy = 0.0
    best_epoch = -1

    for epoch in range(cfg.epochs):     
        running_loss = 0.0
        num_batches = 0

        for i, data in enumerate(trainloader, 0):
    
            # 1. 取出数据
            inputs, labels = data[0].to(device), data[1].to(device)
    
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
            num_batches += 1

        avrg_loss = running_loss / num_batches
        losses.append(avrg_loss)
        accuracy = evaluate_model(net, valloader, device)
        val_accuracies.append(accuracy)
        
        print('epoch %d: loss: %.3f, val_accuracy: %.3f%%' % 
              (epoch+1, avrg_loss, accuracy * 100))
        
        # 保存测试集上准确率最高的模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch
            torch.save(net.state_dict(), f"{save_path}/best_model.pth")
            print(f"   -> 新的最佳模型，测试准确率: {accuracy:.3%}")
      
    print('Finished Training')
    save_training_log(net, losses, val_accuracies, best_epoch, best_accuracy)
    draw(losses, 'loss')
    draw(val_accuracies, 'accuracies')
    run_test()

if __name__ == '__main__':
    train()