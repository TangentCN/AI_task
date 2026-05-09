import torch
import datetime
import winsound
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from models import LeNet, Dropout_LeNet, AlexNet_CIFAR10, VGG11_CIFAR10
from torch import optim
from data import get_data_loaders, get_path
from test import run_test
from config import Config

def get_model(name):
    '''根据config提供的信息选择模型'''
    match name:
        case 'LeNet':
            model = LeNet() 
        case 'Dropout_LeNet':
            model = Dropout_LeNet()
        case 'AlexNet_CIFAR10':
            model = AlexNet_CIFAR10()
        case 'VGG11_CIFAR10':
            model = VGG11_CIFAR10()

    return model

def draw(losses, name):
    '''画图模块'''
    save_path = f"{get_path()}/{name.title()}_in_the_process.png"

    inputs = []
    for i in range(len(losses)):
        inputs.append(i+1)

    plt.style.use('seaborn-v0_8-darkgrid')      # 使用样式
    fig, ax = plt.subplots()                    # 初始化界面fig与图表ax
    ax.plot(inputs, losses, linewidth = 1)      # 指定输入输出，粗细
    ax.set_title(f'{name.title()} in the Training Process', fontsize=24) # 标题
    ax.set_xlabel('epochs', fontsize=14)        # x标题
    ax.set_ylabel(f'{name.title()}', fontsize=14) # y标题
    ax.tick_params(axis='both', labelsize=14)   # 刻度

    plt.savefig(save_path)

def save_training_log(net, losses, val_accuracies,learning_rates, best_epoch, best_accuracy):
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
        f.write(f"早停耐心 (early_stopping_patience): {cfg.early_stopping_patience}\n")
        f.write(f"训练轮数 (epochs): {len(losses)}")
        if len(losses) < cfg.epochs:
            f.write("(早停启用)\n")
        else:
            f.write("\n")
        f.write(f"学习率 (learning_rate): {cfg.learning_rate}\n")
        f.write(f"学习率衰减相关参数:\n")
        f.write(f"\tfactor: {cfg.factor}\n")
        f.write(f"\tpatience: {cfg.patience}\n")
        f.write(f"\tthreshold: {cfg.threshold}\n")
        f.write(f"动量 (momentum): {cfg.momentum}\n")
        if net.regu:
            f.write(f"权重衰减 (weight_decay): {cfg.weight_decay}\n")
            f.write(f"随机丢弃概率 (dropout_rate): {cfg.dropout_rate}\n")
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

        if learning_rates:
            f.write(f"\nLR记录:\n")
            f.write(f"{'=' * 30}\n")
            for i, lr in enumerate(learning_rates, 1):
                f.write(f"第{i:3d}次记录: {lr:.4f}\n")
    
    print(f"训练日志已保存至: {log_path}")

def evaluate_model(net, dataloader, device):
    '''
    在验证集上评估模型准确率

    该模块在升级到Dropout_LeNet时同步加入
    '''
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
    '''
    训练主程序

    LeNet 基本训练模块: 训练主循环, 设备选择, 损失函数, SGD优化器, 损失记录, 绘图和日志记录
    Dropout_LeNet 更新: SGD优化器加入权重衰减系数, 加入验证集并只保存验证最优的模型
    AlexNet 更新: 加入学习率控制器, 加入早停
    '''
    cfg = Config()
    # 解决设备调用问题
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print('device: on gpu')
    else:
        print('device: on cpu')
    net = get_model(cfg.model_name).to(device)  # 有GPU就用GPU

    # 初始化各项变量
    save_path = get_path()
        # 交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
        # 使用SGD（随机梯度下降）优化
    optimizer = optim.SGD(
        net.parameters(),
        lr=cfg.learning_rate,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay
        )
        # 学习率动态衰减：监控准确度 (mode='max')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',              # 监控准确度，越大越好
        factor=cfg.factor,       # 学习率 * n
        patience=cfg.patience,   # n个epoch无改善后衰减
        threshold=cfg.threshold, # 判定是否改善的阈值
    )
    trainloader, valloader, _ = get_data_loaders()
    losses = []
    val_accuracies = []
    learning_rates = []
    ln_learning_rates = []
    best_accuracy = 0.0
    best_epoch = -1
    counter = 0 # 早停监测器
    winsound.Beep(1000,500) # 准备好了你就响一声
    # 训练主循环(以epoch为单位)
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
        # 计算本epoch的各项统计数据并存储，报告状态
        avrg_loss = running_loss / num_batches
        losses.append(avrg_loss)
        accuracy = evaluate_model(net, valloader, device)
        val_accuracies.append(accuracy)
        current_lr = optimizer.param_groups[0]['lr']
        ln_current_lr = np.log(current_lr)
        learning_rates.append(current_lr)
        ln_learning_rates.append(ln_current_lr)
        print('epoch %d: loss: %.3f lr: %.5f val_accuracy: %.3f%%' % 
              (epoch+1, avrg_loss, current_lr, accuracy * 100))
        
        # 保存测试集上准确率最高的模型
        if accuracy >= best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch
            counter = 0
            torch.save(net.state_dict(), f"{save_path}/best_model.pth")
            print(f"   -> 新的最佳模型，测试准确率: {accuracy:.3%}")
        # 早停监测
        else:
            counter += 1
            print(f"No improvement for {counter} epochs")
            if counter >= cfg.early_stopping_patience:
                print("Early stopping triggered!")
                break
        # 每个epoch结束后更新学习率调度器，传入当前准确度
        scheduler.step(accuracy)
    # 结束处理  
    print('Finished Training')
    winsound.Beep(1000,500) # 结束了就响一声
    save_training_log(net, losses, val_accuracies, learning_rates, best_epoch, best_accuracy)
    draw(losses, 'loss')
    draw(val_accuracies, 'accuracies')
    draw(ln_learning_rates, 'lr(ln)')
    run_test() #顺便运行测试

if __name__ == '__main__':
    train()