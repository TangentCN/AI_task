import torch
from data import get_data_loaders, get_path
from model import Dropout_Net
from config import Config

def predict(testloader, net, device='cpu'):
    '''在测试集上评估模型'''
    correct = 0 # 预测正确的图片数
    total = 0 # 总共的图片数

    net.eval()
    net.to(device)

    with torch.no_grad(): # 正向传播时不计算梯度
        for data in testloader:
            # 1. 取出数据
            images, labels = data
            # 2. 正向传播，得到输出结果
            outputs = net(images)
            # 3. 从输出中得到模型预测
            _, predicted = torch.max(outputs, 1)
            # 4. 计算性能指标
            total += labels.size(0)
            correct += (predicted == labels).sum()
    
    print('测试集中的准确率为: %d %%' % (100 * correct / total))

def load_model(weight_path, device='cpu'):
    '''加载训练好的模型'''
    net = Dropout_Net()
    # 加载权重参数（需要路径）
    net.load_state_dict(torch.load(weight_path, map_location=device))
    # 选择设备
    net.to(device)
    # 调整到推理模式
    net.eval()

    return net

def run_test():
    '''完整测试流程'''
    cfg = Config()
    # 1.获取设备信息：有独显优先，没有就用CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 2.获取测试集
    _, test_loader = get_data_loaders()
    # 3.获取模型权重参数（用于模型初始化）
    epoch = cfg.epochs
    weight_path = f"{get_path()}/epoch_{epoch + 1}_model.pth"
    # 4.模型初始化
    net = load_model(weight_path, device)
    # 5.正式测试，输出结果
    predict(test_loader, net, device)

if __name__=='__main__':
    run_test()