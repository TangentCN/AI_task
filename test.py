import torch
from data import get_data_loaders, get_path
from config import Config
from model import LeNet, Dropout_LeNet, AlexNet_CIFAR10

def predict(testloader, net, device='cpu'):
    '''在测试集上评估模型'''
    correct = 0 # 预测正确的图片数
    total = 0 # 总共的图片数

    net.eval()
    net.to(device)

    all_predictions = []
    all_labels = []

    with torch.no_grad(): # 正向传播时不计算梯度
        for data in testloader:
            # 1. 取出数据
            images, labels = data[0].to(device), data[1].to(device)
            # 2. 正向传播，得到输出结果
            outputs = net(images)
            # 3. 从输出中得到模型预测
            _, predicted = torch.max(outputs, 1)
            # 4. 
            all_predictions.append(predicted.cpu())
            all_labels.append(labels.cpu())
    
     # 拼接所有批次的结果
    y_pred = torch.cat(all_predictions)
    y_true = torch.cat(all_labels)
    
    # 计算准确率
    correct = (y_pred == y_true).sum().item()
    total = y_true.size(0)
    accuracy = correct / total
    
    # 获取类别数
    num_classes = torch.max(y_true).item() + 1
    
    # 计算混淆矩阵
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    
    # 计算每个类别的TP, FP, FN
    precision_list = []
    recall_list = []
    f1_list = []
    
    for i in range(num_classes):
        tp = cm[i, i].item()
        fp = cm[:, i].sum().item() - tp
        fn = cm[i, :].sum().item() - tp
        
        # 精确率
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        # 召回率
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        # F1分数
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
    
    # 计算加权平均（按各类别样本数加权）
    class_counts = cm.sum(dim=1).float()
    total_samples = class_counts.sum()
    
    weights = class_counts / total_samples
    weighted_precision = torch.tensor(precision_list).dot(weights).item()
    weighted_recall = torch.tensor(recall_list).dot(weights).item()
    weighted_f1 = torch.tensor(f1_list).dot(weights).item()
 
    # 返回性能指标
    return {
        'accuracy': accuracy,
        'precision': weighted_precision,
        'recall': weighted_recall,
        'f1_score': weighted_f1,
        'per_class_precision': precision_list,
        'per_class_recall': recall_list,
        'per_class_f1': f1_list,
        'confusion_matrix': cm.numpy(),
        'predictions': y_pred.numpy(),
        'true_labels': y_true.numpy()
    }

def get_model(name):
    '''根据config提供的信息选择模型'''
    match name:
        case 'LeNet':
            model = LeNet() 
        case 'Dropout_LeNet':
            model = Dropout_LeNet()
        case 'AlexNet_CIFAR10':
            model = AlexNet_CIFAR10()

    return model

def load_model(cfg, weight_path, device='cpu'):
    '''加载训练好的模型'''
    net = get_model(cfg.model_name)
    # 加载权重参数（需要路径）
    net.load_state_dict(torch.load(weight_path, map_location=device))
    # 选择设备
    net.to(device)
    # 调整到推理模式
    net.eval()

    return net

def save_test_log(log_dict):
    '''保存测试日志'''
    log_path = f"{get_path()}/test_log.txt"
    accuracy = log_dict['accuracy']
    weighted_precision = log_dict['precision']
    weighted_recall = log_dict['recall']
    weighted_f1 = log_dict['f1_score']

    with open(log_path, 'w', encoding='utf-8') as f:
        f.write('='*50 + '\n')
        f.write('测试集评估结果:\n')
        f.write('='*50 + '\n')
        f.write(f'准确率 (Accuracy):    {accuracy:.4f} ({100*accuracy:.2f}%)\n')
        f.write(f'精确率 (Precision):   {weighted_precision:.4f}\n')
        f.write(f'召回率 (Recall):      {weighted_recall:.4f}\n')
        f.write(f'F1分数 (F1-Score):    {weighted_f1:.4f}\n')
        f.write('='*50)

    print(f"测试日志已保存至{log_path}")

def run_test():
    '''完整测试流程'''
    cfg = Config()
    # 1.获取设备信息：有独显优先，没有就用CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 2.获取测试集
    _, _, test_loader = get_data_loaders()
    # 3.获取模型权重参数（用于模型初始化）
    epoch = cfg.epochs
    weight_path = f"{get_path()}/best_model.pth"
    # 4.模型初始化
    net = load_model(cfg, weight_path, device)
    # 5.正式测试，输出结果
    save_test_log(predict(test_loader, net, device))
    print('Finished testing!')

if __name__=='__main__':
    run_test()