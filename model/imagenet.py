import os
import pandas as pd
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import models
import torch.optim as optim

import wandb


#只用一次
# os.mkdir('output')# 存放结果文件
# os.mkdir('checkpoint')# 存放训练得到的模型权重
# os.mkdir('图表')# 存放生成的图表

# 忽略烦人的红色提示
import warnings
def main():
    os.environ["WANDB_API_KEY"] = '3e09fbea2e93817854c0274c469eb7e83eed3f9b'
    warnings.filterwarnings("ignore")

    # 有 GPU 就用 GPU，没有就用 CPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 训练集图像预处理：缩放裁剪、图像增强、转 Tensor、归一化
    train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                          ])

    # 测试集图像预处理-RCTN：缩放、裁剪、转 Tensor、归一化
    test_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(
                                             mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                                         ])

    dataset_dir = '../view-robot/dataset/dataset_split'

    train_path = os.path.join(dataset_dir, 'train')
    test_path = os.path.join(dataset_dir, 'val')

    # 载入训练集
    train_dataset = datasets.ImageFolder(train_path, train_transform)
    # 载入测试集
    test_dataset = datasets.ImageFolder(test_path, test_transform)
    # print('训练集图像数量', len(train_dataset))
    # print('类别个数', len(train_dataset.classes))
    # print('各类别名称', train_dataset.classes)
    # print('测试集图像数量', len(test_dataset))
    # print('类别个数', len(test_dataset.classes))
    # print('各类别名称', test_dataset.classes)

    # 各类别名称
    class_names = train_dataset.classes
    n_class = len(class_names)
    # # 映射关系：类别 到 索引号
    # train_dataset.class_to_idx
    # 映射关系：索引号 到 类别
    idx_to_labels = {y: x for x, y in train_dataset.class_to_idx.items()}

    # 保存为本地的 npy 文件
    np.save('../tests/idx_to_labels.npy', idx_to_labels)
    np.save('../tests/labels_to_idx.npy', train_dataset.class_to_idx)

    from torch.optim import lr_scheduler
    BATCH_SIZE = 32
    # 训练集的数据加载器
    train_loader = DataLoader(train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=4
                              )

    # 测试集的数据加载器
    test_loader = DataLoader(test_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=False,
                             num_workers=4
                             )

    model = models.resnet18(pretrained=True)  # 载入预训练模型
    model.fc = nn.Linear(model.fc.in_features, n_class)
    optimizer = optim.Adam(model.parameters())

    model = model.to(device)
    # 交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    # 训练轮次 Epoch
    EPOCHS = 30
    # 学习率降低策略
    lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score
    def train_one_batch(images, labels):
        '''
        运行一个 batch 的训练，返回当前 batch 的训练日志
        '''

        # 获得一个 batch 的数据和标注
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)  # 输入模型，执行前向预测
        loss = criterion(outputs, labels)  # 计算当前 batch 中，每个样本的平均交叉熵损失函数值

        # 优化更新权重
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 获取当前 batch 的标签类别和预测类别
        _, preds = torch.max(outputs, 1)  # 获得当前 batch 所有图像的预测类别
        preds = preds.cpu().numpy()
        loss = loss.detach().cpu().numpy()
        outputs = outputs.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

        log_train = {}
        log_train['epoch'] = epoch
        log_train['batch'] = batch_idx
        # 计算分类评估指标
        log_train['train_loss'] = loss
        log_train['train_accuracy'] = accuracy_score(labels, preds)
        # log_train['train_precision'] = precision_score(labels, preds, average='macro')
        # log_train['train_recall'] = recall_score(labels, preds, average='macro')
        # log_train['train_f1-score'] = f1_score(labels, preds, average='macro')

        return log_train

    def evaluate_testset():
        '''
        在整个测试集上评估，返回分类评估指标日志
        '''

        loss_list = []
        labels_list = []
        preds_list = []

        with torch.no_grad():
            for images, labels in test_loader:  # 生成一个 batch 的数据和标注
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)  # 输入模型，执行前向预测

                # 获取整个测试集的标签类别和预测类别
                _, preds = torch.max(outputs, 1)  # 获得当前 batch 所有图像的预测类别
                preds = preds.cpu().numpy()
                loss = criterion(outputs, labels)  # 由 logit，计算当前 batch 中，每个样本的平均交叉熵损失函数值
                loss = loss.detach().cpu().numpy()
                outputs = outputs.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()

                loss_list.append(loss)
                labels_list.extend(labels)
                preds_list.extend(preds)

        log_test = {}
        log_test['epoch'] = epoch

        # 计算分类评估指标
        log_test['test_loss'] = np.mean(loss_list)
        log_test['test_accuracy'] = accuracy_score(labels_list, preds_list)
        log_test['test_precision'] = precision_score(labels_list, preds_list, average='macro')
        log_test['test_recall'] = recall_score(labels_list, preds_list, average='macro')
        log_test['test_f1-score'] = f1_score(labels_list, preds_list, average='macro')

        # print("Labels range:", np.min(labels_list), np.max(labels_list))  # 打印标签范围
        # print("Predictions range:", np.min(preds_list), np.max(preds_list))  # 打印预测范围

        return log_test

    epoch = 0
    batch_idx = 0
    best_test_accuracy = 0

    # 训练日志-训练集
    df_train_log = pd.DataFrame()
    log_train = {}
    log_train['epoch'] = 0
    log_train['batch'] = 0
    images, labels = next(iter(train_loader))
    log_train.update(train_one_batch(images, labels))
    df_train_log = df_train_log._append(log_train, ignore_index=True)

    # 训练日志-测试集
    df_test_log = pd.DataFrame()
    log_test = {}
    log_test['epoch'] = 0
    log_test.update(evaluate_testset())
    df_test_log = df_test_log._append(log_test, ignore_index=True)

    # 确保 wandb 初始化正确
    wandb.init(project='fruit30', name=time.strftime('%m%d%H%M%S'))

    # 获得一个 batch 的数据和标注
    images, labels = next(iter(train_loader))
    images = images.to(device)
    labels = labels.to(device)
    # 输入模型，执行前向预测
    outputs = model(images)
    # 由 logit，计算当前 batch 中，每个样本的平均交叉熵损失函数值
    loss = criterion(outputs, labels)
    # 反向传播“三部曲”
    optimizer.zero_grad()  # 清除梯度
    loss.backward()  # 反向传播
    optimizer.step()  # 优化更新
    # 获得当前 batch 所有图像的预测类别
    _, preds = torch.max(outputs, 1)

    for epoch in range(1, EPOCHS + 1):
        print(f'Epoch {epoch}/{EPOCHS}')

        ## 训练阶段
        model.train()
        for images, labels in tqdm(train_loader):  # 获得一个 batch 的数据和标注
            batch_idx += 1
            log_train = train_one_batch(images, labels) # 传递 batch_idx 和 epoch
            df_train_log = df_train_log._append(log_train, ignore_index=True)
            wandb.log(log_train)

        lr_scheduler.step()

        ## 测试阶段
        model.eval()
        log_test = evaluate_testset()
        df_test_log = df_test_log._append(log_test, ignore_index=True)
        wandb.log(log_test)

        # 保存最新的最佳模型文件
        if log_test['test_accuracy'] > best_test_accuracy:
            # 删除旧的最佳模型文件(如有)
            old_best_checkpoint_path = 'checkpoint/best-{:.3f}.pth'.format(best_test_accuracy)
            if os.path.exists(old_best_checkpoint_path):
                os.remove(old_best_checkpoint_path)
            # 保存新的最佳模型文件
            best_test_accuracy = log_test['test_accuracy']
            new_best_checkpoint_path = 'checkpoint/best-{:.3f}.pth'.format(log_test['test_accuracy'])
            torch.save(model.state_dict(), new_best_checkpoint_path)  # 保存模型的状态字典
            print('保存新的最佳模型', 'checkpoint/best-{:.3f}.pth'.format(best_test_accuracy))

    df_train_log.to_csv('训练日志-训练集.csv', index=False)
    df_test_log.to_csv('训练日志-测试集.csv', index=False)

    # 载入最佳模型作为当前模型
    model.load_state_dict(torch.load('checkpoint/best-{:.3f}.pth'.format(best_test_accuracy)))

    model.eval()
    print(evaluate_testset())


if __name__ == '__main__':
    main()


