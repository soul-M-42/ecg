import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm  # Import tqdm
from utils.tools import clisa_loss, save_img
from sklearn.metrics import f1_score, accuracy_score
ce_loss = nn.BCELoss()

def compute_score(outputs, targets):
    N = outputs.shape[0]
    """
    计算二分类的 F1 分数。

    参数:
    outputs (torch.Tensor): 模型的输出值，范围为 0 到 1。
    targets (torch.Tensor): 真实标签，取值为 0 或 1。

    返回:
    float: 计算得到的 F1 分数。
    """
    # 将 outputs 转换为二进制预测值
    predicted = (outputs >= 0.5).int()

    # 将 tensors 转换为 numpy 数组，以便使用 sklearn 计算 F1 分数
    predicted_np = predicted.cpu().numpy()
    targets_np = targets.cpu().numpy()
    # print(f'Abnormal predicted/real: {predicted_np.sum()}/{targets.sum()}')
    # print(f'Normal predicted/real: {N-predicted_np.sum()}/{N-targets.sum()}')

    # 计算 F1 分数
    f1 = f1_score(targets_np, predicted_np, average='binary')
    acc = accuracy_score(targets_np, predicted_np)
    
    return f1, acc

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    f1_train = []
    acc_train = []

    for data, target in tqdm(dataloader, desc="Training", total=len(dataloader)):
        data, target = data.to(device), target.to(device)

        # 前向传播
        outputs, feature = model(data)
        loss = 0
        if('class' in criterion):
            loss = loss + ce_loss(outputs.float(), target.float())
        if('clisa' in criterion):
            loss_clisa = clisa_loss(feature, target)
            loss = loss + loss_clisa
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计
        total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        f1,acc = compute_score(outputs, target)
        f1_train.append(f1)
        acc_train.append(acc)
        # save_img(torch.stack([outputs, target]), 'train_logits.png')
    f1 = np.mean(f1_train)
    acc = np.mean(acc_train)
    return avg_loss, loss_clisa, f1, acc

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    f1_val = []
    acc_val = []

    with torch.no_grad():
        for data, target in tqdm(dataloader, desc="Validating", total=len(dataloader)):
            data, target = data.to(device), target.to(device)

            # 前向传播
            outputs, feature = model(data)
            loss = 0
            if('class' in criterion):
                loss = loss + ce_loss(outputs.float(), target.float())
            if('clisa' in criterion):
                loss_clisa = clisa_loss(feature, target)
                loss = loss + loss_clisa

            # 统计
            total_loss += loss.item()
            avg_loss = total_loss / len(dataloader)
            f1,acc = compute_score(outputs, target)
            f1_val.append(f1)
            acc_val.append(acc)
            # save_img(torch.stack([outputs, target]), 'val_logits.png')
    f1 = np.mean(f1_val)
    acc = np.mean(acc_val)
    return avg_loss, loss_clisa, f1, acc
