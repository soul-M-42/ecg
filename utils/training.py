import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm  # Import tqdm
from utils.tools import clisa_loss
ce_loss = nn.BCELoss()


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for data, target in tqdm(dataloader, desc="Training", total=len(dataloader)):
        data, target = data.to(device), target.to(device)

        # 前向传播
        outputs, feature = model(data)
        loss = 0
        if(criterion == 'class_ce'):
            loss = loss + ce_loss(outputs.float(), target.float())
        else:
            raise('Loss not implemented ERROR')
        # clisa loss
        loss_clisa = clisa_loss(feature, target)
        loss = loss + loss_clisa
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计
        total_loss += loss.item()
        outputs[outputs >= 0.5] = 1
        outputs[outputs < 0.5] = 0
        correct += (outputs == target).sum().item()
        total += target.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy, loss_clisa

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in tqdm(dataloader, desc="Validating", total=len(dataloader)):
            data, target = data.to(device), target.to(device)

            # 前向传播
            outputs, feature = model(data)
            loss = 0
            if(criterion == 'class_ce'):
                loss = loss + ce_loss(outputs.float(), target.float())
            else:
                raise('Loss not implemented ERROR')
            # clisa loss
            loss_clisa = clisa_loss(feature, target)
            loss = loss + loss_clisa
            # 统计
            total_loss += loss.item()
            outputs[outputs >= 0.5] = 1
            outputs[outputs < 0.5] = 0
            correct += (outputs == target).sum().item()
            total += target.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy, loss_clisa
