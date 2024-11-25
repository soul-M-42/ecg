import torch
import torch.nn as nn
import torch.nn.functional as F
ce_loss = nn.BCELoss()


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for data, target in dataloader:
        data, target = data.to(device), target.to(device)

        # 前向传播
        outputs = model(data)
        loss = 0
        if(criterion == 'class_ce'):
            loss = loss + ce_loss(outputs.float(), target.float())
        else:
            raise('Loss not implemented ERROR')
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
    return avg_loss, accuracy

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)

            # 前向传播
            outputs = model(data)
            loss = 0
            if(criterion == 'class_ce'):
                loss = loss + ce_loss(outputs.float(), target.float())
            else:
                raise('Loss not implemented ERROR')

            # 统计
            total_loss += loss.item()
            outputs[outputs >= 0.5] = 1
            outputs[outputs < 0.5] = 0
            correct += (outputs == target).sum().item()
            total += target.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy
