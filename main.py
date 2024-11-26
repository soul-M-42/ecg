import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torchvision import transforms
from tqdm import tqdm  # Import tqdm

# 从外部导入自定义模块
from utils.data import get_dataloaders  # 你需要在 `dataset.py` 中定义 CustomDataset
from utils.training import train_one_epoch, validate
from utils.model import Mscnn      # 你需要在 `model.py` 中定义 CustomModel

data_path = 'C:/EEE5046 Signal/PR2/ecg'
NUM_EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 1e-3
import torch
import numpy as np
import random

def set_seed(seed: int):
    # 设置 PyTorch 随机数种子
    torch.manual_seed(seed)
    # 设置 CPU 和 GPU 上的随机数种子
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 对所有 GPU 设置种子（如果有多个 GPU）
    
    # 设置 NumPy 随机数种子
    np.random.seed(seed)
    
    # 设置 Python 的 random 模块种子
    random.seed(seed)

    # 确保所有操作的随机数生成是确定性的
    torch.backends.cudnn.deterministic = True  # 确保 GPU 上的操作是确定性的
    torch.backends.cudnn.benchmark = False  # 禁用 cudnn 自动优化，避免非确定性行为



def main():
    acc_train = []
    acc_val = []
    for cv in range(4):
        print('\n')
        acc_train_best = 0
        acc_val_best = 0
        print(f'Fold {cv}')
        train_loader, val_loader = get_dataloaders(data_path, cv=0)
        model = Mscnn(1, 1).to(DEVICE)
        optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = 'class_ce'
        
        # Wrap the epoch loop with tqdm for progress bar
        for epoch in range(NUM_EPOCHS):
            print(f'Fold {cv} Epoch {epoch}')
            train_loss, train_acc, loss_clisa = train_one_epoch(
                model, train_loader, criterion, optimizer, DEVICE
            )
            print(f"Train Loss: {train_loss:.4f} Train Accuracy: {train_acc:.2f}% Clisa Loss: {loss_clisa:.2f}")
            acc_train_best = max(acc_train_best, train_acc)

            # 验证阶段
            val_loss, val_acc, loss_clisa = validate(model, val_loader, criterion, DEVICE)
            print(f"Val Loss: {val_loss:.4f} Val Accuracy: {val_acc:.2f}% Clisa Loss: {loss_clisa:.2f}")
            acc_val_best = max(acc_val_best, val_acc)  # Corrected to use acc_val_best
        
        # Add the best accuracy for each fold to the lists
        acc_train.append(acc_train_best)
        acc_val.append(acc_val_best)
    
    # Print the final results
    for cv in range(4):
        print(f'Fold {cv} Train Accuracy: {acc_train[cv]:.2f}%, Validation Accuracy: {acc_val[cv]:.2f}%')

if __name__ == "__main__":
    set_seed(42)
    main()
