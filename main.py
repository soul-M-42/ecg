import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torchvision import transforms

# 从外部导入自定义模块
from utils.data import get_dataloaders  # 你需要在 `dataset.py` 中定义 CustomDataset
from utils.training import train_one_epoch, validate
from utils.model import Mscnn      # 你需要在 `model.py` 中定义 CustomModel

data_path = 'C:/EEE5046 Signal/PR2/ecg'
NUM_EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 1e-3

def main():
    for cv in range(4):
        train_loader, val_loader = get_dataloaders(data_path, cv=0)
        model = Mscnn(1, 1).to(DEVICE)
        optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = 'class_ce'
        for epoch in range(NUM_EPOCHS):
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}")

            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, DEVICE
            )
            print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")

            # 验证阶段
            val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
            print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")
        
    return

if __name__ == "__main__":
    main()