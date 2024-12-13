import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
BCE = torch.nn.BCEWithLogitsLoss()
CEL = torch.nn.CrossEntropyLoss()

def save_img(data, path, cmap=True):
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    # 创建图像，确保每个数据点为一个像素，保持最大分辨率
    plt.imshow(data, aspect='auto', interpolation='nearest',)
    if cmap:
        plt.colorbar()

    # 调整图像边距以去除空白
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # 保存图像时去除边距
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()

def clisa_loss(features, target):
    import torch
import torch.nn.functional as F

def clisa_loss(features, target, vis=False):
    B = features.shape[0]
    
    # Compute pairwise cosine similarity matrix
    cos_sim = torch.matmul(features, features.T)
    
    # Compute target similarity matrix (exclude diagonal)
    sim_target = target ^ target.T  # XOR to generate a similarity matrix
    sim_target = ~(sim_target.bool())  # Convert to boolean and negate (1 for similar, 0 for dissimilar)
    sim_target = sim_target.float()  # Convert back to float for BCE

    # Visualization section
    if vis:
        # Optionally save images for visualization
        fea = features.cpu().detach()
        save_img(fea[:, :1024], 'fea.png')  # Assuming save_img is defined elsewhere for saving feature images
        save_img(cos_sim.cpu().detach().view(B, B-1), 'sim.png')  # Save cosine similarity as an image
        save_img(sim_target.cpu().detach().view(B, B-1), 'sim_tar.png')  # Save target similarity as an image
    
    # Mask to get upper triangular part excluding diagonal
    mask = torch.triu(torch.ones(B, B), diagonal=1).bool()  # Upper triangular mask without diagonal
    
    # Apply mask to get the upper triangular cosine similarities and target similarities
    cos_sim = cos_sim[mask].reshape(-1)  # Flatten the upper triangular part
    sim_target = sim_target[mask].reshape(-1)  # Flatten the upper triangular part
    
    # Compute binary cross-entropy loss for the upper triangular part
    loss = F.binary_cross_entropy_with_logits(cos_sim, sim_target)
    
    return loss

