import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
def clisa_loss(features, target):
    B = features.shape[0]
    norms = features.norm(p=2, dim=1, keepdim=True)  # 计算每行（即每个向量）的 L2 范数，形状为 B × 1
    cos_sim = torch.mm(features, features.T) / (norms * norms.T)
    sim_target = target ^ target.T
    sim_target = ~(sim_target.bool())
    sim_target = sim_target.float()
    sim_target = 2 * (sim_target - 0.5)
    mask = ~torch.eye(B, dtype=torch.bool)
    cos_sim = cos_sim[mask].reshape(B, B-1)
    sim_target = sim_target[mask].reshape(B, B-1)
    # plt.imshow(cos_sim.cpu().detach())
    # plt.savefig('sim.png')
    # plt.imshow(sim_target.cpu().detach())
    # plt.savefig('sim_tar.png')
    # plt.close()
    loss = F.mse_loss(cos_sim, sim_target)
    return loss