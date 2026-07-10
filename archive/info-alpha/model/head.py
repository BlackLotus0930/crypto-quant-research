"""Δt-FiLM + 单调分位头（设计 §4.3）。base + cumsum(softplus(Δ)) 保证分位不交叉。"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantileHead(nn.Module):
    def __init__(self, d, K=21, dt_dim=16):
        super().__init__()
        self.dt = nn.Sequential(nn.Linear(1, dt_dim), nn.GELU(), nn.Linear(dt_dim, 2 * d))
        self.base = nn.Linear(d, 1)
        self.deltas = nn.Linear(d, K - 1)

    def forward(self, h, dt):
        # h [B,N,d], dt [B]
        g, b = self.dt(dt.view(-1, 1).float()).chunk(2, -1)         # FiLM
        h = h * (1 + g).unsqueeze(1) + b.unsqueeze(1)
        base = self.base(h)                                         # [B,N,1]
        inc = F.softplus(self.deltas(h))                            # [B,N,K-1] > 0
        return torch.cat([base, base + torch.cumsum(inc, -1)], -1)  # [B,N,K] 单调
