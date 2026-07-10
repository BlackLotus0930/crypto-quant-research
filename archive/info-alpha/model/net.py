"""最小双轴 attention 编码器（设计 §4.2）。

资产轴 any-variate（masked, 无资产位置 = 置换等变）+ 时间轴 causal。读 anchor 日 token。
用 F.scaled_dot_product_attention（flash/mem-efficient kernel，不物化 N×N 矩阵）→ 省显存、可放大 N。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoid(T, d):
    pe = torch.zeros(T, d)
    pos = torch.arange(T).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class Attn(nn.Module):
    """多头注意力（SDPA）。asset 轴用 key_padding（加性 mask）；time 轴用 is_causal。二者不同时用。"""
    def __init__(self, d, heads, p=0.1):
        super().__init__()
        self.h, self.dh, self.p = heads, d // heads, p
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)

    def forward(self, x, key_padding=None, is_causal=False):
        Bx, S, d = x.shape
        q, k, v = self.qkv(x).view(Bx, S, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)  # 各 [Bx,h,S,dh]
        attn_mask = None
        if key_padding is not None:                       # [Bx,S] True=pad
            attn_mask = torch.zeros(Bx, 1, 1, S, dtype=q.dtype, device=x.device)
            attn_mask = attn_mask.masked_fill(key_padding[:, None, None, :], float("-inf"))
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal,
                                           dropout_p=self.p if self.training else 0.0)
        return self.proj(o.transpose(1, 2).reshape(Bx, S, d))


class Block(nn.Module):
    def __init__(self, d, heads, p=0.1):
        super().__init__()
        self.an = nn.LayerNorm(d); self.aattn = Attn(d, heads, p)
        self.tn = nn.LayerNorm(d); self.tattn = Attn(d, heads, p)
        self.mn = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d), nn.Dropout(p))

    @staticmethod
    def _safe_kpm(kpm):
        empty = kpm.all(1)
        kpm = kpm.clone(); kpm[empty] = False     # 全 pad 行临时设不 pad，输出会被 *m 清零
        return kpm

    def forward(self, h, xmask):
        B, T, N, d = h.shape
        m = xmask.unsqueeze(-1)
        # 资产轴：[B*T, N, d]，masked any-variate（无位置）
        a = self.an(h).reshape(B * T, N, d)
        kpm = self._safe_kpm((~xmask).reshape(B * T, N))
        ao = self.aattn(a, key_padding=kpm)
        h = h + torch.nan_to_num(ao.reshape(B, T, N, d), nan=0.0, posinf=0.0, neginf=0.0) * m
        # 时间轴：[B*N, T, d]，causal（只用 causal，不加 key_padding → 无全 -inf 行 → 无 NaN 梯度）
        t = self.tn(h).permute(0, 2, 1, 3).reshape(B * N, T, d)
        to = self.tattn(t, is_causal=True)
        h = h + torch.nan_to_num(to.reshape(B, N, T, d).permute(0, 2, 1, 3), nan=0.0, posinf=0.0, neginf=0.0) * m
        h = h + self.mlp(self.mn(h)) * m
        return h


class Encoder(nn.Module):
    def __init__(self, C=5, d=64, L=2, heads=4, T=64):
        super().__init__()
        self.embed = nn.Linear(C, d)
        self.register_buffer("pe", sinusoid(T, d))
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(L)])
        self.d = d

    def forward(self, x, xmask):
        B, T, N, _ = x.shape
        h = self.embed(x) * xmask.unsqueeze(-1)
        h = h + self.pe[:T].view(1, T, 1, self.d)
        for blk in self.blocks:
            h = blk(h, xmask)
        return h[:, -1]                              # anchor 日 token [B,N,d]
