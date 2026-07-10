"""掩码可信度验证(不重写逻辑，验证不变量→敢信)。三条信任关键属性：
  ① 有限性：含整列缺失/整时间步缺失时，输出无 NaN/Inf。
  ② 无泄漏：改"缺失资产"的输入值，**不**改变"在场资产"的输出（缺失股不污染真预测）。
  ③ 置换等变：打乱资产顺序，输出同样打乱（any-variate 内核）。
跑：PYTHONUTF8=1 .venv/Scripts/python.exe test_masking.py
"""
import torch
from model.train import Net

torch.manual_seed(0)
B, T, N, C, d, L = 2, 8, 12, 7, 32, 2
net = Net(d=d, L=L, T=T, C=C).eval()          # eval→关 dropout，确定性
dt = torch.full((B,), 3.0)

x = torch.randn(B, T, N, C)
xmask = torch.ones(B, T, N, dtype=torch.bool)
absent = [2, 5, 9]                              # 这几个 slot 全时段缺失
present = [i for i in range(N) if i not in absent]
xmask[:, :, absent] = False
xmask[:, 3, :] = False                          # 再制造一个"整时间步全缺失"的极端行
x = x * xmask.unsqueeze(-1)                      # 缺失格清零（与管线一致）

with torch.no_grad():
    out1 = net(x, xmask, dt)                     # [B,N,K]

# ① 有限性
finite = torch.isfinite(out1).all().item()
print(f"① 有限性（含整列/整时间步缺失）：{'✓' if finite else '✗ 出现 NaN/Inf'}")

# ② 无泄漏：只改缺失资产的输入值
x2 = x.clone()
x2[:, :, absent] = torch.randn(B, T, len(absent), C)   # 缺失格塞任意值
x2 = x2 * xmask.unsqueeze(-1)
with torch.no_grad():
    out2 = net(x2, xmask, dt)
leak = (out1[:, present] - out2[:, present]).abs().max().item()
print(f"② 无泄漏（缺失资产输入变→在场输出变？）：max|Δ|={leak:.2e}  {'✓' if leak < 1e-5 else '✗ 缺失股泄漏进真预测!'}")

# ③ 置换等变：打乱资产轴
perm = torch.randperm(N)
with torch.no_grad():
    out_p = net(x[:, :, perm], xmask[:, :, perm], dt)
equiv = (out_p - out1[:, perm]).abs().max().item()
print(f"③ 置换等变（打乱资产→输出同样打乱？）：max|Δ|={equiv:.2e}  {'✓' if equiv < 1e-5 else '✗ 不等变!'}")

ok = finite and leak < 1e-5 and equiv < 1e-5
print(f"\n掩码可信度：{'✓ 三条全过，敢信' if ok else '✗ 有不变量被破坏，需精准修'}")
