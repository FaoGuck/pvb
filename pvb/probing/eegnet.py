# 来自 S2R 505b298 的 s2r/probing/eegnet.py，2026-11-26 复制（机械改名：S2R_→PRJ_、包名 s2r→项目名）
"""E105：被试内监督 EEGNet（总纲 C2，Lawhern 2018 的 EEGNet-8,2），纯 torch 实现，不依赖 braindecode。

做法要点（与 S8 的线性探测同一套折、同一批试验、同一组置换）
- 外层：留一 block 作测试（D1 6 折）；剩下 5 个 block 里取"测试 block 的下一个 block"（循环）作早停验证（总纲 6.8 第 5 条），
  其余 4 个 block 训练；早停选验证损失最小的 epoch，用该 epoch 的模型预测测试 block
- 输入：S3 的 clean_<W>（250 Hz、1–45 Hz、平均参考），降采样到 125 Hz；每个导联用训练集（4 个 block）的均值/标准差标准化
- 目标：回归 = mep_resid（训练集上 z 分数）+ MSE；分类 = block 内中位数切分 + BCE
- 置换：真实目标与 P 个置换目标作为 K = 1 + P 个**独立模型**打包训练（PackedEEGNet：分组卷积 + 按模型分开的全连接层），
  每个模型自己的参数、BatchNorm 统计、Adam 状态、早停；与分别训练 K 次在数学上等价，只是快得多
- 指标：各外层折的测试预测拼接后算（Spearman / AUC），与 cv.nested_lobo 相同
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import cv


class PackedEEGNet(nn.Module):
    """K 个独立的 EEGNet-F1,D 打包成一个网络。输入 (B, 1, C, T)（所有模型同一输入），输出 (B, K)。

    通道排列：每一层的通道按模型连续存放（模型 k 占 [k·m, (k+1)·m)），分组卷积保证模型之间不共享参数；
    BatchNorm 是逐通道的，所以每个模型的 BN 也独立。
    """

    def __init__(self, K: int, n_ch: int, n_t: int, F1: int = 8, D: int = 2, F2: int = 16,
                 k1: int = 64, k3: int = 16, pool1: int = 4, pool2: int = 8, dropout: float = 0.25):
        super().__init__()
        self.K, self.F2 = K, F2
        self.conv1 = nn.Conv2d(1, K * F1, (1, k1), padding="same", bias=False)                 # 时间滤波
        self.bn1 = nn.BatchNorm2d(K * F1)
        self.conv2 = nn.Conv2d(K * F1, K * F1 * D, (n_ch, 1), groups=K * F1, bias=False)       # 逐滤波器的空间滤波
        self.bn2 = nn.BatchNorm2d(K * F1 * D)
        self.pool1 = nn.AvgPool2d((1, pool1))
        self.conv3 = nn.Conv2d(K * F1 * D, K * F1 * D, (1, k3), padding="same", groups=K * F1 * D, bias=False)
        self.conv4 = nn.Conv2d(K * F1 * D, K * F2, 1, groups=K, bias=False)                    # 可分离卷积的逐点部分（按模型分组）
        self.bn3 = nn.BatchNorm2d(K * F2)
        self.pool2 = nn.AvgPool2d((1, pool2))
        self.drop = nn.Dropout(dropout)
        with torch.no_grad():
            t_out = self._features(torch.zeros(2, 1, n_ch, n_t)).shape[-1]
        self.n_feat = F2 * t_out
        self.fc_w = nn.Parameter(torch.empty(K, self.n_feat))
        self.fc_b = nn.Parameter(torch.zeros(K))
        nn.init.kaiming_uniform_(self.fc_w, a=5 ** 0.5)

    def _features(self, x):
        h = self.bn1(self.conv1(x))
        h = self.drop(self.pool1(F.elu(self.bn2(self.conv2(h)))))
        h = self.conv4(self.conv3(h))
        h = self.drop(self.pool2(F.elu(self.bn3(h))))
        return h                                                    # (B, K·F2, 1, t_out)

    def forward(self, x):
        h = self._features(x)
        B = h.shape[0]
        h = h.reshape(B, self.K, -1)                                # 每个模型自己的 F2·t_out 个特征
        return torch.einsum("bkf,kf->bk", h, self.fc_w) + self.fc_b

    @torch.no_grad()
    def apply_maxnorm(self, spatial: float | None, dense: float | None):
        """EEGNet 原文：空间滤波器范数 ≤ 1，全连接层范数 ≤ 0.25（每个滤波器 / 每个模型各自约束）。"""
        if spatial:
            w = self.conv2.weight                                   # (K·F1·D, 1, C, 1)
            n = w.flatten(1).norm(dim=1).clamp(min=spatial)
            w.mul_((spatial / n).view(-1, 1, 1, 1))
        if dense:
            n = self.fc_w.norm(dim=1).clamp(min=dense)
            self.fc_w.mul_((dense / n).view(-1, 1))


def _loss_cols(out: torch.Tensor, y: torch.Tensor, task: str) -> torch.Tensor:
    """out, y: (B, K) → 每个模型的平均损失 (K)。"""
    if task == "regression":
        return ((out - y) ** 2).mean(0)
    return F.binary_cross_entropy_with_logits(out, y, reduction="none").mean(0)


def fit_fold(Xtr, Ytr, Xva, Yva, Xte, task: str, hp: dict, seed: int, dev: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """一折：训练 K = Ytr.shape[1] 个独立模型，按各自的验证损失早停。
    返回 (测试预测 n_te × K, 每个模型选中的 epoch K)。"""
    K = Ytr.shape[1]
    g = torch.Generator(device="cpu").manual_seed(seed)
    torch.manual_seed(seed)
    n_ch, n_t = Xtr.shape[1], Xtr.shape[2]
    net = PackedEEGNet(K, n_ch, n_t, hp["F1"], hp["D"], hp["F2"], hp["k1"], hp["k3"], hp["pool1"], hp["pool2"],
                       hp["dropout"]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=hp["lr"], weight_decay=hp.get("weight_decay", 0.0))
    Xtr_, Xva_, Xte_ = (torch.as_tensor(a, dtype=torch.float32, device=dev)[:, None] for a in (Xtr, Xva, Xte))
    Ytr_ = torch.as_tensor(Ytr, dtype=torch.float32, device=dev)
    Yva_ = torch.as_tensor(Yva, dtype=torch.float32, device=dev)
    best = torch.full((K,), float("inf"), device=dev)
    best_ep = torch.zeros(K, dtype=torch.long, device=dev)
    pred_te = torch.zeros(len(Xte), K, device=dev)
    since = torch.zeros(K, dtype=torch.long, device=dev)
    n, bs = len(Xtr), int(hp["batch"])
    for ep in range(int(hp["max_epochs"])):
        net.train()
        order = torch.randperm(n, generator=g).to(dev)
        for i in range(0, n, bs):
            idx = order[i: i + bs]
            if len(idx) < 2:                                       # BatchNorm 需要 ≥ 2 个样本
                continue
            loss = _loss_cols(net(Xtr_[idx]), Ytr_[idx], task).sum()   # 求和：每个模型的梯度只来自自己的损失
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            net.apply_maxnorm(hp.get("maxnorm_spatial"), hp.get("maxnorm_dense"))
        net.eval()
        with torch.no_grad():
            vl = _loss_cols(net(Xva_), Yva_, task)
            imp = vl < best
            if imp.any():
                pt = net(Xte_)
                pred_te[:, imp] = pt[:, imp]
                best = torch.where(imp, vl, best)
                best_ep = torch.where(imp, torch.full_like(best_ep, ep), best_ep)
            since = torch.where(imp, torch.zeros_like(since), since + 1)
        if bool((since >= int(hp["patience"])).all()):
            break
    return pred_te.cpu().numpy(), best_ep.cpu().numpy()


def zscore_channels(Xtr: np.ndarray, *others: np.ndarray):
    """每个导联用训练集（试验 × 时间）的均值 / 标准差标准化。"""
    mu = Xtr.mean(axis=(0, 2), keepdims=True)
    sd = Xtr.std(axis=(0, 2), keepdims=True) + 1e-8
    return [(a - mu) / sd for a in (Xtr, *others)]


def lobo_eegnet(X: np.ndarray, Y: np.ndarray, blocks: np.ndarray, task: str, hp: dict, seed: int,
                dev: torch.device, pack: int | None = None) -> dict:
    """X: n × C × T；Y: n × K（第 0 列真实目标）。返回 {'pred': n × K, 'metric': K, 'epochs': {折: 真实目标选中的 epoch}}。
    pack：一次打包训练的模型数上限（显存不够时调小；结果不变，只是分几批）。"""
    n, K = Y.shape
    pack = pack or K
    ub = np.unique(blocks)
    pred = np.zeros((n, K), dtype=np.float32)
    epochs = {}
    for j, b_out in enumerate(ub):
        b_val = ub[(j + 1) % len(ub)]                               # 早停验证：测试 block 的下一个 block（循环）
        te = np.where(blocks == b_out)[0]
        va = np.where(blocks == b_val)[0]
        tr = np.where((blocks != b_out) & (blocks != b_val))[0]
        Xtr, Xva, Xte = zscore_channels(X[tr], X[va], X[te])
        Ytr, Yva = Y[tr].astype(np.float32), Y[va].astype(np.float32)
        if task == "regression":                                    # 目标在训练集上 z 分数（每列各自）
            mu, sd = Ytr.mean(0, keepdims=True), Ytr.std(0, keepdims=True) + 1e-8
            Ytr, Yva = (Ytr - mu) / sd, (Yva - mu) / sd
        for c0 in range(0, K, pack):
            sl = slice(c0, min(K, c0 + pack))
            p, ep = fit_fold(Xtr, Ytr[:, sl], Xva, Yva[:, sl], Xte, task, hp, seed * 1009 + j * 101 + c0, dev)
            pred[te, sl] = p
            if c0 == 0:
                epochs[int(b_out)] = int(ep[0])
    P = torch.as_tensor(pred)
    Yt = torch.as_tensor(Y, dtype=torch.float32)
    metric = cv.spearman_cols(P, Yt) if task == "regression" else cv.auc_cols(P, Yt)
    return {"pred": pred, "metric": metric.numpy(), "epochs": epochs}
