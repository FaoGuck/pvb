# 来自 S2R 505b298 的 s2r/probing/hussain_lda.py，2026-11-26 复制（机械改名：S2R_→PRJ_、包名 s2r→项目名）
"""E102：Hussain & Quentin 2022 的分类流程（MVPA-Light 式收缩 LDA + 卡方排序逐个加特征 + 网格搜索）。

对每个被试：在每个训练折内按卡方 −log p 给 168 个特征排序；对 k = 1…168（前 k 个特征）× 100 个 λ 训练 LDA，
在测试折上算 AUC；各折平均后取整个网格的最大值作为该被试的 AUC（论文做法，本身偏乐观）。
置换检验时整套网格对每组打乱的标签重跑，所以显著性已考虑"在网格里挑最大值"的偏差。

LDA（MVPA-Light）：Σ_λ = (1−λ)Σ + λ·ν·I，ν = trace(Σ)/k，Σ 为合并类内协方差；w = Σ_λ⁻¹(μ₁ − μ₀)。
对固定 k 只做一次特征分解，100 个 λ 一起算。
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def chi2_scores(Xtr: np.ndarray, ytr: np.ndarray, bins: int) -> np.ndarray:
    """每个特征按训练折内分位数分 bins 箱，与 0/1 标签做 bins×2 卡方独立性检验，返回 −log p（越大越相关）。"""
    n, p = Xtr.shape
    r = np.argsort(np.argsort(Xtr, axis=0), axis=0)
    B = np.minimum((r * bins) // n, bins - 1)                           # n × p，等频分箱
    oh = B[:, :, None] == np.arange(bins)[None, None, :]                # n × p × bins
    O1 = oh[ytr == 1].sum(0).astype(float)                              # p × bins
    O0 = oh[ytr == 0].sum(0).astype(float)
    rowt = O1 + O0
    n1, n0 = (ytr == 1).sum(), (ytr == 0).sum()
    E1, E0 = rowt * n1 / n, rowt * n0 / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi = np.nansum((O1 - E1) ** 2 / E1 + (O0 - E0) ** 2 / E0, axis=1)
    return -stats.chi2.logsf(chi, df=(bins - 1))


def _auc_matrix(S: np.ndarray, y: np.ndarray) -> np.ndarray:
    """S: n × L 的分数 → 每列 AUC。"""
    r = np.argsort(np.argsort(S, axis=0), axis=0) + 1
    n1 = y.sum()
    n0 = len(y) - n1
    return ((r * y[:, None]).sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)


def fold_grid_auc(Xtr, ytr, Xte, yte, lambdas: np.ndarray, kmax: int, bins: int, zscore: bool = True) -> np.ndarray:
    """返回 kmax × L 的测试 AUC。zscore：用训练折的均值、标准差标准化（论文未写；收缩项 ν·I 对量纲敏感）。"""
    if zscore:
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd = np.where(sd > 0, sd, 1.0)
        Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    order = np.argsort(-chi2_scores(Xtr, ytr, bins), kind="stable")
    out = np.empty((kmax, len(lambdas)))
    X1, X0 = Xtr[ytr == 1], Xtr[ytr == 0]
    m1, m0 = X1.mean(0), X0.mean(0)
    C1, C0 = X1 - m1, X0 - m0
    Sw_full = (C1.T @ C1 + C0.T @ C0) / (len(ytr) - 2)
    d_full = m1 - m0
    lam = lambdas[:, None]
    for k in range(1, kmax + 1):
        idx = order[:k]
        e, V = np.linalg.eigh(Sw_full[np.ix_(idx, idx)])
        e = np.clip(e, 0, None)
        nu = e.mean()
        denom = (1 - lam) * e[None, :] + lam * nu + 1e-300              # L × k
        coef = (V.T @ d_full[idx])[None, :] / denom                     # L × k
        S = (Xte[:, idx] @ V) @ coef.T                                  # n_te × L
        out[k - 1] = _auc_matrix(S, yte)
    return out


def grid_search(X: np.ndarray, y: np.ndarray, folds: list, lambdas: np.ndarray, kmax: int, bins: int,
                zscore: bool = True):
    """folds: [(train_idx, test_idx), ...]。返回 (最佳 AUC, 最佳 k, 最佳 λ)。"""
    A = np.mean([fold_grid_auc(X[tr], y[tr], X[te], y[te], lambdas, kmax, bins, zscore) for tr, te in folds], axis=0)
    i, j = np.unravel_index(np.argmax(A), A.shape)
    return float(A[i, j]), int(i + 1), float(lambdas[j])
