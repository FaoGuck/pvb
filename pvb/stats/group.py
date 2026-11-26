# 来自 S2R 505b298 的 s2r/stats/group.py，2026-11-26 复制（机械改名：S2R_→PRJ_、包名 s2r→项目名）
"""S9 组水平统计（总纲 S9）：单样本 / 配对 Wilcoxon、配对秩二列相关（效应量）、自助法置信区间、BH-FDR。

约定
- 输入是"每个被试一个数"的向量（如提升 = 指标 − 置换均值，或两个预测器提升之差）
- 对随机水平：提升 > 0 的单侧检验；预测器之间：差值的双侧检验
- 效应量：配对秩二列相关 r = (R⁺ − R⁻) / (R⁺ + R⁻)，R 为 |d| 的秩和（去掉 0）；−1 ~ 1，0 = 无效应
- 置信区间：按被试有放回重抽样 10000 次，中位数的百分位区间（种子固定）
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def rank_biserial(d: np.ndarray) -> float:
    d = np.asarray(d, float)
    d = d[np.isfinite(d) & (d != 0)]
    if len(d) == 0:
        return np.nan
    r = stats.rankdata(np.abs(d))
    rp, rm = r[d > 0].sum(), r[d < 0].sum()
    return float((rp - rm) / (rp + rm))


def wilcoxon_p(d: np.ndarray, alternative: str) -> float:
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) < 6 or np.all(d == 0):
        return np.nan
    return float(stats.wilcoxon(d, alternative=alternative).pvalue)


def boot_median_ci(d: np.ndarray, n_boot: int = 10000, level: float = 0.95, seed: int = 0) -> tuple[float, float]:
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    med = np.median(d[idx], axis=1)
    a = (1 - level) / 2
    return float(np.quantile(med, a)), float(np.quantile(med, 1 - a))


def bh_fdr(p: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg 校正后的 q 值；NaN 保持 NaN、不计入检验数。"""
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    m = ok.sum()
    if m == 0:
        return q
    ps = p[ok]
    order = np.argsort(ps)
    ranked = ps[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(ranked, 1.0)
    q[ok] = out
    return q


def summarize(d: np.ndarray, alternative: str, seed: int = 0) -> dict:
    """一组被试差值（或提升）的汇总：n、中位数、自助 CI、更好人数、Wilcoxon p、效应量。"""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    lo, hi = boot_median_ci(d, seed=seed)
    return {"n": len(d), "median": float(np.median(d)) if len(d) else np.nan, "ci_lo": lo, "ci_hi": hi,
            "n_pos": int((d > 0).sum()), "p": wilcoxon_p(d, alternative), "r_rb": rank_biserial(d)}


def partial_spearman(y: np.ndarray, x: np.ndarray, z: np.ndarray) -> float:
    """控制 z 后 y 与 x 的偏 Spearman（秩上做线性回归残差的相关）。"""
    y, x, z = (stats.rankdata(a) for a in (y, x, z))
    Z = np.column_stack([np.ones_like(z), z])
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    return float(np.corrcoef(ry, rx)[0, 1])
