"""T2 平凡与经典基线（00 第 4 节 P205）与统一的评估（第 5.2 节）。

单位 = 刺激对 p × 记录电极 r（评估集已排除刺激电极与相邻电极）；目标 = N1 有无。每名患者分别：
  B0 距离        d、log d、1/d（刺激对中点到 r 的 MNI 距离）
  B1 接收端先验  r 作为记录电极、对“可用折”里其他刺激对的 N1 出现率与 N1 窗最小值的均值
  B2 连接度      观测段（只来自“可用折”刺激对的段）上 r 与 p 的两个电极之间各连接度的均值与最大值
  B3 = B0 + B1 + B2
“可用折”：测试折 f 的单位 → 除 f 外的折；训练单位（在折 g）→ 除 f、g 外的折。于是任何单位的特征都不含
它自己刺激对的标签或观测段，且训练与测试的特征按同一方式构造。
预测头：标准化 + L2 逻辑回归；C 由训练折上的留一折（按折分组）内层 CV 选（AUROC）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

FEATURE_SETS = {"B0": ["B0"], "B1": ["B1"], "B2": ["B2"], "B0+B1": ["B0", "B1"], "B0+B2": ["B0", "B2"],
                "B3": ["B0", "B1", "B2"]}


class ObsPool:
    """把一名患者各 run 的 owner 累加量合到患者级通道表上，按“可用 owner 集合”求平均连接度。"""

    def __init__(self, accs: list[dict]):
        chans = sorted({c for a in accs for c in a["channels"]})
        self.chans = chans
        self.ci = {c: i for i, c in enumerate(chans)}
        self.measures = list(accs[0]["measures"])
        owners = sorted({o for a in accs for o in a["owners"]})
        self.owners = owners
        oi = {o: i for i, o in enumerate(owners)}
        n, m = len(chans), len(self.measures)
        self.sums = np.zeros((len(owners), m, n, n))
        self.counts = np.zeros((len(owners), n, n))
        for a in accs:
            idx = np.array([self.ci[c] for c in a["channels"]])
            for k, o in enumerate(a["owners"]):
                self.sums[oi[o]][:, idx[:, None], idx[None, :]] += a["sums"][k]
                self.counts[oi[o]][np.ix_(idx, idx)] += a["counts"][k]
        self._cache: dict = {}

    def mean(self, owner_mask: np.ndarray) -> np.ndarray:
        key = owner_mask.tobytes()
        if key not in self._cache:
            s = self.sums[owner_mask].sum(axis=0)
            c = self.counts[owner_mask].sum(axis=0)
            with np.errstate(all="ignore"):
                M = s / c[None]
            d = np.arange(M.shape[1])
            M[:, d, d] = np.nan                            # 自身连接无意义（置换时 r 可能正是刺激电极）
            self._cache[key] = M
        return self._cache[key]


def b0(dist: np.ndarray) -> np.ndarray:
    d = np.maximum(dist, 1.0)
    return np.stack([d, np.log(d), 1.0 / d], axis=1)


def b1(units: pd.DataFrame, source: pd.DataFrame, allowed_pairs: set) -> np.ndarray:
    """units 的接收端先验：source（患者的全部单位）里“可用折”刺激对的标签，按记录电极平均。"""
    tr = source[source.stim_pair.isin(allowed_pairs)]
    g = tr.groupby("channel").agg(rate=("n1_detected", "mean"), amp=("winmin_uv", "mean"))
    return g.reindex(units.channel.to_numpy()).to_numpy(float)


def b2(units: pd.DataFrame, pool: ObsPool, owner_mask: np.ndarray) -> np.ndarray:
    M = pool.mean(owner_mask)                          # (n_meas, n_ch, n_ch)
    ci = pool.ci
    n = len(units)
    out = np.full((n, 2 * M.shape[0]), np.nan)
    r = np.array([ci.get(c, -1) for c in units.channel])
    e = np.array([[ci.get(x, -1) for x in p.split("-")] for p in units.stim_pair])
    ok = (r >= 0) & (e >= 0).all(axis=1)
    v1 = M[:, e[ok, 0], r[ok]].T
    v2 = M[:, e[ok, 1], r[ok]].T
    out[ok, : M.shape[0]] = np.nanmean(np.stack([v1, v2]), axis=0)
    out[ok, M.shape[0]:] = np.nanmax(np.stack([v1, v2]), axis=0)
    return out


def build_features(units: pd.DataFrame, fold_of_pair: dict, pool: ObsPool | None, excl_folds: set,
                   b2_pair_map: dict | None = None, source: pd.DataFrame | None = None) -> dict:
    """对 units（同一组“可用折”）构造 B0/B1/B2。excl_folds = 不可用的折；source = 患者全部单位（B1 的标签来源，
    只取可用折的刺激对）。b2_pair_map：置换时 刺激对 → 取其 B2 的刺激对。"""
    allowed = {p for p, f in fold_of_pair.items() if f not in excl_folds}
    src = units if source is None else source
    feats = {"B0": b0(units.dist_mm.to_numpy(float)), "B1": b1(units, src, allowed)}
    if pool is not None:
        om = np.array([o in allowed for o in pool.owners])
        u = units if b2_pair_map is None else units.assign(stim_pair=units.stim_pair.map(b2_pair_map))
        feats["B2"] = b2(u, pool, om)
    return feats


def _fit_predict(Xtr, ytr, Xte, C):
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    Xtr = np.where(np.isfinite(Xtr), Xtr, med); Xte = np.where(np.isfinite(Xte), Xte, med)
    if len(np.unique(ytr)) < 2:                        # 训练集只有一类：给常数预测
        return np.full(len(Xte), float(np.mean(ytr)))
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(Xtr), ytr)
    return m.predict_proba(sc.transform(Xte))[:, 1]


def safe_auc(y, s) -> float:
    y = np.asarray(y, bool)
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def cv_predict(units: pd.DataFrame, fold_of_pair: dict, pool: ObsPool | None, sets: list[str], cfg_b: dict,
               fixed_C: dict | None = None, b2_pair_map: dict | None = None) -> tuple[dict, dict]:
    """外层 K 折（按刺激对的折）。返回 {set: 每个单位的预测分数}、{(set, fold): 选中的 C}。"""
    fold = units.stim_pair.map(fold_of_pair).to_numpy()
    y = units.n1_detected.to_numpy(bool)
    K = int(np.nanmax(fold)) + 1
    pred = {s: np.full(len(units), np.nan) for s in sets}
    chosen = {}
    for f in range(K):
        te = fold == f
        if not te.any():
            continue
        Fte = build_features(units[te], fold_of_pair, pool, {f}, b2_pair_map, source=units)
        tr_parts = {}
        for g in range(K):
            if g == f or not (fold == g).any():
                continue
            tr_parts[g] = build_features(units[fold == g], fold_of_pair, pool, {f, g}, b2_pair_map, source=units)
        gs = sorted(tr_parts)
        ytr = np.concatenate([y[fold == g] for g in gs])
        gtr = np.concatenate([np.full((fold == g).sum(), g) for g in gs])
        for s in sets:
            cols = FEATURE_SETS[s]
            Xtr = np.concatenate([np.hstack([tr_parts[g][c] for c in cols]) for g in gs])
            Xte = np.hstack([Fte[c] for c in cols])
            if fixed_C is not None:
                C = fixed_C[(s, f)]
            else:
                best, C = -1.0, cfg_b["C_grid"][0]
                for c in cfg_b["C_grid"]:
                    sc = []
                    for h in gs:                                   # 内层：在训练折上留一折
                        tr_h, va_h = gtr != h, gtr == h
                        if 0 < ytr[va_h].sum() < va_h.sum() and 0 < ytr[tr_h].sum() < tr_h.sum():
                            sc.append(safe_auc(ytr[va_h], _fit_predict(Xtr[tr_h], ytr[tr_h], Xtr[va_h], c)))
                    v = np.nanmean(sc) if sc else -1.0
                    if v > best:
                        best, C = v, c
            chosen[(s, f)] = C
            pred[s][te] = _fit_predict(Xtr, ytr, Xte, C)
    return pred, chosen
