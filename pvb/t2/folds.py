"""T2 的折划分与评估集（00 第 5.2 节第 1、2 条）。

- 共用一个电极的刺激对必须同折：刺激对按共用电极连成连通组（并查集），以组为单位贪心分到 K 折，
  每次把最大的未分配组放进当前刺激对最少的折（同样大小时按随机种子打乱的顺序）
- 评估集排除：记录电极属于刺激对本身，或与任一刺激电极“直接相邻”（MNI 距离 ≤ adjacent_factor × 该患者电极
  最近邻距离的中位数）
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def pair_components(pairs: list[str]) -> dict[str, int]:
    """无序刺激对 → 连通组编号（共用电极即相连）。"""
    par = {p: p for p in pairs}

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    by_el: dict[str, list[str]] = {}
    for p in pairs:
        for e in p.split("-"):
            by_el.setdefault(e, []).append(p)
    for ps in by_el.values():
        for q in ps[1:]:
            par[find(q)] = find(ps[0])
    roots = sorted({find(p) for p in pairs})
    rid = {r: i for i, r in enumerate(roots)}
    return {p: rid[find(p)] for p in pairs}


def assign_folds(pairs: list[str], k: int = 5, seed: int = 0) -> dict[str, int]:
    """刺激对 → 折号（0…k-1）。组不可拆。"""
    comp = pair_components(sorted(set(pairs)))
    groups: dict[int, list[str]] = {}
    for p, c in comp.items():
        groups.setdefault(c, []).append(p)
    rng = np.random.default_rng(seed)
    order = list(groups)
    rng.shuffle(order)
    order.sort(key=lambda c: -len(groups[c]))          # 稳定排序：大组先放，同大小按随机顺序
    load = [0] * k
    out = {}
    for c in order:
        f = int(np.argmin(load))
        for p in groups[c]:
            out[p] = f
        load[f] += len(groups[c])
    return out


def adjacency_threshold(el_xyz: np.ndarray, factor: float) -> float:
    """电极坐标 (n, 3) → factor × 最近邻距离的中位数（mm）。"""
    x = el_xyz[np.isfinite(el_xyz).all(axis=1)]
    d = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    return float(factor * np.median(d.min(axis=1)))


def eval_mask(units: pd.DataFrame, el: pd.DataFrame, thr_mm: float) -> np.ndarray:
    """units 需有 stim_pair、channel 列；el 以大写电极名为索引、含 x/y/z。返回 True = 进入评估集。"""
    xyz = el[["x", "y", "z"]]
    keep = np.ones(len(units), dtype=bool)
    for i, (p, ch) in enumerate(zip(units.stim_pair.to_numpy(), units.channel.to_numpy())):
        e1, e2 = p.split("-")
        if ch in (e1, e2) or ch not in xyz.index:
            keep[i] = False
            continue
        r = xyz.loc[ch].to_numpy(float)
        for e in (e1, e2):
            if e in xyz.index and np.linalg.norm(r - xyz.loc[e].to_numpy(float)) <= thr_mm:
                keep[i] = False
                break
    return keep
