"""00 第 7.2 节 test_t2_folds：共用电极的刺激对同折；被排除的记录电极不在评估集；特征不含自己刺激对的信息。
运行：python tests/test_t2_folds.py（或 pytest）
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from pvb.t2 import baselines as bl, folds as fo  # noqa: E402

CFG_B = dict(C_grid=[0.1, 1.0], inner_k=4, n_perm=2, perm_seed=0)


def grid():
    """6 条 8 触点电极条（10 mm 间距），每条 7 个相邻刺激对 + 2 个孤立对。"""
    rows, pairs = [], []
    for s, lab in enumerate("ABCDEF"):
        for i in range(1, 9):
            rows.append((f"{lab}{i:02d}", 10.0 * i, 15.0 * s, 0.0))
        pairs += [f"{lab}{i:02d}-{lab}{i + 1:02d}" for i in range(1, 8)]
    rows += [("Z01", 200.0, 0, 0), ("Z02", 210.0, 0, 0), ("Z03", 300.0, 0, 0), ("Z04", 310.0, 0, 0)]
    pairs += ["Z01-Z02", "Z03-Z04"]
    el = pd.DataFrame(rows, columns=["NAME", "x", "y", "z"]).set_index("NAME")
    return el, pairs


def test_shared_electrode_same_fold():
    el, pairs = grid()
    f = fo.assign_folds(pairs, 5, 0)
    by_el = {}
    for p in pairs:
        for e in p.split("-"):
            by_el.setdefault(e, set()).add(f[p])
    assert all(len(v) == 1 for v in by_el.values())
    assert set(f.values()) == set(range(5))


def test_eval_mask_excludes_stim_and_adjacent():
    el, _ = grid()
    thr = fo.adjacency_threshold(el[["x", "y", "z"]].to_numpy(float), 1.5)
    assert abs(thr - 15.0) < 1e-6                                   # 最近邻 10 mm × 1.5
    u = pd.DataFrame({"stim_pair": ["A03-A04"] * 6, "channel": ["A03", "A02", "A05", "A06", "B03", "C08"]})
    keep = fo.eval_mask(u, el, thr)
    # A03 = 刺激电极；A02、A05 距刺激电极 10 mm；B03 距 A03 15 mm（≤ 15 → 相邻）；A06 距 A04 20 mm；C08 远
    assert keep.tolist() == [False, False, False, True, False, True]


def units_and_pool():
    el, pairs = grid()
    rng = np.random.default_rng(0)
    rows = []
    for p in pairs:
        e1, e2 = p.split("-")
        mid = el.loc[[e1, e2]].to_numpy(float).mean(axis=0)
        for ch in el.index:
            if ch in (e1, e2):
                continue
            d = float(np.linalg.norm(el.loc[ch].to_numpy(float) - mid))
            y = rng.random() < np.exp(-d / 15)
            rows.append(("r1", "S", p, ch, y, -300.0 * y + rng.normal(0, 30), d))
    u = pd.DataFrame(rows, columns=["run_id", "subject", "stim_pair", "channel", "n1_detected", "winmin_uv", "dist_mm"])
    chans = list(el.index)
    n = len(chans)
    acc = dict(owners=np.array(pairs), measures=np.array(["m1", "m2"]), channels=np.array(chans),
               sums=rng.random((len(pairs), 2, n, n)).astype(np.float32), counts=np.ones((len(pairs), n, n), np.int32))
    return u, bl.ObsPool([acc]), pairs


def test_test_features_ignore_own_labels_and_segments():
    u, pool, pairs = units_and_pool()
    f = fo.assign_folds(pairs, 5, 0)
    te = u[u.stim_pair.map(f) == 0]
    F0 = bl.build_features(te, f, pool, {0}, source=u)
    assert np.isfinite(F0["B1"]).mean() > 0.9 and np.isfinite(F0["B2"]).mean() > 0.9   # 特征确实有值（11-26 曾因 B1 全为 NaN 而“逐位不变”）
    # 改掉测试折自己的标签 → B1 不变；改掉测试折 owner 的观测累加量 → B2 不变
    u2 = u.copy()
    m = u2.stim_pair.map(f) == 0
    u2.loc[m, "n1_detected"] = ~u2.loc[m, "n1_detected"]
    u2.loc[m, "winmin_uv"] = 9999.0
    pool2 = bl.ObsPool([dict(owners=np.array(pool.owners), measures=np.array(pool.measures), channels=np.array(pool.chans),
                             sums=pool.sums.copy(), counts=pool.counts.copy())])
    for k, o in enumerate(pool2.owners):
        if f[o] == 0:
            pool2.sums[k] += 1000.0
    F1 = bl.build_features(u2[m], f, pool2, {0}, source=u2)
    for key in ["B0", "B1", "B2"]:
        assert np.array_equal(F0[key], F1[key], equal_nan=True), key


def test_cv_runs_and_distance_baseline_beats_chance():
    u, pool, pairs = units_and_pool()
    f = fo.assign_folds(pairs, 5, 0)
    pred, chosen = bl.cv_predict(u, f, pool, ["B0", "B3"], CFG_B)
    assert bl.safe_auc(u.n1_detected, pred["B0"]) > 0.7
    assert np.isfinite(pred["B3"]).all() and len(chosen) == 10


if __name__ == "__main__":
    for t in [test_shared_electrode_same_fold, test_eval_mask_excludes_stim_and_adjacent,
              test_test_features_ignore_own_labels_and_segments, test_cv_runs_and_distance_baseline_beats_chance]:
        t(); print("OK", t.__name__)
