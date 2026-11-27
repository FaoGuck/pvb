#!/usr/bin/env python
"""P205：D5 平凡与经典基线 B0 / B1 / B2 / B3（00 第 4 节 P205）与 G0-T2 ③④ 的判定。

每名患者：
  1. 单位 = P202 长表的 刺激对 × 记录电极，去掉与刺激电极直接相邻的（pvb/t2/folds.py：≤ 1.5 × 最近邻距离中位数）
  2. 折：共用电极的刺激对同折，K = 5（folds.assign_folds；观测段的 owner 也一起分折，保证 B2 只用训练折的段）
  3. 外层 5 折交叉验证，特征组合 B0、B1、B2、B0+B1、B0+B2、B3（pvb/t2/baselines.py），预测拼接后算 AUROC（N1 有无）
  4. 上限：同一评估集上，一半的幅度预测另一半 N1 判定的 AUROC（定义同 P203）
  5. ④ 的置换零分布：n_perm 次在患者内随机打乱 刺激对 → B2 特征 的对应（B0、B1 不动），C 固定为真实模型每折选中的值；
     增量 = [AUROC(B3) − AUROC(B0+B1)] − 置换下同一差值的均值
G0-T2（00 第 8 节，锁定）：③ B0+B1 的 AUROC 中位 ≤ 0.90 且 ≤ 上限 − 0.05；④ 增量 > 0 的患者 ≥ 4 / 5。

用法（workflows/pvb-p205-baselines-gate.yaml；只读 data/t2/d5，不读原始信号）：
    PRJ_STEP = scripts/p205_baselines.py   PRJ_ARGS = --config configs/datasets/p_d5_ds004080.yaml
输出：$PRJ_OUT/runs/P205_baselines_d5/<时间>_<git7>/  auroc.tsv、perm.tsv、preds.parquet、folds.tsv、metrics.csv、report.md、DONE
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CODE = Path(os.environ["PRJ_CODE"])
sys.path.insert(0, str(CODE))
from pvb.probing import runs  # noqa: E402
from pvb.t2 import baselines as bl, folds as fo  # noqa: E402

EXP_ID = "P205_baselines_d5"
SETS = list(bl.FEATURE_SETS)


def ceiling_auc(u: pd.DataFrame) -> float:
    a = bl.safe_auc(u.n1_detected_even, -u.winmin_odd_uv.to_numpy())
    b = bl.safe_auc(u.n1_detected_odd, -u.winmin_even_uv.to_numpy())
    return float(np.nanmean([a, b]))


def run_subject(sub: str, cfg: dict, data: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    t0 = time.time()
    lab = pd.read_parquet(data / cfg["labels_dir"] / f"{sub}_labels.parquet")
    obs_dir = data / "t2" / "d5" / "observation"
    el = pd.read_parquet(obs_dir / f"{sub}_electrodes.parquet").set_index("NAME")
    thr = fo.adjacency_threshold(el[["x", "y", "z"]].to_numpy(float), cfg["folds"]["adjacent_factor"])
    keep = fo.eval_mask(lab, el, thr)
    u = lab[keep & np.isfinite(lab.dist_mm) & lab.winmin_uv.notna()].reset_index(drop=True)
    accs = [dict(np.load(f, allow_pickle=False)) for f in sorted(glob.glob(str(obs_dir / sub / "*_obs.npz")))]
    pool = bl.ObsPool(accs) if accs else None
    pairs = sorted(set(u.stim_pair) | (set(pool.owners) if pool else set()))
    fold_of = fo.assign_folds(pairs, cfg["folds"]["k"], cfg["folds"]["seed"])
    n_ne = len(set(fold_of[p] for p in u.stim_pair.unique()))
    min_ne = cfg["folds"].get("min_nonempty_folds", cfg["folds"]["k"])
    if n_ne < min_ne:                                   # 2026-11-27 排除规则（00 第 14 节）：只看结构
        note = f"不可评估：按组分折后非空折 {n_ne} < {min_ne}（刺激对经共用电极连成 {len(set(fo.pair_components(pairs).values()))} 组）"
        return dict(subject=sub, n_units=len(u), n_units_all=len(lab), n_pairs=u.stim_pair.nunique(), note=note), \
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), f"SKIP {sub} {note}"
    pred, chosen = bl.cv_predict(u, fold_of, pool, SETS, cfg["baselines"])
    y = u.n1_detected.to_numpy(bool)
    auc = {s: bl.safe_auc(y, pred[s]) for s in SETS}
    ceil = ceiling_auc(u)
    rng = np.random.default_rng(cfg["baselines"]["perm_seed"])
    up = sorted(u.stim_pair.unique())
    perm_rows = []
    for i in range(cfg["baselines"]["n_perm"]):
        mp = dict(zip(up, rng.permutation(up)))
        pp, _ = bl.cv_predict(u, fold_of, pool, ["B3"], cfg["baselines"], fixed_C=chosen, b2_pair_map=mp)
        perm_rows.append(dict(subject=sub, perm=i, auc_B3=bl.safe_auc(y, pp["B3"])))
    perm = pd.DataFrame(perm_rows)
    gain = auc["B3"] - auc["B0+B1"]
    gain_null = float((perm.auc_B3 - auc["B0+B1"]).mean())
    p_perm = float((1 + (perm.auc_B3 >= auc["B3"]).sum()) / (1 + len(perm)))
    res = dict(subject=sub, n_units=len(u), n_units_all=len(lab), n_pairs=u.stim_pair.nunique(), n1_rate=float(y.mean()),
               adj_thr_mm=round(thr, 2), n_folds_used=len(set(fold_of[p] for p in up)), ceiling_auroc=ceil,
               **{f"auc_{s}": v for s, v in auc.items()},
               gain_B3_vs_B01=gain, gain_null_mean=gain_null, increment=gain - gain_null, p_perm=p_perm,
               seconds=round(time.time() - t0, 1))
    preds = u[["run_id", "subject", "stim_pair", "channel", "dist_mm", "n1_detected"]].assign(
        fold=u.stim_pair.map(fold_of), **{f"p_{s}": pred[s] for s in SETS})
    comp = fo.pair_components(pairs)
    folds = pd.DataFrame([dict(subject=sub, stim_pair=p, fold=f, comp=comp[p]) for p, f in fold_of.items()])
    msg = (f"OK  {sub}  单位 {len(u)}/{len(lab)}  相邻阈值 {thr:.1f} mm  上限 {ceil:.3f}  "
           + "  ".join(f"{s} {auc[s]:.3f}" for s in SETS) + f"  增量 {gain - gain_null:+.4f}（p={p_perm:.3f}）  {res['seconds']} s")
    return res, perm, preds, folds, msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/datasets/p_d5_ds004080.yaml")
    ap.add_argument("--subjects", default="")
    ap.add_argument("--procs", type=int, default=5)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(CODE / a.config, encoding="utf-8"))
    subs = [s for s in a.subjects.split(",") if s] or cfg["subjects_gate"]
    data = Path(os.environ["PRJ_DATA"])
    d = runs.new_run_dir(Path(os.environ["PRJ_OUT"]), EXP_ID)
    runs.write_snapshot(d, {**cfg, "_args": vars(a), "_subjects": subs})
    log = runs.Log(d)
    R, P, Q, F = [], [], [], []
    with ProcessPoolExecutor(max_workers=max(1, min(a.procs, len(subs)))) as ex:   # 按患者并行
        futs = {ex.submit(run_subject, s, cfg, data): s for s in subs}
        for fu in as_completed(futs):
            try:                                        # 逐患者容错（S2R 约定；2026-11-27 扩到 74 人时一个患者崩溃拖垮了全部）
                r, p, q, f, msg = fu.result()
            except Exception as e:  # noqa: BLE001
                r, p, q, f, msg = dict(subject=futs[fu], note=f"FAIL {e!r}"[:300]), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), \
                    f"FAIL {futs[fu]} {e!r}"
            R.append(r); P.append(p); Q.append(q); F.append(f); log(msg)
    R.sort(key=lambda r: r["subject"])
    res = pd.DataFrame(R)
    res.to_csv(d / "auroc.tsv", sep="\t", index=False)
    pd.concat(P).to_csv(d / "perm.tsv", sep="\t", index=False)
    pd.concat(Q).to_parquet(d / "preds.parquet", index=False)
    pd.concat(F).to_csv(d / "folds.tsv", sep="\t", index=False)
    met = [dict(exp_id=EXP_ID, run_id=os.environ["PRJ_RUN_ID"], git_commit=runs.git_commit(), track="T2", dataset=cfg["dataset"],
                subject=r.subject, model=c.replace("auc_", ""), feature_set=c.replace("auc_", ""), fold="all",
                stim_pair="all", metric="auroc", value=r[c], n_pairs=r.n_pairs, n_targets=r.n_units, seed=cfg["folds"]["seed"])
           for _, r in res[res["auc_B3"].notna()].iterrows() for c in [f"auc_{s}" for s in SETS] + ["ceiling_auroc"]]   # 不可评估的患者不进长表
    pd.DataFrame(met).to_csv(d / "metrics.csv", index=False)
    med01, medc = float(res["auc_B0+B1"].median()), float(res.ceiling_auroc.median())
    g3 = (med01 <= 0.90) and (med01 <= medc - 0.05)
    n_inc = int((res.increment > 0).sum())
    g4 = n_inc >= 4
    cols = ["subject", "n_units", "n1_rate", "ceiling_auroc"] + [f"auc_{s}" for s in SETS] + ["increment", "p_perm"]
    rep = [f"# {EXP_ID}", "", res[cols].round(4).to_string(index=False), "",
           f"- B0+B1 AUROC 中位 {med01:.4f}；上限中位 {medc:.4f}（同一评估集）→ **G0-T2 ③（≤ 0.90 且 ≤ 上限 − 0.05 = {medc - 0.05:.4f}）："
           f"{'通过' if g3 else '不通过'}**",
           f"- B2 在 B0+B1 之上有增量（相对置换零分布均值 > 0）的患者 {n_inc} / {len(res)} → **G0-T2 ④（≥ 4 / 5）：{'通过' if g4 else '不通过'}**",
           f"- B3 − B0+B1 原始差值：{res.gain_B3_vs_B01.round(4).tolist()}；置换零分布均值：{res.gain_null_mean.round(4).tolist()}"]
    (d / "report.md").write_text("\n".join(rep) + "\n", encoding="utf-8")
    log("\n".join(rep))
    log(f"SUMMARY exp={EXP_ID} subjects={len(res)} G0_T2_3={'PASS' if g3 else 'FAIL'} G0_T2_4={'PASS' if g4 else 'FAIL'}")
    runs.mark_done(d)


if __name__ == "__main__":
    main()
