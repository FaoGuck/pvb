#!/usr/bin/env python
"""P205b：D5 响应电极的幅度基线（判据写死在 docs/00 第 14 节，2026-11-26，运行前）。

为什么：P205 显示“N1 有无”的 AUROC 里经典基线 B3 离 split-half 上限只剩约 0.03；00 第 3 节的第二个主指标是
“响应电极中幅度的 Spearman”。这一步给它配上经典基线和上限，判断 T2 还有没有给模型留出的余量。

每名患者：
  1. 评估集 = P205 的评估集（去相邻、距离有限、窗内最小值非空）中全数据 N1 判为有的单位
  2. 折 = 与 P205 完全相同（全部刺激对 + 观测段 owner，folds.assign_folds）
  3. 目标 = log(max(−winmin_uv, 1))；特征 B0 / B1 / B2 / B0+B1 / B0+B2 / B3（baselines.py，可用折规则不变；
     B1 的标签来源 = P205 的全部评估单位）；预测头 = 标准化 + 岭回归，α 内层按 Spearman 选
  4. 指标 = 外层预测拼接后与目标的 Spearman；上限 = 同一批单位两半窗内最小值的 Spearman（不做 SB 校正）
  5. 另报：B2 在 B0+B1 之上的增量（置换 刺激对 → B2 特征，α 固定为真实模型选中的值）
判据：5 名患者“上限 − B3”的中位数 ≥ amplitude.headroom_thr（0.10）→ PASS（T2 以幅度为主指标继续）；否则 FAIL（T2 降为附录）。
有患者无法评估（有 N1 的单位 < min_units）时不判通过。

用法（workflows/pvb-p205b-amplitude-gate.yaml；只读 data/t2/d5）：
    PRJ_STEP = scripts/p205b_amplitude.py   PRJ_ARGS = --config configs/datasets/p_d5_ds004080.yaml --procs 5
输出：$PRJ_OUT/runs/P205b_amplitude_d5/<时间>_<git7>/  amp.tsv、perm.tsv、preds.parquet、metrics.csv、report.md、DONE
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

EXP_ID = "P205b_amplitude_d5"
SETS = list(bl.FEATURE_SETS)


def amp_target(winmin_uv) -> np.ndarray:
    return np.log(np.maximum(-np.asarray(winmin_uv, float), 1.0))


def run_subject(sub: str, cfg: dict, data: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame, str]:
    t0 = time.time()
    cb = {**cfg["baselines"], **cfg["amplitude"]}
    lab = pd.read_parquet(data / cfg["labels_dir"] / f"{sub}_labels.parquet")
    obs_dir = data / "t2" / "d5" / "observation"
    el = pd.read_parquet(obs_dir / f"{sub}_electrodes.parquet").set_index("NAME")
    thr = fo.adjacency_threshold(el[["x", "y", "z"]].to_numpy(float), cfg["folds"]["adjacent_factor"])
    keep = fo.eval_mask(lab, el, thr)
    u = lab[keep & np.isfinite(lab.dist_mm) & lab.winmin_uv.notna()].reset_index(drop=True)   # = P205 的评估集
    accs = [dict(np.load(f, allow_pickle=False)) for f in sorted(glob.glob(str(obs_dir / sub / "*_obs.npz")))]
    pool = bl.ObsPool(accs) if accs else None
    pairs = sorted(set(u.stim_pair) | (set(pool.owners) if pool else set()))
    fold_of = fo.assign_folds(pairs, cfg["folds"]["k"], cfg["folds"]["seed"])           # 与 P205 相同
    r = u[u.n1_detected.astype(bool)].reset_index(drop=True)
    r = r.assign(amp_log=amp_target(r.winmin_uv))
    base = dict(subject=sub, n_resp=len(r), n_units=len(u), n_pairs_resp=r.stim_pair.nunique(), adj_thr_mm=round(thr, 2))
    n_ne = len(set(fold_of[p] for p in u.stim_pair.unique()))
    min_ne = cfg["folds"].get("min_nonempty_folds", cfg["folds"]["k"])
    if n_ne < min_ne:                                   # 2026-11-27 排除规则（00 第 14 节），与 P205 相同
        note = f"不可评估：按组分折后非空折 {n_ne} < {min_ne}（刺激对经共用电极连成 {len(set(fo.pair_components(pairs).values()))} 组）"
        return {**base, "note": note}, pd.DataFrame(), pd.DataFrame(), f"SKIP {sub} {note}"
    if len(r) < cb["min_units"]:
        return {**base, "note": f"有 N1 的单位 < {cb['min_units']}，不评估"}, pd.DataFrame(), pd.DataFrame(), \
            f"SKIP {sub} n_resp={len(r)}"
    ceil = bl.safe_spearman(r.winmin_odd_uv, r.winmin_even_uv)
    pred, chosen = bl.cv_predict(r, fold_of, pool, SETS, cb, target="amp_log", task="reg", source=u)
    rho = {s: bl.safe_spearman(r.amp_log, pred[s]) for s in SETS}
    rng = np.random.default_rng(cb["perm_seed"])
    up = sorted(r.stim_pair.unique())
    perm_rows = []
    for i in range(cb["n_perm"]):
        mp = dict(zip(up, rng.permutation(up)))
        pp, _ = bl.cv_predict(r, fold_of, pool, ["B3"], cb, fixed_C=chosen, b2_pair_map=mp,
                              target="amp_log", task="reg", source=u)
        perm_rows.append(dict(subject=sub, perm=i, rho_B3=bl.safe_spearman(r.amp_log, pp["B3"])))
    perm = pd.DataFrame(perm_rows)
    gain = rho["B3"] - rho["B0+B1"]
    gain_null = float((perm.rho_B3 - rho["B0+B1"]).mean())
    p_perm = float((1 + (perm.rho_B3 >= rho["B3"]).sum()) / (1 + len(perm)))
    res = {**base, "ceiling_rho": ceil, **{f"rho_{s}": v for s, v in rho.items()}, "headroom": ceil - rho["B3"],
           "gain_B3_vs_B01": gain, "gain_null_mean": gain_null, "increment": gain - gain_null, "p_perm": p_perm,
           "folds_used": len(set(fold_of[p] for p in up)), "seconds": round(time.time() - t0, 1)}
    preds = r[["run_id", "subject", "stim_pair", "channel", "dist_mm", "winmin_uv", "amp_log"]].assign(
        fold=r.stim_pair.map(fold_of), **{f"p_{s}": pred[s] for s in SETS})
    msg = (f"OK  {sub}  有 N1 的单位 {len(r)}/{len(u)}（{r.stim_pair.nunique()} 对）  上限 {ceil:.3f}  "
           + "  ".join(f"{s} {rho[s]:.3f}" for s in SETS)
           + f"  余量 {ceil - rho['B3']:+.3f}  B2 增量 {gain - gain_null:+.4f}（p={p_perm:.3f}）  {res['seconds']} s")
    return res, perm, preds, msg


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
    R, P, Q = [], [], []
    with ProcessPoolExecutor(max_workers=max(1, min(a.procs, len(subs)))) as ex:
        futs = {ex.submit(run_subject, s, cfg, data): s for s in subs}
        for fu in as_completed(futs):
            try:                                        # 逐患者容错（S2R 约定；2026-11-27）
                r, p, q, msg = fu.result()
            except Exception as e:  # noqa: BLE001
                r, p, q, msg = dict(subject=futs[fu], note=f"FAIL {e!r}"[:300]), pd.DataFrame(), pd.DataFrame(), f"FAIL {futs[fu]} {e!r}"
            R.append(r); P.append(p); Q.append(q); log(msg)
    R.sort(key=lambda r: r["subject"])
    res = pd.DataFrame(R)
    res.to_csv(d / "amp.tsv", sep="\t", index=False)
    pd.concat(P).to_csv(d / "perm.tsv", sep="\t", index=False)
    pd.concat(Q).to_parquet(d / "preds.parquet", index=False)
    ev = res[res.ceiling_rho.notna()] if "ceiling_rho" in res else res.iloc[0:0]
    met = [dict(exp_id=EXP_ID, run_id=os.environ["PRJ_RUN_ID"], git_commit=runs.git_commit(), track="T2",
                dataset=cfg["dataset"], subject=r.subject, model=c.replace("rho_", ""), feature_set=c.replace("rho_", ""),
                fold="all", stim_pair="all", metric="spearman_amp_resp", value=r[c], n_pairs=r.n_pairs_resp,
                n_targets=r.n_resp, seed=cfg["folds"]["seed"])
           for _, r in ev.iterrows() for c in [f"rho_{s}" for s in SETS] + ["ceiling_rho"]]
    pd.DataFrame(met).to_csv(d / "metrics.csv", index=False)
    thr = cfg["amplitude"]["headroom_thr"]
    n_ev = len(ev)
    med_h = float(ev.headroom.median()) if n_ev else float("nan")
    ok = n_ev == len(subs) and med_h >= thr          # 有患者无法评估 → 不判通过
    rep = [f"# {EXP_ID}", "", f"- 可评估患者 {n_ev} / {len(subs)}"]
    if n_ev:
        cols = ["subject", "n_resp", "ceiling_rho"] + [f"rho_{s}" for s in SETS] + ["headroom", "increment", "p_perm"]
        rep[2:2] = [ev[cols].round(4).to_string(index=False), ""]
        rep += [f"- 上限中位 {ev.ceiling_rho.median():.4f}；B0 {ev.rho_B0.median():.4f}、B0+B1 {ev['rho_B0+B1'].median():.4f}、"
                f"B0+B2 {ev['rho_B0+B2'].median():.4f}、B3 {ev.rho_B3.median():.4f}",
                f"- （另报）B2 在 B0+B1 之上的增量 > 0 的患者 {int((ev.increment > 0).sum())} / {n_ev}"]
    rep.append(f"- 上限 − B3 的中位数 {med_h:.4f}（判据 ≥ {thr}）→ **P205b："
               f"{'通过（T2 以幅度为主指标继续）' if ok else '不通过（T2 降为附录）'}**")
    (d / "report.md").write_text("\n".join(rep) + "\n", encoding="utf-8")
    log("\n".join(rep))
    log(f"SUMMARY exp={EXP_ID} subjects={n_ev}/{len(subs)} headroom_med={med_h:.4f} P205b={'PASS' if ok else 'FAIL'}")
    runs.mark_done(d)


if __name__ == "__main__":
    main()
