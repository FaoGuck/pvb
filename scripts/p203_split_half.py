#!/usr/bin/env python
"""P203：D5 的上限——每个刺激对的奇数次 / 偶数次试验平均之间的 split-half 可靠度（00 第 4 节 P203；G0-T2 ②③）。

只读 P202 的长表（$PRJ_DATA/t2/d5/ccep_labels/<sub>_labels.parquet），不读信号；本地即可运行（几秒）。
**定义（2026-11-26 在看到奇偶两半数据之前写死，00 第 14 节）**，每名患者分别计算，单位 = 刺激对 × 记录电极：
  幅度      a_odd / a_even = 奇 / 偶数次平均波形在 N1 窗（9–100 ms）内的最小值（winmin_odd_uv / winmin_even_uv，总有定义）
  可靠度    rel = Spearman–Brown(ρ)，ρ = Spearman(a_odd, a_even)，SB(ρ) = 2ρ / (1 + ρ)                       → G0-T2 ② 第 1 部分（≥ 0.7）
  残差可靠度 两半各自减去“按距离分 20 个等样本量区间的区间中位数”（距离 = 刺激对中点到记录电极，两半分开拟合），
            再算 SB(Spearman)；单元 < 200 个时区间数 = 单元数 // 10                                         → G0-T2 ② 第 2 部分（≥ 0.5）
            （ρ ≤ 0 时 Spearman–Brown 无意义，记为 ρ 本身）
  上限 AUROC 用 −a_odd 预测偶数半的 N1 判定、−a_even 预测奇数半的 N1 判定，两者平均                              → G0-T2 ③ 的上限
  判定一致  奇偶两半 N1 判定的一致率与 Cohen κ
G0-T2 ② 以 5 名患者的中位数判断。

用法（本地）：
  PowerShell： $env:PRJ_CODE="D:\\proj_LLM\\proj\\proj_PVB\\pvb"; $env:PRJ_DATA="D:\\proj_LLM\\proj\\proj_PVB\\data";
               $env:PRJ_OUT="D:\\proj_LLM\\proj\\proj_PVB\\out"; $env:PRJ_GIT=(git -C pvb rev-parse --short HEAD); $env:PRJ_RUN_ID="local"
               python pvb\\scripts\\p203_split_half.py
输出：$PRJ_OUT/runs/P203_split_half_d5/<时间>_<git7>/  per_subject.tsv、metrics.csv、report.md、DONE
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import rankdata

CODE = Path(os.environ["PRJ_CODE"])
sys.path.insert(0, str(CODE))
from pvb.probing import runs  # noqa: E402

EXP_ID = "P203_split_half_d5"
N_DIST_BINS = 20


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def sb(r: float) -> float:
    """Spearman–Brown；ρ ≤ 0 时校正没有意义，原样返回 ρ（运行前写死）。"""
    return 2 * r / (1 + r) if r > 0 else float(r)


def auroc(score: np.ndarray, y: np.ndarray) -> float:
    y = y.astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def kappa(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    po = float((a == b).mean())
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def resid_by_distance(a: np.ndarray, d: np.ndarray, n_bins: int = N_DIST_BINS) -> np.ndarray:
    n_bins = max(1, min(n_bins, len(a) // 10))          # 单元少于 200 个时减少区间数（每区间 ≥ 10 个）
    q = np.quantile(d, np.linspace(0, 1, n_bins + 1))
    idx = np.clip(np.searchsorted(q, d, side="right") - 1, 0, n_bins - 1)
    med = np.array([np.median(a[idx == k]) if np.any(idx == k) else 0.0 for k in range(n_bins)])
    return a - med[idx]


def per_subject(g: pd.DataFrame) -> dict:
    g = g[np.isfinite(g.winmin_odd_uv) & np.isfinite(g.winmin_even_uv) & np.isfinite(g.dist_mm)]
    ao, ae, d = g.winmin_odd_uv.to_numpy(), g.winmin_even_uv.to_numpy(), g.dist_mm.to_numpy()
    rho = spearman(ao, ae)
    rho_res = spearman(resid_by_distance(ao, d), resid_by_distance(ae, d))
    yo, ye = g.n1_detected_odd.to_numpy(), g.n1_detected_even.to_numpy()
    au = np.nanmean([auroc(-ao, ye), auroc(-ae, yo)])
    return dict(n_units=len(g), n_pairs=g.stim_pair.nunique() if "stim_pair" in g else np.nan,
                rho=rho, rel_sb=sb(rho), rho_resid=rho_res, rel_resid_sb=sb(rho_res),
                ceiling_auroc=au, det_agree=float((yo == ye).mean()), det_kappa=kappa(yo, ye),
                n1_rate_full=float(g.n1_detected.mean()), n1_rate_odd=float(yo.mean()), n1_rate_even=float(ye.mean()),
                median_trials=float(g.n_trials.median()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/datasets/p_d5_ds004080.yaml")
    ap.add_argument("--subjects", default="")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(CODE / a.config, encoding="utf-8"))
    subs = [s for s in a.subjects.split(",") if s] or cfg["subjects_gate"]
    lab_dir = Path(os.environ["PRJ_DATA"]) / cfg["labels_dir"]
    d = runs.new_run_dir(Path(os.environ["PRJ_OUT"]), EXP_ID)
    runs.write_snapshot(d, {**cfg, "_args": vars(a), "_subjects": subs, "_n_dist_bins": N_DIST_BINS})
    log = runs.Log(d)
    rows = []
    for s in subs:
        lab = pd.read_parquet(lab_dir / f"{s}_labels.parquet")
        need = {"winmin_odd_uv", "winmin_even_uv", "n1_detected_odd", "n1_detected_even"}
        if not need <= set(lab.columns):
            raise SystemExit(f"{s} 的长表缺少 {need - set(lab.columns)}：先用新版 P202 重跑")
        rows.append(dict(subject=s, **per_subject(lab)))
    df = pd.DataFrame(rows)
    df.to_csv(d / "per_subject.tsv", sep="\t", index=False)
    med = df[["rel_sb", "rel_resid_sb", "ceiling_auroc", "det_kappa"]].median()
    g2 = bool(med.rel_sb >= 0.7 and med.rel_resid_sb >= 0.5)
    met = [dict(exp_id=EXP_ID, run_id=os.environ["PRJ_RUN_ID"], git_commit=runs.git_commit(), track="T2",
                dataset=cfg["dataset"], subject=r.subject, metric=m, value=r[m], n_pairs=r.n_pairs, n_targets=r.n_units)
           for _, r in df.iterrows() for m in ["rel_sb", "rel_resid_sb", "ceiling_auroc", "det_agree", "det_kappa"]]
    pd.DataFrame(met).to_csv(d / "metrics.csv", index=False)
    rep = [f"# {EXP_ID}", "", df.round(3).to_string(index=False), "",
           f"- 5 名患者中位数：rel_sb = {med.rel_sb:.3f}，rel_resid_sb = {med.rel_resid_sb:.3f}，"
           f"上限 AUROC = {med.ceiling_auroc:.3f}，判定 κ = {med.det_kappa:.3f}",
           f"- **G0-T2 ②（rel_sb 中位 ≥ 0.7 且 rel_resid_sb 中位 ≥ 0.5）：{'通过' if g2 else '不通过'}**",
           f"- G0-T2 ③ 要用的上限 AUROC（中位）= {med.ceiling_auroc:.3f}；B0 + B1 的 AUROC 中位需 ≤ 0.90 且 ≤ 上限 − 0.05 = {med.ceiling_auroc - 0.05:.3f}（P205）"]
    (d / "report.md").write_text("\n".join(rep) + "\n", encoding="utf-8")
    log("\n".join(rep))
    log(f"SUMMARY exp={EXP_ID} subjects={len(df)} G0_T2_2={'PASS' if g2 else 'FAIL'}")
    runs.mark_done(d)


if __name__ == "__main__":
    main()
