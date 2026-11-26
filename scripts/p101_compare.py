#!/usr/bin/env python
"""P101 回归测试的比较（本地只读；00 第 4 节 P101、第 8 节 G0-T1 ①）。

PVB 用复制来的代码在云上原样重跑 S2R 的 E101 / E103 / E110（workflows/pvb-p101-e101/e103/e110.yaml），
本脚本把 PVB 的 metrics.csv 与 S2R 的参考运行（out/summary/p1_report.md 第 3 行列出的运行）逐行对齐比较：
  键 = subject × predictor × window × pool × layer × target × metric × pca
  通过条件（G0-T1 ①，锁定）：真实目标的指标 value 逐行 |差| ≤ 0.005，且两边的键完全一致
  另列（不作通过条件）：置换均值 null_mean、p_perm、n_trials 的差
用法（本地，读 pull 回来的 out 与只读的 S2R out）：
    set PRJ_CODE / PRJ_OUT / PRJ_RUN_ID / PRJ_GIT 后
    python -X utf8 scripts/p101_compare.py [--pvb-runs E101_classical_d1=<目录名>,...] [--s2r-out D:/proj_LLM/proj/proj_S2R/out]
输出：$PRJ_OUT/runs/P101_regression_s2r/<时间>_<git7>/  compare.tsv、per_exp.tsv、report.md、DONE；最后一行 SUMMARY
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(os.environ["PRJ_CODE"])
sys.path.insert(0, str(CODE))
from pvb.probing import runs  # noqa: E402

EXP_ID = "P101_regression_s2r"
# S2R 的参考运行 = out/summary/p1_report.md 第 3 行（S9 统计实际使用的运行）
S2R_REF = {"E101_classical_d1": "20261121-1029_c96f6d3",
           "E103_fm_d1": "20261123-0524_86c721d",
           "E110_history_d1": "20261123-2114_03d0871"}
KEY = ["subject", "predictor", "window", "pool", "layer", "target", "metric", "pca"]


def latest_done(root: Path) -> str | None:
    ds = sorted(p.name for p in root.glob("*") if (p / "DONE").is_file()) if root.is_dir() else []
    return ds[-1] if ds else None


def load(p: Path) -> pd.DataFrame:
    m = pd.read_csv(p / "metrics.csv", dtype={c: str for c in KEY})
    return m


def compare(exp: str, a: pd.DataFrame, b: pd.DataFrame, tol: float) -> tuple[pd.DataFrame, dict]:
    """a = PVB，b = S2R。"""
    da, db = a.duplicated(KEY).sum(), b.duplicated(KEY).sum()
    m = a.merge(b, on=KEY, how="outer", suffixes=("_pvb", "_s2r"), indicator=True)
    both = m[m._merge == "both"].copy()
    for c in ["value", "null_mean", "p_perm", "n_trials"]:
        both[f"d_{c}"] = (both[f"{c}_pvb"].astype(float) - both[f"{c}_s2r"].astype(float)).abs()
    both.insert(0, "exp", exp)
    s = dict(exp=exp, rows_pvb=len(a), rows_s2r=len(b), matched=len(both),
             only_pvb=int((m._merge == "left_only").sum()), only_s2r=int((m._merge == "right_only").sum()),
             dup_keys=int(da + db), subjects=both.subject.nunique(),
             max_d_value=float(both.d_value.max()) if len(both) else np.nan,
             n_over_tol=int((both.d_value > tol).sum()) if len(both) else 0,
             nan_mismatch=int((both.value_pvb.isna() != both.value_s2r.isna()).sum()),
             max_d_null_mean=float(both.d_null_mean.max()) if len(both) else np.nan,
             max_d_p_perm=float(both.d_p_perm.max()) if len(both) else np.nan,
             max_d_n_trials=float(both.d_n_trials.max()) if len(both) else np.nan)
    s["ok"] = bool(s["matched"] > 0 and s["only_pvb"] == 0 and s["only_s2r"] == 0 and s["dup_keys"] == 0
                   and s["n_over_tol"] == 0 and s["nan_mismatch"] == 0)
    cols = ["exp"] + KEY + ["value_pvb", "value_s2r", "d_value", "d_null_mean", "d_p_perm", "d_n_trials"]
    return both[cols], s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pvb-runs", default="", help="exp=目录名,…；默认取 PVB out/runs/<exp>/ 下最新带 DONE 的目录")
    ap.add_argument("--pvb-out", default=os.environ.get("PRJ_OUT", ""))
    ap.add_argument("--s2r-out", default="D:/proj_LLM/proj/proj_S2R/out")
    ap.add_argument("--tol", type=float, default=0.005)
    a = ap.parse_args()
    pvb_out, s2r_out = Path(a.pvb_out), Path(a.s2r_out)
    pick = dict(kv.split("=", 1) for kv in a.pvb_runs.split(",") if kv)
    d = runs.new_run_dir(Path(os.environ["PRJ_OUT"]), EXP_ID)
    log = runs.Log(d)
    rows, stats = [], []
    for exp, ref in S2R_REF.items():
        run = pick.get(exp) or latest_done(pvb_out / "runs" / exp)
        if run is None:
            stats.append(dict(exp=exp, ok=False, note="PVB 没有带 DONE 的运行")); log(f"FAIL {exp}：PVB 没有带 DONE 的运行")
            continue
        c, s = compare(exp, load(pvb_out / "runs" / exp / run), load(s2r_out / "runs" / exp / ref), a.tol)
        s.update(pvb_run=run, s2r_run=ref)
        rows.append(c); stats.append(s)
        log(f"{'OK  ' if s['ok'] else 'FAIL'} {exp}  PVB {run} vs S2R {ref}  对齐 {s['matched']} 行（仅 PVB {s['only_pvb']}、仅 S2R {s['only_s2r']}）"
            f"  被试 {s['subjects']}  最大|差| {s['max_d_value']:.2e}（超 {a.tol} 的 {s['n_over_tol']} 行）"
            f"  置换均值最大差 {s['max_d_null_mean']:.2e}  p 最大差 {s['max_d_p_perm']:.2e}")
    st = pd.DataFrame(stats)
    st.to_csv(d / "per_exp.tsv", sep="\t", index=False)
    if rows:
        pd.concat(rows).to_csv(d / "compare.tsv", sep="\t", index=False)
    ok = bool(len(st) == len(S2R_REF) and st.ok.all())
    mx = float(pd.concat(rows).d_value.max()) if rows else float("nan")
    rep = [f"# {EXP_ID}", "", st.to_string(index=False), "",
           f"- 逐被试真实目标指标的最大 |PVB − S2R| = {mx:.2e}（容差 {a.tol}）",
           f"- **G0-T1 ①（P101 回归测试）：{'通过' if ok else '不通过'}**",
           "- 置换均值、p 值的差只列出，不作通过条件（00 第 8 节）"]
    (d / "report.md").write_text("\n".join(rep) + "\n", encoding="utf-8")
    log("\n".join(rep))
    log(f"SUMMARY exp={EXP_ID} max_abs_diff={mx:.3e} tol={a.tol} G0_T1_1={'PASS' if ok else 'FAIL'}")
    runs.mark_done(d)


if __name__ == "__main__":
    main()
