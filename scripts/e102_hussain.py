#!/usr/bin/env python
# 来自 S2R 505b298 的 scripts/e102_hussain.py，2026-11-26 复制（机械改名：S2R_→PRJ_、包名 s2r→项目名）
"""E102：复现 Hussain & Quentin 2022 的分类流程（细节与偏差见 configs/experiments/E102_hussain_d1.yaml 开头注释）。

用法（workflows/s2r-e102-hussain.yaml）:
    PRJ_STEP = scripts/e102_hussain.py
    PRJ_ARGS = --exp configs/experiments/E102_hussain_d1.yaml
    PRJ_DATA_IN = data/interim/d1          # 读 S3 h5 的 master + S5 qc 表（剔除标记、MEP）

计算量：每人每种交叉验证 (1 + 500 次置换) × 折数 × 168 × 100 个 LDA；按被试 × 变体多进程并行（只用 CPU）。
输出: $PRJ_OUT/runs/E102_hussain_d1/<时间戳>_<git7>/
    metrics.csv（总纲 7.3 长表；额外列 variant、best_k、best_lambda、null_mean、null_p95、significant）
    nulls.npz（每人每变体的置换 AUC 分布）、features/sub-XXX_hussain.parquet（168 个特征，供核对）
    report.md、config.yaml、env.txt、log.txt、DONE
本地测试：--n-perm 5 --subjects 283 --workers 2 --out-root <临时目录>
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CODE = Path(os.environ.get("PRJ_CODE", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(CODE))
from pvb.features import hussain as hf  # noqa: E402
from pvb.probing import cv, hussain_lda as hl, runs  # noqa: E402


def make_folds(y, blocks, vspec, seed):
    if vspec["cv"] == "stratified_kfold":
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=int(vspec["k"]), shuffle=True, random_state=seed)
        return list(skf.split(np.zeros(len(y)), y))
    if vspec["cv"] == "leave_one_block_out":
        return [(np.where(blocks != b)[0], np.where(blocks == b)[0]) for b in np.unique(blocks)]
    raise ValueError(vspec["cv"])


def job(a):
    """一个被试 × 一个变体：真实标签 + n_perm 次置换的完整网格搜索。"""
    from threadpoolctl import threadpool_limits

    sub, vname, vspec, X, y, blocks, cfg, n_perm = a
    with threadpool_limits(1):
        t0 = time.time()
        lam = cv.grid(cfg["lda"]["lambdas"])
        kmax, bins, z = int(cfg["lda"]["max_features"]), int(cfg["lda"]["chi2_bins"]), bool(cfg["lda"].get("zscore", True))
        seed = int(cfg["seed"])
        rng = np.random.default_rng(seed * 100003 + int(sub) * 7 + (1 if vname == "block6" else 0))
        pb = blocks if vspec["permute_within"] == "block" else np.zeros(len(y), dtype=int)
        perm = cv.permutation_index(pb, n_perm, rng)
        auc, k, lb = hl.grid_search(X, y, make_folds(y, blocks, vspec, seed), lam, kmax, bins, z)
        null = np.empty(n_perm)
        for i in range(n_perm):
            yp = y[perm[i]]
            null[i] = hl.grid_search(X, yp, make_folds(yp, blocks, vspec, seed + 1 + i), lam, kmax, bins, z)[0]
        return sub, vname, auc, k, lb, null, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--subjects", nargs="*")
    ap.add_argument("--n-perm", type=int)
    ap.add_argument("--workers", type=int, default=0, help="进程数，默认 = CPU 核数")
    ap.add_argument("--out-root")
    args = ap.parse_args()

    data = Path(os.environ["PRJ_DATA"])
    d1_dir = data / "interim" / "d1"
    out_root = Path(args.out_root or os.environ.get("PRJ_OUT", CODE.parent / "out"))
    cfg = yaml.safe_load(open(CODE / args.exp, encoding="utf-8"))
    if args.n_perm is not None:
        cfg["permutation"]["n"] = args.n_perm
    n_perm = int(cfg["permutation"]["n"])
    workers = args.workers or os.cpu_count() or 1
    rd = runs.new_run_dir(out_root, cfg["exp_id"])
    runs.write_snapshot(rd, {**cfg, "_resolved": {"workers": workers, "subjects_arg": args.subjects,
                                                 "git": runs.git_commit(), "run_id": os.environ.get("PRJ_RUN_ID")}})
    (rd / "features").mkdir()
    log = runs.Log(rd)
    subs = args.subjects or sorted(p.name[4:7] for p in (d1_dir / "qc").glob("sub-*_qc.parquet"))
    log(f"[E102] run_dir={rd} subjects={len(subs)} n_perm={n_perm} workers={workers}")

    # 1. 特征与标签（主进程，顺序做）
    jobs, ninfo, errors = [], {}, {}
    for s in subs:
        try:
            ai, X, names = hf.subject_features(d1_dir / "epochs_pre" / f"sub-{s}_pre.h5", cfg["features"])
            q = pd.read_parquet(d1_dir / "qc" / f"sub-{s}_qc.parquet")
            keep = np.ones(len(q), dtype=bool)
            for c in cfg["trials"]["require"]:
                keep &= q[c].astype(bool).to_numpy()
            for c in cfg["trials"]["exclude_flags"]:
                keep &= ~q[c].astype(bool).to_numpy()
            q = q[keep]
            pos = pd.Series(np.arange(len(ai)), index=ai)
            Xs = X[pos.loc[q["aligned_idx"]].to_numpy()]
            mep = q[cfg["label"]["column"]].to_numpy(dtype=float)
            y = (mep > np.median(mep)).astype(int)
            blocks = q["block"].to_numpy()
            fdf = pd.DataFrame(Xs, columns=names)
            fdf.insert(0, "aligned_idx", q["aligned_idx"].to_numpy())
            fdf.insert(1, "y", y)
            fdf.to_parquet(rd / "features" / f"sub-{s}_hussain.parquet", index=False)
            ninfo[s] = {"n_trials": len(y), "n_removed": int((~keep).sum())}
            for vname, vspec in cfg["variants"].items():
                jobs.append((s, vname, vspec, Xs, y, blocks, cfg, n_perm))
            log(f"[E102] sub-{s} 特征 OK n={len(y)}（Hussain 规则剔除 {ninfo[s]['n_removed']}）")
        except Exception:  # noqa: BLE001
            errors[s] = traceback.format_exc()
            log(f"[E102] sub-{s} FAIL（特征）\n{errors[s]}")

    # 2. 网格搜索 + 置换（多进程）
    alpha = float(cfg["permutation"]["alpha"])
    rows, nulls = [], {}
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        for res in pool.imap_unordered(job, jobs):
            s, vname, auc, k, lb, null, sec = res
            p95 = float(np.percentile(null, 100 * (1 - alpha))) if n_perm else np.nan
            p = cv.perm_p(np.concatenate([[auc], null]))
            nulls[f"sub-{s}_{vname}"] = null
            rows.append({"exp_id": cfg["exp_id"], "run_id": os.environ.get("PRJ_RUN_ID", "local"),
                         "git_commit": runs.git_commit(), "dataset": cfg["dataset"], "subject": f"sub-{s}",
                         "predictor": "hussain_lda", "window": "W05h", "pool": "na", "layer": "na",
                         "target": "y_bin_subject", "metric": "auc", "value": auc, "p_perm": p,
                         "n_trials": ninfo[s]["n_trials"], "seed": cfg["seed"], "variant": vname, "best_k": k,
                         "best_lambda": lb, "null_mean": float(np.mean(null)) if n_perm else np.nan, "null_p95": p95,
                         "significant": bool(auc > p95) if n_perm else False, "n_perm": n_perm})
            log(f"[E102] sub-{s} {vname}: AUC={auc:.3f} k={k} λ={lb:.3g} 置换均值={rows[-1]['null_mean']:.3f} "
                f"95%={p95:.3f} p={p:.3f} {sec:.0f}s")

    met = pd.DataFrame(rows).sort_values(["variant", "subject"])
    if len(met):
        met.to_csv(rd / "metrics.csv", index=False)
        np.savez_compressed(rd / "nulls.npz", **nulls)
        from scipy import stats
        lines = [f"# E102 复现 Hussain 2022 结果概况", "", f"- 运行目录 {rd.name}；git {runs.git_commit()}；置换 {n_perm} 次",
                 f"- 被试 {met['subject'].nunique()} / {len(subs)}；失败: {', '.join(errors) or '无'}",
                 "- 论文：16/20 人显著（80%），AUC 0.62 ± 0.01（0.53–0.72），置换均值 0.55", "",
                 "| 交叉验证 | 显著人数 | AUC 均值 ± 标准误 | 范围 | 置换均值 | 对置换均值 Wilcoxon p（单侧） |",
                 "|---|---|---|---|---|---|"]
        for v, g in met.groupby("variant", sort=False):
            a = g["value"].to_numpy()
            pw = stats.wilcoxon(a - g["null_mean"].to_numpy(), alternative="greater")[1] if len(a) > 5 else np.nan
            lines.append(f"| {v} | {int(g['significant'].sum())}/{len(g)} | {a.mean():.3f} ± {a.std(ddof=1) / np.sqrt(len(a)):.3f} | "
                         f"{a.min():.2f}–{a.max():.2f} | {g['null_mean'].mean():.3f} | {pw:.3g} |")
        lines += ["", "strat5 = 照论文的分层 5 折随机划分；block6 = 留一 block。两者差距反映 MEP 慢漂移对随机折的影响。",
                  "每人的 AUC 是网格（168 × 100）里各折平均 AUC 的最大值（论文做法，偏乐观）；置换分布同样取最大值，所以显著性是公平的。"]
        (rd / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not errors and len(met) == len(jobs):
        runs.mark_done(rd)
    log(f"SUMMARY_E102 rows={len(met)}/{len(jobs)} errors={len(errors)} run_dir={rd}")
    sys.exit(1 if errors or len(met) != len(jobs) else 0)


if __name__ == "__main__":
    main()
