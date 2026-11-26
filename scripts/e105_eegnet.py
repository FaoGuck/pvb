#!/usr/bin/env python
# 来自 S2R 505b298 的 scripts/e105_eegnet.py，2026-11-26 复制（机械改名：S2R_→PRJ_、包名 s2r→项目名）
"""E105：被试内监督 EEGNet（总纲 C2，"可达上限"参考）。算法见 s2r/probing/eegnet.py。

用法（workflows/s2r-e105-eegnet.yaml）:
    PRJ_STEP = scripts/e105_eegnet.py
    PRJ_ARGS = --exp configs/experiments/E105_eegnet_d1.yaml
    数据：data/interim/d1/（epochs_pre 的 h5 取 clean_<W>；qc 取 keep_W3、mep_resid、block）
输出: $PRJ_OUT/runs/<exp_id>/<时间戳>_<git7>/  metrics.csv、preds.parquet、report.md、config.yaml、env.txt、log.txt、DONE
      格式与 s08_probe.py 相同（predictor = eegnet_<W>，pool/layer = na，pca = none，n_features = 网络参数量/模型）
      额外列 epochs_median：真实目标 6 折选中 epoch 的中位数（看早停是否正常）
并行：--procs N（默认 = GPU 数）个进程按被试分工；每个进程内 1 + n_perm 个模型打包训练，GPU 已基本吃满，一般每卡 1 个进程即可
置换：取 S8 同一随机序列（种子 (seed, 被试号)、n_from 次）的前 n 个，所以与 E101–E108 的前 n 个置换完全相同
本地测试：--subjects 283 --n-perm 5 --procs 1 --pack 6 --out-root <临时目录>
"""
from __future__ import annotations

import argparse
import json
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
sys.path.insert(0, str(CODE / "scripts"))
from pvb.probing import cv, eegnet, runs  # noqa: E402

CHANCE = {"spearman": 0.0, "auc": 0.5}
_DEV = None


def _init_worker(q):
    import torch

    global _DEV
    _DEV = torch.device(q.get())


def load_base(data: Path, sub: str, keep_col: str) -> pd.DataFrame:
    q = pd.read_parquet(data / "interim" / "d1" / "qc" / f"sub-{sub}_qc.parquet")
    q = q[["aligned_idx", "block", keep_col, "mep_resid"]]
    return q[q[keep_col].astype(bool)].sort_values("aligned_idx").reset_index(drop=True)


def load_window(data: Path, sub: str, base: pd.DataFrame, window: str, fs_out: int) -> np.ndarray:
    """保留试验的 clean_<window>，行顺序与 base 一致，降采样到 fs_out → (n, C, T) float32。"""
    import h5py
    from scipy import signal

    with h5py.File(data / "interim" / "d1" / "epochs_pre" / f"sub-{sub}_pre.h5", "r") as f:
        fs = int(f.attrs["clean_sfreq"])
        ai = f["aligned_idx"][:]
        ok = ai >= 0                                                # 未对齐 EMG 的试验 aligned_idx = −1（可重复，如 767、780），先去掉
        pos = pd.Series(np.where(ok)[0], index=ai[ok])
        idx = pos.reindex(base["aligned_idx"]).to_numpy()
        if np.isnan(idx.astype(float)).any():
            raise ValueError(f"sub-{sub}: 有保留试验在 h5 里找不到")
        idx = idx.astype(int)
        order = np.argsort(idx)                                     # h5 要求递增下标
        X = np.empty((len(idx),) + f[f"clean_{window}"].shape[1:], dtype=np.float32)
        X[order] = f[f"clean_{window}"][idx[order]]
    if fs % fs_out:
        raise ValueError(f"clean 采样率 {fs} 不是 {fs_out} 的整数倍")
    q = fs // fs_out
    if q > 1:                                                       # clean 已 1–45 Hz 带通，resample_poly 自带抗混叠
        X = signal.resample_poly(X.astype(np.float64), 1, q, axis=-1).astype(np.float32)
    return X


def hp_for(cfg: dict) -> dict:
    e = dict(cfg["eegnet"])
    k1 = int(round(e.pop("kern1_s") * e["fs"]))
    e["k1"] = k1 + 1 - k1 % 2                                       # 时间卷积核 ≈ 0.5 s，取奇数（same 填充对称；125 Hz 下 63 点）
    return e


def run_subject(args):
    s, cfg, data, n_perm, pack = args
    import torch

    dev = _DEV or cv.device()
    torch.backends.cudnn.benchmark = True
    t0 = time.time()
    base_row = {"exp_id": cfg["exp_id"], "run_id": os.environ.get("PRJ_RUN_ID", "local"), "git_commit": runs.git_commit(),
                "dataset": cfg["dataset"]}
    try:
        hp = hp_for(cfg)
        m = load_base(data, s, cfg["trials"]["keep_column"])
        blocks = m["block"].to_numpy()
        rng = np.random.default_rng(int(cfg["seed"]) * 100003 + int(s))
        n_from = int(cfg["permutation"].get("n_from", n_perm))
        perm = cv.permutation_index(blocks, max(n_from, n_perm), rng)[:n_perm]   # 与 S8 的前 n_perm 个置换相同
        targets = {}
        for tname, tspec in cfg["targets"].items():
            y = m[tspec["column"]].to_numpy(dtype=float)
            if tspec.get("split") == "block_median":
                y = cv.block_median_split(y, blocks)
            targets[tname] = (tspec, y, cv.with_permutations(y, perm))
        mrows, prows = [], []
        for pname, ps in cfg["predictors"].items():
            X = load_window(data, s, m, ps["window"], int(hp["fs"]))
            n_params = None
            for ti, (tname, (tspec, y, Y)) in enumerate(targets.items()):
                metric = "spearman" if tspec["task"] == "regression" else "auc"
                seed = int(cfg["seed"]) * 100003 + int(s) * 10 + ti
                r = eegnet.lobo_eegnet(X, Y, blocks, tspec["task"], hp, seed, dev, pack)
                if n_params is None:
                    net = eegnet.PackedEEGNet(1, X.shape[1], X.shape[2], hp["F1"], hp["D"], hp["F2"], hp["k1"], hp["k3"],
                                              hp["pool1"], hp["pool2"], hp["dropout"])
                    n_params = sum(p.numel() for p in net.parameters())
                mrows.append({**base_row, "subject": f"sub-{s}", "predictor": pname, "window": ps["window"], "pool": "na",
                              "layer": "na", "target": tname, "metric": metric, "value": float(r["metric"][0]),
                              "p_perm": cv.perm_p(r["metric"]), "n_trials": len(m), "seed": int(cfg["seed"]), "pca": "none",
                              "n_features": n_params, "n_perm": n_perm, "chance": CHANCE[metric],
                              "null_mean": float(np.mean(r["metric"][1:])) if n_perm else np.nan,
                              "epochs_median": float(np.median(list(r["epochs"].values())))})
                prows.append(pd.DataFrame({"subject": f"sub-{s}", "aligned_idx": m["aligned_idx"], "block": blocks,
                                           "predictor": pname, "layer": "na", "seed": int(cfg["seed"]), "pca": "none",
                                           "target": tname, "y_true": y, "y_pred": r["pred"][:, 0]}))
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        line = (f"[E105] sub-{s} OK n={len(m)} {time.time() - t0:.0f}s dev={dev}  " +
                "  ".join(f"{r['predictor']}/{r['target']}={r['value']:.3f}(p={r['p_perm']:.3f},ep={r['epochs_median']:.0f})"
                          for r in mrows))
        return s, mrows, prows, line, None
    except Exception:  # noqa: BLE001
        return s, [], [], f"[E105] sub-{s} FAIL dev={dev}", traceback.format_exc()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--subjects", nargs="*")
    ap.add_argument("--n-perm", type=int, help="覆盖配置里的置换次数（本地测试用）")
    ap.add_argument("--procs", type=int, default=0, help="并行进程数，默认 = 可见 GPU 数（无 GPU 时 1）")
    ap.add_argument("--pack", type=int, help="一次打包训练的模型数上限（默认 = 1 + n_perm；显存不够时调小，结果不变）")
    ap.add_argument("--max-epochs", type=int, help="覆盖配置（本地测试用）")
    ap.add_argument("--out-root", help="默认 $PRJ_OUT")
    args = ap.parse_args()

    import torch
    from s08_probe import report

    data = Path(os.environ["PRJ_DATA"])
    out_root = Path(args.out_root or os.environ.get("PRJ_OUT", CODE.parent / "out"))
    cfg = yaml.safe_load(open(CODE / args.exp, encoding="utf-8"))
    if args.n_perm is not None:
        cfg["permutation"]["n"] = args.n_perm
    if args.max_epochs is not None:
        cfg["eegnet"]["max_epochs"] = args.max_epochs
    n_perm = int(cfg["permutation"]["n"])
    pack = args.pack or cfg["eegnet"].get("pack") or (1 + n_perm)
    n_gpu = torch.cuda.device_count()
    procs = args.procs or max(1, n_gpu)
    devs = [f"cuda:{i % n_gpu}" for i in range(procs)] if n_gpu else ["cpu"] * procs
    rd = runs.new_run_dir(out_root, cfg["exp_id"])
    runs.write_snapshot(rd, {**cfg, "_resolved": {"devices": devs, "subjects_arg": args.subjects, "git": runs.git_commit(),
                                                 "run_id": os.environ.get("PRJ_RUN_ID"), "pack": pack}})
    log = runs.Log(rd)
    subs = args.subjects or sorted(p.name[4:7] for p in (data / "interim" / "d1" / "qc").glob("sub-*_qc.parquet"))
    log(f"[E105] exp={cfg['exp_id']} run_dir={rd} gpus={n_gpu} procs={procs} subjects={len(subs)} n_perm={n_perm} "
        f"pack={pack} predictors={list(cfg['predictors'])} hp={json.dumps(hp_for(cfg))}")

    jobs = [(s, cfg, data, n_perm, pack) for s in subs]
    mrows, prows, errors = [], [], {}
    t0 = time.time()
    if procs == 1:
        results = map(run_subject, jobs)
    else:
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        for d in devs:
            q.put(d)
        pool = ctx.Pool(procs, initializer=_init_worker, initargs=(q,))
        results = pool.imap_unordered(run_subject, jobs)
    for s, mr, pr, line, err in results:
        log(line + f"  [{time.time() - t0:.0f}s]")
        if err:
            errors[s] = err
            log(err)
        mrows += mr
        prows += pr
    if procs > 1:
        pool.close()
        pool.join()

    met = pd.DataFrame(mrows)
    if len(met):
        met = met.sort_values(["predictor", "target", "subject"])
        cols = runs.METRIC_COLUMNS + [c for c in met.columns if c not in runs.METRIC_COLUMNS]
        met[cols].to_csv(rd / "metrics.csv", index=False)
        pd.concat(prows, ignore_index=True).to_parquet(rd / "preds.parquet", index=False)
        report(met, rd / "report.md",
               [f"# {cfg['exp_id']} 结果概况", "", f"- 运行目录: {rd.name}；git {runs.git_commit()}；GPU {n_gpu} 张、{procs} 个进程；"
                f"用时 {time.time() - t0:.0f} s",
                f"- 被试 {met['subject'].nunique()} / {len(subs)}；失败: {', '.join(errors) or '无'}；置换 {n_perm} 次"
                f"（p 值分辨率 {1 / (1 + n_perm):.3f}）",
                f"- 早停选中 epoch（真实目标，各被试 6 折中位数）的中位数：" +
                "；".join(f"{k} {g['epochs_median'].median():.0f}" for k, g in met.groupby(['predictor', 'target']))])
    if not errors:
        runs.mark_done(rd)
    log(f"SUMMARY_E105 subjects_ok={len(subs) - len(errors)}/{len(subs)} run_dir={rd}")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
