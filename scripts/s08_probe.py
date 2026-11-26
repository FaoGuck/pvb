#!/usr/bin/env python
# 来自 S2R 505b298 的 scripts/s08_probe.py，2026-11-26 复制（机械改名：S2R_→PRJ_、包名 s2r→项目名）
"""S8：被试内探测与评估（总纲 6.8）。预测器可以是经典特征（E101）或基础模型表征（E103 起）。

用法（workflows/s2r-e101-classical.yaml、s2r-e103-fm.yaml、s2r-e104-random.yaml）:
    PRJ_STEP = scripts/s08_probe.py
    PRJ_ARGS = --exp configs/experiments/<实验>.yaml
    数据：data/interim/d1/qc/（S5，含 keep_W3、mep_resid、block）+ 经典特征 data/features/d1/classical/ 或表征 data/features/d1/embeddings/

预测器（配置 predictors: 下每一项）
    type: classical（默认）  prefixes / columns 选 S6 特征列
    type: history            （E110，"上一个 MEP"基线）lags（[1] / [1,2,3]）、kind（causal：ln MEP 减去它及之前 window 个试验的中位数，
                             只用过去；resid：直接用 S4 的 mep_resid，其去漂移用了以当前试验为中心的窗口，只作核对）、window（默认 25）
    任何 classical / embedding 预测器可加 history: {lags, kind, window}，把历史特征拼在后面；history_unpenalized（默认 true）让这几列不受惩罚
    type: embedding          dir（embeddings 下的子目录，如 reve、random_init_reve）、window（W4/W2/W1）、pool（meanpool/last1s/c4）、
                             layers（[last] / all / 整数列表，每层一个预测器）、seeds（随机初始化的种子列表，文件名带 _seed<k>）
    type: stack              （P2-B，07 第 11 节）两层堆叠：eeg = 第一层的脑电特征（classical 写法，或 {type: embedding, dir, window,
                             pool, layer}），history = 第二层的历史特征；见 cv.nested_stack。只做回归目标；PCA 只作用于第一层
                             metrics.csv 额外列 stack_b_hist、stack_b_eeg（真实目标 6 折第二层系数的平均）、stack_n_eeg_zero
输出: $PRJ_OUT/runs/<exp_id>/<时间戳>_<git7>/
    metrics.csv    长表（总纲 7.3），每个被试 × 预测器 × 层 × 种子 × PCA × 目标一行；额外列 pca、n_features、n_perm、chance、null_mean
    preds.parquet  每个试验的外层测试预测（只存真实目标的预测）
    report.md      组水平概况；正式统计在 S9
    config.yaml、env.txt、log.txt、DONE
并行：--procs N（默认 = 可见 GPU 数）个进程按被试分工，第 i 个进程用第 i mod n_gpu 张卡；每个被试的置换由 (seed, 被试号) 决定，与进程无关，
     所以并行与否、跑在哪张卡上，结果都相同；不同实验之间同一被试的置换也相同（可做配对比较）
本地测试：--n-perm 20 --subjects 283 --procs 1 --emb-dir <临时目录> --out-root <临时目录>
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
from pvb.probing import cv, runs  # noqa: E402

CHANCE = {"spearman": 0.0, "auc": 0.5}


# ---------- 预测器 ----------
def expand_predictors(cfg: dict) -> list[dict]:
    """把配置展开成一个个具体预测器（嵌入类按层、按种子展开）。"""
    out = []
    for pname, ps in cfg["predictors"].items():
        if ps.get("type") == "stack":
            e = ps["eeg"]
            emb = e.get("type", "classical") == "embedding"
            out.append({"name": pname, "type": "stack", "spec": ps, "window": e.get("window", "na"),
                        "pool": e["pool"] if emb else "na", "layer": e.get("layer", "last") if emb else "na",
                        "seed": cfg["seed"], "file_seed": e.get("seed")})
            continue
        if ps.get("type", "classical") in ("classical", "history"):
            out.append({"name": pname, "type": ps.get("type", "classical"), "spec": ps, "window": ps.get("window", "na"),
                        "pool": "na", "layer": "na", "seed": cfg["seed"]})
            continue
        for seed in ps.get("seeds", [None]):
            for layer in (ps.get("layers") or ["last"]) if ps.get("layers") != "all" else ["all"]:
                out.append({"name": pname, "type": "embedding", "spec": ps, "window": ps["window"], "pool": ps["pool"],
                            "layer": layer, "seed": cfg["seed"] if seed is None else seed, "file_seed": seed})
    return out


def select_columns(df: pd.DataFrame, spec: dict) -> list[str]:
    cols = [c for c in df.columns if any(c.startswith(p) for p in spec.get("prefixes", []))]
    cols += [c for c in spec.get("columns", []) if c not in cols]
    missing = [c for c in spec.get("columns", []) if c not in df.columns]
    if missing or not cols:
        raise ValueError(f"特征列缺失: {missing or spec}")
    return cols


def load_base(data: Path, sub: str, keep_col: str) -> pd.DataFrame:
    q = pd.read_parquet(data / "interim" / "d1" / "qc" / f"sub-{sub}_qc.parquet")
    q = q[["aligned_idx", "block", keep_col, "mep_resid", "emg_valid"]]
    return q[q[keep_col].astype(bool)].sort_values("aligned_idx").reset_index(drop=True)


def classical_matrix(data: Path, sub: str, base: pd.DataFrame, spec: dict, cache: dict) -> np.ndarray:
    if "classical" not in cache:
        f = pd.read_parquet(data / "features" / "d1" / "classical" / f"sub-{sub}_classical.parquet")
        m = base[["aligned_idx"]].merge(f.drop(columns=["subject", "block", "eeg_idx"]), on="aligned_idx", how="left")
        if m.drop(columns="aligned_idx").isna().any().any():
            raise ValueError(f"sub-{sub}: 有保留试验在经典特征表里找不到")
        cache["classical"] = m
    m = cache["classical"]
    return m[select_columns(m, spec)].to_numpy(dtype=float)


def history_matrix(data: Path, sub: str, base: pd.DataFrame, h: dict, cache: dict) -> np.ndarray:
    """E110："上一个 MEP"等反应历史特征，行顺序与 base 一致 → (n, len(lags))。
    用全部对齐试验（含被 QC 剔除的：它们确实发生过），只在 block 内往前看；缺失（block 开头、EMG 无效）填 0。"""
    key = ("history", h.get("kind", "causal"), tuple(h["lags"]), int(h.get("window", 25)))
    if key not in cache:
        q = pd.read_parquet(data / "interim" / "d1" / "qc" / f"sub-{sub}_qc.parquet")
        q = q.sort_values("aligned_idx").reset_index(drop=True)
        ok = q["emg_valid"].astype(bool) & (q["mep_raw_uv"] > 0)
        lnm = np.where(ok, np.log(q["mep_raw_uv"].where(q["mep_raw_uv"] > 0, 1.0)), np.nan)
        if h.get("kind", "causal") == "causal":
            w = int(h.get("window", 25))
            s_ = pd.Series(lnm)
            past_med = s_.groupby(q["block"]).transform(lambda v: v.rolling(w, min_periods=1).median())
            val = pd.Series(lnm - past_med.to_numpy())                   # 试验 t 相对它及之前 w−1 个试验的中位数，只用 ≤ t
        elif h["kind"] == "resid":
            val = q["mep_resid"].where(ok)
        else:
            raise ValueError(h["kind"])
        cols = {f"lag{k}": val.groupby(q["block"]).shift(k) for k in h["lags"]}   # 试验 t 用 t−k 的值
        H = pd.DataFrame({"aligned_idx": q["aligned_idx"], **cols}).fillna(0.0)
        cache[key] = base[["aligned_idx"]].merge(H, on="aligned_idx", how="left").drop(columns="aligned_idx").to_numpy(float)
    return cache[key]


def embedding_tensor(emb_dir: Path, sub: str, base: pd.DataFrame, p: dict, cache: dict) -> np.ndarray:
    """返回保留试验的 (n, n_layers, dim) float32，行顺序与 base 一致。"""
    d = emb_dir / p["spec"]["dir"]
    fn = f"sub-{sub}_{p['window']}_{p['pool']}" + ("" if p["file_seed"] is None else f"_seed{p['file_seed']}") + ".npy"
    key = str(d / fn)          # 11-23 修：原来只用文件名作键，三个模型的文件同名，后读的模型复用了 REVE 的缓存
    if key not in cache:
        E = np.load(d / fn)
        ai = pd.read_parquet(d / f"sub-{sub}_trials.parquet")["aligned_idx"].to_numpy()
        pos = pd.Series(np.arange(len(ai)), index=ai)
        idx = pos.reindex(base["aligned_idx"]).to_numpy()
        if np.isnan(idx.astype(float)).any():
            raise ValueError(f"sub-{sub}: 有保留试验在 {fn} 里找不到")
        cache[key] = E[idx.astype(int)].astype(np.float32)
    return cache[key]


# ---------- 每个被试（可在子进程里运行） ----------
_DEV = None


def _init_worker(q):
    import torch

    global _DEV
    _DEV = torch.device(q.get())


def run_subject(args):
    s, cfg, data, emb_dir, n_perm = args
    import torch

    dev = _DEV or cv.device()
    t0 = time.time()
    base_row = {"exp_id": cfg["exp_id"], "run_id": os.environ.get("PRJ_RUN_ID", "local"), "git_commit": runs.git_commit(),
                "dataset": cfg["dataset"]}
    try:
        m = load_base(data, s, cfg["trials"]["keep_column"])
        blocks = m["block"].to_numpy()
        rng = np.random.default_rng(int(cfg["seed"]) * 100003 + int(s))
        perm = cv.permutation_index(blocks, n_perm, rng)          # 同一被试所有预测器、所有实验共用同一组置换
        targets = {}
        for tname, tspec in cfg["targets"].items():
            y = m[tspec["column"]].to_numpy(dtype=float)
            if tspec.get("split") == "block_median":
                y = cv.block_median_split(y, blocks)
            targets[tname] = (tspec, y, cv.with_permutations(y, perm))
        cache, mrows, prows = {}, [], []
        for p in expand_predictors(cfg):
            if p["type"] == "history":
                Xs = [("na", history_matrix(data, s, m, p["spec"], cache))]
            elif p["type"] == "classical":
                Xs = [("na", classical_matrix(data, s, m, p["spec"], cache))]
            elif p["type"] == "stack":                                      # P2-B：第一层脑电 + 第二层历史
                e = p["spec"]["eeg"]
                if e.get("type", "classical") == "classical":
                    Xs = [("na", classical_matrix(data, s, m, e, cache))]
                else:
                    E = embedding_tensor(emb_dir, s, m, {"spec": e, "window": e["window"], "pool": e["pool"],
                                                         "file_seed": e.get("seed")}, cache)
                    li = E.shape[1] - 1 if p["layer"] == "last" else int(p["layer"])
                    Xs = [(li, E[:, li])]
                H_stack = history_matrix(data, s, m, p["spec"]["history"], cache)
            else:
                E = embedding_tensor(emb_dir, s, m, p, cache)
                L = E.shape[1]
                lay = list(range(L)) if p["layer"] == "all" else [L - 1 if p["layer"] == "last" else int(p["layer"])]
                Xs = [(li, E[:, li]) for li in lay]
            boost = None
            if p["spec"].get("history") and p["type"] != "stack":          # 脑电特征 + 反应历史（E110）
                H = history_matrix(data, s, m, p["spec"]["history"], cache)
                Xs = [(li, np.hstack([X, H])) for li, X in Xs]
                if p["spec"].get("history_unpenalized", True):
                    boost = list(range(Xs[0][1].shape[1] - H.shape[1], Xs[0][1].shape[1]))
            for li, X in Xs:
                for tname, (tspec, y, Y) in targets.items():
                    metric = "spearman" if tspec["task"] == "regression" else "auc"
                    if p["type"] == "stack" and tspec["task"] != "regression":
                        continue                                            # 堆叠只做 y_cont（07 第 11.3 节）
                    for pca in cfg["cv"]["pca"]:
                        k = None if pca in (None, "none") else int(pca)
                        if k is not None and X.shape[1] <= k:
                            continue
                        extra = {}
                        if p["type"] == "stack":
                            r = cv.nested_stack(X, H_stack, Y, blocks, cfg, k, dev)
                            extra = {"stack_b_hist": float(np.mean(r["coef"][:, 1])),      # 真实目标 6 折的平均系数
                                     "stack_b_eeg": float(np.mean(r["coef"][:, -1])),
                                     "stack_n_eeg_zero": int(np.sum(r["coef"][:, -1] == 0))}   # 脑电系数被约束为 0 的折数
                        else:
                            r = cv.nested_lobo(X, Y, blocks, tspec["task"], cfg, k, dev, boost)
                        mrows.append({**base_row, "subject": f"sub-{s}", "predictor": p["name"], "window": p["window"],
                                      "pool": p["pool"], "layer": li, "target": tname, "metric": metric,
                                      "value": float(r["metric"][0]), "p_perm": cv.perm_p(r["metric"]), "n_trials": len(m),
                                      "seed": p["seed"], "pca": "none" if k is None else k,
                                      "n_features": X.shape[1] + (H_stack.shape[1] if p["type"] == "stack" else 0),
                                      "n_perm": n_perm, "chance": CHANCE[metric],
                                      "null_mean": float(np.mean(r["metric"][1:])) if n_perm else np.nan, **extra})
                        prows.append(pd.DataFrame({"subject": f"sub-{s}", "aligned_idx": m["aligned_idx"], "block": blocks,
                                                   "predictor": p["name"], "layer": str(li), "seed": p["seed"],
                                                   "pca": "none" if k is None else str(k), "target": tname,
                                                   "y_true": y, "y_pred": r["pred"][:, 0]}))
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        line = (f"[S8] sub-{s} OK n={len(m)} {time.time() - t0:.0f}s dev={dev}  " +
                "  ".join(f"{r['predictor']}/L{r['layer']}/s{r['seed']}/{r['pca']}/{r['target']}={r['value']:.3f}"
                          f"(p={r['p_perm']:.3f})" for r in mrows if r["layer"] in ("na",) or len(mrows) <= 40))
        return s, mrows, prows, line, None
    except Exception:  # noqa: BLE001
        return s, [], [], f"[S8] sub-{s} FAIL dev={dev}", traceback.format_exc()


# ---------- 报告 ----------
def report(met: pd.DataFrame, path: Path, header: list[str]):
    from scipy import stats

    keys = ["predictor", "window", "pool", "layer", "pca", "target", "metric"]
    lines = header + ["", "相对置换均值的提升 = 指标 − 该被试置换分布的均值（留一 block 交叉验证在无信号时 Spearman 期望约 −0.03，不是 0）。", "",
                      "| 预测器 | 窗口 | 池化 | 层 | 种子 | PCA | 目标 | 指标 | n | 中位数 | 四分位距 | p<0.05 人数 | 相对置换均值提升（中位数） | 提升 Wilcoxon p（单侧） |",
                      "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]

    def row(g, seed_label):
        v = g["value"].to_numpy()
        gain = (g["value"] - g["null_mean"]).to_numpy()
        pw = stats.wilcoxon(gain, alternative="greater")[1] if len(v) > 5 else np.nan
        q1, q3 = np.percentile(v, [25, 75])
        nsig = f"{int((g['p_perm'] < 0.05).sum())}/{len(v)}" if "p_perm" in g else "—"
        k = g.iloc[0]
        return (f"| {k.predictor} | {k.window} | {k.pool} | {k.layer} | {seed_label} | {k.pca} | {k.target} | {k.metric} | {len(v)} | "
                f"{np.median(v):.3f} | {q1:.3f}–{q3:.3f} | {nsig} | {np.median(gain):+.3f} | {pw:.3g} |")

    for _, g in met.groupby(keys, sort=False):
        seeds = sorted(g["seed"].unique())
        if len(seeds) > 1:
            for sd, gs in g.groupby("seed"):
                lines.append(row(gs, sd))
            gm = g.groupby("subject", as_index=False).agg(
                **{c: (c, "first") for c in keys}, value=("value", "mean"), null_mean=("null_mean", "mean"))
            lines.append(row(gm, f"平均（{len(seeds)} 个）"))
        else:
            lines.append(row(g, seeds[0]))
    lines += ["", "每个被试的指标来自 6 折留一 block 的拼接测试预测；p_perm 为 block 内打乱目标、整套嵌套交叉验证重跑的置换 p 值。"
              "种子平均行：先对每个被试的多个随机初始化种子取平均，再做组水平统计。正式统计（FDR、配对比较、自助法置信区间）在 S9。"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--subjects", nargs="*")
    ap.add_argument("--n-perm", type=int, help="覆盖配置里的置换次数（本地测试用）")
    ap.add_argument("--procs", type=int, default=0, help="并行进程数，默认 = 可见 GPU 数（无 GPU 时 1）")
    ap.add_argument("--emb-dir", help="表征目录，默认 $PRJ_DATA/features/d1/embeddings")
    ap.add_argument("--out-root", help="默认 $PRJ_OUT")
    args = ap.parse_args()

    import torch

    data = Path(os.environ["PRJ_DATA"])
    emb_dir = Path(args.emb_dir) if args.emb_dir else data / "features" / "d1" / "embeddings"
    out_root = Path(args.out_root or os.environ.get("PRJ_OUT", CODE.parent / "out"))
    cfg = yaml.safe_load(open(CODE / args.exp, encoding="utf-8"))
    if args.n_perm is not None:
        cfg["permutation"]["n"] = args.n_perm
    n_perm = int(cfg["permutation"]["n"])
    n_gpu = torch.cuda.device_count()
    procs = args.procs or max(1, n_gpu)
    devs = [f"cuda:{i % n_gpu}" for i in range(procs)] if n_gpu else ["cpu"] * procs
    rd = runs.new_run_dir(out_root, cfg["exp_id"])
    runs.write_snapshot(rd, {**cfg, "_resolved": {"devices": devs, "subjects_arg": args.subjects, "git": runs.git_commit(),
                                                 "run_id": os.environ.get("PRJ_RUN_ID"),
                                                 "n_predictor_units": len(expand_predictors(cfg))}})
    log = runs.Log(rd)
    subs = args.subjects or sorted(p.name[4:7] for p in (data / "interim" / "d1" / "qc").glob("sub-*_qc.parquet"))
    log(f"[S8] exp={cfg['exp_id']} run_dir={rd} gpus={n_gpu} procs={procs} subjects={len(subs)} n_perm={n_perm} "
        f"predictors={len(expand_predictors(cfg))}")

    jobs = [(s, cfg, data, emb_dir, n_perm) for s in subs]
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
        met = met.sort_values(["predictor", "seed", "layer", "pca", "target", "subject"],
                              key=lambda c: c.map(lambda v: f"{v:04d}" if isinstance(v, (int, np.integer)) else str(v)))
        cols = runs.METRIC_COLUMNS + [c for c in met.columns if c not in runs.METRIC_COLUMNS]
        met[cols].to_csv(rd / "metrics.csv", index=False)
        pd.concat(prows, ignore_index=True).to_parquet(rd / "preds.parquet", index=False)
        report(met, rd / "report.md",
               [f"# {cfg['exp_id']} 结果概况", "", f"- 运行目录: {rd.name}；git {runs.git_commit()}；GPU {n_gpu} 张、{procs} 个进程；"
                f"用时 {time.time() - t0:.0f} s",
                f"- 被试 {met['subject'].nunique()} / {len(subs)}；失败: {', '.join(errors) or '无'}；置换 {n_perm} 次"])
    if not errors:
        runs.mark_done(rd)
    log(f"SUMMARY_S8 subjects_ok={len(subs) - len(errors)}/{len(subs)} run_dir={rd}")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
