#!/usr/bin/env python
"""P206 第 1 步：观测段 → 冻结模型表征 → 按归属刺激对累加（docs/06 第 3 节；框架 2026-11-27）。

每个 run：观测段与 P204 完全相同（observation.segment_table / segment_valid，参数 configs/datasets/p_d5_ds004080.yaml 的
observation）；每段只读一次信号，所有 --models 共用；每个模型的累加量存一个 npz（pvb/t2/fm.py）。
防泄漏：累加按 owner 分开存，评估时只对可用折的 owner 求平均（p206_eval.py）；tests/test_t2_p206.py 做两项替换测试。
已存在的 npz 跳过（断点续跑；共享规则第 8 条：超时被杀不回传，分批跑时下一批自动接着做）。

多卡：--workers N --gpus G → 启动 N 个子进程，第 i 个用卡 i % G，按 run 体积从大到小轮流分配；每个子进程只加载一次模型。
同一个环境里的模型才能一起跑（MOMENT 在 pvb-env-moment-1，其余在 pvb-env-2）。

用法（容器里；本地冒烟见 tests/test_t2_p206.py）：
    PRJ_STEP = scripts/p206_extract.py
    PRJ_ARGS = --models chronos,chronos-rand --subjects sub-ccepAgeUMCU01,… --workers 4 --gpus 4
输出：$PRJ_DATA/t2/d5/fm/<模型>/<患者>/<run_id>_emb.npz       owners、channels、emb_sums、emb_counts、sim_sums、sim_counts、
                                                                n_segments、dim、seconds_model、seconds_read、n_chseg
      $PRJ_OUT/runs/P206_extract_d5/<时间>_<git7>/  runs.tsv（每个 run × 模型一行：耗时、通道段数、吞吐）、report.md、DONE
SUMMARY 行：SUMMARY exp=P206_extract_d5 models=… ok=…/… ch_seg_per_s_med=…（C 步按它定 D 的卡数和超时）
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CODE = Path(os.environ["PRJ_CODE"])
sys.path.insert(0, str(CODE))
from pvb.probing import runs  # noqa: E402
from pvb.t2 import encoders as encs, fm, io_d5, observation as obs  # noqa: E402

EXP_ID = "P206_extract_d5"


def out_path(fm_root: Path, model: str, meta) -> Path:
    return fm_root / model / meta.subject / f"{meta.run_id}_emb.npz"


def run_files(root: str, subs: list[str]) -> list[str]:
    files = [f for s in subs for f in io_d5.find_runs(root, s)]
    size = lambda f: os.path.getsize(f[: -len("_events.tsv")] + "_ieeg.eeg") if os.path.exists(f[: -len("_events.tsv")] + "_ieeg.eeg") else 0  # noqa: E731
    return sorted(files, key=lambda f: (-size(f), f))


def worker(files: list[str], a, cfg: dict, enc_cfg: dict, log, shard: int) -> pd.DataFrame:
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:                                 # 只用 spec_test 的本地冒烟
        dev = "cpu"
    fm_root = Path(os.environ["PRJ_DATA"]) / cfg["p206"]["fm_dir"]
    models = [m for m in a.models.split(",") if m]
    t0 = time.time()
    E = {m: encs.make(m, enc_cfg, a.models_dir, dev) for m in models}
    log(f"[shard {shard}] 设备 {dev}；加载 {models} 用时 {time.time() - t0:.1f} s；{len(files)} 个 run")
    for m, e in E.items():
        if getattr(e, "n_reset", 0):
            log(f"[shard {shard}]   {m}：随机初始化，重置 {e.n_reset} 个子模块")
    root = os.path.join(os.environ["PRJ_DATA"], cfg["raw_dir"])
    xyz_cache: dict = {}
    rows = []
    for f in files:
        try:
            meta = io_d5.read_run_meta(f)
            todo = {m: e for m, e in E.items() if not out_path(fm_root, m, meta).exists()}
            if not todo:
                rows.append(dict(run_id=meta.run_id, subject=meta.subject, model="*", status="skip_exists")); continue
            t1 = time.time()
            good = io_d5.good_channels(meta, cfg["channels"]["types"], cfg["channels"]["status_good"])
            rd = io_d5.RawReader(meta)
            picks = rd.picks(good)
            chans = [rd.names[i] for i in picks]
            segs = obs.segment_table(meta.stim, meta.sfreq, rd.n_times, cfg["observation"])
            if meta.subject not in xyz_cache:
                el = io_d5.read_electrodes(root, meta.subject)
                xyz_cache[meta.subject] = {n: tuple(v) for n, v in el[["x", "y", "z"]].iterrows()}
            acc = fm.owner_embed_accumulators(rd, chans, picks, segs, meta.sfreq, cfg["observation"], todo,
                                              xyz_cache[meta.subject])
            wall = time.time() - t1
            for m, ac in acc.items():
                p = out_path(fm_root, m, meta)
                p.parent.mkdir(parents=True, exist_ok=True)
                tmp = p.with_name(p.stem + ".tmp.npz")
                np.savez_compressed(tmp, **ac)
                os.replace(tmp, p)                      # 写完再改名：被杀时不会留下半个文件被当成“已完成”
                r = dict(run_id=meta.run_id, subject=meta.subject, model=m, status="ok", sfreq=meta.sfreq, n_ch=len(chans),
                         n_segments=len(segs), dim=int(ac["dim"]), n_chseg=int(ac["n_chseg"]),
                         seconds_model=round(float(ac["seconds_model"]), 1),
                         seconds_read=round(float(ac["seconds_read"]), 1), seconds_wall=round(wall, 1),
                         mb=round(p.stat().st_size / 2**20, 2))
                r["ch_seg_per_s"] = round(r["n_chseg"] / max(r["seconds_model"], 1e-6), 1)
                rows.append(r)
                log(f"[shard {shard}] OK  {meta.run_id}  {m}  段 {len(segs)}  通道 {len(chans)}  D={r['dim']}  "
                    f"模型 {r['seconds_model']} s（{r['ch_seg_per_s']} 通道段/s）  读 {r['seconds_read']} s  {r['mb']} MB")
        except Exception as e:  # noqa: BLE001  逐 run 容错（S2R 约定）
            rows.append(dict(run_id=os.path.basename(f), model="*", status="error", error=repr(e)[:300]))
            log(f"[shard {shard}] ERR {os.path.basename(f)}: {e!r}\n{traceback.format_exc()}")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/datasets/p_d5_ds004080.yaml")
    ap.add_argument("--models", required=True, help="逗号分隔，见 configs/models/p206_encoders.yaml；加 -rand 为随机初始化")
    ap.add_argument("--subjects", default="", help="逗号分隔；空 = 闸门 5 人；all = raw 目录下全部患者")
    ap.add_argument("--models-dir", default="")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--gpus", type=int, default=0)
    ap.add_argument("--shard", type=int, default=-1, help="内部用：子进程编号")
    ap.add_argument("--run-dir", default="", help="内部用")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(CODE / a.config, encoding="utf-8"))
    enc_cfg = yaml.safe_load(open(CODE / cfg["p206"]["encoders_config"], encoding="utf-8"))
    a.models_dir = a.models_dir or os.path.join(os.environ["PRJ_DATA"], "models")
    root = os.path.join(os.environ["PRJ_DATA"], cfg["raw_dir"])
    if a.subjects == "all":
        subs = sorted(p.name for p in Path(root).glob("sub-*") if p.is_dir())
    else:
        subs = [s for s in a.subjects.split(",") if s] or cfg["subjects_gate"]
    files = run_files(root, subs)

    if a.shard >= 0:                                    # 子进程
        df = worker(files[a.shard::a.workers], a, cfg, enc_cfg, lambda *x: print(*x, flush=True), a.shard)
        if df.empty:                                    # 分片比 run 多时这一片是空的：也要写表头，汇总时才读得进来
            df = pd.DataFrame(columns=["run_id", "model", "status"])
        df.to_csv(Path(a.run_dir) / f"shard_{a.shard}.tsv", sep="\t", index=False)
        return

    d = runs.new_run_dir(Path(os.environ["PRJ_OUT"]), EXP_ID)
    runs.write_snapshot(d, {**cfg, "_encoders": enc_cfg, "_args": vars(a), "_subjects": subs})
    log = runs.Log(d)
    log(f"{EXP_ID}: 模型 {a.models}；{len(subs)} 名患者、{len(files)} 个 run；workers {a.workers}、gpus {a.gpus}")
    t0 = time.time()
    if a.workers <= 1:
        df = worker(files, a, cfg, enc_cfg, log, -1)
    else:
        procs = []
        for i in range(a.workers):
            env = dict(os.environ)
            if a.gpus > 0:
                env["CUDA_VISIBLE_DEVICES"] = str(i % a.gpus)
            cmd = [sys.executable, "-X", "utf8", __file__] + [x for x in sys.argv[1:]] + ["--shard", str(i), "--run-dir", str(d)]
            procs.append(subprocess.Popen(cmd, env=env))
        rcs = [p.wait() for p in procs]
        parts = [pd.read_csv(d / f"shard_{i}.tsv", sep="\t") for i in range(a.workers) if (d / f"shard_{i}.tsv").exists()]
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["status"])
        if any(rcs) or len(parts) < a.workers:
            log(f"子进程返回码 {rcs}；回收到 {len(parts)} / {a.workers} 份结果")
            df = pd.concat([df, pd.DataFrame([dict(run_id=f"shard_{i}", model="*", status="error", error=f"rc={rc}")
                                              for i, rc in enumerate(rcs) if rc])], ignore_index=True)
    df.to_csv(d / "runs.tsv", sep="\t", index=False)
    ok = df[df.status == "ok"] if "status" in df else df.iloc[0:0]
    n_err = int((df.status == "error").sum()) if "status" in df else 0
    n_skip = int((df.status == "skip_exists").sum()) if "status" in df else 0
    rep = [f"# {EXP_ID}", "", f"- 模型 {a.models}；run {len(files)}；新算 {ok.run_id.nunique()}、已存在跳过 {n_skip}、出错 {n_err}；"
           f"总用时 {(time.time() - t0) / 60:.1f} min（workers {a.workers}）", ""]
    if len(ok):
        g = ok.groupby("model").agg(runs=("run_id", "nunique"), dim=("dim", "first"), n_chseg=("n_chseg", "sum"),
                                    sec_model=("seconds_model", "sum"), sec_read=("seconds_read", "sum"),
                                    thr_med=("ch_seg_per_s", "median"), mb=("mb", "sum"))
        rep += ["每个模型（吞吐 = 通道段 / 模型秒；C 步用它估 D 的卡数和超时）：", "", g.round(1).to_string(), ""]
    (d / "report.md").write_text("\n".join(rep) + "\n", encoding="utf-8")
    log("\n".join(rep))
    thr = float(ok.ch_seg_per_s.median()) if len(ok) else float("nan")
    log(f"SUMMARY exp={EXP_ID} models={a.models} ok={len(files) - n_err}/{len(files)} new={ok.run_id.nunique()} "
        f"skip={n_skip} err={n_err} ch_seg_per_s_med={thr:.1f}")
    if n_err == 0 and files:
        runs.mark_done(d)


if __name__ == "__main__":
    main()
