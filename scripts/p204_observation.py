#!/usr/bin/env python
"""P204：D5 观测段 + 按归属刺激对累加的连接度（00 第 4 节 P204；B2 基线的输入）。

规则见 pvb/t2/observation.py 文件头与 configs/datasets/p_d5_ds004080.yaml 的 observation 一节
（T_tail = 2.0 s；屏蔽 owner 与下一个脉冲的刺激电极及饱和通道；长段切块）。防泄漏由 tests/test_t2_observation.py 检验。

用法（workflows/pvb-p204-obs-gate.yaml）：
    PRJ_STEP = scripts/p204_observation.py   PRJ_ARGS = --config configs/datasets/p_d5_ds004080.yaml --procs 8
输出：$PRJ_DATA/t2/d5/observation/<sub>/<run_id>_obs.npz      owners、measures、channels、sums、counts、n_segments
      $PRJ_DATA/t2/d5/observation/<sub>/<run_id>_segments.parquet 每个子段一行
      $PRJ_DATA/t2/d5/observation/<sub>_electrodes.parquet       电极坐标（P205 判相邻用，免得再挂原始数据）
      $PRJ_OUT/runs/P204_observation_d5/<时间>_<git7>/  runs.tsv、report.md、log.txt、DONE
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CODE = Path(os.environ["PRJ_CODE"])
sys.path.insert(0, str(CODE))
from pvb.probing import runs  # noqa: E402
from pvb.t2 import io_d5, observation as obs  # noqa: E402

EXP_ID = "P204_observation_d5"


def process_run(events_tsv: str, cfg: dict, out_dir: str) -> dict:
    t0 = time.time()
    meta = io_d5.read_run_meta(events_tsv)
    good = io_d5.good_channels(meta, cfg["channels"]["types"], cfg["channels"]["status_good"])
    rd = io_d5.RawReader(meta)
    picks = rd.picks(good)
    chans = [rd.names[i] for i in picks]
    segs = obs.segment_table(meta.stim, meta.sfreq, rd.n_times, cfg["observation"])
    acc = obs.owner_accumulators(rd, chans, picks, segs, meta.sfreq, cfg["observation"])
    od = Path(out_dir) / meta.subject
    od.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(od / f"{meta.run_id}_obs.npz", **{k: v for k, v in acc.items() if k != "masked_frac_mean"})
    segs.assign(run_id=meta.run_id).to_parquet(od / f"{meta.run_id}_segments.parquet", index=False)
    return dict(run_id=meta.run_id, subject=meta.subject, sfreq=meta.sfreq, n_ch=len(chans), n_pulses=len(meta.stim),
                n_segments=len(segs), obs_total_min=round(segs.length_s.sum() / 60, 1),
                seg_len_med_s=round(float(segs.length_s.median()), 3) if len(segs) else np.nan,
                n_owners=len(acc["owners"]), masked_frac_mean=round(acc["masked_frac_mean"], 4),
                seconds=round(time.time() - t0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/datasets/p_d5_ds004080.yaml")
    ap.add_argument("--subjects", default="")
    ap.add_argument("--procs", type=int, default=0)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(CODE / a.config, encoding="utf-8"))
    subs = [s for s in a.subjects.split(",") if s] or cfg["subjects_gate"]
    root = os.path.join(os.environ["PRJ_DATA"], cfg["raw_dir"])
    out_dir = os.path.join(os.environ["PRJ_DATA"], "t2", "d5", "observation")
    d = runs.new_run_dir(Path(os.environ["PRJ_OUT"]), EXP_ID)
    runs.write_snapshot(d, {**cfg, "_args": vars(a), "_subjects": subs})
    log = runs.Log(d)
    files = [f for s in subs for f in io_d5.find_runs(root, s)]
    log(f"{EXP_ID}: {len(subs)} 名患者，{len(files)} 个 run；T_tail = {cfg['observation']['t_tail']} s")
    for s in subs:
        el = io_d5.read_electrodes(root, s)
        os.makedirs(out_dir, exist_ok=True)
        el[["x", "y", "z"]].reset_index().to_parquet(Path(out_dir) / f"{s}_electrodes.parquet", index=False)
    rows = []
    with ProcessPoolExecutor(max_workers=min(a.procs or cfg.get("procs", 8), max(1, len(files)))) as ex:
        futs = {ex.submit(process_run, f, cfg, out_dir): f for f in files}
        for fu in as_completed(futs):
            try:
                r = fu.result(); rows.append(r)
                log(f"OK  {r['run_id']}  段 {r['n_segments']}（{r['obs_total_min']} min，中位 {r['seg_len_med_s']} s）"
                    f"  owner {r['n_owners']}  屏蔽 {r['masked_frac_mean']:.3f}  {r['seconds']} s")
            except Exception as e:
                rows.append(dict(run_id=os.path.basename(futs[fu]), error=repr(e)))
                log(f"ERR {os.path.basename(futs[fu])}: {e!r}\n{traceback.format_exc()}")
    df = pd.DataFrame(rows).sort_values("run_id")
    df.to_csv(d / "runs.tsv", sep="\t", index=False)
    n_err = int(df["error"].notna().sum()) if "error" in df else 0
    rep = f"# {EXP_ID}\n\n{df.to_string(index=False)}\n"
    (d / "report.md").write_text(rep, encoding="utf-8")
    log(rep)
    log(f"SUMMARY exp={EXP_ID} ok={len(df) - n_err}/{len(files)}")
    if n_err == 0 and files:
        runs.mark_done(d)


if __name__ == "__main__":
    main()
