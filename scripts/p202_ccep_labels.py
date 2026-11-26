#!/usr/bin/env python
"""P202：D5（ds004080）CCEP 标签——每个 run、每个刺激对、每个记录电极的 N1 有无 / 幅度 / 潜伏期（00 第 4 节 P202）。

算法复现 ds004080 作者的 mnl_ccepAge（细节与出处见 configs/datasets/p_d5_ds004080.yaml 和 pvb/t2/ccep.py）。
同时存下每个刺激对的平均波形（全部 / 奇数次 / 偶数次试验），给 P203（split-half 上限）和 P204（T_tail）用，
以后不必再读原始信号。与 derivatives/av_ccep 的逐项核对在 P201 看清 .mat 结构后另写（p202b）。

防泄漏（00 第 6 节第 3 条）：标签只用刺激后的数据；这里不产生任何观测段特征。

用法（workflows/pvb-p202-ccep-gate.yaml）：
    PRJ_STEP = scripts/p202_ccep_labels.py
    PRJ_ARGS = --config configs/datasets/p_d5_ds004080.yaml            （默认 = 闸门 5 名患者）
             [--subjects sub-ccepAgeUMCU01,sub-ccepAgeUMCU02] [--procs 8]
输入：$PRJ_DATA/raw/ds004080/<sub>/ses-*/ieeg/*（BrainVision + sidecar）
输出：$PRJ_DATA/t2/d5/ccep_labels/<sub>/<run_id>_ccep.npz      平均波形与 N1（见 save_npz）
      $PRJ_DATA/t2/d5/ccep_labels/<sub>_labels.parquet         长表：每行 = run × 刺激对 × 记录电极
      $PRJ_OUT/runs/P202_ccep_labels_d5/<时间>_<git7>/          config、env、log、runs.tsv、metrics.csv、report.md、DONE
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
from pvb.t2 import ccep, io_d5  # noqa: E402

EXP_ID = "P202_ccep_labels_d5"
DIST_BINS = [0, 10, 20, 30, 40, 60, 1000]


def process_run(events_tsv: str, cfg: dict, root: str, out_dir: str) -> tuple[dict, pd.DataFrame]:
    t0 = time.time()
    meta = io_d5.read_run_meta(events_tsv)
    good = io_d5.good_channels(meta, cfg["channels"]["types"], cfg["channels"]["status_good"])
    rd = io_d5.RawReader(meta)
    picks = rd.picks(good)
    chans = [rd.names[i] for i in picks]
    if abs(rd.sfreq - meta.sfreq) > 1e-6:
        raise ValueError(f"采样率不一致：vhdr {rd.sfreq} vs json {meta.sfreq}")
    sf = rd.sfreq
    e = cfg["epoch"]
    off = int(round(e["tmin"] * sf))
    n_samp = int(round((e["tmax"] - e["tmin"]) * sf))
    tt = (np.arange(n_samp) + off) / sf
    el = io_d5.read_electrodes(root, meta.subject)
    xyz = el.reindex(chans)[["x", "y", "z"]].to_numpy(float)

    st = meta.stim
    rows, skipped = [], {"few_trials": 0, "stim_el_bad": 0}
    A, O, E, P, NT = [], [], [], [], []
    for pair, g in st.groupby("pair", sort=True):
        if len(g) < cfg["stim"]["min_trials"]:
            skipped["few_trials"] += 1
            continue
        e1, e2 = io_d5.pair_electrodes(pair)
        if cfg["stim"]["drop_pair_if_stim_el_bad"] and not (e1 in good and e2 in good):
            skipped["stim_el_bad"] += 1
            continue
        trials = rd.segments(picks, g["sample0"].to_numpy() + off, n_samp)
        ok = np.isfinite(trials).all(axis=(1, 2))
        trials = trials[ok]
        if trials.shape[0] < cfg["stim"]["min_trials"]:
            skipped["few_trials"] += 1
            continue
        avg = ccep.average(trials, tt, cfg)
        halves = [ccep.average(trials[0::2], tt, cfg), ccep.average(trials[1::2], tt, cfg)]
        stim_idx = [i for i, c in enumerate(chans) if c in (e1, e2)]
        for x in [avg] + halves:
            x[stim_idx] = np.nan                         # mnl：刺激电极本身不评估
        n1 = ccep.detect_n1(avg, tt, cfg["n1"])
        tail = ccep.tail_time(avg, tt, sf, cfg["tail"])
        a_s, tts = ccep.to_store(avg, tt, sf, cfg["store"])
        o_s, _ = ccep.to_store(halves[0], tt, sf, cfg["store"])
        e_s, _ = ccep.to_store(halves[1], tt, sf, cfg["store"])
        A.append(a_s); O.append(o_s); E.append(e_s); P.append(pair); NT.append(trials.shape[0])
        mid = np.nanmean(el.reindex([e1, e2])[["x", "y", "z"]].to_numpy(float), axis=0)
        dist = np.linalg.norm(xyz - mid, axis=1)
        cur = pd.to_numeric(g["electrical_stimulation_current"], errors="coerce").median() * 1000
        for c, ch in enumerate(chans):
            if c in stim_idx:
                continue
            rows.append((meta.run_id, meta.subject, pair, ch, trials.shape[0], bool(n1["detected"][c]),
                         n1["latency_s"][c] * 1000, n1["amp_uv"][c], n1["sd_uv"][c], tail[c], dist[c], cur))
    lab = pd.DataFrame(rows, columns=["run_id", "subject", "stim_pair", "channel", "n_trials", "n1_detected",
                                      "n1_latency_ms", "n1_amp_uv", "base_sd_uv", "tail_s", "dist_mm", "current_ma"])
    od = Path(out_dir) / meta.subject
    od.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        od / f"{meta.run_id}_ccep.npz",
        avg=np.stack(A) if A else np.zeros((0, len(chans), 0), np.float32),
        avg_odd=np.stack(O) if O else np.zeros((0, len(chans), 0), np.float32),
        avg_even=np.stack(E) if E else np.zeros((0, len(chans), 0), np.float32),
        tt=tts if A else np.zeros(0), stim_pairs=np.array(P), channels=np.array(chans),
        n_trials=np.array(NT), sfreq_native=sf, sfreq_store=cfg["store"]["sfreq"])
    summ = dict(run_id=meta.run_id, subject=meta.subject, sfreq=sf, n_good_ch=len(chans),
                n_stim_events=len(st), n_pairs_all=st["pair"].nunique(), n_pairs_used=len(P),
                skip_few_trials=skipped["few_trials"], skip_stim_el_bad=skipped["stim_el_bad"],
                n_eval=len(lab), n1_rate=float(lab.n1_detected.mean()) if len(lab) else np.nan,
                seconds=round(time.time() - t0, 1))
    return summ, lab


def report(runs_df: pd.DataFrame, lab: pd.DataFrame, cfg: dict) -> tuple[str, pd.DataFrame]:
    lab = lab.copy()
    lab["dist_bin"] = pd.cut(lab.dist_mm, DIST_BINS, right=False)
    det = lab[lab.n1_detected]
    lines = [f"# {EXP_ID}", "",
             f"- run {len(runs_df)} 个（失败 {int(runs_df.get('error', pd.Series(dtype=str)).notna().sum())}），"
             f"刺激对 {int(runs_df.n_pairs_used.sum())} 个，评估的 刺激对 × 电极 {len(lab)} 个，N1 出现率 {lab.n1_detected.mean():.3f}",
             "", "## 每名患者", ""]
    per = lab.groupby("subject").agg(n_pairs=("stim_pair", "nunique"), n_eval=("n1_detected", "size"),
                                     n1_rate=("n1_detected", "mean")).reset_index()
    per = per.merge(det.groupby("subject").agg(lat_med_ms=("n1_latency_ms", "median"),
                                               amp_med_uv=("n1_amp_uv", "median"),
                                               tail_med_s=("tail_s", "median"),
                                               tail_p90_s=("tail_s", lambda x: x.quantile(0.9)),
                                               tail_p95_s=("tail_s", lambda x: x.quantile(0.95)),
                                               tail_gt1s=("tail_s", lambda x: (x > 1.0).mean())).reset_index(),
                    on="subject", how="left")
    lines += [per.round(3).to_string(index=False), "",
              "## 合理性检查 1：N1 出现率随距离下降（00 第 7.3 节第 1 条）", ""]
    bydist = lab.groupby(["subject", "dist_bin"], observed=False).n1_detected.mean().unstack().round(3)
    lines += [bydist.to_string(), "",
              "## CCEP 持续时间 → T_tail（P204；只看有 N1 的 刺激对 × 电极）", "",
              f"- tail_s = 整流平均波形（{cfg['tail']['smooth_s']*1000:g} ms 平滑）最后一次超过 {cfg['tail']['sd_mult']:g}×基线 SD 的时刻",
              f"- 全体：中位 {det.tail_s.median():.3f} s，90% {det.tail_s.quantile(0.9):.3f} s，95% {det.tail_s.quantile(0.95):.3f} s，"
              f"> 1.0 s 的比例 {(det.tail_s > 1.0).mean():.3f}"]
    metrics = []
    for _, r in per.iterrows():
        for m in ["n1_rate", "lat_med_ms", "amp_med_uv", "tail_med_s", "tail_p95_s", "tail_gt1s"]:
            metrics.append(dict(exp_id=EXP_ID, run_id=os.environ["PRJ_RUN_ID"], git_commit=runs.git_commit(),
                                track="T2", dataset=cfg["dataset"], subject=r.subject, metric=m, value=r[m],
                                n_pairs=r.n_pairs, n_targets=r.n_eval))
    return "\n".join(lines) + "\n", pd.DataFrame(metrics)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/datasets/p_d5_ds004080.yaml")
    ap.add_argument("--subjects", default="", help="逗号分隔；默认 = 配置里的 subjects_gate")
    ap.add_argument("--procs", type=int, default=0)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(CODE / a.config, encoding="utf-8"))
    subs = [s for s in a.subjects.split(",") if s] or cfg["subjects_gate"]
    root = os.path.join(os.environ["PRJ_DATA"], cfg["raw_dir"])
    out_dir = os.path.join(os.environ["PRJ_DATA"], cfg["labels_dir"])
    d = runs.new_run_dir(Path(os.environ["PRJ_OUT"]), EXP_ID)
    runs.write_snapshot(d, {**cfg, "_args": vars(a), "_subjects": subs})
    log = runs.Log(d)
    files = [f for s in subs for f in io_d5.find_runs(root, s)]
    log(f"{EXP_ID}: {len(subs)} 名患者，{len(files)} 个 run；root={root}")
    procs = a.procs or cfg.get("procs", 8)
    summs, labs = [], []
    with ProcessPoolExecutor(max_workers=min(procs, max(1, len(files)))) as ex:
        futs = {ex.submit(process_run, f, cfg, root, out_dir): f for f in files}
        for fu in as_completed(futs):
            f = futs[fu]
            try:
                s, lab = fu.result()
                summs.append(s); labs.append(lab)
                log(f"OK  {s['run_id']}  sfreq={s['sfreq']:g}  ch={s['n_good_ch']}  pairs={s['n_pairs_used']}/{s['n_pairs_all']}"
                    f"  N1={s['n1_rate']:.3f}  {s['seconds']} s")
            except Exception as e:  # 逐 run 容错（S2R 约定）
                summs.append(dict(run_id=os.path.basename(f), error=repr(e)))
                log(f"ERR {os.path.basename(f)}: {e!r}\n{traceback.format_exc()}")
    runs_df = pd.DataFrame(summs).sort_values("run_id")
    runs_df.to_csv(d / "runs.tsv", sep="\t", index=False)
    lab = pd.concat(labs, ignore_index=True) if labs else pd.DataFrame()
    for s, g in lab.groupby("subject"):
        g.to_parquet(Path(out_dir) / f"{s}_labels.parquet", index=False)
    n_ok = int(runs_df.get("error", pd.Series([np.nan] * len(runs_df))).isna().sum())
    if len(lab):
        rep, met = report(runs_df, lab, cfg)
        (d / "report.md").write_text(rep, encoding="utf-8")
        met.to_csv(d / "metrics.csv", index=False)
        log(rep)
    log(f"SUMMARY exp={EXP_ID} ok={n_ok}/{len(files)} pairs={int(runs_df.get('n_pairs_used', pd.Series([0])).fillna(0).sum())} "
        f"eval={len(lab)}")
    if n_ok == len(files) and len(files) > 0:
        runs.mark_done(d)


if __name__ == "__main__":
    main()
