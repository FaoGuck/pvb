#!/usr/bin/env python
"""P202b：我们的 N1（P202）与 ds004080 作者发布的 derivatives/av_ccep/ccepData_V*.mat 逐项核对（P202 的验收）。

作者的结构（P201 打印、本地 loadmat 核对）：ccepData[患者].run[k] 有 runName、n1_peak_sample（通道 × 刺激对，
MATLAB 1 起的样本号，指向 tt；无 N1 = NaN）、channel_names、stimpair_names（按首次出现的方向命名）、tt（-2…+3 s）。
对齐：run 名 → 我们的 run_id；刺激对按无序对（pvb.t2.io_d5.pair_name）；通道名大写。
潜伏期 = tt[n1_peak_sample - 1]（秒）。只比较我们评估过的 刺激对 × 电极（good ECOG、非刺激电极）。

只读、本地运行（几秒）：
  $env:PRJ_CODE / PRJ_DATA / PRJ_OUT / PRJ_GIT / PRJ_RUN_ID 同 p203；python pvb\\scripts\\p202b_compare_avccep.py
输出：$PRJ_OUT/runs/P202b_avccep_compare_d5/<时间>_<git7>/  per_subject.tsv、mismatch_examples.tsv、report.md、DONE
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.io import loadmat

CODE = Path(os.environ["PRJ_CODE"])
sys.path.insert(0, str(CODE))
from pvb.probing import runs  # noqa: E402
from pvb.t2 import io_d5  # noqa: E402

EXP_ID = "P202b_avccep_compare_d5"


def author_table(mat_path: Path, subjects: list[str]) -> pd.DataFrame:
    m = loadmat(mat_path, squeeze_me=True, struct_as_record=False)["ccepData"]
    rows = []
    for s in np.atleast_1d(m):
        if s.id not in subjects:
            continue
        for r in np.atleast_1d(s.run):
            mm = re.match(r"(sub-[^_]+)_(ses-[^_]+)_task-[^_]+_run-([^_]+)_", r.runName)
            run_id = f"{mm.group(1)}_{mm.group(2)}_run-{mm.group(3)}"
            n1 = np.atleast_2d(np.asarray(r.n1_peak_sample, float))
            chans = [str(c).upper() for c in np.atleast_1d(r.channel_names)]
            pairs = [io_d5.pair_name(p) for p in np.atleast_1d(r.stimpair_names)]
            tt = np.asarray(r.tt, float)
            for j, p in enumerate(pairs):
                for i, c in enumerate(chans):
                    v = n1[i, j]
                    rows.append((run_id, s.id, p, c, np.isfinite(v),
                                 float(tt[int(v) - 1] * 1000) if np.isfinite(v) else np.nan))
    df = pd.DataFrame(rows, columns=["run_id", "subject", "stim_pair", "channel", "a_detected", "a_latency_ms"])
    # 同一 run 里同一无序对出现两次（两个方向分开存）时，任一方向有 N1 即算有，潜伏期取有的那个
    return (df.sort_values("a_detected", ascending=False)
              .drop_duplicates(["run_id", "stim_pair", "channel"], keep="first"))


def kappa(a, b) -> float:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = (a == b).mean(); pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def compare(lab: pd.DataFrame, au: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    j = lab.merge(au, on=["run_id", "subject", "stim_pair", "channel"], how="left", indicator=True)
    only_ours = int((j._merge == "left_only").sum())
    j = j[j._merge == "both"].copy()
    j["a_detected"] = j.a_detected.astype(bool)
    rows = []
    for s, g in j.groupby("subject"):
        o, a = g.n1_detected.to_numpy(bool), g.a_detected.to_numpy(bool)
        both = g[o & a]
        dlat = (both.n1_latency_ms - both.a_latency_ms).abs()
        sr = 1000.0 / 512                                   # 512 Hz 的 1 个样本（2048 Hz 更细）
        rows.append(dict(subject=s, n=len(g), ours_rate=o.mean(), author_rate=a.mean(), agree=(o == a).mean(),
                         kappa=kappa(o, a), sens_vs_author=(o & a).sum() / max(1, a.sum()),
                         prec_vs_author=(o & a).sum() / max(1, o.sum()), n_both=len(both),
                         lat_absdiff_med_ms=float(dlat.median()) if len(both) else np.nan,
                         lat_within_1smp512=float((dlat <= sr + 1e-6).mean()) if len(both) else np.nan))
    o, a = j.n1_detected.to_numpy(bool), j.a_detected.to_numpy(bool)
    tot = dict(n=len(j), only_ours=only_ours, agree=float((o == a).mean()), kappa=kappa(o, a),
               sens=float((o & a).sum() / max(1, a.sum())), prec=float((o & a).sum() / max(1, o.sum())))
    mis = j[o != a][["run_id", "stim_pair", "channel", "dist_mm", "n_trials", "n1_detected", "n1_amp_uv", "base_sd_uv",
                     "n1_latency_ms", "a_detected", "a_latency_ms", "sat_frac_n1win"]]
    return pd.DataFrame(rows), mis, tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/datasets/p_d5_ds004080.yaml")
    ap.add_argument("--subjects", default="")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(CODE / a.config, encoding="utf-8"))
    subs = [s for s in a.subjects.split(",") if s] or cfg["subjects_gate"]
    data = Path(os.environ["PRJ_DATA"])
    lab = pd.concat([pd.read_parquet(data / cfg["labels_dir"] / f"{s}_labels.parquet") for s in subs], ignore_index=True)
    d = runs.new_run_dir(Path(os.environ["PRJ_OUT"]), EXP_ID)
    runs.write_snapshot(d, {**cfg, "_args": vars(a), "_subjects": subs})
    log = runs.Log(d)
    rep = [f"# {EXP_ID}", ""]
    for ver in ["V1", "V2"]:
        au = author_table(data / cfg["raw_dir"] / "derivatives" / "av_ccep" / f"ccepData_{ver}.mat", subs)
        per, mis, tot = compare(lab, au)
        per.to_csv(d / f"per_subject_{ver}.tsv", sep="\t", index=False)
        mis.sort_values("dist_mm").head(300).to_csv(d / f"mismatch_examples_{ver}.tsv", sep="\t", index=False)
        lo = mis[mis.n1_detected & ~mis.a_detected]; hi = mis[~mis.n1_detected & mis.a_detected]
        rep += [f"## 对照 ccepData_{ver}.mat", "",
                f"- 对上的 刺激对 × 电极 {tot['n']} 个（我们有、作者没有的 {tot['only_ours']} 个）",
                f"- 一致率 {tot['agree']:.4f}，κ {tot['kappa']:.4f}；以作者为准：我们的敏感度 {tot['sens']:.4f}，精确度 {tot['prec']:.4f}",
                f"- 不一致 {len(mis)} 个：我们有、作者没有 {len(lo)}（我们的幅度中位 {lo.n1_amp_uv.median():.0f} µV）；"
                f"作者有、我们没有 {len(hi)}",
                "", per.round(4).to_string(index=False), ""]
    (d / "report.md").write_text("\n".join(rep) + "\n", encoding="utf-8")
    log("\n".join(rep))
    log(f"SUMMARY exp={EXP_ID} subjects={len(subs)}")
    runs.mark_done(d)


if __name__ == "__main__":
    main()
