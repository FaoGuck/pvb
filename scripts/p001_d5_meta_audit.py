"""P001 / V4c：ds004080（D5）元数据审计——只读 json / tsv，不读信号。

读 $PRJ_DATA/raw/ds004080/ 下每个 run 的 *_ieeg.json、*_events.tsv、*_channels.tsv 和每名患者的 *_electrodes.tsv，统计：
  - 采样率、记录时长、通道数（ECoG / SEEG）、坏导数
  - 刺激：刺激对（无序对，正负极性合并，与 mnl_ccepAge 一致）、每对脉冲数、电流、频率
  - 相邻脉冲间隔（同一 run 内按时间排序的全部脉冲，不论是否同一对），以及按 00 第 8 节 G0-T2 ① 的观测段长度
    观测段 = 间隔 − T_tail − 10 ms（T_tail 默认 1.0 s，P201 后再定），≥ 1 s 才算可用
  - 电极坐标完整性（V4e）
并按 00 第 8 节锁定的规则选出 G0-T2 的 5 名闸门患者：按被试编号顺序，取前 5 名刺激对数 ≥ 20 的患者。

本地运行（只读分析，几秒）：
  PowerShell： $env:PRJ_DATA="D:\\proj_LLM\\proj\\proj_PVB\\data"; $env:PRJ_OUT="D:\\proj_LLM\\proj\\proj_PVB\\out"; python scripts/p001_d5_meta_audit.py
输出：$PRJ_OUT/summary/p001_d5_runs.tsv（每 run 一行）、p001_d5_subjects.tsv（每名患者一行）、p001_d5_meta_audit.md（汇总）
"""
import glob
import json
import os
import re

import numpy as np
import pandas as pd

T_TAIL = 1.0      # s，P204 默认值
PRE_GAP = 0.010   # s，观测段在下一个脉冲前 10 ms 结束
MIN_OBS = 1.0     # s，观测段最短长度（P204）
MIN_PAIRS = 20    # G0-T2 闸门患者的刺激对数下限
N_GATE = 5

ROOT = os.path.join(os.environ["PRJ_DATA"], "raw", "ds004080")
OUT = os.path.join(os.environ["PRJ_OUT"], "summary")


def unordered(site):
    a = [s.strip() for s in str(site).split("-")]
    return "-".join(sorted(a)) if len(a) == 2 else str(site)


def audit_run(ev_path):
    base = ev_path[: -len("_events.tsv")]
    m = re.search(r"(sub-[^_]+)_(ses-[^_]+)_task-SPESclin_run-(\d+)", os.path.basename(base))
    sub, ses, run = m.group(1), m.group(2), m.group(3)
    info = json.load(open(base + "_ieeg.json", encoding="utf-8"))
    ch = pd.read_csv(base + "_channels.tsv", sep="\t")
    ev = pd.read_csv(base + "_events.tsv", sep="\t")
    st = ev[ev.trial_type == "electrical_stimulation"].copy()
    st["pair"] = st.electrical_stimulation_site.map(unordered)
    st = st.sort_values("onset")
    isi = np.diff(st.onset.values)
    obs = isi - T_TAIL - PRE_GAP
    per_pair = st.groupby("pair").size()
    # 同一刺激对内部相邻脉冲的间隔（刺激频率）
    within = st.groupby("electrical_stimulation_site").onset.apply(lambda x: np.diff(np.sort(x.values))).explode().astype(float)
    ieeg_types = ch[ch.type.isin(["ECOG", "SEEG"])]
    cur = pd.to_numeric(st.electrical_stimulation_current, errors="coerce")
    return dict(
        subject=sub, session=ses, run=run,
        day=int(run[:2]), clock=run[2:],
        sfreq=info.get("SamplingFrequency"), duration_s=info.get("RecordingDuration"),
        n_ecog=info.get("ECOGChannelCount", 0), n_seeg=info.get("SEEGChannelCount", 0),
        n_ieeg_ch=len(ieeg_types), n_bad=int((ieeg_types.status != "good").sum()),
        n_pulses=len(st), n_pairs=per_pair.size, n_sites_ordered=st.electrical_stimulation_site.nunique(),
        pulses_per_pair_min=int(per_pair.min()) if len(per_pair) else 0,
        pulses_per_pair_med=float(per_pair.median()) if len(per_pair) else 0,
        pulses_per_pair_max=int(per_pair.max()) if len(per_pair) else 0,
        isi_within_med=float(np.median(within)) if len(within) else np.nan,
        isi_within_min=float(np.min(within)) if len(within) else np.nan,
        isi_all_p05=float(np.percentile(isi, 5)) if len(isi) else np.nan,
        isi_all_med=float(np.median(isi)) if len(isi) else np.nan,
        obs_ok_frac=float(np.mean(obs >= MIN_OBS)) if len(obs) else np.nan,
        obs_total_s=float(np.clip(obs, 0, None)[obs >= MIN_OBS].sum()) if len(obs) else 0.0,
        current_mA=";".join(sorted({f"{c*1000:g}" for c in cur.dropna()})),
        freq_hz=";".join(sorted({str(f) for f in st.electrical_stimulation_frequency.dropna().unique()})),
        stim_type=";".join(sorted(st.electrical_stimulation_type.dropna().astype(str).unique())),
        n_artefact=int((ev.trial_type == "artefact").sum()),
    ), set(per_pair.index)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows, pairs_by_sub = [], {}
    for p in sorted(glob.glob(os.path.join(ROOT, "sub-*", "ses-*", "ieeg", "*_events.tsv"))):
        r, prs = audit_run(p)
        rows.append(r)
        pairs_by_sub.setdefault(r["subject"], set()).update(prs)
    runs = pd.DataFrame(rows)
    runs.to_csv(os.path.join(OUT, "p001_d5_runs.tsv"), sep="\t", index=False)

    subs = []
    for sub, g in runs.groupby("subject"):
        el = glob.glob(os.path.join(ROOT, sub, "ses-*", "ieeg", "*_electrodes.tsv"))
        n_el = n_nocoord = 0
        for e in el:
            t = pd.read_csv(e, sep="\t")
            xyz = t[["x", "y", "z"]].apply(pd.to_numeric, errors="coerce")
            n_el += len(t); n_nocoord += int(xyz.isna().any(axis=1).sum())
        subs.append(dict(
            subject=sub, n_runs=len(g), sfreq=";".join(sorted({f"{s:g}" for s in g.sfreq})),
            n_pairs=len(pairs_by_sub[sub]), n_pulses=int(g.n_pulses.sum()),
            obs_ok_frac=float(np.average(g.obs_ok_frac.fillna(0), weights=g.n_pulses.clip(lower=1))),
            n_electrodes=n_el, n_no_coord=n_nocoord, n_ecog=int(g.n_ecog.max()), n_seeg=int(g.n_seeg.max()),
        ))
    subs = pd.DataFrame(subs)
    subs["num"] = subs.subject.str.extract(r"(\d+)$").astype(int)
    subs = subs.sort_values("num").drop(columns="num")
    gate = subs[subs.n_pairs >= MIN_PAIRS].head(N_GATE).subject.tolist()
    subs["gate"] = subs.subject.isin(gate)
    subs.to_csv(os.path.join(OUT, "p001_d5_subjects.tsv"), sep="\t", index=False)

    q = lambda s: f"{s.min():g} / {s.median():g} / {s.max():g}"
    with_stim = runs[runs.n_pulses > 0]
    md = [
        "# P001 / V4c：ds004080 元数据审计（`scripts/p001_d5_meta_audit.py`，只读 json/tsv）", "",
        f"- 患者 {subs.shape[0]} 名，run {runs.shape[0]} 个（有刺激的 {with_stim.shape[0]} 个）",
        f"- 采样率（Hz）各取值的 run 数：{runs.sfreq.value_counts().sort_index().to_dict()}",
        f"- 记录时长 s（最小 / 中位 / 最大）：{q(runs.duration_s)}",
        f"- 通道：ECoG 患者 {(subs.n_ecog > 0).sum()}，SEEG 患者 {(subs.n_seeg > 0).sum()}；每 run iEEG 通道数 {q(runs.n_ieeg_ch)}；坏导 {q(runs.n_bad)}",
        f"- 每名患者刺激对数（无序对）：{q(subs.n_pairs)}；≥ {MIN_PAIRS} 对的患者 {(subs.n_pairs >= MIN_PAIRS).sum()} 名",
        f"- 每对脉冲数（每 run 的中位数，跨 run 最小 / 中位 / 最大）：{q(with_stim.pulses_per_pair_med)}；单对最少 {with_stim.pulses_per_pair_min.min()}、最多 {with_stim.pulses_per_pair_max.max()}",
        f"- 同一刺激位点内相邻脉冲间隔 s（每 run 中位数，最小 / 中位 / 最大）：{q(with_stim.isi_within_med)}；全体最小 {with_stim.isi_within_min.min():.3f}",
        f"- 相邻脉冲间隔（不论是否同一对）的 5% 分位数 s（跨 run 最小 / 中位 / 最大）：{q(with_stim.isi_all_p05)}",
        f"- 刺激电流 mA：{sorted(set(';'.join(runs.current_mA).split(';')) - {''})}；频率 Hz：{sorted(set(';'.join(runs.freq_hz).split(';')) - {''})}；类型：{sorted(set(';'.join(runs.stim_type).split(';')) - {''})}",
        f"- 电极坐标：共 {subs.n_electrodes.sum()} 个电极，缺坐标 {subs.n_no_coord.sum()} 个；有缺坐标的患者 {(subs.n_no_coord > 0).sum()} 名",
        "",
        f"## G0-T2 ①（T_tail = {T_TAIL} s：观测段 = 间隔 − T_tail − {PRE_GAP*1000:g} ms ≥ {MIN_OBS} s 的间隔占比 ≥ 80%）",
        f"- 每名患者的占比（最小 / 中位 / 最大）：{q(subs.obs_ok_frac)}；占比 ≥ 0.8 的患者 {(subs.obs_ok_frac >= 0.8).sum()} / {subs.shape[0]}",
        "",
        f"## G0-T2 闸门患者（按编号顺序，前 {N_GATE} 名刺激对数 ≥ {MIN_PAIRS}）",
        subs[subs.gate].to_markdown(index=False) if hasattr(subs, "to_markdown") else subs[subs.gate].to_string(index=False),
    ]
    open(os.path.join(OUT, "p001_d5_meta_audit.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
