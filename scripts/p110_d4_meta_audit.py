"""P110 第 2 步 / V5d：ds004024（D4）元数据审计——只读 json / tsv，不读信号。

读 $PRJ_DATA/raw/ds004024/ 下每个 run 的 *_eeg.json、*_events.tsv、*_channels.tsv，统计：
  - 每个 spTMS run：采样率、记录时长、通道数、EMG 两导是否在且 good、坏 EEG 导数
  - 脉冲：events.tsv 里 trial_type = "Stimulus/A" 或 "SB/A"（value 都是 3；CON008 的 30 个 run 全用 SB/A）的时刻
    （"Out/A" 紧跟其后 1 个样本，另计数核对；全体 Out/A 27 753 = Stimulus/A 25 353 + SB/A 2 400）；
    每 run 脉冲数（期望 80）、相邻间隔（期望 5.0 s）、第一个脉冲前有多少秒、最后一个脉冲后到记录结束多少秒
  - 刺激前窗口：与上一个脉冲相隔 ≥ PRE_WIN + PRE_GAP_PREV 的脉冲才算“窗口可取”（每 run 第一个脉冲看到记录起点的距离）
  - 强度：按 dataset_description.json，run 01–06 = 100% rMT、07–12 = 110% rMT
  - rest、ccPAS run 只计数和时长
  ⚠ 元数据里**没有刺激侧（左 / 右 M1）和时间点（ccPAS 前 / 后 10 / 后 60 min）**：events 只有脉冲标记，scans.tsv 的 acq_time 全是 n/a。
    刺激侧要在第 3 步从 EMG 判断（左 M1 → 右侧 FDI 有 MEP），时间点只能按 run 编号假设，这里不做推断
本地运行（只读分析，几秒），Git Bash：
  PRJ_DATA=/d/proj_LLM/proj/proj_PVB/data PRJ_OUT=/d/proj_LLM/proj/proj_PVB/out python -X utf8 scripts/p110_d4_meta_audit.py
输出：$PRJ_OUT/summary/p110_d4_runs.tsv（每个 run 一行）、p110_d4_subjects.tsv（每人一行）、p110_d4_meta_audit.md（汇总）
"""
import fnmatch
import glob
import json
import os
import re

import numpy as np
import pandas as pd

PRE_WIN = 4.0        # s，最长的刺激前窗口（与 D1 的 W4 同长，S2R 08 第 2 节）
PRE_GAP_PREV = 0.5   # s，窗口起点离上一个脉冲至少这么远（避开上一个脉冲的伪迹和 MEP）
N_EXPECT = 80        # 每 run 期望脉冲数（S2R 08 第 2 节）
ISI_EXPECT = 5.0     # s
MIN_TRIALS = 200     # G0-T1 ②：每人可用试验数下限（锁定）
MIN_SUBJ = 10        # G0-T1 ②：满足下限的被试数（锁定）
META_PATS = "*.json *.tsv *.vhdr *.vmrk CHANGES README*".split()
PULSE_TYPES = {"Stimulus/A", "SB/A"}
KNOWN_TYPES = PULSE_TYPES | {"Out/A", "New Segment/"}

ROOT = os.path.join(os.environ["PRJ_DATA"], "raw", "ds004024")
MAN = os.path.join(os.environ["PRJ_DATA"], "raw", "_manifest", "ds004024.keys.tsv")
OUT = os.path.join(os.environ["PRJ_OUT"], "summary")

RX = re.compile(r"(sub-[^_]+)_ses-([^_]+)_task-([^_]+)_run-(\d+)")


def read_json(p):
    try:
        with open(p, encoding="utf-8-sig") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def audit_run(ev_path):
    base = ev_path[: -len("_events.tsv")]
    sub, ses, task, run = RX.search(os.path.basename(base)).groups()
    js = read_json(base + "_eeg.json")
    r = dict(subject=sub, session=ses, task=task, run=int(run),
             sfreq=js.get("SamplingFrequency", np.nan), duration_s=js.get("RecordingDuration", np.nan),
             n_eeg=js.get("EEGChannelCount", np.nan), n_emg=js.get("EMGChannelCount", np.nan),
             json_ok=bool(js))
    ch_path = base + "_channels.tsv"
    if os.path.isfile(ch_path):
        ch = pd.read_csv(ch_path, sep="\t", encoding="utf-8-sig")
        st = dict(zip(ch["name"], ch["status"]))
        r.update(emg_right=st.get("EMG Right", "缺"), emg_left=st.get("EMG Left", "缺"),
                 n_bad_eeg=int(((ch["type"] == "EEG") & (ch["status"] != "good")).sum()), channels_ok=True)
    else:
        r.update(emg_right="无文件", emg_left="无文件", n_bad_eeg=np.nan, channels_ok=False)
    ev = pd.read_csv(ev_path, sep="\t", encoding="utf-8-sig")
    tt = ev["trial_type"].astype(str)
    r["other_types"] = ";".join(sorted(set(tt) - KNOWN_TYPES))
    r["pulse_marker"] = ";".join(sorted(set(tt) & PULSE_TYPES))
    if task != "spTMS":
        r.update(n_pulse=int(tt.isin(PULSE_TYPES).sum()))
        return r
    on = np.sort(ev.loc[tt.isin(PULSE_TYPES), "onset"].to_numpy(float))
    r.update(n_pulse=len(on), n_out=int((tt == "Out/A").sum()), intensity_rmt=100 if int(run) <= 6 else 110)
    if len(on):
        isi = np.diff(on)
        prev = np.concatenate([[on[0]], isi])            # 第一个脉冲：到记录起点的距离
        r.update(first_onset_s=on[0], tail_after_last_s=r["duration_s"] - on[-1] if r["duration_s"] == r["duration_s"] else np.nan,
                 isi_med=float(np.median(isi)) if len(isi) else np.nan,
                 isi_min=float(isi.min()) if len(isi) else np.nan, isi_max=float(isi.max()) if len(isi) else np.nan,
                 n_pause=int((isi > ISI_EXPECT + 0.05).sum()),     # run 中途暂停（不影响窗口）
                 n_short=int((isi < ISI_EXPECT - 0.05).sum()),     # 间隔不足 5 s（后一个脉冲的窗口可能不干净）
                 n_prewin_ok=int((prev >= PRE_WIN + PRE_GAP_PREV).sum()))
    return r


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = [audit_run(p) for p in sorted(glob.glob(os.path.join(ROOT, "sub-*", "ses-*", "eeg", "*_events.tsv")))]
    runs = pd.DataFrame(rows).sort_values(["subject", "session", "task", "run"])
    runs.to_csv(os.path.join(OUT, "p110_d4_runs.tsv"), sep="\t", index=False)
    sp = runs[runs.task == "spTMS"]

    # 本地缺失的元数据文件（与清单比）
    k = pd.read_csv(MAN, sep="\t", header=None, names=["key", "bytes"])
    want = k[k.key.apply(lambda s: any(fnmatch.fnmatch(s, p) for p in META_PATS))]
    missing = [x for x in want.key if not os.path.isfile(os.path.join(ROOT, x))]
    eeg_keys = k[k.key.str.contains("task-spTMS") & k.key.str.endswith("_eeg.eeg")].copy()
    eeg_keys["gib"] = eeg_keys.bytes / 2**30

    g = sp.groupby("subject")
    subs = pd.DataFrame({
        "sessions": g.session.nunique(),
        "runs_100": g.apply(lambda d: int((d.intensity_rmt == 100).sum())),
        "runs_110": g.apply(lambda d: int((d.intensity_rmt == 110).sum())),
        "pulses_100": g.apply(lambda d: int(d.loc[d.intensity_rmt == 100, "n_pulse"].sum())),
        "pulses_110": g.apply(lambda d: int(d.loc[d.intensity_rmt == 110, "n_pulse"].sum())),
        "prewin_ok_100": g.apply(lambda d: int(d.loc[d.intensity_rmt == 100, "n_prewin_ok"].sum())),
        "runs_not_80": g.apply(lambda d: int((d.n_pulse != N_EXPECT).sum())),
    })
    # 刺激侧未知；若左右各半（数据集说明：每个时间点左右各一个 run），一侧的上限 = 一半
    subs["one_side_100_upper"] = subs.prewin_ok_100 // 2
    subs["one_side_all_upper"] = (subs.prewin_ok_100 + g.apply(lambda d: int(d.loc[d.intensity_rmt == 110, "n_prewin_ok"].sum()))) // 2
    subs.to_csv(os.path.join(OUT, "p110_d4_subjects.tsv"), sep="\t")

    odd = sp[(sp.n_pulse != N_EXPECT) | (sp.n_short > 0) | (sp.n_out != sp.n_pulse) | ~sp.json_ok | ~sp.channels_ok
             | (sp.emg_right != "good") | (sp.emg_left != "good")]
    small = eeg_keys[eeg_keys.gib < 1.5]
    other = runs[runs.task != "spTMS"].groupby("task").agg(runs=("run", "size"), dur_med_s=("duration_s", "median"))
    L = ["# P110 第 2 步：D4（ds004024）元数据审计（只读 json / tsv，不读信号）", "",
         f"- spTMS run：{len(sp)} 个（清单 348）；被试 {sp.subject.nunique()} 人；会话 {sp.groupby(['subject', 'session']).ngroups} 个",
         f"- 采样率：{sorted(sp.sfreq.dropna().unique().tolist())} Hz；EEG 导数 {sorted(sp.n_eeg.dropna().unique().tolist())}、EMG {sorted(sp.n_emg.dropna().unique().tolist())}",
         f"- 每 run 脉冲数分布：{sp.n_pulse.value_counts().sort_index().to_dict()}；Out/A 与 Stimulus/A 数不同的 run：{int((sp.n_out != sp.n_pulse).sum())}",
         f"- 脉冲标记：{sp.pulse_marker.value_counts().to_dict()}；其他标记（不影响）：{sp.other_types[sp.other_types != ''].value_counts().to_dict()}",
         f"- 相邻间隔：中位 {sp.isi_med.median():.4f} s，全体最小 {sp.isi_min.min():.4f}、最大 {sp.isi_max.max():.4f}；"
         f"有中途暂停（> 5.05 s）的 run {int((sp.n_pause > 0).sum())} 个（共 {int(sp.n_pause.sum())} 次）；有间隔 < 4.95 s 的 run {int((sp.n_short > 0).sum())} 个（共 {int(sp.n_short.sum())} 次）",
         f"- 第一个脉冲前：中位 {sp.first_onset_s.median():.2f} s（最小 {sp.first_onset_s.min():.2f}）；最后一个脉冲后到记录结束：中位 {sp.tail_after_last_s.median():.2f} s（最小 {sp.tail_after_last_s.min():.2f}）",
         f"- 刺激前 {PRE_WIN:.0f} s 窗口可取（离上一个脉冲 ≥ {PRE_WIN + PRE_GAP_PREV:.1f} s）的脉冲：{int(sp.n_prewin_ok.sum())} / {int(sp.n_pulse.sum())}",
         f"- 坏 EEG 导：每 run 中位 {sp.n_bad_eeg.median():.0f}，最多 {sp.n_bad_eeg.max():.0f}；EMG 两导不是 good 的 run：{int(((sp.emg_right != 'good') | (sp.emg_left != 'good')).sum())}",
         f"- 信号文件 < 1.5 GiB 的 spTMS run（疑似短 run）：{len(small)} 个 " + "；".join(f"{a.split('/')[-1]} {b:.2f} GiB" for a, b in zip(small.key, small.gib)),
         f"- 其他任务：{other.round(1).to_dict('index')}",
         f"- 本地缺失的元数据文件：{len(missing)} 个" + ("（" + "；".join(os.path.basename(x) for x in missing) + "）" if missing else ""),
         "- ⚠ 元数据里**没有刺激侧和时间点**（events 只有脉冲标记，scans.tsv 的 acq_time 全是 n/a）→ 刺激侧在第 3 步从 EMG 判断",
         "", "## 每人（脉冲数 = 元数据里的上限，还没做 MEP 质控）", "",
         subs.to_markdown() if hasattr(subs, "to_markdown") else subs.to_string(), "",
         f"- 一侧、只用 100% rMT 的上限 ≥ {MIN_TRIALS} 的被试：{int((subs.one_side_100_upper >= MIN_TRIALS).sum())} / {len(subs)}；"
         f"一侧、100% + 110% 的上限 ≥ {MIN_TRIALS}：{int((subs.one_side_all_upper >= MIN_TRIALS).sum())} / {len(subs)}（G0-T1 ② 要求 ≥ {MIN_SUBJ} 人，判定要等第 3 步的 MEP 质控）",
         "", "## 需要留意的 spTMS run（脉冲数 ≠ 80、有 < 4.95 s 的间隔、标记数对不上、元数据文件缺失、EMG 不是 good）", "",
         odd[["subject", "session", "run", "n_pulse", "n_out", "n_prewin_ok", "isi_min", "n_short", "duration_s", "json_ok", "channels_ok", "emg_right", "emg_left"]].to_string(index=False) if len(odd) else "无"]
    txt = "\n".join(L) + "\n"
    with open(os.path.join(OUT, "p110_d4_meta_audit.md"), "w", encoding="utf-8") as f:
        f.write(txt)
    print(txt)
    print(f"SUMMARY exp=P110_meta runs={len(sp)} subjects={sp.subject.nunique()} missing_meta={len(missing)} odd_runs={len(odd)}")


if __name__ == "__main__":
    main()
