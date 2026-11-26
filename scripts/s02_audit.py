#!/usr/bin/env python
# 来自 S2R 505b298 的 scripts/s02_audit.py，2026-11-26 复制（机械改名：S2R_→PRJ_、包名 s2r→项目名）
"""S2R S2 数据审计（D1 = OpenNeuro ds002094）。

只读原始数据，不做任何处理，回答总纲 6 节 S2 的问题：
  1. 被试、run、任务（rest / tmseeg1 / tmseeg2）分别有哪些
  2. 采样率、EEG 通道名、肌电通道名、参考电极
  3. 每个记录的脉冲数、刺激间隔分布
  4. 事件标记含义；脉冲标记与 TMS 伪迹是否对齐（标记是否精确）
  5. block 边界（按异常长间隔切分）
另加：肌电是否存在、MEP 粗算（脉冲后 20-40 ms 峰峰值）、与总纲预期逐条对比。

用法: python scripts/s02_audit.py [<raw_root> <out_dir>]
不给参数时用 $PRJ_DATA/raw 和 $PRJ_DATA/audit/d1（总纲 3.1：路径只从环境变量来）
"""
import os
import sys
import re
import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mne  # noqa: E402

mne.set_log_level("ERROR")

# ---------------- 参数（与总纲一致） ----------------
PULSE_MIN_COUNT = 20          # 一个标记出现次数 >= 这个值才可能是脉冲
PULSE_ISI_RANGE = (1.0, 20.0)  # 候选脉冲标记的间隔中位数范围（秒）
W4 = 4.01                     # 刺激前窗口长度（秒），截止于脉冲前 10 ms
GUARD = 0.9                   # 窗口起点须晚于上一脉冲 + 0.9 s（总纲 6.3 第 2 条）
MEP_WIN = (0.020, 0.040)      # MEP 峰峰值窗口（秒）
PRE_EMG_WIN = (-0.105, -0.005)  # 预激活检查窗口（秒）
PRE_EMG_THR = 50.0            # µV
ALIGN_WIN = (-0.010, 0.010)   # 在标记附近找 TMS 伪迹的窗口（秒）
EP_TMIN, EP_TMAX = -0.12, 0.10
EMG_NAME = re.compile(r"(emg|fdi|apb|adm|exg|mep|ecr|fcr|aux)", re.I)
ENT = re.compile(r"(sub|ses|task|acq|run)-([A-Za-z0-9]+)")

if len(sys.argv) >= 3:
    RAW, OUT = Path(sys.argv[1]), Path(sys.argv[2])
else:
    RAW = Path(os.environ["PRJ_DATA"]) / "raw"
    OUT = Path(os.environ["PRJ_DATA"]) / "audit" / "d1"
OUT.mkdir(parents=True, exist_ok=True)
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)
TOP = OUT / "toplevel"
TOP.mkdir(exist_ok=True)
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def find_root(p):
    if (p / "dataset_description.json").exists():
        return p
    for pat in ("*/dataset_description.json", "*/*/dataset_description.json"):
        hits = sorted(p.glob(pat))
        if hits:
            return hits[0].parent
    return p


def entities(name):
    return dict(ENT.findall(name))


def md_table(df, floatfmt="{:.3g}"):
    if df is None or len(df) == 0:
        return "（无）\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append("" if np.isnan(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def q(x, p):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, p)) if x.size else float("nan")


ROOT = find_root(RAW)
log("RAW =", RAW, " ROOT =", ROOT)
errors = []

# ---------------- 1. 文件清单 ----------------
rows = []
for f in sorted(ROOT.rglob("*")):
    if not f.is_file():
        continue
    rel = f.relative_to(ROOT).as_posix()
    e = entities(f.name)
    suf = "".join(f.suffixes[-2:]) if f.suffix == ".gz" else f.suffix
    rows.append(dict(path=rel, sub=e.get("sub", ""), ses=e.get("ses", ""),
                     task=e.get("task", ""), run=e.get("run", ""),
                     ext=suf, size=f.stat().st_size))
inv = pd.DataFrame(rows)
inv.to_csv(OUT / "files_inventory.tsv", sep="\t", index=False)
log(f"files: {len(inv)}  total {inv['size'].sum() / 1e9:.2f} GB")

# 顶层小文本文件原样复制
for f in sorted(ROOT.iterdir()):
    if f.is_file() and f.stat().st_size < 2_000_000:
        try:
            (TOP / f.name).write_bytes(f.read_bytes())
        except Exception as ex:  # noqa: BLE001
            errors.append(f"copy {f.name}: {ex}")

# eeg sidecar json 展平成表
side_rows = []
for f in sorted(ROOT.rglob("*_eeg.json")):
    try:
        d = json.loads(f.read_text(encoding="utf-8", errors="replace"))
        r = {"path": f.relative_to(ROOT).as_posix()}
        for k, v in d.items():
            r[k] = json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
        side_rows.append(r)
    except Exception as ex:  # noqa: BLE001
        errors.append(f"json {f}: {ex}")
side = pd.DataFrame(side_rows)
side.to_csv(OUT / "eeg_sidecars.tsv", sep="\t", index=False)

# ---------------- 2. 逐个记录 ----------------
rec_rows, ann_rows, chan_rows, trial_rows = [], [], [], []
waves = []  # (label, times_ms, mean_emg_uv, align_trace)

vhdrs = sorted(ROOT.rglob("*.vhdr"))
log(f"vhdr files: {len(vhdrs)}")

for i, vh in enumerate(vhdrs):
    rel = vh.relative_to(ROOT).as_posix()
    e = entities(vh.name)
    rec = dict(path=rel, sub=e.get("sub", ""), ses=e.get("ses", ""),
               task=e.get("task", ""), run=e.get("run", ""), error="")
    label = f"{rec['sub']}|{rec['task']}" + (f"|r{rec['run']}" if rec["run"] else "")
    try:
        raw = mne.io.read_raw_brainvision(vh, preload=False)
        sf = float(raw.info["sfreq"])
        types = raw.get_channel_types()
        names = raw.ch_names
        rec.update(sfreq=sf, n_ch=len(names), duration_s=raw.n_times / sf,
                   highpass=raw.info["highpass"], lowpass=raw.info["lowpass"])

        # channels.tsv（若有）给出的类型优先
        ctsv = vh.with_name(vh.name.replace("_eeg.vhdr", "_channels.tsv"))
        bids_types = {}
        if ctsv.exists():
            ct = pd.read_csv(ctsv, sep="\t")
            if "name" in ct and "type" in ct:
                bids_types = dict(zip(ct["name"].astype(str), ct["type"].astype(str)))
        emg = [n for n, t in zip(names, types)
               if t == "emg" or bids_types.get(n, "").upper() == "EMG" or EMG_NAME.search(n)]
        eeg = [n for n, t in zip(names, types) if n not in emg and t == "eeg"]
        rec.update(n_eeg=len(eeg), n_emg=len(emg), emg_channels=";".join(emg),
                   has_channels_tsv=ctsv.exists())
        orig_units = getattr(raw, "_orig_units", {}) or {}
        for n, t in zip(names, types):
            chan_rows.append(dict(path=rel, sub=rec["sub"], task=rec["task"], name=n,
                                  mne_type=t, bids_type=bids_types.get(n, ""),
                                  unit=orig_units.get(n, ""), role="emg" if n in emg else
                                  ("eeg" if n in eeg else "other")))

        # 标记统计
        ann = raw.annotations
        desc = np.asarray(ann.description)
        onset = np.asarray(ann.onset, float)
        stats = []
        for d in sorted(set(desc)):
            o = np.sort(onset[desc == d])
            isi = np.diff(o)
            med = float(np.median(isi)) if isi.size else float("nan")
            stats.append((d, len(o), med))
            ann_rows.append(dict(path=rel, sub=rec["sub"], task=rec["task"],
                                 description=d, count=len(o), isi_median=med,
                                 first_onset=float(o[0]) if len(o) else np.nan))
        cands = [s for s in stats if s[1] >= PULSE_MIN_COUNT
                 and PULSE_ISI_RANGE[0] <= s[2] <= PULSE_ISI_RANGE[1]]
        pulse_desc = max(cands, key=lambda s: s[1])[0] if cands else ""
        rec.update(n_annot=len(desc), n_desc=len(stats), pulse_desc=pulse_desc,
                   n_pulse_candidates=len(cands))

        # BIDS events.tsv 对照
        etsv = vh.with_name(vh.name.replace("_eeg.vhdr", "_events.tsv"))
        if etsv.exists():
            et = pd.read_csv(etsv, sep="\t")
            rec["events_tsv_rows"] = len(et)
            for col in ("trial_type", "value"):
                if col in et:
                    rec[f"events_tsv_{col}"] = ";".join(
                        f"{k}:{v}" for k, v in et[col].astype(str).value_counts().head(8).items())

        if not pulse_desc:
            rec_rows.append(rec)
            log(f"[{i + 1}/{len(vhdrs)}] {label}: no pulse marker; descs={[(s[0], s[1]) for s in stats][:6]}")
            continue

        ev, _ = mne.events_from_annotations(raw, event_id={pulse_desc: 1})
        ev = ev[np.argsort(ev[:, 0])]
        t_p = (ev[:, 0] - raw.first_samp) / sf
        isi = np.diff(t_p)
        gap_thr = max(3 * np.median(isi), 15.0)
        block = np.concatenate([[0], np.cumsum(isi > gap_thr)])
        bsizes = np.bincount(block)
        isi_prev = np.concatenate([[t_p[0]], isi])  # 首个脉冲用距记录起点的时间
        # 块内第一个试验：上一次脉冲来自上一个 block，同样适用防泄漏规则
        keep_isi = isi_prev >= (W4 + GUARD)
        rec.update(n_pulses=len(t_p), isi_median=float(np.median(isi)),
                   isi_min=float(isi.min()), isi_p5=q(isi, 5), isi_p95=q(isi, 95),
                   isi_max=float(isi.max()), isi_cv=float(np.std(isi[isi <= gap_thr]) /
                                                          np.mean(isi[isi <= gap_thr])),
                   n_blocks=len(bsizes), block_sizes=";".join(map(str, bsizes)),
                   n_keep_isi_rule=int(keep_isi.sum()))

        # 截片：所有通道，脉冲前 120 ms 到后 100 ms
        ep = mne.Epochs(raw, ev, event_id=1, tmin=EP_TMIN, tmax=EP_TMAX, baseline=None,
                        preload=True, reject_by_annotation=False, picks="all",
                        proj=False, event_repeated="drop")
        sel = ep.selection
        X = ep.get_data(picks="all") * 1e6  # µV（MNE 内部单位 V）
        tt = ep.times
        ch = ep.ch_names
        rec["n_epochs"] = len(sel)

        # 标记对齐：EEG 一阶差分绝对值的通道平均，在标记 ±10 ms 内找峰
        ie = [ch.index(n) for n in eeg if n in ch]
        aw = (tt >= ALIGN_WIN[0]) & (tt <= ALIGN_WIN[1])
        align_trace = None
        if ie:
            dd = np.abs(np.diff(X[:, ie, :], axis=2)).mean(axis=1)  # (n, t-1)
            td = tt[1:]
            awd = (td >= ALIGN_WIN[0]) & (td <= ALIGN_WIN[1])
            off_ms = td[awd][np.argmax(dd[:, awd], axis=1)] * 1e3
            base = dd[:, td < -0.02].mean(axis=1)
            peak = dd[:, awd].max(axis=1)
            rec.update(align_offset_ms_median=float(np.median(off_ms)),
                       align_offset_ms_iqr=q(off_ms, 75) - q(off_ms, 25),
                       align_frac_within_1ms=float(np.mean(np.abs(off_ms - np.median(off_ms)) <= 1.0)),
                       artifact_to_baseline_ratio=float(np.median(peak / np.maximum(base, 1e-12))))
            m = dd.mean(axis=0)
            align_trace = m / (np.median(m[td < -0.02]) + 1e-12)
        del aw

        # MEP 粗算
        mw = (tt >= MEP_WIN[0]) & (tt <= MEP_WIN[1])
        pw = (tt >= PRE_EMG_WIN[0]) & (tt <= PRE_EMG_WIN[1])
        emg_mean = None
        for k, n in enumerate([n for n in emg if n in ch]):
            x = X[:, ch.index(n), :]
            mep = np.ptp(x[:, mw], axis=1)
            pre = np.ptp(x[:, pw], axis=1)
            if k == 0:
                emg_mean = x.mean(axis=0)
                rec.update(mep_ch=n, mep_median_uv=float(np.median(mep)),
                           mep_p10_uv=q(mep, 10), mep_p90_uv=q(mep, 90),
                           mep_frac_gt50uv=float(np.mean(mep > 50)),
                           mep_logsd=float(np.std(np.log(np.maximum(mep, 1e-3)))),
                           pre_emg_median_uv=float(np.median(pre)),
                           pre_emg_frac_gt50uv=float(np.mean(pre > PRE_EMG_THR)))
            for j, s in enumerate(sel):
                trial_rows.append(dict(sub=rec["sub"], task=rec["task"], run=rec["run"],
                                       emg_ch=n, trial=int(s), onset_s=float(t_p[s]),
                                       isi_prev_s=float(isi_prev[s]), block=int(block[s]),
                                       keep_isi_rule=bool(keep_isi[s]),
                                       mep_pp_uv=float(mep[j]), pre_pp_uv=float(pre[j])))
        if not emg:
            for s in sel:
                trial_rows.append(dict(sub=rec["sub"], task=rec["task"], run=rec["run"],
                                       emg_ch="", trial=int(s), onset_s=float(t_p[s]),
                                       isi_prev_s=float(isi_prev[s]), block=int(block[s]),
                                       keep_isi_rule=bool(keep_isi[s]),
                                       mep_pp_uv=np.nan, pre_pp_uv=np.nan))
        waves.append(dict(label=label, t_ms=tt * 1e3, emg=emg_mean,
                          td_ms=tt[1:] * 1e3, align=align_trace, isi=isi, gap=gap_thr))
        del X, ep
        log(f"[{i + 1}/{len(vhdrs)}] {label}: sf={sf:g} ch={len(names)} emg={emg} "
            f"pulses={len(t_p)} blocks={rec['block_sizes']} isi_med={rec['isi_median']:.2f}s "
            f"mep_med={rec.get('mep_median_uv', float('nan')):.0f}uV "
            f"align={rec.get('align_offset_ms_median', float('nan')):.2f}ms")
    except Exception as ex:  # noqa: BLE001
        rec["error"] = f"{type(ex).__name__}: {ex}"
        errors.append(f"{rel}: {rec['error']}\n{traceback.format_exc()}")
        log(f"[{i + 1}/{len(vhdrs)}] {label}: ERROR {rec['error']}")
    rec_rows.append(rec)

rec = pd.DataFrame(rec_rows)
ann = pd.DataFrame(ann_rows)
chans = pd.DataFrame(chan_rows)
trials = pd.DataFrame(trial_rows)
rec.to_csv(OUT / "recordings.tsv", sep="\t", index=False)
ann.to_csv(OUT / "annotations.tsv", sep="\t", index=False)
chans.to_csv(OUT / "channels_all.tsv", sep="\t", index=False)
trials.to_csv(OUT / "trials_preview.tsv", sep="\t", index=False)

# 通道集合去重
chan_sets = pd.DataFrame()
if len(chans):
    cs = chans.groupby("path")["name"].apply(lambda s: ",".join(s))
    chan_sets = cs.value_counts().rename_axis("channel_set").reset_index(name="n_recordings")
    chan_sets["n_ch"] = chan_sets["channel_set"].str.count(",") + 1
    chan_sets.to_csv(OUT / "channel_sets.tsv", sep="\t", index=False)

# ---------------- 3. 图 ----------------
def grid(n, w=2.6, h=1.9, ncol=8):
    ncol = min(ncol, max(n, 1))
    nrow = int(np.ceil(n / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(w * ncol, h * nrow), squeeze=False)
    for a in axs.ravel()[n:]:
        a.axis("off")
    return fig, axs.ravel()


wv = waves[:120]
if wv:
    fig, axs = grid(len(wv))
    for a, w in zip(axs, wv):
        a.hist(np.clip(w["isi"], 0, 15), bins=60, color="0.4")
        a.axvline(W4 + GUARD, color="r", lw=0.8)
        a.set_title(w["label"], fontsize=7)
        a.tick_params(labelsize=6)
    fig.suptitle(f"ISI histogram (clipped at 15 s); red = {W4 + GUARD:.2f} s rule", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "isi_hist.png", dpi=110)
    plt.close(fig)

    fig, axs = grid(len(wv))
    for a, w in zip(axs, wv):
        if w["emg"] is not None:
            m = (w["t_ms"] >= -20) & (w["t_ms"] <= 80)
            a.plot(w["t_ms"][m], w["emg"][m], lw=0.7, color="k")
            a.axvspan(MEP_WIN[0] * 1e3, MEP_WIN[1] * 1e3, color="orange", alpha=0.25)
        else:
            a.text(0.5, 0.5, "no EMG", ha="center", transform=a.transAxes)
        a.set_title(w["label"], fontsize=7)
        a.tick_params(labelsize=6)
    fig.suptitle("Mean EMG around pulse (µV, ms); orange = MEP window", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "mep_waveforms.png", dpi=110)
    plt.close(fig)

    fig, axs = grid(len(wv))
    for a, w in zip(axs, wv):
        if w["align"] is not None:
            m = (w["td_ms"] >= -15) & (w["td_ms"] <= 25)
            a.semilogy(w["td_ms"][m], w["align"][m], lw=0.7)
            a.axvline(0, color="r", lw=0.6)
        a.set_title(w["label"], fontsize=7)
        a.tick_params(labelsize=6)
    fig.suptitle("Mean |dEEG/dt| / pre-pulse baseline (log); red = marker", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "pulse_alignment.png", dpi=110)
    plt.close(fig)

if len(trials) and trials["mep_pp_uv"].notna().any():
    tr = trials.dropna(subset=["mep_pp_uv"])
    tr = tr[tr["emg_ch"] == tr.groupby(["sub", "task", "run"])["emg_ch"].transform("first")]
    groups = list(tr.groupby(["sub", "task", "run"], sort=True))[:120]
    fig, axs = grid(len(groups))
    for a, ((s, t, r), g) in zip(axs, groups):
        y = np.log10(np.maximum(g["mep_pp_uv"].values, 1))
        a.plot(np.arange(len(y)), y, ".", ms=1.5, color="k")
        for b in np.where(np.diff(g["block"].values) != 0)[0]:
            a.axvline(b + 0.5, color="r", lw=0.6)
        a.set_title(f"{s}|{t}", fontsize=7)
        a.tick_params(labelsize=6)
    fig.suptitle("log10 MEP (µV) vs trial; red = block boundary", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "mep_trend.png", dpi=110)
    plt.close(fig)

# ---------------- 4. AUDIT.md ----------------
subs = sorted(s for s in inv["sub"].unique() if s)
pul = rec[rec.get("n_pulses", pd.Series(dtype=float)).fillna(0) > 0] if "n_pulses" in rec else rec.iloc[0:0]


def rng(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    return f"{np.median(s):.4g}（{s.min():.4g}–{s.max():.4g}）" if len(s) else "—"


task_tab = (rec.pivot_table(index="sub", columns="task", values="path", aggfunc="count", fill_value=0)
            .reset_index() if len(rec) else pd.DataFrame())
task_cnt = rec.groupby("task").agg(n_rec=("path", "count"), n_sub=("sub", "nunique")).reset_index() \
    if len(rec) else pd.DataFrame()
ref_vals = side["EEGReference"].astype(str).value_counts().to_dict() if "EEGReference" in side else {}
sf_vals = rec["sfreq"].value_counts().to_dict() if "sfreq" in rec else {}
emg_vals = rec["emg_channels"].fillna("").value_counts().to_dict() if "emg_channels" in rec else {}
desc_tot = (ann.groupby("description")["count"].agg(["sum", "count"]).reset_index()
            .rename(columns={"sum": "total", "count": "n_recordings"}).sort_values("total", ascending=False)
            if len(ann) else pd.DataFrame())

per_sub_pulses = pul.groupby("sub")["n_pulses"].sum() if len(pul) else pd.Series(dtype=float)
per_sub_keep = pul.groupby("sub")["n_keep_isi_rule"].sum() if len(pul) else pd.Series(dtype=float)

expect = pd.DataFrame([
    ("被试数", "20", str(len(subs))),
    ("每人脉冲数", "600（6 × 100）", rng(per_sub_pulses)),
    ("每个记录的 block 数", "6", rng(pul["n_blocks"]) if len(pul) else "—"),
    ("刺激间隔中位数（秒）", "5（±15% 抖动）", rng(pul["isi_median"]) if len(pul) else "—"),
    ("间隔变异系数", "≈0.09（均匀 ±15%）", rng(pul["isi_cv"]) if len(pul) else "—"),
    ("EEG 通道数", "30", rng(rec["n_eeg"]) if "n_eeg" in rec else "—"),
    ("肌电通道", "左手 FDI", "; ".join(f"{k or '（无）'}×{v}" for k, v in emg_vals.items()) or "—"),
    ("采样率（Hz）", "【待核实】", "; ".join(f"{k:g}×{v}" for k, v in sf_vals.items()) or "—"),
    ("任务", "rest / tmseeg1 / tmseeg2", ", ".join(sorted(t for t in rec["task"].unique() if t))
     if len(rec) else "—"),
    ("满足 6.3 间隔规则的试验数/人", "—", rng(per_sub_keep)),
], columns=["项目", "总纲预期", "实际（中位数，范围）"])

rec_cols = [c for c in ["sub", "task", "run", "sfreq", "n_eeg", "emg_channels", "duration_s", "pulse_desc",
                        "n_pulses", "isi_median", "isi_min", "isi_max", "n_blocks", "block_sizes",
                        "n_keep_isi_rule", "align_offset_ms_median", "align_frac_within_1ms",
                        "mep_median_uv", "mep_frac_gt50uv", "pre_emg_frac_gt50uv", "error"] if c in rec]

md = []
md.append("# D1（ds002094）S2 审计报告\n")
md.append(f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}；数据根目录：`{ROOT}`；用时 {time.time() - T0:.0f} s")
md.append(f"- 文件 {len(inv)} 个，共 {inv['size'].sum() / 1e9:.2f} GB；被试目录 {len(subs)} 个；vhdr {len(vhdrs)} 个；"
          f"读取失败 {int((rec['error'] != '').sum()) if 'error' in rec else 0} 个")
md.append(f"- 判为脉冲标记的规则：出现 ≥ {PULSE_MIN_COUNT} 次且间隔中位数在 {PULSE_ISI_RANGE} 秒内，取次数最多者\n")
md.append("## 0. 与总纲预期的对比\n")
md.append(md_table(expect))
md.append("## 1. 被试与任务\n")
md.append(md_table(task_cnt))
md.append("<details><summary>每个被试的记录数</summary>\n\n" + md_table(task_tab) + "\n</details>\n")
md.append("## 2. 采样率、通道、参考\n")
md.append(f"- 采样率：{sf_vals}")
md.append(f"- 参考（eeg.json 的 EEGReference）：{ref_vals or '（无 sidecar 或无此字段）'}")
md.append(f"- 肌电通道（按 channels.tsv / MNE 类型 / 名称识别）：{emg_vals}")
md.append(f"- 不同的通道集合：{len(chan_sets)} 种（详见 `channel_sets.tsv`）\n")
if len(chan_sets):
    md.append(md_table(chan_sets[["n_ch", "n_recordings", "channel_set"]].head(5)))
md.append("## 3. 每个记录的脉冲与间隔\n")
md.append(md_table(rec[rec_cols]))
md.append("## 4. 标记\n")
md.append(md_table(desc_tot.head(30)))
md.append("- 对齐检查：`align_offset_ms_median` = EEG 伪迹峰（差分最大处）相对标记的时间；"
          "`align_frac_within_1ms` = 偏离中位数 ≤ 1 ms 的比例。图见 `figures/pulse_alignment.png`\n")
md.append("## 5. 图\n")
md.append("- `figures/isi_hist.png` 间隔直方图；`figures/mep_waveforms.png` 平均肌电波形；"
          "`figures/mep_trend.png` MEP 随试验走势与 block 边界；`figures/pulse_alignment.png` 标记对齐\n")
md.append("## 6. 顶层文件\n")
for name in ("dataset_description.json", "participants.tsv", "participants.json", "README", "README.md", "CHANGES"):
    p = TOP / name
    if p.exists():
        txt = p.read_text(encoding="utf-8", errors="replace")
        lines = txt.splitlines()
        md.append(f"### {name}\n\n```\n" + "\n".join(lines[:80]) +
                  ("\n…（截断）" if len(lines) > 80 else "") + "\n```\n")
if len(side):
    md.append("### eeg.json 字段取值（去重）\n")
    for c in side.columns:
        if c == "path":
            continue
        vals = side[c].astype(str).value_counts()
        md.append(f"- `{c}`：" + "; ".join(f"{k}×{v}" for k, v in vals.head(5).items()))
md.append("\n## 7. 错误\n")
md.append("```\n" + ("\n".join(errors) if errors else "（无）") + "\n```\n")
(OUT / "AUDIT.md").write_text("\n".join(md), encoding="utf-8")

summary = dict(n_files=int(len(inv)), n_subjects=len(subs), n_vhdr=len(vhdrs),
               n_errors=len(errors), sfreq=list(sf_vals.keys()), emg=emg_vals,
               tasks=task_cnt.to_dict("records") if len(task_cnt) else [],
               pulses_per_subject=per_sub_pulses.describe().to_dict() if len(per_sub_pulses) else {},
               seconds=round(time.time() - T0, 1))
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str),
                                  encoding="utf-8")
print("\n" + "\n".join(md[:12]))
log("AUDIT_DONE")
