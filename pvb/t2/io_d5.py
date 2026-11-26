"""D5 = ds004080 的读取：BIDS sidecar（json / tsv）+ BrainVision 信号（mne，不整段载入内存）。

约定
- 刺激对用**无序对**命名：两个电极名大写后排序，以 "-" 连接（mnl_ccepAge 合并正负极性；它按第一次出现的顺序命名，
  这里按字母序，只影响名字不影响分组）
- 脉冲时刻 = 事件的 onset（秒）× 采样率，四舍五入成样本号（0 起）；P201 会核对它和 sample_start 列、
  以及和刺激伪迹的对齐
- 信号单位统一成 µV（mne 读出来是 V）
"""
from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

RUN_RE = re.compile(r"(sub-[^_]+)_(ses-[^_]+)_task-([^_]+)_run-([^_]+)_events\.tsv$")


def pair_name(site: str) -> str:
    a = [s.strip().upper() for s in str(site).split("-")]
    return "-".join(sorted(a)) if len(a) == 2 else str(site).upper()


def pair_electrodes(pair: str) -> list[str]:
    return pair.split("-")


@dataclass
class RunMeta:
    subject: str
    session: str
    run: str
    base: str                     # 不含后缀的路径前缀
    sfreq: float
    duration_s: float
    channels: pd.DataFrame        # channels.tsv（加了 NAME 大写列）
    stim: pd.DataFrame            # 刺激事件（加了 pair、sample0 列）
    events: pd.DataFrame          # 全部事件
    json: dict = field(default_factory=dict)

    @property
    def run_id(self) -> str:
        return f"{self.subject}_{self.session}_run-{self.run}"


def find_runs(root: str, subject: str) -> list[str]:
    """返回该患者全部 run 的 events.tsv 路径（按名字排序）。"""
    return sorted(glob.glob(os.path.join(root, subject, "ses-*", "ieeg", f"{subject}_*_events.tsv")))


def read_run_meta(events_tsv: str) -> RunMeta:
    m = RUN_RE.search(os.path.basename(events_tsv))
    if not m:
        raise ValueError(f"不认识的文件名: {events_tsv}")
    sub, ses, _task, run = m.groups()
    base = events_tsv[: -len("_events.tsv")]
    info = json.load(open(base + "_ieeg.json", encoding="utf-8"))
    ch = pd.read_csv(base + "_channels.tsv", sep="\t")
    ch["NAME"] = ch["name"].astype(str).str.upper()
    ev = pd.read_csv(base + "_events.tsv", sep="\t")
    sfreq = float(info["SamplingFrequency"])
    st = ev[ev["trial_type"] == "electrical_stimulation"].copy()
    st["pair"] = st["electrical_stimulation_site"].map(pair_name)
    st["sample0"] = np.round(st["onset"].astype(float) * sfreq).astype(np.int64)
    st = st.sort_values("onset").reset_index(drop=True)
    return RunMeta(sub, ses, run, base, sfreq, float(info.get("RecordingDuration", np.nan)), ch, st, ev, info)


def good_channels(meta: RunMeta, types=("ECOG",), status_good="good") -> list[str]:
    ch = meta.channels
    ok = ch["type"].astype(str).str.upper().isin([t.upper() for t in types]) & (ch["status"] == status_good)
    return ch.loc[ok, "NAME"].tolist()


def read_electrodes(root: str, subject: str) -> pd.DataFrame:
    """电极坐标（MNI）。多个 session 时合并，名字大写作索引。"""
    fs = sorted(glob.glob(os.path.join(root, subject, "ses-*", "ieeg", f"{subject}_*_electrodes.tsv")))
    el = pd.concat([pd.read_csv(f, sep="\t") for f in fs], ignore_index=True)
    el["NAME"] = el["name"].astype(str).str.upper()
    for c in ["x", "y", "z"]:
        el[c] = pd.to_numeric(el[c], errors="coerce")
    return el.drop_duplicates("NAME", keep="last").set_index("NAME")


class RawReader:
    """按需读取一段信号（不整段载入）。"""

    def __init__(self, meta: RunMeta):
        import mne
        self.raw = mne.io.read_raw_brainvision(meta.base + "_ieeg.vhdr", preload=False, verbose="ERROR")
        self.sfreq = float(self.raw.info["sfreq"])
        self.n_times = int(self.raw.n_times)
        self.names = [n.upper() for n in self.raw.ch_names]
        self._idx = {n: i for i, n in enumerate(self.names)}

    def picks(self, names: list[str]) -> list[int]:
        return [self._idx[n] for n in names if n in self._idx]

    def segments(self, picks: list[int], starts: np.ndarray, n_samp: int) -> np.ndarray:
        """返回 (n_seg, n_ch, n_samp) float32 µV；超出记录范围的段整段为 NaN。"""
        out = np.full((len(starts), len(picks), n_samp), np.nan, dtype=np.float32)
        for k, s0 in enumerate(starts):
            s0 = int(s0)
            if s0 < 0 or s0 + n_samp > self.n_times:
                continue
            out[k] = (self.raw.get_data(picks=picks, start=s0, stop=s0 + n_samp) * 1e6).astype(np.float32)
        return out
