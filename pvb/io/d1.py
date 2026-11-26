# 来自 S2R 505b298 的 s2r/io/d1.py，2026-11-26 复制（机械改名：S2R_→PRJ_、包名 s2r→项目名）
"""D1（OpenNeuro ds002094）：找文件、读脉冲、与 EMG 对齐。

只做读取和登记，不做任何信号处理。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml


def load_config(code_root: Path | str) -> dict:
    p = Path(code_root) / "configs" / "datasets" / "d1_ds002094.yaml"
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def list_subjects(raw_root: Path) -> list[str]:
    """返回 ['283', '711', ...]（不带 sub- 前缀）。"""
    return sorted(p.name[4:] for p in Path(raw_root).glob("sub-*") if p.is_dir())


def vhdr_path(raw_root: Path, sub: str, task: str) -> Path:
    return Path(raw_root) / f"sub-{sub}" / "eeg" / f"sub-{sub}_task-{task}_eeg.vhdr"


def tms_recordings(raw_root: Path, sub: str, tasks: list[str]) -> list[tuple[str, Path]]:
    """按合并顺序返回存在的 TMS 记录 [(task, vhdr), ...]。"""
    out = [(t, vhdr_path(raw_root, sub, t)) for t in tasks]
    out = [(t, p) for t, p in out if p.exists()]
    if not out:
        raise FileNotFoundError(f"sub-{sub} 没有 TMS 记录")
    return out


def pulse_samples(raw, pulse_desc: str) -> np.ndarray:
    """脉冲标记的样本序号（相对 raw 第一个样本，可直接用于 raw.get_data(start, stop)）。"""
    ann = raw.annotations
    onsets = np.asarray(ann.onset)[np.asarray(ann.description) == pulse_desc]
    samp = np.round(onsets * raw.info["sfreq"]).astype(np.int64)
    return np.sort(samp)


def emg_alignment(sub: str, n_total: int, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """按官方 README 的规则，从 EEG 合并序列里标出要删的试验。

    返回 aligned_idx（0 起；被删的为 -1）和 drop_reason（'' 表示保留）。
    """
    rule = (cfg.get("emg_alignment") or {}).get(str(sub), {}) or {}
    drop = set(int(i) for i in rule.get("drop", []))
    if "drop_from" in rule:
        drop |= set(range(int(rule["drop_from"]), n_total + 1))
    reason = np.array(["emg_align" if (i + 1) in drop else "" for i in range(n_total)], dtype=object)
    aligned = np.full(n_total, -1, dtype=np.int64)
    keep = reason == ""
    aligned[keep] = np.arange(int(keep.sum()))
    return aligned, reason
