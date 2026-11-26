# 来自 S2R 505b298 的 s2r/preprocess/badchan.py，2026-11-26 复制（机械改名：S2R_→PRJ_、包名 s2r→项目名）
"""坏导判定（只用 rest 记录）与球面样条插值矩阵。"""
from __future__ import annotations

import numpy as np


def detect_bads_rest(raw_rest, picks: list[str], flat_uv: float, robust_z: float) -> tuple[list[str], dict]:
    """rest 记录 1–45 Hz 后各导联标准差；过平或 log 标准差稳健 z 分数过大的视为坏导。"""
    r = raw_rest.copy().pick(picks).load_data()
    r.filter(1.0, 45.0, verbose="ERROR")
    sd = r.get_data().std(axis=1) * 1e6
    lg = np.log(np.maximum(sd, 1e-6))
    mad = np.median(np.abs(lg - np.median(lg))) * 1.4826
    z = (lg - np.median(lg)) / (mad if mad > 0 else 1.0)
    bads = [ch for ch, s, zz in zip(picks, sd, z) if s < flat_uv or abs(zz) > robust_z]
    return bads, {ch: (round(float(s), 3), round(float(zz), 2)) for ch, s, zz in zip(picks, sd, z)}


def interp_matrix(ch_names: list[str], bads: list[str]) -> np.ndarray:
    """返回 M（n_ch × n_ch）：clean = M @ x 即完成坏导插值（好导不变）。

    用 MNE 的 interpolate_bads 作用在单位矩阵上得到，之后每个试验只需矩阵乘法。
    """
    import mne

    n = len(ch_names)
    if not bads:
        return np.eye(n)
    info = mne.create_info(ch_names, 1000.0, "eeg")
    ep = mne.EpochsArray(np.eye(n)[None] * 1e-6, info, verbose="ERROR")
    ep.set_montage(mne.channels.make_standard_montage("standard_1005"), match_case=False, on_missing="raise")
    ep.info["bads"] = list(bads)
    ep.interpolate_bads(reset_bads=True, verbose="ERROR")
    return ep.get_data()[0] * 1e6
