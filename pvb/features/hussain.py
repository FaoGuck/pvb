# 来自 S2R 505b298 的 s2r/features/hussain.py，2026-11-26 复制（机械改名：S2R_→PRJ_、包名 s2r→项目名）
"""E102 用的 Hussain & Quentin 2022 式频谱特征（28 导 × 6 频段 = 168）。参数见 configs/experiments/E102_hussain_d1.yaml。

只用 S3 的 master（脉冲前、1000 Hz、未滤波）窗口内的数据：
取最后 window_s → 窗口内降到 500 Hz → 平均参考 → 去均值 + 线性去趋势 → Welch（MATLAB pwelch 默认分段）→ 频段内平均。
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
from scipy import signal


def matlab_welch(x: np.ndarray, fs: float, w: dict) -> tuple[np.ndarray, np.ndarray]:
    """模仿 MATLAB pwelch 默认：最多 n_segments 段、50% 重叠 → 段长 floor(N / 4.5)；对称 Hamming 窗；补零到 df_hz 网格。"""
    N = x.shape[-1]
    nseg = int(w.get("n_segments", 8))
    ov = float(w.get("overlap", 0.5))
    L = int(np.floor(N / ((nseg - 1) * (1 - ov) + 1)))
    win = signal.get_window(w.get("window", "hamming"), L, fftbins=False)
    nfft = int(round(fs / float(w["df_hz"])))
    return signal.welch(x, fs=fs, window=win, nperseg=L, noverlap=int(np.floor(L * ov)), nfft=nfft,
                        detrend=False, axis=-1)


def subject_features(h5_path: Path, fc: dict) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """返回 aligned_idx（升序）、X（n × 28·6）、特征名。"""
    with h5py.File(h5_path, "r") as f:
        ch = json.loads(f.attrs["ch_names"])
        fs_m = int(f.attrs["master_sfreq"])
        ai = f["aligned_idx"][:]
        rows = np.where(ai >= 0)[0]
        rows = rows[np.argsort(ai[rows])]
        n_win = int(round(fc["window_s"] * fs_m))
        seg = f["master"][:, :, -n_win:][rows].astype(np.float64)        # n × ch × 500
    q = fs_m // int(fc["fs"])
    x = signal.resample_poly(seg, 1, q, axis=-1, padtype="line")
    if fc.get("reference") == "average":
        x = x - x.mean(axis=1, keepdims=True)
    x = signal.detrend(x, axis=-1, type=fc.get("detrend", "linear"))
    fr, P = matlab_welch(x, fc["fs"], fc["welch"])
    lo_ex, hi_ex = fc["exclude_hz"]
    feats, names = [], []
    for band, (lo, hi) in fc["bands"].items():
        m = (fr >= lo) & (fr <= hi) & ~((fr > lo_ex) & (fr < hi_ex))
        v = P[..., m].mean(axis=-1)                                      # n × ch
        if fc.get("log_power"):
            v = np.log10(v)
        feats.append(v)
        names += [f"{band}_{c}" for c in ch]
    X = np.concatenate(feats, axis=1)
    return ai[rows], X, names
