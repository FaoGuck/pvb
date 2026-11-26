# 来自 S2R 505b298 的 s2r/preprocess/epoch.py，2026-11-26 复制（机械改名：S2R_→PRJ_、包名 s2r→项目名）
"""S3 刺激前片段：截取与片段内处理。

防泄漏（总纲 6.3）的做法：
1. 先按脉冲时刻截断：每个试验只读 [脉冲 - 0.01 - W, 脉冲 - 0.01) 这一段原始样本；
2. 之后的所有处理（去均值、降采样、陷波、带通、插值、重参考）都只作用在这一段上，
   边界用片段自身做对称延拓（reflect），不碰片段以外的任何样本；
3. 被试级别的东西（坏导、插值矩阵）只用 rest 记录（无 TMS）估计。
因此，某个试验的输出只取决于它自己窗口里的样本。leakage_selftest() 用模拟数据逐样本验证这一点。
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import signal

ReadFn = Callable[[int, int], np.ndarray]  # read(start, stop) -> (n_ch, stop - start)，单位 µV


def wname(w: float) -> str:
    """4.0 -> 'W4'，0.5 -> 'W05'。"""
    return "W" + (f"{w:g}".replace(".", "") if w < 1 else f"{int(w)}")


def _pad_reflect(x: np.ndarray, p: int) -> np.ndarray:
    return np.pad(x, [(0, 0)] * (x.ndim - 1) + [(p, p)], mode="reflect")


def to_master(seg: np.ndarray, fs_in: int, fs_master: int) -> np.ndarray:
    """原始片段 → 主片段：去均值 + 抗混叠降采样（resample_poly），不做其他滤波。"""
    if fs_in % fs_master:
        raise ValueError("fs_in 必须是 fs_master 的整数倍")
    q = fs_in // fs_master
    n = seg.shape[-1]
    if n % q:
        raise ValueError("片段长度必须是降采样倍数的整数倍")
    x = seg - seg.mean(axis=-1, keepdims=True)
    p = min(n - 1, fs_in // 10) // q * q          # 0.1 s 的对称延拓，取 q 的整数倍
    y = signal.resample_poly(_pad_reflect(x, p), 1, q, axis=-1, padtype="line")
    return y[..., p // q: p // q + n // q]


def clean_filters(fs: int, notch_hz: float, notch_q: float, band_hz, order: int):
    b, a = signal.iirnotch(notch_hz, notch_q, fs=fs)
    sos_notch = signal.tf2sos(b, a)
    sos_band = signal.butter(order, band_hz, btype="bandpass", fs=fs, output="sos")
    return np.vstack([sos_notch, sos_band])


def clean_window(x: np.ndarray, fs: int, fs_out: int, sos: np.ndarray, spatial: np.ndarray) -> np.ndarray:
    """主片段的末端窗口 → 干净版本。x: (..., n_ch, n)。

    窗口内去均值 → 对称延拓 → 陷波 + 带通（零相位）→ 降采样 → 裁回 → 空间矩阵（插值 + 平均参考）。
    """
    q = fs // fs_out
    n = x.shape[-1]
    if n % q:
        raise ValueError("窗口长度必须是降采样倍数的整数倍")
    x = x - x.mean(axis=-1, keepdims=True)
    p = (n - 1) // q * q
    xp = signal.sosfiltfilt(sos, _pad_reflect(x, p), axis=-1, padtype=None)
    y = signal.resample_poly(xp, 1, q, axis=-1, padtype="line")[..., p // q: p // q + n // q]
    return np.einsum("ij,...jt->...it", spatial, y)


def spatial_matrix(n_ch: int, interp: np.ndarray | None = None, average_ref: bool = True) -> np.ndarray:
    """插值矩阵（坏导由好导线性组合得到）再乘平均参考矩阵。"""
    m = np.eye(n_ch) if interp is None else np.asarray(interp, dtype=float)
    if average_ref:
        m = (np.eye(n_ch) - np.full((n_ch, n_ch), 1.0 / n_ch)) @ m
    return m


def epoch_recording(read: ReadFn, n_samples: int, pulses: np.ndarray, fs_in: int, ep: dict,
                    spatial: np.ndarray, n_ch: int) -> dict:
    """一个 TMS 记录的全部试验。返回 master、各窗口 clean、每个窗口的 guard 标记。

    read 只会被调用在 [start, stop) 上，且 stop = 脉冲 - end_offset。
    """
    off = int(round(ep["end_offset_s"] * fs_in))
    wm = ep["master_window_s"]
    n_in = int(round(wm * fs_in))
    fs_m = int(ep["master_sfreq"])
    cl = ep["clean"]
    sos = clean_filters(fs_m, cl["notch_hz"], cl["notch_q"], cl["band_hz"], cl["filter_order"])
    n_tr = len(pulses)
    master = np.full((n_tr, n_ch, int(round(wm * fs_m))), np.nan, dtype=np.float32)
    ok_read = np.zeros(n_tr, dtype=bool)
    for k, pk in enumerate(pulses):
        stop = int(pk) - off
        start = stop - n_in
        if start < 0 or stop > n_samples:
            continue
        assert stop <= pk - off, "窗口越过了脉冲前截止点"
        seg = np.asarray(read(start, stop), dtype=np.float64)
        master[k] = to_master(seg, fs_in, fs_m)
        ok_read[k] = True
    out = {"master": master, "ok_read": ok_read}
    prev = np.concatenate([[np.nan], pulses[:-1].astype(float)])
    for w in ep["windows_s"]:
        nm = int(round(w * fs_m))
        cw = np.full((n_tr, n_ch, int(round(w * cl["sfreq"]))), np.nan, dtype=np.float32)
        if ok_read.any():
            cw[ok_read] = clean_window(master[ok_read, :, -nm:].astype(np.float64), fs_m, int(cl["sfreq"]),
                                       sos, spatial)
        out[f"clean_{wname(w)}"] = cw
        start_w = pulses - off - int(round(w * fs_in))
        guard = np.where(np.isnan(prev), start_w >= 0, start_w >= prev + ep["guard_s"] * fs_in)
        out[f"guard_ok_{wname(w)}"] = guard & ok_read
    return out


def leakage_selftest(n_trials_check: int = 6, seed: int = 0) -> tuple[bool, str]:
    """总纲 6.3 第 4 条：把某个脉冲之后的数据全部换成噪声，重跑整个记录，
    该试验及其之前所有试验的输出必须逐样本完全相同。"""
    rng = np.random.default_rng(seed)
    fs, n_ch = 5000, 6
    ep = {"end_offset_s": 0.01, "master_window_s": 4.0, "master_sfreq": 1000,
          "windows_s": [4.0, 3.0, 1.0, 0.5], "guard_s": 0.9,
          "clean": {"sfreq": 250, "notch_hz": 60, "notch_q": 30, "band_hz": [1.0, 45.0], "filter_order": 4}}
    isi = rng.uniform(4.25, 5.75, 20)
    pulses = (np.cumsum(isi) * fs).astype(np.int64) + 5 * fs
    n = int(pulses[-1] + 3 * fs)
    t = np.arange(n) / fs
    data = rng.normal(0, 10, (n_ch, n)) + 20 * np.sin(2 * np.pi * 10 * t) + 5 * np.sin(2 * np.pi * 60 * t)
    for pk in pulses:                       # 脉冲后的伪迹 + 诱发响应，幅度远大于背景
        data[:, pk:pk + 50] += 5000
        data[:, pk:pk + 1500] += 80 * np.sin(2 * np.pi * 8 * t[: min(1500, n - pk)])
    spatial = spatial_matrix(n_ch)
    base = epoch_recording(lambda s, e: data[:, s:e], n, pulses, fs, ep, spatial, n_ch)
    keys = [k for k in base if k.startswith(("master", "clean_"))]
    bad = []
    for k in rng.choice(len(pulses), size=min(n_trials_check, len(pulses)), replace=False):
        d2 = data.copy()
        d2[:, pulses[k]:] = rng.normal(0, 1000, (n_ch, n - pulses[k]))
        alt = epoch_recording(lambda s, e: d2[:, s:e], n, pulses, fs, ep, spatial, n_ch)
        for key in keys:
            a, b = base[key][: k + 1], alt[key][: k + 1]
            if not np.array_equal(a, b, equal_nan=True):
                bad.append(f"trial {k} {key}")
    return (not bad), ("PASS" if not bad else "FAIL: " + "; ".join(bad[:10]))
