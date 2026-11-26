# 来自 S2R 505b298 的 s2r/targets/mep.py，2026-11-26 复制（机械改名：S2R_→PRJ_、包名 s2r→项目名）
"""S4 预测目标：从 EMG .mat 读单次肌电，算 MEP 幅度和各类辅助指标。

D1 的 EMG 文件（sourcedata/emg/sub-XXX/XXX_00k.mat，k = 0..5 = block）：
MATLAB v5 结构体，字段 Data（试验 × 采样点 × 肌肉，mV）、t（ms，-25 → +100）、
SamplingRate（5000）、Label（{FDI, ADM}）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _field(s, name):
    v = getattr(s, name) if hasattr(s, name) else s[name]
    return np.asarray(v).squeeze() if not isinstance(v, str) else v


def load_emg_mat(path: Path) -> dict:
    """返回 {'data': (n_trials, n_times, n_muscles) mV, 't_ms': (n_times,), 'labels': [...], 'sfreq': float}。"""
    from scipy.io import loadmat

    m = loadmat(str(path), squeeze_me=True, struct_as_record=False)
    cands = [v for k, v in m.items() if not k.startswith("__") and hasattr(v, "_fieldnames") and "Data" in v._fieldnames]
    if len(cands) != 1:
        raise ValueError(f"{path.name}: 找不到唯一的含 Data 字段的结构体，变量有 {[k for k in m if not k.startswith('__')]}")
    s = cands[0]
    data = np.asarray(s.Data, dtype=float)
    t = np.asarray(s.t, dtype=float).ravel()
    labels = [str(x) for x in np.atleast_1d(s.Label)] if "Label" in s._fieldnames else []
    sf = float(np.asarray(s.SamplingRate).ravel()[0]) if "SamplingRate" in s._fieldnames else np.nan
    if data.ndim == 2:                       # 只有 1 个试验时 squeeze 会去掉一维
        data = data[None]
    if data.shape[1] != t.size:              # 保险：时间轴必须是第 2 维
        ax = [i for i, n in enumerate(data.shape) if n == t.size]
        if not ax:
            raise ValueError(f"{path.name}: Data 形状 {data.shape} 与 t 长度 {t.size} 对不上")
        data = np.moveaxis(data, ax[0], 1)
    return {"data": data, "t_ms": t, "labels": labels, "sfreq": sf}


def emg_files(emg_root: Path, sub: str) -> list[Path]:
    fs = sorted((Path(emg_root) / f"sub-{sub}").glob(f"{sub}_[0-9][0-9][0-9].mat"))
    if not fs:
        raise FileNotFoundError(f"sub-{sub}: 没有 EMG .mat 文件")
    return fs


def _ptp(x, t, win):
    m = (t >= win[0]) & (t <= win[1])
    return x[:, m].max(axis=1) - x[:, m].min(axis=1)


def _rms(x, t, win):
    m = (t >= win[0]) & (t <= win[1])
    seg = x[:, m] - x[:, m].mean(axis=1, keepdims=True)
    return np.sqrt((seg ** 2).mean(axis=1))


def block_running_median(y: np.ndarray, block: np.ndarray, window: int) -> np.ndarray:
    """每个 block 内单独做居中滑动中位数（边缘窗口自动缩短），不跨 block。"""
    out = np.full(len(y), np.nan)
    for b in np.unique(block):
        i = np.where(block == b)[0]
        out[i] = pd.Series(y[i]).rolling(window, center=True, min_periods=1).median().to_numpy()
    return out


def subject_targets(sub: str, emg_root: Path, tg: dict) -> tuple[pd.DataFrame, dict]:
    """一个被试的全部试验。aligned_idx 与 S3 的 trials 表一一对应（按 block 文件顺序拼接）。"""
    rows, waves, info = [], [], {}
    scale = float(tg["emg_unit_scale"])
    for b, f in enumerate(emg_files(emg_root, sub)):
        e = load_emg_mat(f)
        lab = [x.upper() for x in e["labels"]]
        mi = lab.index(tg["muscle"].upper()) if tg["muscle"].upper() in lab else 0
        x = e["data"][:, :, mi] * scale                       # µV
        t = e["t_ms"]
        other = [i for i in range(e["data"].shape[2]) if i != mi]
        x2 = e["data"][:, :, other[0]] * scale if other else None
        w = tg["mep_window_ms"]
        mep = _ptp(x, t, w)
        imax = np.argmax(np.abs(x[:, (t >= w[0]) & (t <= w[1])]), axis=1)
        lat = t[(t >= w[0]) & (t <= w[1])][imax]
        pre_pp = _ptp(x, t, tg["pre_window_ms"])
        pre_rms = _rms(x, t, tg["pre_window_ms"])
        cw = (t >= tg["wave_corr_window_ms"][0]) & (t <= tg["wave_corr_window_ms"][1])
        waves.append(x[:, cw])
        flat = (x == 0).mean(axis=1) > float(tg.get("emg_flat_frac", 0.5))
        for k in range(x.shape[0]):
            rows.append({"block": b, "trial_in_block": k, "emg_flat": bool(flat[k]),
                         "mep_raw_uv": mep[k], "mep_peak_ms": lat[k],
                         "pre_pp_uv": pre_pp[k], "pre_rms_uv": pre_rms[k],
                         "other_mep_uv": (_ptp(x2[k:k + 1], t, w)[0] if x2 is not None else np.nan)})
        info.setdefault("files", []).append(f"{f.name}:{x.shape[0]}")
        info["sfreq"], info["t_range_ms"] = e["sfreq"], [float(t[0]), float(t[-1])]
        info["muscle_used"] = e["labels"][mi] if e["labels"] else f"index {mi}"

    df = pd.DataFrame(rows)
    df.insert(0, "subject", f"sub-{sub}")
    df.insert(1, "aligned_idx", np.arange(len(df)))
    n_valid = (tg.get("emg_n_valid") or {}).get(str(sub))
    df["emg_valid"] = ~df["emg_flat"] & ((df["aligned_idx"] < int(n_valid)) if n_valid else True)
    info["n_flat"] = int(df["emg_flat"].sum())
    info["n_valid"] = int(df["emg_valid"].sum())
    v = df["emg_valid"].to_numpy()
    # 单次波形与个人平均波形的相关（Hussain 2022 的剔除指标之一；此处只计算，不剔除）
    W = np.vstack(waves)
    mean_w = W[v].mean(axis=0)                 # 个人平均波形只用有效试验
    Wc, mc = W - W.mean(axis=1, keepdims=True), mean_w - mean_w.mean()
    df["mep_wave_r"] = (Wc @ mc) / (np.linalg.norm(Wc, axis=1) * np.linalg.norm(mc) + 1e-12)

    # 以下只在有效试验上计算；无效试验（EMG 中断、全 0）的目标为 NaN
    y = np.full(len(df), np.nan)
    y[v] = np.log(np.maximum(df.loc[v, "mep_raw_uv"].to_numpy(), 1.0))  # 自然对数；< 1 µV 截到 1 µV 防 log(0)
    med = np.full(len(df), np.nan)
    med[v] = block_running_median(y[v], df.loc[v, "block"].to_numpy(), int(tg["detrend_window"]))
    df["mep_log"] = y
    df["mep_trend"] = med
    df["mep_resid"] = y - med                                   # y_cont：去漂移后的连续目标
    bmed = df[v].groupby("block")["mep_resid"].median()
    df["y_bin"] = np.where(v, (df["mep_resid"] > df["block"].map(bmed)).astype(float), np.nan)  # block 内中位数切分（主分析）
    df["y_bin_subject"] = np.where(v, (y > np.nanmedian(y)).astype(float), np.nan)  # 被试内中位数切分（复现 Hussain）
    return df, info
