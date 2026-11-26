# 来自 S2R 505b298 的 s2r/features/classical.py，2026-11-26 复制（机械改名：S2R_→PRJ_、包名 s2r→项目名）
"""S6 经典特征（总纲 S6；参数见配置 features_classical: 一节）。

全部特征只用 S3 已截好的脉冲前窗口（防泄漏 6.3 由 S3 保证）；这里不做任何跨试验的标准化（留给 S8 训练折内）。

列名
    pow_<W>_<band>_<ch>   各导联 log10 频段功率（clean_<W>，平均参考），W ∈ 配置的 windows（W1、W05；11-23 起加 W2、W4）
    loc_<W>_<band>        C4 Hjorth 拉普拉斯后的 log10 频段功率（刺激侧运动区）
    glob_<W>_<band>       全脑：先对 28 导的 PSD 求平均，再取 log10 频段功率
    slope_<W> / offset_<W>  1/f：全脑平均 PSD 的 log10 P = offset + slope × log10 f（slope 为负）
    mu_phase              脉冲时刻 C4 拉普拉斯 mu 相位（弧度，0 = 波峰，±π = 波谷），只用过去样本 + AR 外推
    mu_sin / mu_cos       sin、cos（S8 用这两列）
    mu_amp                脉冲时刻外推信号的包络幅值（µV）
    phase_check_err       自检：在 −1.0 s 处假装脉冲，估计相位与两侧数据"真值"的绝对误差（弧度）
    phase_check_amp       自检点的包络幅值（µV），用来看误差是否随 mu 幅值下降
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
from scipy import linalg, signal


# ---------- 功率 ----------
def welch_psd(x: np.ndarray, fs: float, w: dict) -> tuple[np.ndarray, np.ndarray]:
    """x: (..., T)。返回 freqs, psd (..., F)。窗口短于 nperseg 时用整段。"""
    T = x.shape[-1]
    nper = min(int(w["nperseg"]), T)
    nover = min(int(w["noverlap"]), nper - 1)
    f, p = signal.welch(x, fs=fs, window="hann", nperseg=nper, noverlap=nover, nfft=int(w["nfft"]),
                        axis=-1, detrend="constant")
    return f, p


def band_logpower(f: np.ndarray, psd: np.ndarray, bands: dict) -> dict[str, np.ndarray]:
    out = {}
    for name, (lo, hi) in bands.items():
        m = (f >= lo) & (f <= hi)
        out[name] = np.log10(psd[..., m].mean(axis=-1))
    return out


def aperiodic_fit(f: np.ndarray, psd: np.ndarray, fit_hz, exclude_hz) -> tuple[np.ndarray, np.ndarray]:
    """psd: (n, F)。逐试验线性拟合 log10 P ~ log10 f，返回 slope, offset。"""
    m = (f >= fit_hz[0]) & (f <= fit_hz[1])
    for lo, hi in exclude_hz:
        m &= ~((f >= lo) & (f <= hi))
    X = np.column_stack([np.ones(m.sum()), np.log10(f[m])])
    Y = np.log10(psd[:, m]).T                      # F × n
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)   # 2 × n
    return coef[1], coef[0]


def hjorth(x: np.ndarray, ch: list[str], center: str, neighbors: list[str]) -> np.ndarray:
    """x: (n, ch, T) → (n, T)。权重和为 0，与参考无关。"""
    c = ch.index(center)
    nb = [ch.index(k) for k in neighbors]
    return x[:, c] - x[:, nb].mean(axis=1)


# ---------- 相位（只用过去样本） ----------
def _ar_yule_walker(y: np.ndarray, order: int) -> np.ndarray:
    y = y - y.mean()
    n = len(y)
    r = np.array([np.dot(y[: n - k], y[k:]) / n for k in range(order + 1)])
    return linalg.solve_toeplitz(r[:order], r[1: order + 1])


def _ar_forecast(y: np.ndarray, a: np.ndarray, steps: int) -> np.ndarray:
    p = len(a)
    buf = list(y[-p:])
    out = np.empty(steps)
    for i in range(steps):
        v = float(np.dot(a, buf[::-1][:p]))
        out[i] = v
        buf.append(v)
        buf.pop(0)
    return out


def phase_design(fs_in: int, ph: dict) -> dict:
    q = fs_in // int(ph["fs"])
    if q * int(ph["fs"]) != fs_in:
        raise ValueError("master 采样率必须是相位采样率的整数倍")
    b = signal.firwin(int(ph["fir_numtaps"]), ph["band_hz"], pass_zero=False, fs=ph["fs"])
    return {"q": q, "b": b, "fs": int(ph["fs"]), "edge": int(round(ph["edge_s"] * ph["fs"])),
            "tail": int(round(ph["tail_s"] * ph["fs"])), "order": int(ph["ar_order"])}


def phase_at(seg: np.ndarray, pulse_pos_in: int, d: dict) -> tuple[float, float]:
    """seg: 一段只含过去样本的信号（fs_in）；pulse_pos_in: 脉冲在 seg 坐标中的位置（≥ len(seg)）。
    降采样 → 窗口内零相位 FIR 带通 → 丢末端 edge → AR 外推到脉冲后 tail → Hilbert → 取脉冲点相位与包络。"""
    if pulse_pos_in % d["q"]:
        raise ValueError("脉冲位置必须落在降采样网格上")
    x = signal.resample_poly(seg - seg.mean(), 1, d["q"], padtype="line")
    p = pulse_pos_in // d["q"]
    y = signal.filtfilt(d["b"], [1.0], x)
    yc = y[: len(y) - d["edge"]]
    a = _ar_yule_walker(yc, d["order"])
    z = np.concatenate([yc, _ar_forecast(yc, a, p - len(yc) + 1 + d["tail"])])
    an = signal.hilbert(z)
    return float(np.angle(an[p])), float(np.abs(an[p]))


def true_phase(full: np.ndarray, pos_in: int, d: dict) -> float:
    """自检用"真值"：用 pos 两侧数据（整段零相位滤波 + Hilbert）。只在脉冲前窗口内部取点，不涉及刺激后数据。"""
    x = signal.resample_poly(full - full.mean(), 1, d["q"], padtype="line")
    y = signal.filtfilt(d["b"], [1.0], x)
    return float(np.angle(signal.hilbert(y)[pos_in // d["q"]]))


def circ_absdiff(a, b):
    return np.abs(np.angle(np.exp(1j * (np.asarray(a) - np.asarray(b)))))


# ---------- 每个被试 ----------
def subject_features(h5_path: Path, fc: dict) -> tuple[dict, dict, dict]:
    """返回 (列 → 数组, PSD 包, 信息)。行 = h5 中 aligned_idx ≥ 0 的试验，按 aligned_idx 排序。"""
    lp = fc["laplacian"]
    cols: dict[str, np.ndarray] = {}
    psd_pack: dict[str, np.ndarray] = {}
    with h5py.File(h5_path, "r") as f:
        ch = json.loads(f.attrs["ch_names"])
        bads = json.loads(f.attrs.get("bads", "[]"))
        fs_clean = float(f.attrs["clean_sfreq"])
        fs_m = int(f.attrs["master_sfreq"])
        win_m = float(f.attrs["master_window_s"])
        end_off = float(f.attrs["end_offset_s"])
        ai = f["aligned_idx"][:]
        rows = np.where(ai >= 0)[0]
        rows = rows[np.argsort(ai[rows])]
        cols["aligned_idx"] = ai[rows]
        cols["block"] = f["block"][:][rows]
        cols["eeg_idx"] = f["eeg_idx"][:][rows]

        # 1. 功率（clean 窗口）
        for w in fc["windows"]:
            X = f[f"clean_{w}"][:][rows].astype(np.float64)            # n × ch × T
            fr, P = welch_psd(X, fs_clean, fc["welch"])                 # n × ch × F
            for band, v in band_logpower(fr, P, fc["bands"]).items():
                for j, c in enumerate(ch):
                    cols[f"pow_{w}_{band}_{c}"] = v[:, j]
            _, PL = welch_psd(hjorth(X, ch, lp["center"], lp["neighbors"]), fs_clean, fc["welch"])
            for band, v in band_logpower(fr, PL, fc["bands"]).items():
                cols[f"loc_{w}_{band}"] = v
            Pg = P.mean(axis=1)                                         # n × F
            for band, v in band_logpower(fr, Pg, fc["bands"]).items():
                cols[f"glob_{w}_{band}"] = v
            ap = fc["aperiodic"]
            cols[f"slope_{w}"], cols[f"offset_{w}"] = aperiodic_fit(fr, Pg, ap["fit_hz"], ap["exclude_hz"])
            keep_f = (fr >= 1) & (fr <= fc["welch"]["psd_fmax"])
            psd_pack[f"psd_{w}"] = P[..., keep_f].astype(np.float32)
            psd_pack[f"psdlap_{w}"] = PL[..., keep_f].astype(np.float32)
            psd_pack["freqs"] = fr[keep_f]

        # 2. 相位（master，C4 拉普拉斯）
        ph = fc["phase"]
        idx = sorted({ch.index(lp["center"]), *[ch.index(k) for k in lp["neighbors"]]})
        M = f["master"][:, idx, :][rows].astype(np.float64)            # n × 5 × 4000
    sub_ch = [ch[i] for i in idx]
    L = hjorth(M, sub_ch, lp["center"], lp["neighbors"])               # n × Tm
    d = phase_design(fs_m, ph)
    Tm = L.shape[1]
    n_win = int(round(ph["window_s"] * fs_m))
    pulse = int(round((win_m + end_off) * fs_m))                       # 脉冲在 master 坐标中的位置（= 4010）
    chk = pulse + int(round(ph["check_at_s"] * fs_m))                  # 自检的假脉冲位置（= 3010）
    gap = pulse - Tm                                                    # 窗口末端到脉冲的样本数
    n = len(L)
    phase, amp, err, camp = np.empty(n), np.empty(n), np.empty(n), np.empty(n)
    for i in range(n):
        phase[i], amp[i] = phase_at(L[i, Tm - n_win:], n_win + gap, d)
        est, camp[i] = phase_at(L[i, chk - gap - n_win: chk - gap], n_win + gap, d)
        err[i] = circ_absdiff(est, true_phase(L[i], chk, d))
    cols["mu_phase"], cols["mu_sin"], cols["mu_cos"] = phase, np.sin(phase), np.cos(phase)
    cols["mu_amp"], cols["phase_check_err"], cols["phase_check_amp"] = amp, err, camp

    psd_pack["aligned_idx"] = cols["aligned_idx"]
    psd_pack["ch_names"] = np.array(ch)
    info = {"n": int(n), "ch_names": ch, "bads": bads, "fs_clean": fs_clean, "fs_master": fs_m,
            "pulse_pos_master": pulse, "phase_check_pos_master": chk,
            "phase_check_err_median_deg": round(float(np.degrees(np.median(err))), 1)}
    if bads:
        info["warning"] = "有坏导：master 未插值，若坏导在 C4 邻域，相位特征不可信"
    return cols, psd_pack, info
