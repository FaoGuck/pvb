"""CCEP 平均与 N1 检测（复现 mnl_ccepAge；参数见 configs/datasets/p_d5_ds004080.yaml）。

处理一个刺激对（一个 run 内）：
  trials (n_trial, n_ch, n_t) µV，时间轴 tt（秒，0 = 脉冲）
  1. 每个试验、每个通道减去 trial_baseline 窗的中位数
  2. 对试验求平均（NaN 忽略）→ 再减 avg_baseline 窗的中位数
  3. N1：在 [search 起点, search_tmax] 上找负峰（prominence ≥ peak_prominence_uv），只保留 tmin < t ≤ tmax 的，
     取最负的一个；|幅度| ≥ thresh_sd × max(SD(平均波形在 sd_baseline 窗), sd_floor_uv) 且 ≤ max_abs_uv 才算有 N1
奇数次 / 偶数次试验各自走 1–2 步得到两半平均（P203 split-half 用）。
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks, resample_poly


def _win(tt: np.ndarray, tmin: float, tmax: float) -> np.ndarray:
    return (tt > tmin) & (tt < tmax)


def baseline_subtract(x: np.ndarray, tt: np.ndarray, tmin: float, tmax: float, stat: str = "median") -> np.ndarray:
    """x (..., n_t)：沿最后一维减去基线窗的中位数 / 均值。"""
    w = _win(tt, tmin, tmax)
    f = np.nanmedian if stat == "median" else np.nanmean
    with np.errstate(all="ignore"):
        return x - f(x[..., w], axis=-1, keepdims=True)


def average(trials: np.ndarray, tt: np.ndarray, cfg: dict) -> np.ndarray:
    """trials (n_trial, n_ch, n_t) → 平均波形 (n_ch, n_t)。"""
    tb, ab = cfg["trial_baseline"], cfg["avg_baseline"]
    x = baseline_subtract(trials, tt, tb["tmin"], tb["tmax"], tb.get("stat", "median"))
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            avg = np.nanmean(x, axis=0)
    return baseline_subtract(avg, tt, ab["tmin"], ab["tmax"], ab.get("stat", "median"))


def detect_n1(avg: np.ndarray, tt: np.ndarray, n1: dict) -> dict:
    """avg (n_ch, n_t) → 每个通道的 N1：sample（原采样率下的索引）、latency_s、amp_uv、sd_uv、detected。"""
    n_ch = avg.shape[0]
    out = {k: np.full(n_ch, np.nan) for k in ["sample", "latency_s", "amp_uv", "sd_uv"]}
    out["detected"] = np.zeros(n_ch, dtype=bool)
    i0 = int(np.argmax(tt > n1["tmin"]))
    i_end = int(np.argmax(tt > n1["search_tmax"]))
    i_n1_end = int(np.argmax(tt > n1["tmax"]))
    wsd = _win(tt, *n1["sd_baseline"])
    for c in range(n_ch):
        x = avg[c]
        if not np.isfinite(x[i0:i_end]).all():
            continue
        sd = float(np.nanstd(x[wsd], ddof=1))          # MATLAB std 默认 N-1
        out["sd_uv"][c] = sd
        sd = max(sd, n1["sd_floor_uv"])
        pk, _ = find_peaks(-x[i0:i_end], prominence=n1["peak_prominence_uv"])
        pk = pk + i0
        pk = pk[pk < i_n1_end]
        if pk.size == 0:
            continue
        p = int(pk[np.argmin(x[pk])])                   # 最负的那个
        a = float(x[p])
        out["sample"][c], out["latency_s"][c], out["amp_uv"][c] = p, float(tt[p]), a
        out["detected"][c] = (a < 0) and (abs(a) >= n1["thresh_sd"] * sd) and (abs(a) <= n1["max_abs_uv"])
    return out


def tail_time(avg: np.ndarray, tt: np.ndarray, sfreq: float, tail: dict, sd_win=(-2.0, -0.1)) -> np.ndarray:
    """每个通道：整流 + 滑动平均后，最后一次超过 sd_mult × 基线 SD 的时刻（秒）；从未超过 → 0。"""
    n = max(1, int(round(tail["smooth_s"] * sfreq)))
    k = np.ones(n) / n
    w_sd = _win(tt, *sd_win)
    post = (tt > 0.009) & (tt <= tail["tmax"])
    res = np.full(avg.shape[0], np.nan)
    for c in range(avg.shape[0]):
        x = avg[c]
        if not np.isfinite(x).all():
            continue
        env = np.convolve(np.abs(x), k, mode="same")
        thr = tail["sd_mult"] * np.std(x[w_sd], ddof=1)
        over = np.where(post & (env > thr))[0]
        res[c] = float(tt[over[-1]]) if over.size else 0.0
    return res


def to_store(avg: np.ndarray, tt: np.ndarray, sfreq: float, store: dict) -> tuple[np.ndarray, np.ndarray]:
    """截取存储窗并降采样到 store.sfreq（NaN 通道保持 NaN）。返回 (avg_store float32, tt_store)。"""
    w = (tt >= store["tmin"]) & (tt <= store["tmax"])
    x = avg[:, w].astype(np.float64)
    bad = ~np.isfinite(x).all(axis=1)
    x[bad] = 0.0
    up, down = int(store["sfreq"]), int(round(sfreq))
    g = np.gcd(up, down)
    if up != down:
        x = resample_poly(x, up // g, down // g, axis=1)
    x[bad] = np.nan
    t = store["tmin"] + np.arange(x.shape[1]) / store["sfreq"]
    return x.astype(np.float32), t
