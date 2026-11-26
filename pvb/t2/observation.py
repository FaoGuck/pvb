"""T2 观测段（00 第 4 节 P204、第 6 节第 2 条）与按“归属刺激对”累加的连接度（P205 的 B2 用）。

观测段：同一 run 内按时间排序的相邻脉冲 k、k+1 之间，[t_k + T_tail, t_{k+1} − pre_gap)；最后一个脉冲之后按 5 s 间隔截止。
  - 长度 < min_len 丢弃；长段（两个刺激对之间的空档）切成 n = floor(L / L0) 块等长子段（L0 = 5 − T_tail − pre_gap），
    使每块权重相同
  - **归属（owner）= 前一个脉冲的刺激对**：这一段反映的是刺激过 owner 之后的状态。预测留出刺激对 X 时，
    只能用训练折刺激对 owner 的段（X 自己的段不得进入）
  - 屏蔽：owner 的两个刺激电极、下一个脉冲的两个刺激电极、段内任一样本 |x| ≥ sat_uv 的通道
连接度（每段、每对有效通道，降采样到 cfg.sfreq 后；全部只在段内计算，不读段外任何样本）：
  corr_bb                 宽带 Pearson 相关
  coh_<band>              频段内对互谱求和后的相干 |ΣS_ij|² / (ΣS_ii ΣS_jj)（单段内的频率平均）
  env_<band>              4 阶 Butterworth 带通（sosfiltfilt）→ |Hilbert| → Pearson 相关
按 owner 累加：sums[owner, measure, i, j]、counts[owner, i, j]（i、j 都有效的段数）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, hilbert, resample_poly, sosfiltfilt

from pvb.t2 import io_d5


def measure_names(cfg_obs: dict) -> list[str]:
    return ["corr_bb"] + [f"coh_{b}" for b in cfg_obs["bands"]] + [f"env_{b}" for b in cfg_obs["envelope_bands"]]


def segment_table(stim: pd.DataFrame, sfreq: float, n_times: int, cfg_obs: dict, isi_nominal: float = 5.0) -> pd.DataFrame:
    """stim：read_run_meta 的 stim（按时间排序，含 sample0、pair）。返回每个子段一行（原采样率下的样本区间）。"""
    T, g, Lmin = cfg_obs["t_tail"], cfg_obs["pre_gap"], cfg_obs["min_len"]
    L0 = isi_nominal - T - g
    s0 = stim["sample0"].to_numpy(np.int64)
    pr = stim["pair"].to_numpy()
    rows = []
    for k in range(len(s0)):
        a = s0[k] + int(round(T * sfreq))
        if k + 1 < len(s0):
            b, nxt = s0[k + 1] - int(round(g * sfreq)), pr[k + 1]
        else:
            b, nxt = min(n_times, s0[k] + int(round((isi_nominal - g) * sfreq))), ""
        L = (b - a) / sfreq
        if L < Lmin:
            continue
        n = max(1, int(np.floor(L / L0 + 1e-9)))
        edges = np.linspace(a, b, n + 1).round().astype(np.int64)
        for j in range(n):
            rows.append((k, j, int(edges[j]), int(edges[j + 1]), pr[k], nxt))
    df = pd.DataFrame(rows, columns=["pulse_idx", "chunk", "start", "stop", "owner_pair", "next_pair"])
    df["length_s"] = (df.stop - df.start) / sfreq
    return df


def _corr(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=1, keepdims=True)
    sd = np.sqrt((x ** 2).sum(axis=1))
    sd[sd == 0] = np.nan
    return (x @ x.T) / np.outer(sd, sd)


def segment_measures(x: np.ndarray, sf: float, cfg_obs: dict) -> np.ndarray:
    """x (n_ch, n)，已降采样、单位 µV。返回 (n_meas, n_ch, n_ch)。"""
    out = [_corr(x)]
    xd = x - x.mean(axis=1, keepdims=True)
    X = np.fft.rfft(xd * np.hanning(xd.shape[1])[None, :], axis=1)
    f = np.fft.rfftfreq(xd.shape[1], 1 / sf)
    for lo, hi in cfg_obs["bands"].values():
        w = (f >= lo) & (f < min(hi, sf / 2 - 1e-6))
        if not w.any():
            out.append(np.full((x.shape[0], x.shape[0]), np.nan)); continue
        Xb = X[:, w]
        S = Xb @ Xb.conj().T
        p = np.real(np.diag(S))
        with np.errstate(all="ignore"):
            out.append(np.abs(S) ** 2 / np.outer(p, p))
    for b in cfg_obs["envelope_bands"]:
        lo, hi = cfg_obs["bands"][b]
        sos = butter(4, [lo, min(hi, 0.45 * sf)], btype="band", fs=sf, output="sos")
        env = np.abs(hilbert(sosfiltfilt(sos, xd, axis=1), axis=1))
        out.append(_corr(env))
    return np.stack(out).astype(np.float64)


def owner_accumulators(reader, chans: list[str], picks: list[int], segs: pd.DataFrame, sfreq: float,
                       cfg_obs: dict) -> dict:
    """reader 需有 segments(picks, starts, n_samp)（io_d5.RawReader 或测试用的假读取器）。"""
    meas = measure_names(cfg_obs)
    owners = sorted(segs.owner_pair.unique())
    oi = {o: i for i, o in enumerate(owners)}
    n_ch = len(chans)
    sums = np.zeros((len(owners), len(meas), n_ch, n_ch))
    counts = np.zeros((len(owners), n_ch, n_ch), dtype=np.int32)
    n_seg = np.zeros(len(owners), dtype=np.int32)
    ci = {c: i for i, c in enumerate(chans)}
    up, down = int(cfg_obs["sfreq"]), int(round(sfreq))
    gcd = np.gcd(up, down)
    masked_frac = []
    for r in segs.itertuples(index=False):
        x = reader.segments(picks, np.array([r.start]), r.stop - r.start)[0].astype(np.float64)
        valid = np.isfinite(x).all(axis=1)
        valid &= ~(np.abs(np.nan_to_num(x)) >= cfg_obs["sat_uv"]).any(axis=1)
        drop = []
        if cfg_obs.get("mask_owner_stim", True):
            drop += io_d5.pair_electrodes(r.owner_pair)
        if cfg_obs.get("mask_next_stim", True) and r.next_pair:
            drop += io_d5.pair_electrodes(r.next_pair)
        for e in drop:
            if e in ci:
                valid[ci[e]] = False
        masked_frac.append(1 - valid.mean())
        if valid.sum() < 2:
            continue
        xv = np.nan_to_num(x[valid])
        if up != down:
            xv = resample_poly(xv, up // gcd, down // gcd, axis=1)
        m = segment_measures(xv, float(up), cfg_obs)
        idx = np.where(valid)[0]
        o = oi[r.owner_pair]
        ok = np.isfinite(m).all(axis=0)
        blk = np.ix_(idx, idx)
        sums[o][:, idx[:, None], idx[None, :]] += np.where(ok[None], m, 0.0)
        counts[o][blk] += ok.astype(np.int32)
        n_seg[o] += 1
    return dict(owners=np.array(owners), measures=np.array(meas), channels=np.array(chans),
                sums=sums.astype(np.float32), counts=counts, n_segments=n_seg,
                masked_frac_mean=float(np.mean(masked_frac)) if masked_frac else np.nan)
