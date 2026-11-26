# 来自 S2R 505b298 的 s2r/preprocess/qc.py，2026-11-26 复制（机械改名：S2R_→PRJ_、包名 s2r→项目名）
"""S5 质控：给每个试验打"保留 / 剔除（原因）"标记（总纲 S5；规则 = 2026-11-21 定的"方案 C"）。

输入（每个被试）
    epochs_pre/sub-XXX_trials.parquet  S3 每脉冲一行（aligned_idx < 0 = 与 EMG 对不上的多余脉冲，不进 QC 表）
    epochs_pre/sub-XXX_pre.h5          S3 的 clean_W3 / clean_W4（试验 × 28 导 × 采样点，µV）
    targets/sub-XXX_targets.parquet    S4 每试验一行（MEP、脉冲前肌电、wave_r）

剔除规则（rej_* 列，True = 剔除）
    rej_emg_invalid  S4 emg_valid == False（EMG 中断等）
    rej_guard_W*     S3 guard_ok_W* == False（窗口起点离上一脉冲 < 0.9 s，防泄漏 6.3 第 2 条）
    rej_preact       脉冲前 −25→−5 ms FDI 峰峰值 > pre_pp_max_uv
    rej_mep_mad      block 内 |mep_resid − 中位数| > k × 1.4826 × MAD
    rej_eeg_W*       clean_W* 里，除 eeg_ignore_channels 外任一导联峰峰值 > eeg_ptp_max_uv
    rej_badchan      该被试 S3 坏导占比 > max_bad_frac（被试级，所有试验同值）
    keep_W* = 以上（对应窗口）全为 False

只打标记的列（flag_*，不影响 keep）：flag_hussain_rms、flag_wave_r
眨眼指标：blink_ptp_W* = FP1/FP2 峰峰值的较大者（S8 可做"眨眼能否预测 MEP"的对照）
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

# 原因的先后顺序：画堆叠图时，一个试验只算在第一个命中的原因下
REASON_ORDER = ["emg_invalid", "badchan", "guard", "preact", "mep_mad", "eeg"]


def channel_ptp(h5_path: Path, windows: list[str]) -> tuple[dict[str, np.ndarray], list[str], np.ndarray, list[str]]:
    """读 h5，返回 {窗口: 试验×导联 峰峰值}、导联名、h5 里每行的 aligned_idx、坏导列表。"""
    out = {}
    with h5py.File(h5_path, "r") as f:
        ch = json.loads(f.attrs["ch_names"])
        bads = json.loads(f.attrs.get("bads", "[]"))
        ai = f["aligned_idx"][:]
        for w in windows:
            x = f[f"clean_{w}"][:]            # 600 × 28 × T，float32，一次读入约 50–70 MB
            out[w] = (x.max(axis=-1) - x.min(axis=-1)).astype(np.float32)
    return out, ch, ai, bads


def mad_outlier(x: np.ndarray, k: float, scale: float) -> np.ndarray:
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * scale
    if mad == 0:
        return np.zeros(len(x), dtype=bool)
    return np.abs(x - med) > k * mad


def subject_qc(sub: str, d1_dir: Path, qc: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """返回 (QC 表, 导联峰峰值长表, 信息)。sub 形如 '283'。"""
    s = f"sub-{sub}"
    tr = pd.read_parquet(d1_dir / "epochs_pre" / f"{s}_trials.parquet")
    tg = pd.read_parquet(d1_dir / "targets" / f"{s}_targets.parquet")
    windows = list(qc["windows"])

    # 1. S3 已对齐的试验 ⋈ S4 目标（必须逐个对上，否则报错）
    tr = tr[tr["aligned_idx"] >= 0].copy()
    keep_s3 = ["subject", "aligned_idx", "eeg_idx", "block"] + [f"guard_ok_{w}" for w in windows]
    keep_s4 = ["aligned_idx", "block", "trial_in_block", "emg_valid", "mep_raw_uv", "mep_resid",
               "pre_pp_uv", "pre_rms_uv", "mep_wave_r"]
    df = tr[keep_s3].merge(tg[keep_s4], on="aligned_idx", how="left", suffixes=("", "_s4"), indicator=True)
    n_miss = int((df["_merge"] != "both").sum())
    if n_miss:
        raise ValueError(f"{s}: {n_miss} 个 S3 试验在 S4 里找不到")
    if not (df["block"] == df["block_s4"]).all():
        raise ValueError(f"{s}: S3 与 S4 的 block 编号不一致")
    df = df.drop(columns=["_merge", "block_s4"]).sort_values("aligned_idx").reset_index(drop=True)
    df["emg_valid"] = df["emg_valid"].astype(bool)
    valid = df["emg_valid"].to_numpy()

    # 2. EMG / MEP 规则（只在有效试验上估计统计量）
    df["rej_emg_invalid"] = ~valid
    df["rej_preact"] = valid & (df["pre_pp_uv"] > qc["pre_pp_max_uv"]).to_numpy()
    mm = qc["mep_mad"]
    rej_mad = np.zeros(len(df), dtype=bool)
    for _, idx in df[valid].groupby("block").groups.items():
        idx = np.asarray(idx)
        rej_mad[idx] = mad_outlier(df.loc[idx, mm["column"]].to_numpy(), mm["k"], mm["scale"])
    df["rej_mep_mad"] = rej_mad

    fo = qc["flags_only"]
    rms = df.loc[valid, "pre_rms_uv"].to_numpy()
    q1, q3 = np.percentile(rms, [25, 75])
    rms_thr = float(q3 + fo["hussain_rms_iqr_k"] * (q3 - q1))
    df["flag_hussain_rms"] = valid & (df["pre_rms_uv"] > rms_thr).to_numpy()
    df["flag_wave_r"] = valid & (df["mep_wave_r"] < fo["wave_r_min"]).to_numpy()

    # 3. EEG 规则（clean 窗口逐导联峰峰值）
    ptp, ch, h5_ai, bads = channel_ptp(d1_dir / "epochs_pre" / f"{s}_pre.h5", windows)
    ok = h5_ai >= 0                                          # −1 = 与 EMG 对不上的多余脉冲（767/774/780），可能有多个
    pos = pd.Series(np.arange(len(h5_ai))[ok], index=h5_ai[ok])
    if not pos.index.is_unique:
        raise ValueError(f"{s}: h5 里 aligned_idx 有重复")
    rows = pos.reindex(df["aligned_idx"]).to_numpy()
    if np.isnan(rows.astype(float)).any():
        raise ValueError(f"{s}: 有试验在 h5 里找不到")
    rows = rows.astype(int)
    judge = np.array([c not in set(qc["eeg_ignore_channels"]) for c in ch])
    blink = np.array([c in set(qc["blink_channels"]) for c in ch])
    long = []
    for w in windows:
        p = ptp[w][rows]                                     # 本表试验 × 导联
        over = p > qc["eeg_ptp_max_uv"]
        df[f"rej_eeg_{w}"] = over[:, judge].any(axis=1)
        df[f"n_ch_over_{w}"] = over[:, judge].sum(axis=1)
        df[f"ptp_max_judged_{w}"] = p[:, judge].max(axis=1)
        df[f"blink_ptp_{w}"] = p[:, blink].max(axis=1) if blink.any() else np.nan
        df[f"rej_guard_{w}"] = ~df[f"guard_ok_{w}"].astype(bool)
        lw = pd.DataFrame(p, columns=ch)
        lw.insert(0, "aligned_idx", df["aligned_idx"].to_numpy())
        lw.insert(0, "window", w)
        long.append(lw)
    chptp = pd.concat(long, ignore_index=True)
    chptp.insert(0, "subject", s)

    bad_frac = len(bads) / len(ch)
    df["rej_badchan"] = bad_frac > qc["max_bad_frac"]

    # 4. 汇总：keep 与原因字符串
    for w in windows:
        cols = {"emg_invalid": "rej_emg_invalid", "badchan": "rej_badchan", "guard": f"rej_guard_{w}",
                "preact": "rej_preact", "mep_mad": "rej_mep_mad", "eeg": f"rej_eeg_{w}"}
        R = df[[cols[r] for r in REASON_ORDER]].to_numpy()
        df[f"keep_{w}"] = ~R.any(axis=1)
        df[f"reason_{w}"] = [";".join(r for r, hit in zip(REASON_ORDER, row) if hit) for row in R]
    df = df.drop(columns=[f"guard_ok_{w}" for w in windows])

    info = {"subject": s, "n_trials": len(df), "n_valid": int(valid.sum()), "ch_names": ch, "bads": bads,
            "bad_frac": round(bad_frac, 3), "hussain_rms_thr_uv": round(rms_thr, 3)}
    return df, chptp, info


# QC 表列说明（写进报告）
COLUMNS_DOC = {
    "aligned_idx / eeg_idx / block / trial_in_block": "试验编号（与 S3、S4 一致）",
    "keep_W3 / keep_W4": "True = 保留；W3 主分析，W4 敏感性分析",
    "reason_W3 / reason_W4": "剔除原因，分号连接；空 = 保留",
    "rej_*": "各条剔除规则是否命中",
    "flag_hussain_rms / flag_wave_r": "只打标记（Hussain 规则），不影响 keep",
    "n_ch_over_W* / ptp_max_judged_W*": "参与判定的导联中超阈值的个数 / 最大峰峰值（µV）",
    "blink_ptp_W*": "FP1/FP2 峰峰值的较大者（µV），眨眼指标",
}
