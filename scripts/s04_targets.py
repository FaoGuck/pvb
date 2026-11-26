#!/usr/bin/env python
# 来自 S2R 505b298 的 scripts/s04_targets.py，2026-11-26 复制（机械改名：S2R_→PRJ_、包名 s2r→项目名）
"""S4：构造预测目标 MEP（总纲第 6 节 S4）。

用法（workflows/s2r-s04-targets.yaml）:
    PRJ_STEP = scripts/s04_targets.py
    PRJ_DATA_IN = data/raw/ds002094/sourcedata/emg     # 只拉 EMG（约 16 MB），不拉 EEG

输入:  $PRJ_DATA/raw/ds002094/sourcedata/emg/sub-XXX/XXX_00k.mat
输出:  $PRJ_DATA/interim/d1/targets/
         sub-XXX_targets.parquet   每个试验一行，用 aligned_idx 与 S3 的 trials 表对上
         _s04_<运行编号>_summary.tsv / _report.md
         figures/s04_mep_trend.png    每人 MEP 随试验序号的走势 + block 内滑动中位数
         figures/s04_mep_hist.png     每人去漂移后目标的分布（应近似正态）
列说明见 s2r/targets/mep.py 末尾。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(os.environ.get("PRJ_CODE", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(CODE))
from pvb.io import d1  # noqa: E402
from pvb.targets import mep  # noqa: E402

RUN_ID = os.environ.get("PRJ_RUN_ID", "local")


def figures(all_df: pd.DataFrame, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    subs = sorted(all_df["subject"].unique())
    nc = 5
    nr = int(np.ceil(len(subs) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(3.2 * nc, 2.2 * nr), squeeze=False)
    for ax, s in zip(axs.flat, subs):
        d = all_df[all_df["subject"] == s]
        ax.plot(d["aligned_idx"], d["mep_log"], ".", ms=1.5, color="0.55")
        for b, g in d.groupby("block"):
            ax.plot(g["aligned_idx"], g["mep_trend"], "-", lw=1, color="C3")
        for b in d["block"].unique()[1:]:
            ax.axvline(d.loc[d["block"] == b, "aligned_idx"].min(), color="0.85", lw=0.6)
        ax.set_title(s, fontsize=8)
        ax.tick_params(labelsize=6)
    for ax in list(axs.flat)[len(subs):]:
        ax.axis("off")
    fig.suptitle("ln MEP (µV) vs trial; red = block running median (detrend)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "s04_mep_trend.png", dpi=130)
    plt.close(fig)

    fig, axs = plt.subplots(nr, nc, figsize=(3.2 * nc, 2.0 * nr), squeeze=False)
    for ax, s in zip(axs.flat, subs):
        d = all_df[all_df["subject"] == s]
        ax.hist(d["mep_resid"], bins=30, color="C0", alpha=0.8)
        ax.set_title(s, fontsize=8)
        ax.tick_params(labelsize=6)
    for ax in list(axs.flat)[len(subs):]:
        ax.axis("off")
    fig.suptitle("mep_resid = ln MEP - block running median", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "s04_mep_hist.png", dpi=130)
    plt.close(fig)


def summarize(df: pd.DataFrame, info: dict) -> dict:
    from scipy import stats

    n_all = len(df)
    df = df[df["emg_valid"]]
    y, r = df["mep_log"].to_numpy(), df["mep_resid"].to_numpy()
    lag1 = lambda v: float(np.corrcoef(v[:-1], v[1:])[0, 1]) if len(v) > 2 else np.nan  # noqa: E731
    return {
        "subject": df["subject"].iloc[0], "n_emg": n_all, "n_valid": len(df), "n_flat": info["n_flat"],
        "n_blocks": df["block"].nunique(),
        "files": " ".join(info["files"]), "muscle": info["muscle_used"],
        "mep_median_uv": round(float(df["mep_raw_uv"].median()), 1),
        "mep_p10_uv": round(float(df["mep_raw_uv"].quantile(0.1)), 1),
        "mep_p90_uv": round(float(df["mep_raw_uv"].quantile(0.9)), 1),
        "frac_mep_gt50uv": round(float((df["mep_raw_uv"] > 50).mean()), 3),
        "pre_pp_median_uv": round(float(df["pre_pp_uv"].median()), 1),
        "frac_pre_pp_gt50uv": round(float((df["pre_pp_uv"] > 50).mean()), 3),
        "wave_r_median": round(float(df["mep_wave_r"].median()), 3),
        "frac_wave_r_lt0.4": round(float((df["mep_wave_r"] < 0.4).mean()), 3),
        "drift_spearman": round(float(stats.spearmanr(np.arange(len(y)), y)[0]), 3),
        "lag1_log": round(lag1(y), 3), "lag1_resid": round(lag1(r), 3),
        "skew_log": round(float(stats.skew(y)), 2), "skew_resid": round(float(stats.skew(r)), 2),
        "resid_sd": round(float(np.std(r)), 3),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects", nargs="*", help="如 283 711；默认全部")
    args = ap.parse_args()

    data = Path(os.environ["PRJ_DATA"])
    cfg = d1.load_config(CODE)
    tg = cfg["targets"]
    emg_root = data / tg["emg_dir"]
    out = data / "interim" / "d1" / "targets"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    subs = args.subjects or sorted(p.name[4:] for p in emg_root.glob("sub-*") if p.is_dir())
    print(f"[S4] run={RUN_ID} git={os.environ.get('PRJ_GIT', '?')} subjects={len(subs)} emg={emg_root}", flush=True)

    summ, frames, errors, infos = [], [], {}, {}
    for s in subs:
        try:
            df, info = mep.subject_targets(s, emg_root, tg)
            df.to_parquet(out / f"sub-{s}_targets.parquet", index=False)
            frames.append(df)
            infos[f"sub-{s}"] = info
            r = summarize(df, info)
            summ.append(r)
            print(f"[S4] sub-{s} OK  n={r['n_emg']} valid={r['n_valid']} blocks={r['n_blocks']} MEP中位数={r['mep_median_uv']}µV "
                  f"漂移rho={r['drift_spearman']} lag1 {r['lag1_log']}→{r['lag1_resid']}", flush=True)
        except Exception:  # noqa: BLE001
            errors[s] = traceback.format_exc()
            print(f"[S4] sub-{s} FAIL\n{errors[s]}", flush=True)

    tag = f"_s04_{RUN_ID}"
    sm = pd.DataFrame(summ)
    if frames:
        figures(pd.concat(frames, ignore_index=True), out / "figures")
        sm.to_csv(out / f"{tag}_summary.tsv", sep="\t", index=False)
    first = next(iter(infos.values()), {})
    lines = [f"# S4 预测目标报告 {RUN_ID}", "", f"- git: {os.environ.get('PRJ_GIT', '?')}",
             f"- 成功 {len(summ)} / {len(subs)} 人；失败: {', '.join(errors) or '无'}",
             f"- EMG 格式（第一个被试）: 采样率 {first.get('sfreq')} Hz，时间 {first.get('t_range_ms')} ms，"
             f"肌肉 {first.get('muscle_used')}",
             f"- 参数: MEP 窗 {tg['mep_window_ms']} ms，脉冲前窗 {tg['pre_window_ms']} ms，"
             f"去漂移窗 {tg['detrend_window']} 个试验（block 内）", "",
             "列含义：n_emg = EMG 文件里的试验数；n_valid = 有效试验数（排除 EMG 中断的全 0 试验，统计只用有效试验）；"
             "n_flat = 自动检出的全 0 试验数；mep_median/p10/p90 = MEP 峰峰值分位数；frac_mep_gt50uv = MEP > 50 µV 的比例；"
             "pre_pp = 脉冲前 25→5 ms 肌电峰峰值（预激活）；wave_r = 单次与个人平均波形相关；"
             "drift_spearman = ln MEP 与试验序号的相关（慢漂移）；lag1 = 相邻试验相关，去漂移前→后；"
             "skew = 偏度（接近 0 才近似正态）", ""]
    if len(sm):
        cols = [c for c in sm.columns if c != "files"]
        lines += ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in sm[cols].itertuples(index=False)]
    if errors:
        lines += ["", "## 错误", ""] + [f"### sub-{k}\n```\n{v}\n```" for k, v in errors.items()]
    (out / f"{tag}_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / f"{tag}_config.json").write_text(json.dumps({"run_id": RUN_ID, "git": os.environ.get("PRJ_GIT"),
                                                        "targets": tg, "emg_info": infos}, ensure_ascii=False,
                                                       indent=1), encoding="utf-8")
    print(f"SUMMARY_S4 subjects_ok={len(summ)}/{len(subs)}")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
