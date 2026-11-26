#!/usr/bin/env python
# 来自 S2R 505b298 的 scripts/s06_features_classical.py，2026-11-26 复制（机械改名：S2R_→PRJ_、包名 s2r→项目名）
"""S6：经典特征（总纲 S6；算法见 s2r/features/classical.py，参数见配置 features_classical: 一节）。

用法（workflows/s2r-s06-classical.yaml）:
    PRJ_STEP = scripts/s06_features_classical.py
    PRJ_DATA_IN = data/interim/d1          # h5（约 9 GB）+ S4 targets + S5 qc（方向性检查用）

输入:  $PRJ_DATA/interim/d1/epochs_pre/sub-XXX_pre.h5（配置 windows 对应的 clean_*：W1、W05、W2、W4；master）
       $PRJ_DATA/interim/d1/targets/、qc/（只用于报告里的方向性检查，不进特征）
输出:  $PRJ_DATA/features/d1/classical/（或 --out）
         sub-XXX_classical.parquet  每个对齐试验一行（不按 QC 删；S8 用 qc 的 keep_W3 过滤）
         sub-XXX_psd.npz            1–45 Hz 完整 PSD（各导联、C4 拉普拉斯；每个窗口一份），E102 等可本地重算频段
         _s06_<运行编号>_report.md / _summary.tsv / _direction.tsv / _config.json
         figures/s06_psd.png        每人全脑平均 PSD 与 C4 拉普拉斯 PSD（W1，保留试验平均）
         figures/s06_direction.png  关键特征与 MEP（mep_resid）的被试内 Spearman 相关，每人一个点
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
from pvb.features import classical as cl  # noqa: E402
from pvb.io import d1  # noqa: E402

RUN_ID = os.environ.get("PRJ_RUN_ID", "local")

# 方向性检查（总纲 S6 验收）：这些特征与 mep_resid 的被试内 Spearman
DIRECTION_FEATURES = ["loc_W1_mu", "loc_W1_beta", "glob_W1_mu", "glob_W1_beta", "glob_W1_gamma",
                      "slope_W1", "mu_amp", "mu_cos", "mu_sin"]


def direction_check(feat: pd.DataFrame, d1_dir: Path, sub: str) -> dict | None:
    """在 S5 保留（keep_W3）的试验上，算特征与 mep_resid 的 Spearman。缺 qc/targets 时跳过。"""
    from scipy import stats

    qf = d1_dir / "qc" / f"sub-{sub}_qc.parquet"
    if not qf.exists():
        return None
    q = pd.read_parquet(qf)[["aligned_idx", "keep_W3", "mep_resid"]]
    m = feat.merge(q, on="aligned_idx", how="inner")
    m = m[m["keep_W3"]]
    r = {"subject": f"sub-{sub}", "n_keep": len(m)}
    for c in DIRECTION_FEATURES:
        r[c] = round(float(stats.spearmanr(m[c], m["mep_resid"])[0]), 3)
    # mu 相位：波谷（|phase| > 3π/4）与波峰（|phase| < π/4）的平均 mep_resid 之差
    ph = m["mu_phase"].abs()
    r["trough_minus_peak"] = round(float(m.loc[ph > 3 * np.pi / 4, "mep_resid"].mean()
                                         - m.loc[ph < np.pi / 4, "mep_resid"].mean()), 3)
    return r


def figures(psds: dict, keeps: dict, direc: pd.DataFrame, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for s, pk in psds.items():
        k = keeps.get(s)
        sel = np.isin(pk["aligned_idx"], k) if k is not None else slice(None)
        f = pk["freqs"]
        axs[0].semilogy(f, pk["psd_W1"][sel].mean(axis=(0, 1)), lw=0.8)
        axs[1].semilogy(f, pk["psdlap_W1"][sel].mean(axis=0), lw=0.8, label=s.replace("sub-", ""))
    axs[0].set_title("W1 global mean PSD (kept trials)", fontsize=9)
    axs[1].set_title("W1 C4 Hjorth Laplacian PSD", fontsize=9)
    for ax in axs:
        ax.set_xlabel("Hz")
        ax.axvspan(8, 13, color="0.9")
    axs[1].legend(fontsize=5, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "s06_psd.png", dpi=130)
    plt.close(fig)

    if len(direc):
        cols = DIRECTION_FEATURES + ["trough_minus_peak"]
        fig, ax = plt.subplots(figsize=(10, 4))
        for i, c in enumerate(cols):
            v = direc[c].to_numpy()
            ax.plot(np.full(len(v), i) + np.random.default_rng(i).uniform(-0.15, 0.15, len(v)), v, ".", color="0.4")
            ax.plot([i - 0.3, i + 0.3], [np.median(v)] * 2, "-", color="C3", lw=2)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=30, fontsize=7)
        ax.set_title("within-subject Spearman(feature, mep_resid) on keep_W3 trials; red = median; "
                     "last = mean resid trough − peak", fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "s06_direction.png", dpi=130)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects", nargs="*", help="如 283 711；默认 epochs_pre 里全部 h5")
    ap.add_argument("--out", help="输出目录（默认 $PRJ_DATA/features/d1/classical）")
    args = ap.parse_args()

    from scipy import stats

    data = Path(os.environ["PRJ_DATA"])
    cfg = d1.load_config(CODE)
    fc = cfg["features_classical"]
    d1_dir = data / "interim" / "d1"
    out = Path(args.out) if args.out else data / "features" / "d1" / "classical"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    subs = args.subjects or sorted(p.name[4:7] for p in (d1_dir / "epochs_pre").glob("sub-*_pre.h5"))
    print(f"[S6] run={RUN_ID} git={os.environ.get('PRJ_GIT', '?')} subjects={len(subs)} out={out}", flush=True)

    summ, direc, psds, keeps, infos, errors = [], [], {}, {}, {}, {}
    for s in subs:
        try:
            cols, pk, info = cl.subject_features(d1_dir / "epochs_pre" / f"sub-{s}_pre.h5", fc)
            df = pd.DataFrame(cols)
            df.insert(0, "subject", f"sub-{s}")
            if not np.isfinite(df.drop(columns=["subject"]).to_numpy(dtype=float)).all():
                raise ValueError("特征里有 NaN/Inf")
            df.to_parquet(out / f"sub-{s}_classical.parquet", index=False)
            np.savez_compressed(out / f"sub-{s}_psd.npz", **pk)
            infos[f"sub-{s}"] = info
            psds[f"sub-{s}"] = pk
            r = direction_check(df, d1_dir, s)
            if r:
                direc.append(r)
                q = pd.read_parquet(d1_dir / "qc" / f"sub-{s}_qc.parquet")
                keeps[f"sub-{s}"] = q.loc[q["keep_W3"], "aligned_idx"].to_numpy()
            pk_f = pk["freqs"]
            gm = pk["psd_W1"].mean(axis=(0, 1))
            summ.append({"subject": f"sub-{s}", "n_trials": info["n"], "n_features": df.shape[1] - 4,
                         "alpha_peak_hz_global": float(pk_f[(pk_f >= 6) & (pk_f <= 14)][np.argmax(gm[(pk_f >= 6) & (pk_f <= 14)])]),
                         "slope_W1_median": round(float(df["slope_W1"].median()), 2),
                         "loc_W1_mu_median": round(float(df["loc_W1_mu"].median()), 2),
                         "phase_check_err_median_deg": info["phase_check_err_median_deg"],
                         "phase_check_err_lowamp_deg": round(float(np.degrees(np.median(
                             df.loc[df["phase_check_amp"] < df["phase_check_amp"].median(), "phase_check_err"]))), 1),
                         "phase_check_err_highamp_deg": round(float(np.degrees(np.median(
                             df.loc[df["phase_check_amp"] >= df["phase_check_amp"].median(), "phase_check_err"]))), 1),
                         "phase_uniformity_R": round(float(np.abs(np.exp(1j * df["mu_phase"]).mean())), 3)})
            print(f"[S6] sub-{s} OK  n={info['n']} 特征={df.shape[1] - 4} 相位自检误差中位数={info['phase_check_err_median_deg']}°",
                  flush=True)
        except Exception:  # noqa: BLE001
            errors[s] = traceback.format_exc()
            print(f"[S6] sub-{s} FAIL\n{errors[s]}", flush=True)

    tag = f"_s06_{RUN_ID}"
    sm, dr = pd.DataFrame(summ), pd.DataFrame(direc)
    if summ:
        figures(psds, keeps, dr, out / "figures")
        sm.to_csv(out / f"{tag}_summary.tsv", sep="\t", index=False)
    if len(dr):
        dr.to_csv(out / f"{tag}_direction.tsv", sep="\t", index=False)

    lines = [f"# S6 经典特征报告 {RUN_ID}", "", f"- git: {os.environ.get('PRJ_GIT', '?')}",
             f"- 成功 {len(summ)} / {len(subs)} 人；失败: {', '.join(errors) or '无'}",
             f"- 窗口 {fc['windows']}；频段 {fc['bands']}；Welch {fc['welch']}",
             f"- 局部：{fc['laplacian']['center']} − mean{fc['laplacian']['neighbors']}；1/f 拟合 {fc['aperiodic']}",
             f"- 相位：{fc['phase']}", ""]
    if len(sm):
        lines += ["## 每人概况", "",
                  "phase_check_err = 在 −1.0 s 处假装脉冲、只用其之前数据估计的相位与两侧数据真值的绝对误差中位数（度；"
                  "随机猜为 90°）；lowamp/highamp = mu 幅值低于/高于中位数的试验；phase_uniformity_R = 脉冲时刻相位的"
                  "平均合向量长度（TMS 与 mu 节律无关，应接近 0）", ""]
        lines += ["| " + " | ".join(sm.columns) + " |", "|" + "---|" * len(sm.columns)]
        lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in sm.itertuples(index=False)]
    if len(dr):
        lines += ["", "## 方向性检查（总纲 S6 验收）", "",
                  "S5 保留试验（keep_W3）上，特征与 mep_resid 的被试内 Spearman；组水平：中位数、正相关人数、Wilcoxon 双侧 p。"
                  "文献预期：刺激前 alpha/mu 功率高 → MEP 大（NeuroImage 2025）；mu 波谷时 MEP 较大（Zrenner 2018，"
                  "trough_minus_peak > 0、mu_cos 负相关）。这只是单变量方向检查，不是 E101 的交叉验证结果", ""]
        lines += ["| 特征 | 中位数 | 正/总 | p |", "|---|---|---|---|"]
        for c in DIRECTION_FEATURES + ["trough_minus_peak"]:
            v = dr[c].to_numpy()
            p = stats.wilcoxon(v)[1] if len(v) > 5 else np.nan
            lines.append(f"| {c} | {np.median(v):.3f} | {(v > 0).sum()}/{len(v)} | {p:.3g} |")
        lines += ["", "每人明细见 `" + f"{tag}_direction.tsv" + "`"]
    if errors:
        lines += ["", "## 错误", ""] + [f"### sub-{k}\n```\n{v}\n```" for k, v in errors.items()]
    (out / f"{tag}_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / f"{tag}_config.json").write_text(json.dumps({"run_id": RUN_ID, "git": os.environ.get("PRJ_GIT"),
                                                        "features_classical": fc, "info": infos},
                                                       ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"SUMMARY_S6 subjects_ok={len(summ)}/{len(subs)}")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
