#!/usr/bin/env python
# 来自 S2R 505b298 的 scripts/s10_figures.py，2026-11-26 复制（机械改名：S2R_→PRJ_、包名 s2r→项目名）
"""S10：P1 论文图 F2–F5（总纲 8.2）。风格见 s2r/viz/style.py；用哪些运行沿用 configs/stats/s09_p1.yaml。

用法（本地，几秒）:
    PRJ_DATA=D:/proj_LLM/proj/proj_PVB/data PRJ_OUT=D:/proj_LLM/proj/proj_PVB/out python scripts/s10_figures.py
输出 $PRJ_OUT/figures/：每张图 .pdf + .png（300 dpi）+ 同名 .csv（图中画出的数值）
    F2_data_overview  刺激间隔分布、每人 MEP 大小、每人保留试验数
    F3_main_result    主要预测器的被试内预测力（相对置换均值的提升；每人一个点 + 组中位数），y_cont / y_bin 各一栏
    F4_window_layer   窗口长度扫描（中位数 + 自助 95% CI）、逐层探测（层号按深度归一化）
    F5_pairwise       三个模型对经典 W4 的逐人配对散点；以及上一个 MEP 对经典 W4
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CODE = Path(os.environ.get("PRJ_CODE", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "scripts"))
from pvb.stats import group as gs  # noqa: E402
from pvb.viz import style as st  # noqa: E402
from s09_stats import load_metrics, per_subject, select  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

YLAB = {"y_cont": "Spearman gain over permutation mean", "y_bin": "AUC gain over permutation mean"}


# ---------- F2 ----------
def fig2(data: Path, fd: Path):
    tr = pd.concat([pd.read_parquet(f) for f in glob.glob(str(data / "interim/d1/epochs_pre/sub-*_trials.parquet"))])
    qc = pd.concat([pd.read_parquet(f) for f in glob.glob(str(data / "interim/d1/qc/sub-*_qc.parquet"))])
    isi = tr["isi_prev_s"].dropna()
    isi = isi[isi < 10]                                             # 去掉 block 之间的长停顿
    k = qc[qc["keep_W3"].astype(bool)]
    mep = k.groupby("subject")["mep_raw_uv"].quantile([.25, .5, .75]).unstack()
    mep.columns = ["q1", "median", "q3"]
    keep = qc.groupby("subject")[["keep_W3", "keep_W4"]].sum()
    order = mep.sort_values("median").index
    fig, ax = plt.subplots(1, 3, figsize=(st.WIDTH_2COL, 52 * st.MM), gridspec_kw={"width_ratios": [1.1, 1.3, 1.3]})
    ax[0].hist(isi, bins=np.arange(4.0, 7.01, 0.05), color=st.COLORS["classical"], edgecolor="white", linewidth=0.4)
    ymax = ax[0].get_ylim()[1] * 1.18
    ax[0].set_ylim(0, ymax)
    for x, lab, yy in ((4.01, "W4 (4.01 s)", 0.99), (4.91, "W4 + 0.9 s guard", 0.90)):
        ax[0].axvline(x, color=st.INK2, lw=0.8, ls="--")
        ax[0].text(x + 0.04, ymax * yy, lab, fontsize=6.3, color=st.INK2, va="top")
    ax[0].set_xlim(3.9, 6.1)
    ax[0].set_xlabel("Inter-stimulus interval (s)")
    ax[0].set_ylabel("Pulses (20 subjects)")
    ax[0].set_title("a  ISI distribution", loc="left")
    x = np.arange(len(order))
    m = mep.loc[order]
    ax[1].vlines(x, m["q1"], m["q3"], color=st.COLORS["classical"], lw=1.2)
    ax[1].plot(x, m["median"], "o", color=st.INK, ms=3.5)
    ax[1].set_yscale("log")
    ax[1].set_yticks([200, 500, 1000, 2000, 5000])
    ax[1].set_yticklabels(["200", "500", "1000", "2000", "5000"])
    ax[1].minorticks_off()
    ax[1].set_ylabel("MEP amplitude (µV)")
    ax[1].set_title("b  MEP per subject (median, IQR; kept trials)", loc="left")
    kp = keep.loc[order]
    ax[2].plot(x, kp["keep_W3"], "o", color=st.INK, ms=3.5, label="kept, W3 (main)")
    ax[2].plot(x, kp["keep_W4"], "o", mfc="white", mec=st.INK2, ms=3.5, label="kept, W4 (+ guard)")
    ax[2].set_ylim(0, 620)
    ax[2].set_ylabel("Trials per subject")
    ax[2].set_title("c  Retention after QC", loc="left")
    ax[2].legend(loc="lower left")
    for a in ax[1:]:
        a.set_xticks(x)
        a.set_xticklabels([s.replace("sub-", "") for s in order], rotation=90, fontsize=6)
        a.set_xlabel("Subject (sorted by median MEP)")
    fig.tight_layout()
    out = mep.join(keep).reset_index().rename(columns={"q1": "mep_q1_uv", "median": "mep_median_uv", "q3": "mep_q3_uv"})
    out["isi_median_s"] = float(isi.median())
    out["isi_min_s"] = float(isi.min())
    st.save(fig, fd, "F2_data_overview", out)


# ---------- F3 ----------
F3_ITEMS = [  # (主表标签, 颜色键, 空心=随机, 横轴名)
    ("classical_W4", "classical", False, "Classical"),
    ("reve_W4", "reve", False, "REVE"),
    ("labram_W4", "labram", False, "LaBraM"),
    ("cbramod_W4", "cbramod", False, "CBraMod"),
    ("reve_c4_W4", "reve", False, "REVE\n(C4 pool)"),
    ("random_reve_W4", "reve", True, "REVE\nrandom"),
    ("random_labram_W4", "labram", True, "LaBraM\nrandom"),
    ("random_cbramod_W4", "cbramod", True, "CBraMod\nrandom"),
    ("eegnet_W4", "eegnet", False, "EEGNet\n(n=18)"),
    ("prev1", "history", False, "Prev. MEP"),
    ("classical_W4_prev1", "history", False, "Classical\n+ prev. MEP"),
]


def fig3(M, cfg, fd: Path):
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(2, 1, figsize=(st.WIDTH_2COL, 118 * st.MM), sharex=True)
    rows = []
    for j, t in enumerate(["y_cont", "y_bin"]):
        a = ax[j]
        a.axhline(0, color=st.AXIS, lw=0.8, zorder=0)
        for i, (lab, ck, hollow, name) in enumerate(F3_ITEMS):
            S = select(M, cfg["main"][lab])
            g = S[S["target"] == t]["gain"].to_numpy()
            c = st.COLORS[ck]
            xs = i + rng.uniform(-0.18, 0.18, len(g))
            if hollow:
                a.plot(xs, g, "o", mfc="white", mec=st.lighten(c, 0.2), mew=0.8, ms=3.2, zorder=2)
            else:
                face = c if lab != "classical_W4_prev1" else st.COLORS["classical"]
                a.plot(xs, g, "o", mfc=face, mec="white", mew=0.4, ms=3.4, zorder=2,
                       alpha=0.9 if ck != "history" or lab == "prev1" else 1)
                if lab == "classical_W4_prev1":
                    a.plot(xs, g, "o", mfc="none", mec=st.COLORS["history"], mew=0.8, ms=3.4, zorder=3)
            med = np.median(g)
            a.plot([i - 0.3, i + 0.3], [med, med], color=st.INK, lw=1.6, zorder=4, solid_capstyle="round")
            rows += [{"target": t, "predictor": lab, "subject_idx": k, "gain": v} for k, v in enumerate(g)]
            rows.append({"target": t, "predictor": lab, "subject_idx": "median", "gain": med})
        for x0 in (4.5, 8.5):                                         # 分组：预训练 | 随机 | 其他
            a.axvline(x0, color=st.GRID, lw=0.8, zorder=0)
        a.set_ylabel(YLAB[t])
        a.set_title(("a  Continuous MEP (y_cont)" if t == "y_cont" else "b  High vs low MEP (y_bin)"), loc="left")
        a.set_xticks(range(len(F3_ITEMS)))
        a.set_xticklabels([x[3] for x in F3_ITEMS], fontsize=6.5)
        a.tick_params(axis="x", labelbottom=True)
    fig.text(0.5, -0.01, "W4 (4 s pre-stimulus), last layer; dots = subjects, bars = group median; "
             "hollow = same architecture, random weights (5-seed mean)", ha="center", fontsize=6.5, color=st.INK2)
    fig.tight_layout()
    st.save(fig, fd, "F3_main_result", pd.DataFrame(rows))


# ---------- F4 ----------
def med_ci(g, seed=0):
    lo, hi = gs.boot_median_ci(g, n_boot=10000, seed=seed)
    return float(np.median(g)), lo, hi


def fig4(M, fd: Path, t: str = "y_cont"):
    fig, ax = plt.subplots(1, 2, figsize=(st.WIDTH_2COL, 62 * st.MM), gridspec_kw={"width_ratios": [1, 1.35]})
    rows = []
    wx = {"W05": 0.5, "W1": 1, "W2": 2, "W4": 4}
    series = [("classical", "E106b", lambda w: f"classical_{w}", ["W05", "W1", "W2", "W4"])] + \
             [(m, "E106", (lambda w, m=m: f"{m}_{w}"), ["W1", "W2", "W4"]) for m in ("reve", "labram", "cbramod")]
    off = {"classical": -0.03, "reve": -0.01, "labram": 0.01, "cbramod": 0.03}
    for key, exp, pname, ws in series:
        pts = []
        for w in ws:
            S = per_subject(M[(M["exp"] == exp) & (M["predictor"] == pname(w))])
            m_, lo, hi = med_ci(S[S["target"] == t]["gain"].to_numpy())
            pts.append((wx[w], m_, lo, hi))
            rows.append({"panel": "window", "series": key, "window": w, "median_gain": m_, "ci_lo": lo, "ci_hi": hi})
        p = np.array(pts)
        xs = p[:, 0] * 2 ** off[key]
        ax[0].vlines(xs, p[:, 2], p[:, 3], color=st.COLORS[key], lw=0.8, alpha=0.7)
        ax[0].plot(xs, p[:, 1], "-o", color=st.COLORS[key], ms=4, label=st.NAMES[key])
    ax[0].set_xscale("log", base=2)
    ax[0].set_xticks([0.5, 1, 2, 4])
    ax[0].set_xticklabels(["0.5", "1", "2", "4"])
    ax[0].set_xlabel("Pre-stimulus window (s)")
    ax[0].set_ylabel(YLAB[t])
    ax[0].axhline(0, color=st.AXIS, lw=0.8, zorder=0)
    ax[0].set_title("a  Window length (median, 95% bootstrap CI)", loc="left")
    ax[0].legend(loc="upper left")

    a = ax[1]
    S = per_subject(M[(M["exp"] == "E106b") & (M["predictor"] == "classical_W4")])
    cm, clo, chi = med_ci(S[S["target"] == t]["gain"].to_numpy())
    a.axhspan(clo, chi, color=st.lighten(st.COLORS["classical"], 0.9), lw=0, zorder=0)
    a.axhline(cm, color=st.COLORS["classical"], lw=1.0, zorder=1)
    a.text(0.0, cm + 0.004, "Classical W4 (median, 95% CI band)", color=st.INK2, fontsize=6.3, va="bottom", ha="left")
    rows.append({"panel": "layer", "series": "classical_W4", "layer": "na", "depth": np.nan, "median_gain": cm, "ci_lo": clo, "ci_hi": chi})
    for key, exp, pname, ls, lab in [("reve", "E107", "reve", "-", "REVE"), ("labram", "E107", "labram", "-", "LaBraM"),
                                     ("cbramod", "E107", "cbramod", "-", "CBraMod"),
                                     ("reve", "E107b", "reve_c4", ":", "REVE, C4 pool")]:
        r = M[(M["exp"] == exp) & (M["predictor"] == pname) & (M["target"] == t)]
        g = r.groupby(r["layer"].astype(int))["gain"].median().sort_index()
        depth = g.index / g.index.max()
        a.plot(depth, g.values, ls=ls, color=st.COLORS[key], lw=1.4 if ls == "-" else 1.6, label=lab)
        rows += [{"panel": "layer", "series": f"{exp}:{pname}", "layer": int(L), "depth": float(L / g.index.max()),
                  "median_gain": float(v)} for L, v in g.items()]
    a.axhline(0, color=st.AXIS, lw=0.8, zorder=0)
    a.set_xlabel("Relative depth (0 = input embedding, 1 = last layer)")
    a.set_title("b  Layer-wise probing (W4; median over subjects)", loc="left")
    a.set_ylim(top=chi + 0.07)
    a.legend(loc="upper right", ncol=4, fontsize=6.3, handlelength=1.6, columnspacing=1.0)
    fig.tight_layout()
    st.save(fig, fd, "F4_window_layer", pd.DataFrame(rows))


# ---------- F5 ----------
def fig5(M, cfg, fd: Path, t: str = "y_cont"):
    base = select(M, cfg["main"]["classical_W4"])
    base = base[base["target"] == t].set_index("subject")["gain"]
    panels = [("reve_W4", "reve", "REVE"), ("labram_W4", "labram", "LaBraM"), ("cbramod_W4", "cbramod", "CBraMod"),
              ("prev1", "history", "Prev. MEP")]
    fig, ax = plt.subplots(1, 4, figsize=(st.WIDTH_2COL, 50 * st.MM), sharex=True, sharey=True)
    rows = []
    lim = (-0.12, 0.42)
    for a, (lab, ck, name) in zip(ax, panels):
        S = select(M, cfg["main"][lab])
        y = S[S["target"] == t].set_index("subject")["gain"].reindex(base.index)
        a.plot(lim, lim, color=st.AXIS, lw=0.8, zorder=0)
        a.axhline(0, color=st.GRID, lw=0.6, zorder=0)
        a.axvline(0, color=st.GRID, lw=0.6, zorder=0)
        a.plot(base.values, y.values, "o", mfc=st.COLORS[ck], mec="white", mew=0.4, ms=4.2, zorder=2)
        d = y - base
        rho = pd.Series(base.values).corr(pd.Series(y.values), method="spearman")
        p = gs.wilcoxon_p(d.to_numpy(), "two-sided")
        a.text(0.97, 0.03, f"median diff {np.median(d):+.3f}\nabove diagonal {int((d > 0).sum())}/20\np = {p:.2g}, ρ = {rho:.2f}",
               transform=a.transAxes, va="bottom", ha="right", fontsize=6.2, color=st.INK2)
        a.set_xlim(lim)
        a.set_ylim(lim)
        a.set_aspect("equal")
        a.grid(False)
        a.set_title(name, loc="left")
        a.set_xlabel("Classical W4 gain")
        rows += [{"panel": lab, "subject": s, "classical_W4_gain": float(base[s]), "gain": float(y[s])} for s in base.index]
    ax[0].set_ylabel("Predictor gain (W4)")
    fig.tight_layout()
    st.save(fig, fd, "F5_pairwise", pd.DataFrame(rows))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cfg", default="configs/stats/s09_p1.yaml")
    args = ap.parse_args()
    data = Path(os.environ["PRJ_DATA"])
    out = Path(os.environ.get("PRJ_OUT", CODE.parent / "out"))
    cfg = yaml.safe_load(open(CODE / args.cfg, encoding="utf-8"))
    st.apply()
    fd = out / "figures"
    M = load_metrics(out, cfg["runs"], cfg["pca"])
    fig2(data, fd)
    fig3(M, cfg, fd)
    fig4(M, fd)
    fig5(M, cfg, fd)
    print(f"[S10] F2–F5 → {fd}")


if __name__ == "__main__":
    main()
