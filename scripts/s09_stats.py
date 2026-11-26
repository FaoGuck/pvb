#!/usr/bin/env python
# 来自 S2R 505b298 的 scripts/s09_stats.py，2026-11-26 复制（机械改名：S2R_→PRJ_、包名 s2r→项目名）
"""S9：P1 组水平统计（总纲 S9）。函数见 s2r/stats/group.py，要汇总哪些运行、哪些比较见 configs/stats/s09_p1.yaml。

用法（本地即可，几秒）:
    PRJ_OUT=D:/proj_LLM/proj/proj_PVB/out python scripts/s09_stats.py --cfg configs/stats/s09_p1.yaml
输入: $PRJ_OUT/runs/<运行>/metrics.csv（以及偏相关用的 preds.parquet）
输出: $PRJ_OUT/summary/
    p1_main.csv   主要预测器 × 目标：n、指标中位数与四分位距、被试水平显著人数（置换 p<0.05；以及被试间 BH-FDR q<0.05）、
                  提升（指标 − 置换均值）的中位数、自助 95% CI、单侧 Wilcoxon p、效应量（配对秩二列相关）、主表内 BH-FDR q
    p1_stats.csv  事先计划的配对比较（a − b 的提升之差，双侧）+ 偏相关（对 0 单侧）；同一目标内 BH-FDR
    p1_all.csv    all_from 里每个（预测器 × 窗口 × 池化 × 层）配置：对随机（单侧）、对同窗口经典（双侧）；同一目标内全部一起 BH-FDR
    p1_report.md  可读版
说明: 随机初始化的多个种子先对每个被试取平均（指标、置换均值都平均；被试显著 = 过半种子 p<0.05）；
      不同实验里完全相同的配置（如 E107 末层 = E106 的 W4）在 p1_all 里只保留一个
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CODE = Path(os.environ.get("PRJ_CODE", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(CODE))
from pvb.stats import group as gs  # noqa: E402

TARGETS = ["y_cont", "y_bin"]


def load_metrics(out: Path, runs: dict, pca: str) -> pd.DataFrame:
    parts = []
    for exp, rd in runs.items():
        m = pd.read_csv(out / "runs" / rd / "metrics.csv")
        m = m[m["pca"].astype(str) == str(pca)].copy()
        m["exp"] = exp
        m["layer"] = m["layer"].astype(str)
        m["gain"] = m["value"] - m["null_mean"]
        m["sig"] = (m["p_perm"] < 0.05).astype(float)
        parts.append(m)
    return pd.concat(parts, ignore_index=True)


def per_subject(rows: pd.DataFrame) -> pd.DataFrame:
    """(subject, target) → value、gain、sig（多个种子取平均）、p_perm（种子平均时取中位数，只用于被试间 FDR）。"""
    return rows.groupby(["subject", "target"], as_index=False).agg(
        value=("value", "mean"), gain=("gain", "mean"), sig=("sig", "mean"), p_perm=("p_perm", "median"),
        n_seeds=("seed", "nunique"))


def select(M: pd.DataFrame, spec: dict) -> pd.DataFrame:
    r = M[(M["exp"] == spec["exp"]) & (M["predictor"] == spec["predictor"])]
    if "layer" in spec:
        r = r[r["layer"] == str(spec["layer"])]
    if r.empty:
        raise ValueError(f"找不到 {spec}")
    if r["layer"].nunique() > 1:
        raise ValueError(f"{spec} 有多层，请在配置里写 layer")
    return per_subject(r)


def main_table(M, cfg) -> pd.DataFrame:
    rows = []
    for label, spec in cfg["main"].items():
        S = select(M, spec)
        for t in TARGETS:
            s = S[S["target"] == t]
            if s.empty:
                continue
            g = gs.summarize(s["gain"].to_numpy(), "greater", cfg["seed"])
            q_sub = gs.bh_fdr(s["p_perm"].to_numpy())
            rows.append({"label": label, "exp": spec["exp"], "target": t, "n": g["n"],
                         "value_median": s["value"].median(), "value_q1": s["value"].quantile(.25),
                         "value_q3": s["value"].quantile(.75),
                         "n_sig_perm": int((s["sig"] >= 0.5).sum()), "n_sig_fdr": int((q_sub < 0.05).sum()),
                         "gain_median": g["median"], "gain_ci_lo": g["ci_lo"], "gain_ci_hi": g["ci_hi"],
                         "n_gain_pos": g["n_pos"], "p_vs_chance": g["p"], "r_rb": g["r_rb"]})
    T = pd.DataFrame(rows)
    T["q_vs_chance"] = np.nan
    for t in TARGETS:
        k = T["target"] == t
        T.loc[k, "q_vs_chance"] = gs.bh_fdr(T.loc[k, "p_vs_chance"].to_numpy())
    return T


def paired(Sa: pd.DataFrame, Sb: pd.DataFrame, t: str) -> np.ndarray:
    a = Sa[Sa["target"] == t].set_index("subject")["gain"]
    b = Sb[Sb["target"] == t].set_index("subject")["gain"]
    j = a.index.intersection(b.index)
    return (a[j] - b[j]).to_numpy()


def stats_table(M, cfg, out: Path) -> pd.DataFrame:
    sel = {k: select(M, v) for k, v in cfg["main"].items()}
    rows = []
    for comp in cfg["comparisons"]:
        a, b = comp[0], comp[1]
        fam = comp[2] if len(comp) > 2 else "main"                      # P2：同一族内一起 FDR（07：同一条路的变体一起 FDR）
        for t in TARGETS:
            d = paired(sel[a], sel[b], t)
            if len(d) == 0:
                continue
            g = gs.summarize(d, "two-sided", cfg["seed"])
            rows.append({"kind": "paired", "family": fam, "a": a, "b": b, "target": t, **g})
    for pc in cfg.get("partial", []):
        ea, eh = cfg["main"][pc["eeg"]], cfg["main"][pc["history"]]
        P = []
        for spec, col in ((ea, "eeg"), (eh, "hist")):
            p = pd.read_parquet(out / "runs" / cfg["runs"][spec["exp"]] / "preds.parquet")
            p = p[(p["predictor"] == spec["predictor"]) & (p["target"] == "y_cont") & (p["pca"].astype(str) == str(cfg["pca"]))]
            P.append(p[["subject", "aligned_idx", "y_true", "y_pred"]].rename(columns={"y_pred": col}))
        d = P[0].merge(P[1].drop(columns="y_true"), on=["subject", "aligned_idx"])
        r = d.groupby("subject").apply(lambda g: gs.partial_spearman(g["y_true"].to_numpy(), g["eeg"].to_numpy(),
                                                                    g["hist"].to_numpy()), include_groups=False)
        g = gs.summarize(r.to_numpy(), "greater", cfg["seed"])
        rows.append({"kind": "partial", "family": pc.get("family", "main"), "a": pc["eeg"], "b": f"| {pc['history']}", "target": "y_cont", **g})
    T = pd.DataFrame(rows)
    T["q"] = np.nan
    for (t, fam), k in T.groupby(["target", "family"]).groups.items():
        T.loc[k, "q"] = gs.bh_fdr(T.loc[k, "p"].to_numpy())
    return T


def all_table(M, cfg) -> pd.DataFrame:
    if not cfg.get("all_from"):
        return pd.DataFrame(columns=["target", "p_vs_chance", "p_vs_classical", "q_vs_chance", "q_vs_classical"])
    keys = ["exp", "predictor", "window", "pool", "layer"]
    ref_exp = cfg["classical_ref"]["exp"]
    seen, rows = set(), []
    A = M[M["exp"].isin(cfg["all_from"])]
    for k, r in A.groupby(keys, sort=False):
        S = per_subject(r)
        for t in TARGETS:
            s = S[S["target"] == t].sort_values("subject")
            if s.empty:
                continue
            sig = (k[1], t, tuple(np.round(s["gain"].to_numpy(), 9)))
            if sig[1:] in seen:                                         # 与前面某个配置完全相同（跨实验重复）
                continue
            seen.add(sig[1:])
            g = gs.summarize(s["gain"].to_numpy(), "greater", cfg["seed"])
            row = {**dict(zip(keys, k)), "target": t, "n": g["n"], "value_median": s["value"].median(),
                   "gain_median": g["median"], "gain_ci_lo": g["ci_lo"], "gain_ci_hi": g["ci_hi"],
                   "p_vs_chance": g["p"], "n_sig_perm": int((s["sig"] >= 0.5).sum())}
            ref = M[(M["exp"] == ref_exp) & (M["predictor"] == f"classical_{k[2]}")]
            if not ref.empty and not str(k[1]).startswith("classical") and k[0] != "E110":
                d = paired(S, per_subject(ref), t)
                h = gs.summarize(d, "two-sided", cfg["seed"])
                row.update({"vs_classical": f"classical_{k[2]}", "diff_median": h["median"], "diff_ci_lo": h["ci_lo"],
                            "diff_ci_hi": h["ci_hi"], "n_better": h["n_pos"], "p_vs_classical": h["p"], "r_rb": h["r_rb"]})
            rows.append(row)
    T = pd.DataFrame(rows)
    for c in ("p_vs_chance", "p_vs_classical"):
        T[c.replace("p_", "q_")] = np.nan
        for t in TARGETS:
            k = T["target"] == t
            T.loc[k, c.replace("p_", "q_")] = gs.bh_fdr(T.loc[k, c].to_numpy())
    return T


def fmt(x, d=3):
    return "—" if pd.isna(x) else (f"{x:.2g}" if abs(x) < 0.001 and x != 0 else f"{x:.{d}f}")


def report(main: pd.DataFrame, st: pd.DataFrame, al: pd.DataFrame, path: Path, cfg: dict):
    run_list = ", ".join(f"{k}={v.split('/')[-1]}" for k, v in cfg["runs"].items())
    L = [f"# S9：{cfg.get('title', 'P1（D1）组水平统计')}", "",
         f"- 运行：{run_list}；PCA {cfg['pca']}；自助 {cfg['n_boot']} 次",
         "- 提升 = 指标 − 该被试置换分布均值；效应量 r = 配对秩二列相关；q = BH-FDR（同一目标、同一张表内）", ""]
    for t in TARGETS:
        L += [f"## 主要预测器（{t}）", "",
              "| 预测器 | n | 指标中位数（IQR） | 显著（置换 / FDR） | 提升中位数 [95% CI] | p | q | r |", "|---|---|---|---|---|---|---|---|"]
        for _, r in main[main["target"] == t].iterrows():
            L.append(f"| {r.label} | {r.n} | {fmt(r.value_median)}（{fmt(r.value_q1)}–{fmt(r.value_q3)}） | {r.n_sig_perm} / {r.n_sig_fdr} | "
                     f"{r.gain_median:+.3f} [{r.gain_ci_lo:+.3f}, {r.gain_ci_hi:+.3f}] | {fmt(r.p_vs_chance)} | {fmt(r.q_vs_chance)} | {fmt(r.r_rb, 2)} |")
        L += ["", f"## 计划比较（{t}；q 在同一 FDR 族内计算）", "", "| 族 | 比较 | n | 差值中位数 [95% CI] | a 更好 | p | q | r |",
              "|---|---|---|---|---|---|---|---|"]
        for _, r in st[st["target"] == t].iterrows():
            lab = f"{r.a} − {r.b}" if r.kind == "paired" else f"偏相关 {r.a} {r.b}"
            L.append(f"| {r.family} | {lab} | {r.n} | {r['median']:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] | {r.n_pos}/{r.n} | {fmt(r.p)} | {fmt(r.q)} | {fmt(r.r_rb, 2)} |")
        if not len(al):
            L.append("")
            continue
        a = al[al["target"] == t]
        hv = a.dropna(subset=["p_vs_classical"])
        L += ["", f"## 全部配置（{t}）：{len(a)} 个", "",
              f"- 对随机 q<0.05：{int((a['q_vs_chance'] < 0.05).sum())} 个",
              f"- 与同窗口经典比（{len(hv)} 个模型配置）：q<0.05 且更好 {int(((hv['q_vs_classical'] < 0.05) & (hv['diff_median'] > 0)).sum())} 个；"
              f"q<0.05 且更差 {int(((hv['q_vs_classical'] < 0.05) & (hv['diff_median'] < 0)).sum())} 个；"
              f"未校正 p<0.05 且更好 {int(((hv['p_vs_classical'] < 0.05) & (hv['diff_median'] > 0)).sum())} 个",
              "- 与经典差值最大的 5 个模型配置：", ""]
        for _, r in hv.sort_values("diff_median", ascending=False).head(5).iterrows():
            L.append(f"  - {r.exp} {r.predictor} {r.window} {r['pool']} L{r.layer}：{r.diff_median:+.3f} "
                     f"[{r.diff_ci_lo:+.3f}, {r.diff_ci_hi:+.3f}]，p={fmt(r.p_vs_classical)}，q={fmt(r.q_vs_classical)}")
        L.append("")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cfg", default="configs/stats/s09_p1.yaml")
    ap.add_argument("--out-root", help="默认 $PRJ_OUT")
    args = ap.parse_args()
    out = Path(args.out_root or os.environ.get("PRJ_OUT", CODE.parent / "out"))
    cfg = yaml.safe_load(open(CODE / args.cfg, encoding="utf-8"))
    gs_boot = cfg.get("n_boot", 10000)
    gs.boot_median_ci.__defaults__ = (gs_boot, 0.95, cfg.get("seed", 0))
    M = load_metrics(out, cfg["runs"], cfg["pca"])
    main_t, st, al = main_table(M, cfg), stats_table(M, cfg, out), all_table(M, cfg)
    sd = out / "summary"
    sd.mkdir(parents=True, exist_ok=True)
    pre = cfg.get("prefix", "p1")
    main_t.to_csv(sd / f"{pre}_main.csv", index=False)
    st.to_csv(sd / f"{pre}_stats.csv", index=False)
    if len(al):
        al.to_csv(sd / f"{pre}_all.csv", index=False)
    report(main_t, st, al, sd / f"{pre}_report.md", cfg)
    print(f"[S9] main={len(main_t)} stats={len(st)} all={len(al)} → {sd}")


if __name__ == "__main__":
    main()
