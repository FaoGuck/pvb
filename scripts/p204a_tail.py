#!/usr/bin/env python
"""P204a：确定 T_tail——刺激引起的锁时成分多久消失（00 第 4 节 P204：观测段从“上一个脉冲 + T_tail”开始）。

P202 的 tail_s（整流平均波形最后一次超过 3 SD）有假阳性：远处无 N1 的电极也有 30–35 % > 1 s，且只看到 +3 s。
这里改用无偏的锁时功率：
  每个 刺激对 × 记录电极：试验 x_k(t)（减去各自脉冲前 -0.5…-0.1 s 的中位数，即“上一个间隔末尾”的水平）
  m(t) = 试验均值，v(t) = 试验间方差（N-1），N = 试验数
  锁时功率 p(t) = m(t)^2 − v(t)/N     （E[m^2] = μ^2 + σ^2/N，所以 p 是 μ^2 的无偏估计；没有锁时成分时期望为 0）
  R(t) = p 在 100 ms 格内的均值 ÷ 脉冲前窗口的平均方差    （锁时成分相对背景的“功率信噪比”）
按距离分组（< 20 mm 近处、20–40 mm、≥ 40 mm 远处 = 对照）取每格的中位数。

**T_tail 规则（2026-11-26 运行前写死，见 00 第 14 节）**：对每名患者，取最早的 t（0.5 s 网格，≥ 1.0 s），
使得 t 之后每个 100 ms 格的近处组中位 R < 0.05（锁时幅度 < 背景 SD 的 ≈ 22 %）；T_tail = 5 名患者中的最大值。
另外：被刺激电极在 +0…+4.9 s 每格的饱和比例（|x| ≥ 3000 µV）。

用法：PRJ_STEP = scripts/p204a_tail.py，PRJ_ARGS = --config configs/datasets/p_d5_ds004080.yaml
输出：$PRJ_OUT/runs/P204a_d5_tail/<时间>_<git7>/  curves.tsv、stim_sat.tsv、t_tail.tsv、report.md、figures/<sub>.png、DONE
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CODE = Path(os.environ["PRJ_CODE"])
sys.path.insert(0, str(CODE))
from pvb.probing import runs  # noqa: E402
from pvb.t2 import io_d5  # noqa: E402

EXP_ID = "P204a_d5_tail"
TMIN, TMAX = -0.5, 4.9
PRE = (-0.5, -0.1)
BIN = 0.1
GROUPS = [("near_lt20", 0, 20), ("mid_20_40", 20, 40), ("far_ge40", 40, 1e9)]
R_THR = 0.05
GRID = np.arange(1.0, 4.51, 0.5)


def run_curves(events_tsv: str, cfg: dict, root: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta = io_d5.read_run_meta(events_tsv)
    good = io_d5.good_channels(meta, cfg["channels"]["types"], cfg["channels"]["status_good"])
    rd = io_d5.RawReader(meta)
    picks = rd.picks(good)
    chans = [rd.names[i] for i in picks]
    sf = meta.sfreq
    off = int(round(TMIN * sf)); n_samp = int(round((TMAX - TMIN) * sf))
    tt = (np.arange(n_samp) + off) / sf
    wpre = (tt > PRE[0]) & (tt < PRE[1])
    edges = np.arange(0.0, TMAX + 1e-9, BIN)
    bins = [(tt >= a) & (tt < a + BIN) for a in edges[:-1]]
    el = io_d5.read_electrodes(root, meta.subject)
    xyz = el.reindex(chans)[["x", "y", "z"]].to_numpy(float)
    thr = cfg["n1"]["max_abs_uv"]
    rows, srows = [], []
    for pair, g in meta.stim.groupby("pair"):
        if len(g) < cfg["stim"]["min_trials"]:
            continue
        e1, e2 = io_d5.pair_electrodes(pair)
        if not (e1 in good and e2 in good):
            continue
        tr = rd.segments(picks, g["sample0"].to_numpy() + off, n_samp)
        tr = tr[np.isfinite(tr).all(axis=(1, 2))]
        n = tr.shape[0]
        if n < cfg["stim"]["min_trials"]:
            continue
        stim_idx = [i for i, c in enumerate(chans) if c in (e1, e2)]
        for c in stim_idx:
            s = np.abs(tr[:, c, :]) >= thr
            srows.append([meta.subject, meta.run_id, pair, chans[c]] + [float(s[:, b].any(axis=1).mean()) for b in bins])
        x = tr - np.median(tr[:, :, wpre], axis=2, keepdims=True)
        m = x.mean(axis=0); v = x.var(axis=0, ddof=1)
        p = m ** 2 - v / n
        vb = v[:, wpre].mean(axis=1)
        mid = np.nanmean(el.reindex([e1, e2])[["x", "y", "z"]].to_numpy(float), axis=0)
        dist = np.linalg.norm(xyz - mid, axis=1)
        R = np.stack([p[:, b].mean(axis=1) for b in bins], axis=1) / np.maximum(vb, 1e-12)[:, None]
        for c in range(len(chans)):
            if c in stim_idx or not np.isfinite(dist[c]):
                continue
            rows.append([meta.subject, meta.run_id, pair, chans[c], dist[c], n] + R[c].tolist())
    cols_b = [f"{a:.1f}" for a in edges[:-1]]
    return (pd.DataFrame(rows, columns=["subject", "run_id", "stim_pair", "channel", "dist_mm", "n_trials"] + cols_b),
            pd.DataFrame(srows, columns=["subject", "run_id", "stim_pair", "stim_electrode"] + cols_b))


def t_tail_rule(curve: pd.Series) -> float:
    """curve：index = 格起点（秒，float），值 = 近处组中位 R。返回规则给出的 t（网格上；都不满足 → inf）。"""
    for t in GRID:
        after = curve[curve.index >= t - 1e-9]
        if len(after) and (after < R_THR).all():
            return float(t)
    return float("inf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/datasets/p_d5_ds004080.yaml")
    ap.add_argument("--subjects", default="")
    ap.add_argument("--procs", type=int, default=0)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(CODE / a.config, encoding="utf-8"))
    subs = [s for s in a.subjects.split(",") if s] or cfg["subjects_gate"]
    root = os.path.join(os.environ["PRJ_DATA"], cfg["raw_dir"])
    d = runs.new_run_dir(Path(os.environ["PRJ_OUT"]), EXP_ID)
    runs.write_snapshot(d, {**cfg, "_args": vars(a), "_subjects": subs,
                            "_rule": dict(R_THR=R_THR, GRID=GRID.tolist(), BIN=BIN, PRE=PRE, GROUPS=GROUPS)})
    log = runs.Log(d)
    files = [f for s in subs for f in io_d5.find_runs(root, s)]
    log(f"{EXP_ID}: {len(subs)} 名患者，{len(files)} 个 run")
    cur, sat, n_err = [], [], 0
    with ProcessPoolExecutor(max_workers=min(a.procs or cfg.get("procs", 8), max(1, len(files)))) as ex:
        futs = {ex.submit(run_curves, f, cfg, root): f for f in files}
        for fu in as_completed(futs):
            try:
                c, s = fu.result(); cur.append(c); sat.append(s)
                log(f"OK  {os.path.basename(futs[fu])}  {len(c)} 个 刺激对 × 电极")
            except Exception as e:
                n_err += 1
                log(f"ERR {os.path.basename(futs[fu])}: {e!r}\n{traceback.format_exc()}")
    cur = pd.concat(cur, ignore_index=True); sat = pd.concat(sat, ignore_index=True)
    bcols = [c for c in cur.columns if c[0].isdigit()]
    out, tt_rows = [], []
    for sub, g in cur.groupby("subject"):
        for name, lo, hi in GROUPS:
            gg = g[(g.dist_mm >= lo) & (g.dist_mm < hi)]
            if not len(gg):
                continue
            med = gg[bcols].median()
            for b in bcols:
                out.append(dict(subject=sub, group=name, t0_s=float(b), R_median=med[b],
                                R_q25=gg[b].quantile(0.25), R_q75=gg[b].quantile(0.75), n=len(gg)))
            if name == "near_lt20":
                tt_rows.append(dict(subject=sub, t_rule_s=t_tail_rule(pd.Series(med.values, index=[float(b) for b in bcols]))))
    curves = pd.DataFrame(out); tt = pd.DataFrame(tt_rows)
    curves.to_csv(d / "curves.tsv", sep="\t", index=False); sat.to_csv(d / "stim_sat.tsv", sep="\t", index=False)
    tt.to_csv(d / "t_tail.tsv", sep="\t", index=False)
    T = float(tt.t_rule_s.max()) if len(tt) else float("nan")
    sat_curve = sat.groupby("subject")[bcols].mean()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    (d / "figures").mkdir(exist_ok=True)
    for sub, g in curves.groupby("subject"):
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        for name, gg in g.groupby("group"):
            ax[0].plot(gg.t0_s + BIN / 2, gg.R_median, label=f"{name} (n={int(gg.n.iloc[0])})")
        ax[0].axhline(R_THR, color="k", ls="--", lw=0.8); ax[0].axhline(0, color="0.6", lw=0.5)
        ax[0].set_yscale("symlog", linthresh=0.01); ax[0].set_xlabel("s after pulse"); ax[0].set_ylabel("locked power SNR R (median)")
        tr = tt.set_index("subject").t_rule_s.get(sub, np.nan)
        ax[0].set_title(f"{sub}  rule t = {tr}"); ax[0].legend(fontsize=8)
        if sub in sat_curve.index:
            ax[1].plot([float(b) + BIN / 2 for b in bcols], sat_curve.loc[sub].values)
        ax[1].set_ylim(-0.02, 1.02); ax[1].set_xlabel("s after pulse"); ax[1].set_title("stimulated electrodes: fraction of trials saturated")
        fig.tight_layout(); fig.savefig(d / "figures" / f"{sub}.png", dpi=90); plt.close(fig)

    near = curves[curves.group == "near_lt20"].pivot(index="t0_s", columns="subject", values="R_median")
    far = curves[curves.group == "far_ge40"].pivot(index="t0_s", columns="subject", values="R_median")
    pick = [0.0, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 4.8]
    rep = [f"# {EXP_ID}", "",
           f"- 规则（运行前写死）：近处（< 20 mm）中位 R 在 t 之后每个 100 ms 格都 < {R_THR}；t 取 0.5 s 网格、≥ 1.0 s；T_tail = 各患者最大值",
           f"- 每名患者：{tt.set_index('subject').t_rule_s.to_dict()}",
           f"- **T_tail = {T} s** → 观测段 = 5.0 − T_tail − 0.01 = {5.0 - T - 0.01:.2f} s（G0-T2 ① 要求 ≥ 1 s）", "",
           "## 近处组中位 R（选定时刻）", near.reindex(pick, method="nearest").round(4).to_string(), "",
           "## 远处组（≥ 40 mm，对照）中位 R", far.reindex(pick, method="nearest").round(4).to_string(), "",
           "## 被刺激电极饱和比例（选定时刻）",
           sat_curve[[f"{t:.1f}" for t in [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 4.8]]].round(3).to_string()]
    (d / "report.md").write_text("\n".join(rep) + "\n", encoding="utf-8")
    log("\n".join(rep))
    log(f"SUMMARY exp={EXP_ID} ok={len(files) - n_err}/{len(files)} T_tail={T}")
    if n_err == 0 and files:
        runs.mark_done(d)


if __name__ == "__main__":
    main()
