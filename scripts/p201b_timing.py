#!/usr/bin/env python
"""P201b：D5 刺激时间零点的诊断（P201 的追加检查）。

P201 发现：被刺激的两个电极上 |偏离| 最大的时刻稳定在 ~30 ms（512 / 2048 Hz 都是），且脉冲前这两个通道几乎是平的
（偏离 / 基线比 ~1e12）。怀疑是放大器在刺激时把被刺激通道断开 / 钳住、~30 ms 后才接回，跳变最大，
所以不能用被刺激电极判断对齐。这里改用**没被刺激的邻近电极**：
  1. 每个 run 取均匀分布的 n 个脉冲；对每个脉冲，找离刺激对最近的 4 个 good 记录电极
  2. 在 -20…+20 ms 内找伪迹起点：跳变最大的邻近电极上，|一阶差分| 第一次达到其最大值一半的时刻 → 每个 run 的中位数应在 0 附近（±1 个样本）
  3. 被刺激电极：从脉冲前 100 ms 起，第一次“变平”（相邻样本差 < 1e-3 µV 持续 ≥ 3 ms）和“恢复”的时刻
  4. 每名患者第一个 run 画一张图：被刺激电极 + 4 个邻近电极的单试验叠加与平均（-100…+200 ms）
用法：PRJ_STEP = scripts/p201b_timing.py，PRJ_ARGS = --config configs/datasets/p_d5_ds004080.yaml --n-pulses 30
输出：$PRJ_OUT/runs/P201b_d5_timing/<时间>_<git7>/  runs.tsv、report.md、figures/<run>.png、log.txt、DONE
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

EXP_ID = "P201b_d5_timing"


def flat_runs(x: np.ndarray, sf: float, min_ms: float = 3.0, tol: float = 1e-3):
    """x (n_t,) → [(起点, 终点)] 样本区间：相邻样本差 < tol 持续 ≥ min_ms。"""
    f = np.abs(np.diff(x)) < tol
    out, i, n = [], 0, len(f)
    while i < n:
        if f[i]:
            j = i
            while j < n and f[j]:
                j += 1
            if (j - i) / sf * 1000 >= min_ms:
                out.append((i, j))
            i = j
        else:
            i += 1
    return out


def timing_run(events_tsv: str, cfg: dict, root: str, n_pulses: int, fig_dir: str, plot: bool) -> dict:
    meta = io_d5.read_run_meta(events_tsv)
    good = io_d5.good_channels(meta, cfg["channels"]["types"], cfg["channels"]["status_good"])
    rd = io_d5.RawReader(meta)
    sf = meta.sfreq
    el = io_d5.read_electrodes(root, meta.subject)
    pre, post = int(round(0.1 * sf)), int(round(0.2 * sf))
    tt = (np.arange(pre + post) - pre) / sf * 1000
    st = meta.stim
    sel = st.iloc[np.linspace(0, len(st) - 1, min(n_pulses, len(st))).astype(int)]
    lat_edge, flat_on, flat_off, segs = [], [], [], []
    for _, ev in sel.iterrows():
        e1, e2 = io_d5.pair_electrodes(ev["pair"])
        if e1 not in el.index or e2 not in el.index:
            continue
        mid = el.loc[[e1, e2], ["x", "y", "z"]].to_numpy(float).mean(axis=0)
        cand = [c for c in good if c not in (e1, e2) and c in el.index]
        d = np.linalg.norm(el.loc[cand, ["x", "y", "z"]].to_numpy(float) - mid, axis=1)
        near = [cand[i] for i in np.argsort(d)[:4] if np.isfinite(d[i])]
        names = [e1, e2] + near
        pk = rd.picks(names)
        if len(pk) < 3:
            continue
        seg = rd.segments(pk, np.array([ev["sample0"] - pre]), pre + post)[0]
        if not np.isfinite(seg).all():
            continue
        segs.append((names[: len(pk)], seg))
        w = (tt >= -20) & (tt <= 20)
        dd = np.abs(np.diff(seg[2:], axis=1))                  # 邻近电极
        dd = dd[:, w[1:]]
        r_ = int(np.argmax(dd.max(axis=1)))                 # 跳变最大的邻近电极
        i = int(np.argmax(dd[r_] >= 0.5 * dd[r_].max()))    # 伪迹起点 = 第一次达到最大跳变一半（上升沿）
        lat_edge.append(tt[1:][w[1:]][i])
        for c in range(min(2, seg.shape[0])):                 # 被刺激电极的“变平”区间
            for a, b in flat_runs(seg[c], sf):
                if tt[a] > -50:
                    flat_on.append(tt[a]); flat_off.append(tt[b]); break
    r = dict(run_id=meta.run_id, subject=meta.subject, sfreq=sf, n=len(lat_edge),
             edge_lat_med_ms=float(np.median(lat_edge)) if lat_edge else np.nan,
             edge_lat_iqr_ms=float(np.subtract(*np.percentile(lat_edge, [75, 25]))) if lat_edge else np.nan,
             edge_within_1sample=float(np.mean(np.abs(lat_edge) <= 1000 / sf + 1e-9)) if lat_edge else np.nan,
             stim_flat_frac=len(flat_on) / max(1, 2 * len(segs)),
             stim_flat_on_med_ms=float(np.median(flat_on)) if flat_on else np.nan,
             stim_flat_off_med_ms=float(np.median(flat_off)) if flat_off else np.nan)
    if plot and segs:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names0 = segs[0][0]
        same = [s for n, s in segs if n == names0]
        fig, ax = plt.subplots(len(names0), 2, figsize=(12, 2.0 * len(names0)), squeeze=False)
        for c, nm in enumerate(names0):
            for s in same:
                ax[c, 0].plot(tt, s[c], lw=0.5, alpha=0.5)
            ax[c, 0].set_ylabel(("STIM " if c < 2 else "") + nm, fontsize=8)
            allc = np.stack([s[c] for s in same]).mean(axis=0)
            ax[c, 1].plot(tt, allc, "k", lw=0.8)
            for a in ax[c]:
                a.axvline(0, color="r", lw=0.6); a.axvline(9, color="g", lw=0.5, ls="--"); a.set_xlim(-100, 200)
        ax[0, 0].set_title(f"{meta.run_id}  {sel.iloc[0]['pair']}  single trials (n={len(same)}), uV", fontsize=9)
        ax[0, 1].set_title("mean; red = 0 ms, green dashed = 9 ms (N1 search start)", fontsize=9)
        ax[-1, 0].set_xlabel("ms"); ax[-1, 1].set_xlabel("ms")
        fig.tight_layout()
        os.makedirs(fig_dir, exist_ok=True)
        fig.savefig(os.path.join(fig_dir, f"{meta.run_id}.png"), dpi=90)
        plt.close(fig)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/datasets/p_d5_ds004080.yaml")
    ap.add_argument("--subjects", default="")
    ap.add_argument("--n-pulses", type=int, default=30)
    ap.add_argument("--procs", type=int, default=0)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(CODE / a.config, encoding="utf-8"))
    subs = [s for s in a.subjects.split(",") if s] or cfg["subjects_gate"]
    root = os.path.join(os.environ["PRJ_DATA"], cfg["raw_dir"])
    d = runs.new_run_dir(Path(os.environ["PRJ_OUT"]), EXP_ID)
    runs.write_snapshot(d, {**cfg, "_args": vars(a), "_subjects": subs})
    log = runs.Log(d)
    files = [f for s in subs for f in io_d5.find_runs(root, s)]
    first = {s: next((f for f in files if os.path.basename(f).startswith(s + "_")), None) for s in subs}
    # 每名患者第一个 run 画图；另外 512 Hz 与 2048 Hz 各至少画一个
    plot_set = set(first.values())
    rows = []
    with ProcessPoolExecutor(max_workers=min(a.procs or cfg.get("procs", 8), max(1, len(files)))) as ex:
        futs = {ex.submit(timing_run, f, cfg, root, a.n_pulses, str(d / "figures"), f in plot_set or "run-06" in f): f
                for f in files}
        for fu in as_completed(futs):
            try:
                r = fu.result(); rows.append(r)
                log(f"OK  {r['run_id']}  sf={r['sfreq']:g}  邻近电极最陡边沿 {r['edge_lat_med_ms']:.2f} ms（≤1 样本 {r['edge_within_1sample']:.2f}）"
                    f"  刺激电极变平 {r['stim_flat_frac']:.2f}：{r['stim_flat_on_med_ms']:.1f} → {r['stim_flat_off_med_ms']:.1f} ms")
            except Exception as e:
                rows.append(dict(run_id=os.path.basename(futs[fu]), error=repr(e)))
                log(f"ERR {os.path.basename(futs[fu])}: {e!r}\n{traceback.format_exc()}")
    df = pd.DataFrame(rows).sort_values("run_id")
    df.to_csv(d / "runs.tsv", sep="\t", index=False)
    n_err = int(df["error"].notna().sum()) if "error" in df else 0
    rep = f"# {EXP_ID}\n\n{df.to_string(index=False)}\n"
    (d / "report.md").write_text(rep, encoding="utf-8")
    log(rep)
    log(f"SUMMARY exp={EXP_ID} ok={len(df) - n_err}/{len(files)}")
    if n_err == 0 and files:
        runs.mark_done(d)


if __name__ == "__main__":
    main()
