#!/usr/bin/env python
"""P206 第 2 步：冻结模型表征 vs 距离 / 经典基线（00 第 14 节 2026-11-27 锁定的判据；docs/06 第 3 节；框架 2026-11-27）。

每名患者（评估集、折、可用折、预测头、网格与 P205 / P205b **完全相同**，代码同一份 pvb/t2/baselines.cv_predict）：
  题 B（主）：P205b 的单位（全数据 N1 判为有），目标 log 幅度，岭回归；指标 = 外层预测拼接后的 Spearman
  次要：同一批预测的逐刺激对内 Spearman（刺激对内有 N1 的电极 ≥ within_pair_min，取中位数）
  题 A：P205 的单位，目标 N1 有无，逻辑回归；AUROC，另报按距离分段
  特征组合：基线 B0、B0+B2、B3（与 P205b 数字应一致，作交叉核对）；每个模型 M、B0+M、B3+M
    M = pvb/t2/fm.py 的 Mh（E_r、s、E_r⊙s、|E_r−s|，训练集内 PCA 到 n_pc 维）+ Ms（余弦、表征相似度的均值 / 最大值）
    表征只来自可用折 owner 的观测段（fm.EmbedPool），与 B2 同一规则
组水平（判据，锁定）：每个模型“B0+M − B0”的 Spearman 增量（主）与逐刺激对内 Spearman 增量（次要），组水平 Wilcoxon，
  **所有模型 × 两个指标（含随机初始化版本）一起 BH-FDR**。另报（不作判据）：B3+M − B3、题 A 的 AUROC 增量与分段。
不可评估的患者（非空折 < 5，00 第 14 节）列出、不进入组水平。

用法：PRJ_STEP = scripts/p206_eval.py   PRJ_ARGS = --models chronos,chronos-rand --subjects all --procs 32
输入：$PRJ_DATA/t2/d5/{ccep_labels, observation}（P202、P204）、$PRJ_DATA/<p206.fm_dir>/<模型>/（p206_extract.py）
输出：$PRJ_OUT/runs/P206_eval_d5/<时间>_<git7>/  metrics.csv（00 第 5.2 节长表）、subjects.tsv、group.tsv、
      preds_reg.parquet、preds_clf.parquet、report.md、DONE
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CODE = Path(os.environ["PRJ_CODE"])
sys.path.insert(0, str(CODE))
from pvb.probing import runs  # noqa: E402
from pvb.stats import group  # noqa: E402
from pvb.t2 import baselines as bl, fm, folds as fo  # noqa: E402

EXP_ID = "P206_eval_d5"
BASE_SETS = ["B0", "B0+B2", "B3"]
M_SETS = {"M": ["Mh", "Ms"], "B0+M": ["B0", "Mh", "Ms"], "B3+M": ["B0", "B1", "B2", "Mh", "Ms"]}
FS = {**bl.FEATURE_SETS, **M_SETS}
MAIN, SECOND = "spearman_amp_resp", "spearman_amp_within_pair"


def amp_target(winmin_uv) -> np.ndarray:                # 与 p205b_amplitude.py 相同
    return np.log(np.maximum(-np.asarray(winmin_uv, float), 1.0))


def ceiling_auc(u: pd.DataFrame) -> float:               # 与 p205_baselines.py 相同
    a = bl.safe_auc(u.n1_detected_even, -u.winmin_odd_uv.to_numpy())
    b = bl.safe_auc(u.n1_detected_odd, -u.winmin_even_uv.to_numpy())
    return float(np.nanmean([a, b]))


def bin_names(edges) -> list[str]:
    e = [int(x) if float(x).is_integer() else x for x in edges]
    return [f"auroc_lt{e[0]}"] + [f"auroc_{e[i]}_{e[i + 1]}" for i in range(len(e) - 1)] + [f"auroc_ge{e[-1]}"]


def load_subject(sub: str, cfg: dict, data: Path):
    """与 P205 / P205b 逐行相同的评估集、观测池与折。"""
    lab = pd.read_parquet(data / cfg["labels_dir"] / f"{sub}_labels.parquet")
    obs_dir = data / "t2" / "d5" / "observation"
    el = pd.read_parquet(obs_dir / f"{sub}_electrodes.parquet").set_index("NAME")
    thr = fo.adjacency_threshold(el[["x", "y", "z"]].to_numpy(float), cfg["folds"]["adjacent_factor"])
    keep = fo.eval_mask(lab, el, thr)
    u = lab[keep & np.isfinite(lab.dist_mm) & lab.winmin_uv.notna()].reset_index(drop=True)
    accs = [dict(np.load(f, allow_pickle=False)) for f in sorted(glob.glob(str(obs_dir / sub / "*_obs.npz")))]
    pool = bl.ObsPool(accs) if accs else None
    pairs = sorted(set(u.stim_pair) | (set(pool.owners) if pool else set()))
    fold_of = fo.assign_folds(pairs, cfg["folds"]["k"], cfg["folds"]["seed"])
    return u, pool, pairs, fold_of


def run_subject(sub: str, cfg: dict, data: Path, models: list[str]):
    t0 = time.time()
    pc = cfg["p206"]
    cb_clf, cb_reg = cfg["baselines"], {**cfg["baselines"], **cfg["amplitude"]}
    u, pool, pairs, fold_of = load_subject(sub, cfg, data)
    n_ne = len(set(fold_of[p] for p in u.stim_pair.unique()))
    min_ne = cfg["folds"].get("min_nonempty_folds", cfg["folds"]["k"])
    info = dict(subject=sub, n_units=len(u), n_pairs=u.stim_pair.nunique())
    if n_ne < min_ne:                                   # 2026-11-27 排除规则（00 第 14 节）
        return {**info, "note": f"不可评估：非空折 {n_ne} < {min_ne}"}, [], None, None, f"SKIP {sub} 非空折 {n_ne} < {min_ne}"
    r = u[u.n1_detected.astype(bool)].reset_index(drop=True)
    r = r.assign(amp_log=amp_target(r.winmin_uv))
    info["n_resp"] = len(r)
    if len(r) < cb_reg["min_units"]:
        return {**info, "note": f"有 N1 的单位 < {cb_reg['min_units']}"}, [], None, None, f"SKIP {sub} n_resp={len(r)}"
    y = u.n1_detected.to_numpy(bool)
    bins = bin_names(pc["dist_edges"])
    rows = []

    def add(model, fs, metric, value, n_pairs, n_targets):
        rows.append(dict(subject=sub, model=model, feature_set=fs, metric=metric, value=value, n_pairs=n_pairs,
                         n_targets=n_targets))

    def score(model, preds_r, preds_c, sets):
        for s in sets:
            add(model, s, MAIN, bl.safe_spearman(r.amp_log, preds_r[s]), r.stim_pair.nunique(), len(r))
            wp, n_wp = fm.within_pair_spearman(r, r.amp_log, preds_r[s], pc["within_pair_min"])
            add(model, s, SECOND, wp, n_wp, len(r))
            add(model, s, "auroc", bl.safe_auc(y, preds_c[s]), u.stim_pair.nunique(), len(u))
            for b, v in zip(bins, fm.auc_by_distance(y, preds_c[s], u.dist_mm, pc["dist_edges"])):
                add(model, s, b, v, u.stim_pair.nunique(), len(u))

    wp_c, n_wpc = fm.within_pair_spearman(r, r.winmin_odd_uv, r.winmin_even_uv, pc["within_pair_min"])
    add("ceiling", "split_half", MAIN, bl.safe_spearman(r.winmin_odd_uv, r.winmin_even_uv), r.stim_pair.nunique(), len(r))
    add("ceiling", "split_half", SECOND, wp_c, n_wpc, len(r))
    add("ceiling", "split_half", "auroc", ceiling_auc(u), u.stim_pair.nunique(), len(u))
    pr, _ = bl.cv_predict(r, fold_of, pool, BASE_SETS, cb_reg, target="amp_log", task="reg", source=u)
    pcl, _ = bl.cv_predict(u, fold_of, pool, BASE_SETS, cb_clf)
    score("baseline", pr, pcl, BASE_SETS)
    PR = r[["subject", "stim_pair", "channel", "dist_mm", "amp_log"]].assign(
        fold=r.stim_pair.map(fold_of), **{f"baseline|{s}": pr[s] for s in BASE_SETS})
    PC = u[["subject", "stim_pair", "channel", "dist_mm", "n1_detected"]].assign(
        fold=u.stim_pair.map(fold_of), **{f"baseline|{s}": pcl[s] for s in BASE_SETS})
    notes = []
    for m in models:
        files = sorted(glob.glob(str(data / pc["fm_dir"] / m / sub / "*_emb.npz")))
        if not files:
            notes.append(f"{m}: 没有表征文件"); continue
        ep = fm.EmbedPool([dict(np.load(f, allow_pickle=False)) for f in files])
        kw = dict(feature_sets=FS, extra=fm.make_extra(ep), pca_blocks=("Mh",), n_pc=int(pc["n_pc"]))
        prm, _ = bl.cv_predict(r, fold_of, pool, list(M_SETS), cb_reg, target="amp_log", task="reg", source=u, **kw)
        pcm, _ = bl.cv_predict(u, fold_of, pool, list(M_SETS), cb_clf, **kw)
        score(m, prm, pcm, list(M_SETS))
        PR = PR.assign(**{f"{m}|{s}": prm[s] for s in M_SETS})
        PC = PC.assign(**{f"{m}|{s}": pcm[s] for s in M_SETS})
    info.update(note="; ".join(notes), seconds=round(time.time() - t0, 1))
    v = {(x["model"], x["feature_set"], x["metric"]): x["value"] for x in rows}
    msg = (f"OK  {sub}  单位 {len(u)}（有 N1 {len(r)}）  题B 上限 {v[('ceiling', 'split_half', MAIN)]:.3f}  "
           f"B0 {v[('baseline', 'B0', MAIN)]:.3f}  B3 {v[('baseline', 'B3', MAIN)]:.3f}  "
           + "  ".join(f"{m}:B0+M {v[(m, 'B0+M', MAIN)]:.3f}" for m in models if (m, "B0+M", MAIN) in v)
           + f"  {info['seconds']} s" + (f"  [{info['note']}]" if info["note"] else ""))
    return info, rows, PR, PC, msg


def group_table(met: pd.DataFrame, models: list[str], alternative: str) -> pd.DataFrame:
    """每个模型的逐患者增量 → 组水平。判据族 = 全部模型 × {主, 次要} 的“B0+M − B0”，一起 BH-FDR。"""
    piv = met.pivot_table(index="subject", columns=["model", "feature_set", "metric"], values="value", aggfunc="first")
    comps = [("B0+M", "B0", True), ("B3+M", "B3", False), ("M", "B0", False)]
    metrics = [MAIN, SECOND] + [c for c in met.metric.unique() if c.startswith("auroc")]
    out = []
    for m in models:
        for ms, bs, crit in comps:
            for k in metrics:
                a, b = (m, ms, k), ("baseline", bs, k)
                if a not in piv or b not in piv:
                    continue
                d = (piv[a] - piv[b]).to_numpy(float)
                s = group.summarize(d, alternative)
                out.append(dict(model=m, compare=f"{ms} − {bs}", metric=k, family=crit and k in (MAIN, SECOND), **s))
    g = pd.DataFrame(out)
    if len(g):
        g["q_family"] = np.nan
        fam = g.family.to_numpy(bool)
        g.loc[fam, "q_family"] = group.bh_fdr(g.loc[fam, "p"].to_numpy(float))
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/datasets/p_d5_ds004080.yaml")
    ap.add_argument("--models", required=True)
    ap.add_argument("--subjects", default="", help="逗号分隔；空 = 闸门 5 人；all = 有 P202 标签的全部患者")
    ap.add_argument("--procs", type=int, default=5)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(CODE / a.config, encoding="utf-8"))
    data = Path(os.environ["PRJ_DATA"])
    if a.subjects == "all":
        subs = sorted(Path(f).name[: -len("_labels.parquet")] for f in glob.glob(str(data / cfg["labels_dir"] / "*_labels.parquet")))
    else:
        subs = [s for s in a.subjects.split(",") if s] or cfg["subjects_gate"]
    models = [m for m in a.models.split(",") if m]
    d = runs.new_run_dir(Path(os.environ["PRJ_OUT"]), EXP_ID)
    runs.write_snapshot(d, {**cfg, "_args": vars(a), "_subjects": subs})
    log = runs.Log(d)
    log(f"{EXP_ID}: 模型 {models}；{len(subs)} 名患者；n_pc {cfg['p206']['n_pc']}")
    I, M, PRs, PCs = [], [], [], []
    with ProcessPoolExecutor(max_workers=max(1, min(a.procs, len(subs)))) as ex:
        futs = {ex.submit(run_subject, s, cfg, data, models): s for s in subs}
        for fu in as_completed(futs):
            try:                                        # 逐患者容错
                info, rows, pr, pcl, msg = fu.result()
            except Exception as e:  # noqa: BLE001
                info, rows, pr, pcl, msg = dict(subject=futs[fu], note=f"FAIL {e!r}"[:300]), [], None, None, f"FAIL {futs[fu]} {e!r}"
            I.append(info); M += rows; log(msg)
            if pr is not None:
                PRs.append(pr); PCs.append(pcl)
    info = pd.DataFrame(I).sort_values("subject")
    info.to_csv(d / "subjects.tsv", sep="\t", index=False)
    met = pd.DataFrame(M)
    if len(met):
        long = met.assign(exp_id=EXP_ID, run_id=os.environ["PRJ_RUN_ID"], git_commit=runs.git_commit(), track="T2",
                          dataset=cfg["dataset"], fold="all", stim_pair="all", seed=cfg["folds"]["seed"])
        long[["exp_id", "run_id", "git_commit", "track", "dataset", "subject", "model", "feature_set", "fold", "stim_pair",
              "metric", "value", "n_pairs", "n_targets", "seed"]].to_csv(d / "metrics.csv", index=False)
        pd.concat(PRs, ignore_index=True).to_parquet(d / "preds_reg.parquet", index=False)
        pd.concat(PCs, ignore_index=True).to_parquet(d / "preds_clf.parquet", index=False)
    n_ev = met.subject.nunique() if len(met) else 0
    g = group_table(met, models, cfg["p206"]["alternative"]) if len(met) else pd.DataFrame()
    g.to_csv(d / "group.tsv", sep="\t", index=False)
    rep = [f"# {EXP_ID}", "", f"- 模型 {models}；可评估患者 {n_ev} / {len(subs)}"]
    skipped = info[info.get("note", pd.Series(dtype=str)).fillna("").str.len() > 0] if "note" in info else info.iloc[0:0]
    if len(skipped):
        rep += ["- 不可评估 / 备注：" + "；".join(f"{r.subject} {r.note}" for r in skipped.itertuples())]
    if len(met):
        med = met.groupby(["model", "feature_set", "metric"]).value.median().unstack("metric")
        cols = [c for c in [MAIN, SECOND, "auroc"] + bin_names(cfg["p206"]["dist_edges"]) if c in med]
        rep += ["", "## 各特征组合的患者中位数（基线行应与 P205 / P205b 相同）", "", med[cols].round(4).to_string()]
    if len(g):
        fam = g[g.family]
        rep += ["", "## 判据族（B0+M − B0；主 = 题 B Spearman、次要 = 逐刺激对内 Spearman；全部模型一起 BH-FDR）", "",
                fam[["model", "metric", "n", "median", "ci_lo", "ci_hi", "n_pos", "p", "q_family"]].round(4).to_string(index=False),
                "", "## 另报（不作判据）", "",
                g[~g.family][["model", "compare", "metric", "n", "median", "n_pos", "p"]].round(4).to_string(index=False)]
        if n_ev < 6:
            rep += ["", f"- 注意：只有 {n_ev} 名患者，Wilcoxon 不计算（需要 ≥ 6）；闸门 5 人的运行只用来实测耗时（docs/06 第 4 节 C）"]
    (d / "report.md").write_text("\n".join(rep) + "\n", encoding="utf-8")
    log("\n".join(rep))
    log(f"SUMMARY exp={EXP_ID} subjects={n_ev}/{len(subs)} models={','.join(models)} "
        f"fail={int(info.get('note', pd.Series(dtype=str)).fillna('').str.startswith('FAIL').sum()) if 'note' in info else 0}")
    runs.mark_done(d)


if __name__ == "__main__":
    main()
