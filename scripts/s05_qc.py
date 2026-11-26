#!/usr/bin/env python
# 来自 S2R 505b298 的 scripts/s05_qc.py，2026-11-26 复制（机械改名：S2R_→PRJ_、包名 s2r→项目名）
"""S5：试验质控，给每个试验打"保留 / 剔除（原因）"标记（总纲 S5；规则见 configs 的 qc: 一节）。

用法（workflows/s2r-s05-qc.yaml）:
    PRJ_STEP = scripts/s05_qc.py
    PRJ_DATA_IN = data/interim/d1          # 要读 20 个 h5（约 9 GB）+ S4 targets

输入:  $PRJ_DATA/interim/d1/epochs_pre/sub-XXX_trials.parquet、sub-XXX_pre.h5
       $PRJ_DATA/interim/d1/targets/sub-XXX_targets.parquet
输出:  $PRJ_DATA/interim/d1/qc/（或 --out 指定的目录）
         sub-XXX_qc.parquet       每个试验一行：keep_W3 / keep_W4、剔除原因、各规则标记、眨眼指标
         sub-XXX_chptp.parquet    每个试验 × 窗口一行、每导联一列的峰峰值（µV）——以后改阈值可在本地重算，不必再上云
         _s05_<运行编号>_summary.tsv / _report.md / _config.json
         figures/s05_keep.png          每人保留数（按第一个剔除原因堆叠），W3 与 W4
         figures/s05_eeg_channels.png  每人每导联峰峰值 > 150 µV 的试验比例（W3，含 FP1/FP2）
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
from pvb.preprocess import qc as qcmod  # noqa: E402

RUN_ID = os.environ.get("PRJ_RUN_ID", "local")
COLORS = {"emg_invalid": "0.3", "badchan": "C5", "guard": "C7", "preact": "C1", "mep_mad": "C3", "eeg": "C0"}


def summarize(df: pd.DataFrame, info: dict, qc: dict) -> dict:
    r = {"subject": info["subject"], "n_trials": info["n_trials"], "n_valid": info["n_valid"],
         "bads": ",".join(info["bads"]) or "-"}
    for c in ["rej_emg_invalid", "rej_preact", "rej_mep_mad"]:
        r[c.replace("rej_", "n_")] = int(df[c].sum())
    for w in qc["windows"]:
        r[f"n_guard_{w}"] = int(df[f"rej_guard_{w}"].sum())
        r[f"n_eeg_{w}"] = int(df[f"rej_eeg_{w}"].sum())
    for w in qc["windows"]:
        r[f"keep_{w}"] = int(df[f"keep_{w}"].sum())
        r[f"low_data_{w}"] = bool(r[f"keep_{w}"] < qc["min_keep"])
    r["flag_hussain_rms"] = int(df["flag_hussain_rms"].sum())
    r["flag_wave_r"] = int(df["flag_wave_r"].sum())
    r["hussain_rms_thr_uv"] = info["hussain_rms_thr_uv"]
    w0 = qc["windows"][0]
    r[f"blink_ptp_median_{w0}"] = round(float(df[f"blink_ptp_{w0}"].median()), 1)
    r[f"judged_ptp_median_{w0}"] = round(float(df[f"ptp_max_judged_{w0}"].median()), 1)
    return r


def figures(all_df: pd.DataFrame, all_ch: pd.DataFrame, qc: dict, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    subs = sorted(all_df["subject"].unique())
    labels = [s.replace("sub-", "") for s in subs]
    fig, axs = plt.subplots(1, len(qc["windows"]), figsize=(7 * len(qc["windows"]), 4), squeeze=False)
    for ax, w in zip(axs[0], qc["windows"]):
        first = all_df[f"reason_{w}"].str.split(";").str[0]
        x = np.arange(len(subs))
        bottom = np.array([int(all_df.loc[all_df.subject == s, f"keep_{w}"].sum()) for s in subs])
        ax.bar(x, bottom, color="C2", label="keep")
        for rsn in qcmod.REASON_ORDER:
            h = np.array([int(((all_df.subject == s) & (first == rsn)).sum()) for s in subs])
            if h.sum():
                ax.bar(x, h, bottom=bottom, color=COLORS[rsn], label=rsn)
                bottom = bottom + h
        ax.axhline(qc["min_keep"], color="k", lw=0.8, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_title(f"{w}: kept trials (stacked by first reject reason); dashed = {qc['min_keep']}", fontsize=9)
        ax.legend(fontsize=6, ncol=4, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "s05_keep.png", dpi=130)
    plt.close(fig)

    w0 = qc["windows"][0]
    c = all_ch[all_ch["window"] == w0]
    chans = [k for k in c.columns if k not in ("subject", "window", "aligned_idx")]
    M = np.array([(c.loc[c.subject == s, chans] > qc["eeg_ptp_max_uv"]).mean().to_numpy() for s in subs])
    fig, ax = plt.subplots(figsize=(10, 0.28 * len(subs) + 1.5))
    im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=max(0.05, float(M.max())))
    ax.set_xticks(range(len(chans)))
    ax.set_xticklabels(chans, rotation=90, fontsize=7)
    ax.set_yticks(range(len(subs)))
    ax.set_yticklabels(labels, fontsize=7)
    fig.colorbar(im, ax=ax, label=f"frac ptp > {qc['eeg_ptp_max_uv']} µV")
    ax.set_title(f"clean_{w0} per-channel peak-to-peak; ignored in rule: {qc['eeg_ignore_channels']}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "s05_eeg_channels.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects", nargs="*", help="如 283 711；默认 epochs_pre 里全部有 trials 表的人")
    ap.add_argument("--out", help="输出目录（默认 $PRJ_DATA/interim/d1/qc；本地测试时指到临时目录）")
    args = ap.parse_args()

    data = Path(os.environ["PRJ_DATA"])
    cfg = d1.load_config(CODE)
    qc = cfg["qc"]
    d1_dir = data / "interim" / "d1"
    out = Path(args.out) if args.out else d1_dir / "qc"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    subs = args.subjects or sorted(p.name[4:7] for p in (d1_dir / "epochs_pre").glob("sub-*_trials.parquet"))
    print(f"[S5] run={RUN_ID} git={os.environ.get('PRJ_GIT', '?')} subjects={len(subs)} out={out}", flush=True)

    summ, frames, chs, infos, errors = [], [], [], {}, {}
    for s in subs:
        try:
            df, chptp, info = qcmod.subject_qc(s, d1_dir, qc)
            df.to_parquet(out / f"sub-{s}_qc.parquet", index=False)
            chptp.to_parquet(out / f"sub-{s}_chptp.parquet", index=False)
            frames.append(df)
            chs.append(chptp)
            infos[f"sub-{s}"] = info
            r = summarize(df, info, qc)
            summ.append(r)
            print(f"[S5] sub-{s} OK  n={r['n_trials']} keep_W3={r.get('keep_W3')} keep_W4={r.get('keep_W4')} "
                  f"preact={r['n_preact']} mad={r['n_mep_mad']} eeg_W3={r.get('n_eeg_W3')}", flush=True)
        except Exception:  # noqa: BLE001
            errors[s] = traceback.format_exc()
            print(f"[S5] sub-{s} FAIL\n{errors[s]}", flush=True)

    tag = f"_s05_{RUN_ID}"
    sm = pd.DataFrame(summ)
    if frames:
        figures(pd.concat(frames, ignore_index=True), pd.concat(chs, ignore_index=True), qc, out / "figures")
        sm.to_csv(out / f"{tag}_summary.tsv", sep="\t", index=False)

    lines = [f"# S5 质控报告 {RUN_ID}", "", f"- git: {os.environ.get('PRJ_GIT', '?')}",
             f"- 成功 {len(summ)} / {len(subs)} 人；失败: {', '.join(errors) or '无'}",
             f"- 规则（方案 C）: 脉冲前 FDI 峰峰值 > {qc['pre_pp_max_uv']} µV；block 内 {qc['mep_mad']['column']} 超出中位数 ± "
             f"{qc['mep_mad']['k']}×{qc['mep_mad']['scale']}×MAD；clean 窗口中除 {qc['eeg_ignore_channels']} 外任一导联峰峰值 > "
             f"{qc['eeg_ptp_max_uv']} µV；坏导占比 > {qc['max_bad_frac']}；W* 保护带不满足；EMG 无效",
             f"- 只打标记: Hussain RMS（> Q75 + {qc['flags_only']['hussain_rms_iqr_k']}×IQR）、wave_r < "
             f"{qc['flags_only']['wave_r_min']}",
             f"- 保留数 < {qc['min_keep']} → low_data", ""]
    if len(sm):
        for w in qc["windows"]:
            k = sm[f"keep_{w}"]
            lines.append(f"- keep_{w}: 中位数 {int(k.median())}，最少 {int(k.min())}（{sm.loc[k.idxmin(), 'subject']}），"
                         f"低数据 {int(sm[f'low_data_{w}'].sum())} 人: {', '.join(sm.loc[sm[f'low_data_{w}'], 'subject']) or '无'}")
        lines += ["", "列含义：n_* = 命中该规则的试验数（规则之间有重叠，不能相加）；keep_* = 保留数；"
                  "blink_ptp_median = FP1/FP2 峰峰值中位数；judged_ptp_median = 参与判定导联的最大峰峰值中位数", ""]
        cols = list(sm.columns)
        lines += ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in sm[cols].itertuples(index=False)]
        lines += ["", "## QC 表（sub-XXX_qc.parquet）列说明", ""]
        lines += [f"- `{k}`：{v}" for k, v in qcmod.COLUMNS_DOC.items()]
    if errors:
        lines += ["", "## 错误", ""] + [f"### sub-{k}\n```\n{v}\n```" for k, v in errors.items()]
    (out / f"{tag}_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / f"{tag}_config.json").write_text(json.dumps({"run_id": RUN_ID, "git": os.environ.get("PRJ_GIT"),
                                                        "qc": qc, "info": infos}, ensure_ascii=False, indent=1),
                                            encoding="utf-8")
    print(f"SUMMARY_S5 subjects_ok={len(summ)}/{len(subs)}")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
