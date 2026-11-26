#!/usr/bin/env python
# 来自 S2R 505b298 的 scripts/s03_epoch.py，2026-11-26 复制（机械改名：S2R_→PRJ_、包名 s2r→项目名）
"""S3：切分 D1 的刺激前片段（总纲第 6 节 S3，规则见 6.3 防泄漏）。

用法（在 s2r-run.yaml 里）:
    PRJ_STEP = scripts/s03_epoch.py
    PRJ_ARGS = ""                       # 全部 20 人；调试可写 "--subjects 283 --max-trials 50"
    PRJ_DATA_IN = data/raw/ds002094

输入:  $PRJ_DATA/raw/ds002094/（BIDS 原始数据，只读）
输出:  $PRJ_DATA/interim/d1/epochs_pre/
         sub-XXX_pre.h5          master（1000 Hz，W4，只去均值+降采样）、clean_W4/W3/W2/W1/W05（250 Hz）
         sub-XXX_trials.parquet  每个试验一行：记录、脉冲时刻、间隔、对齐序号、block、guard 标记、幅值
         _s03_<运行编号>_summary.tsv / _report.md / _config.json
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CODE = Path(os.environ.get("PRJ_CODE", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(CODE))
from pvb.io import d1  # noqa: E402
from pvb.preprocess import badchan, epoch  # noqa: E402

RUN_ID = os.environ.get("PRJ_RUN_ID", "local")


def process_subject(sub: str, raw_root: Path, out_dir: Path, cfg: dict, max_trials: int | None) -> dict:
    import h5py
    import mne

    t0 = time.time()
    ep, eeg = cfg["epoch"], cfg["eeg"]
    fs_in = int(eeg["sfreq_in"])
    recs = d1.tms_recordings(raw_root, sub, cfg["tms_tasks"])

    # 通道：以第一个 TMS 记录为准，去掉 TP9/TP10；其余记录必须相同
    raw0 = mne.io.read_raw_brainvision(recs[0][1], preload=False, verbose="ERROR")
    ch_names = [c for c in raw0.ch_names if c.upper() not in {d.upper() for d in eeg["drop_channels"]}]
    n_ch = len(ch_names)

    # 坏导：只用 rest
    rest = d1.vhdr_path(raw_root, sub, cfg["rest_task"])
    if rest.exists():
        rr = mne.io.read_raw_brainvision(rest, preload=False, verbose="ERROR")
        bads, chan_stats = badchan.detect_bads_rest(rr, ch_names, ep["bad_channels"]["flat_uv"],
                                                    ep["bad_channels"]["robust_z"])
    else:
        bads, chan_stats = [], {}
    spatial = epoch.spatial_matrix(n_ch, badchan.interp_matrix(ch_names, bads),
                                   average_ref=ep["clean"]["reference"] == "average")

    parts, rows = [], []
    for task, vhdr in recs:
        raw = mne.io.read_raw_brainvision(vhdr, preload=False, verbose="ERROR")
        if int(raw.info["sfreq"]) != fs_in:
            raise ValueError(f"{vhdr.name} 采样率 {raw.info['sfreq']} != {fs_in}")
        if [c for c in raw.ch_names if c in ch_names] != ch_names:
            raise ValueError(f"{vhdr.name} 通道与第一个记录不一致")
        picks = [raw.ch_names.index(c) for c in ch_names]
        pulses = d1.pulse_samples(raw, cfg["pulse_desc"])
        if max_trials:
            pulses = pulses[:max_trials]

        def read(s, e, raw=raw, picks=picks):
            return raw.get_data(picks=picks, start=s, stop=e) * 1e6

        res = epoch.epoch_recording(read, raw.n_times, pulses, fs_in, ep, spatial, n_ch)
        parts.append(res)
        prev = np.concatenate([[np.nan], pulses[:-1]])
        for k, pk in enumerate(pulses):
            rows.append({"task": task, "trial_in_rec": k, "pulse_sample": int(pk), "onset_s": pk / fs_in,
                         "isi_prev_s": (pk - prev[k]) / fs_in if k else np.nan})

    tr = pd.DataFrame(rows)
    tr.insert(0, "subject", f"sub-{sub}")
    tr.insert(1, "eeg_idx", np.arange(len(tr)))          # 合并后的 0 起序号
    aligned, reason = d1.emg_alignment(sub, len(tr), cfg)
    tr["aligned_idx"] = aligned                          # 与 EMG 试验一一对应的 0 起序号；-1 = 删除
    tr["block"] = np.where(aligned >= 0, aligned // int(cfg["trials_per_block"]), -1)
    tr["drop_reason"] = reason

    cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    tr["ok_read"] = cat["ok_read"]
    for w in ep["windows_s"]:
        wn = epoch.wname(w)
        tr[f"guard_ok_{wn}"] = cat[f"guard_ok_{wn}"]
        cw = cat[f"clean_{wn}"]
        ptp = np.ptp(cw, axis=-1)                       # (trial, ch)，µV
        tr[f"ptp_max_uv_{wn}"] = np.nanmax(ptp, axis=1) if np.isfinite(ptp).any() else np.nan
        tr[f"n_ch_over150_{wn}"] = (ptp > 150).sum(axis=1)

    out_dir.mkdir(parents=True, exist_ok=True)
    h5p = out_dir / f"sub-{sub}_pre.h5"
    with h5py.File(h5p, "w") as f:
        f.create_dataset("master", data=cat["master"])
        for w in ep["windows_s"]:
            f.create_dataset(f"clean_{epoch.wname(w)}", data=cat[f"clean_{epoch.wname(w)}"])
        f.create_dataset("eeg_idx", data=tr["eeg_idx"].to_numpy())
        f.create_dataset("aligned_idx", data=tr["aligned_idx"].to_numpy())
        f.create_dataset("block", data=tr["block"].to_numpy())
        f.attrs.update({
            "subject": f"sub-{sub}", "ch_names": json.dumps(ch_names), "bads": json.dumps(bads),
            "master_sfreq": ep["master_sfreq"], "master_window_s": ep["master_window_s"],
            "clean_sfreq": ep["clean"]["sfreq"], "end_offset_s": ep["end_offset_s"],
            "windows_s": json.dumps(ep["windows_s"]), "units": "uV",
            "master_note": "demean + resample_poly only; original reference (POz); no other filter",
            "clean_note": "per-window demean, reflect-pad, notch 60 + bandpass 1-45 (sosfiltfilt), "
                          "resample 250, spline-interpolate bads (rest-based), average reference",
            "run_id": RUN_ID, "git": os.environ.get("PRJ_GIT", "?"),
            "recordings": json.dumps([t for t, _ in recs]),
        })
    tr.to_parquet(out_dir / f"sub-{sub}_trials.parquet", index=False)

    kept = tr["aligned_idx"] >= 0
    summ = {"subject": f"sub-{sub}", "recordings": "+".join(t for t, _ in recs), "n_pulses": len(tr),
            "n_emg_aligned": int(kept.sum()), "n_blocks": int(tr.loc[kept, "block"].nunique()),
            "n_ok_read": int(tr["ok_read"].sum()), "bads": ",".join(bads) or "-",
            "isi_min_s": round(float(np.nanmin(tr["isi_prev_s"])), 3)}
    for w in ep["windows_s"]:
        wn = epoch.wname(w)
        summ[f"guard_ok_{wn}"] = int((tr[f"guard_ok_{wn}"] & kept).sum())
    summ["ptp_max_uv_W3_median"] = round(float(np.nanmedian(tr.get("ptp_max_uv_W3", np.nan))), 1)
    summ["h5_mb"] = round(h5p.stat().st_size / 1e6, 1)
    summ["seconds"] = round(time.time() - t0, 1)
    summ["chan_stats"] = chan_stats
    return summ


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects", nargs="*", help="如 283 711；默认全部")
    ap.add_argument("--max-trials", type=int, default=None, help="调试用：每个记录只取前 N 个脉冲")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    data = Path(os.environ["PRJ_DATA"])
    cfg = d1.load_config(CODE)
    raw_root = data / cfg["raw_dir"]
    out_dir = data / "interim" / "d1" / "epochs_pre"
    out_dir.mkdir(parents=True, exist_ok=True)

    ok, msg = epoch.leakage_selftest()
    print(f"[S3] leakage selftest: {msg}", flush=True)
    if not ok:
        sys.exit(3)

    subs = args.subjects or d1.list_subjects(raw_root)
    print(f"[S3] run={RUN_ID} git={os.environ.get('PRJ_GIT', '?')} subjects={len(subs)} raw={raw_root}", flush=True)
    results, errors = [], {}
    with ProcessPoolExecutor(max_workers=max(1, min(args.workers, len(subs)))) as ex:
        futs = {ex.submit(process_subject, s, raw_root, out_dir, cfg, args.max_trials): s for s in subs}
        for fu in as_completed(futs):
            s = futs[fu]
            try:
                r = fu.result()
                results.append(r)
                print(f"[S3] sub-{s} OK  pulses={r['n_pulses']} aligned={r['n_emg_aligned']} "
                      f"guardW3={r['guard_ok_W3']} guardW4={r['guard_ok_W4']} bads={r['bads']} {r['seconds']}s",
                      flush=True)
            except Exception:  # noqa: BLE001
                errors[s] = traceback.format_exc()
                print(f"[S3] sub-{s} FAIL\n{errors[s]}", flush=True)

    tag = f"_s03_{RUN_ID}"
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "chan_stats"} for r in results])
    if len(df):
        df = df.sort_values("subject")
        df.to_csv(out_dir / f"{tag}_summary.tsv", sep="\t", index=False)
    (out_dir / f"{tag}_config.json").write_text(json.dumps(
        {"run_id": RUN_ID, "git": os.environ.get("PRJ_GIT"), "args": vars(args), "config": cfg,
         "leakage_selftest": msg, "chan_stats": {r["subject"]: r["chan_stats"] for r in results},
         "errors": errors}, ensure_ascii=False, indent=1), encoding="utf-8")

    lines = [f"# S3 切片报告 {RUN_ID}", "", f"- git: {os.environ.get('PRJ_GIT', '?')}",
             f"- 防泄漏自检: {msg}", f"- 成功 {len(results)} / {len(subs)} 人；失败: {', '.join(errors) or '无'}", ""]
    if len(df):
        cols = [c for c in df.columns if c not in ("recordings",)]
        lines += ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df[cols].itertuples(index=False)]
        lines += ["", "按窗口统计（EMG 对齐后、guard 通过的试验数，中位数 / 最小）："]
        for w in cfg["epoch"]["windows_s"]:
            c = f"guard_ok_{epoch.wname(w)}"
            lines.append(f"- {epoch.wname(w)}: {int(df[c].median())} / {int(df[c].min())}")
    (out_dir / f"{tag}_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[-8:]))
    n_bad = len(errors)
    print(f"SUMMARY_S3 subjects_ok={len(results)}/{len(subs)} leakage={'PASS' if ok else 'FAIL'}")
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
