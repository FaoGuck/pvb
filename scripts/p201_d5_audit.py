#!/usr/bin/env python
"""P201：D5（ds004080）信号层面的审计（00 第 4 节 P201；元数据层面已在 p001_d5_meta_audit.py 做过）。

每个 run 检查：
  1. vhdr 的采样率 / 时长与 ieeg.json 一致；信号文件里的通道名与 channels.tsv 一致（多出 / 缺少的列出）
  2. 事件时刻：onset × 采样率 与 sample_start 列差几个样本（决定用哪一列）
  3. **刺激伪迹对齐**：在被刺激的两个电极上，脉冲前后 ±50 ms 里 |信号 − 基线中位数| 最大的时刻应在 0 附近
     （这一步验证我们切出来的 epoch 时间零点是对的；不对的话 N1 潜伏期全部错位）
另外把 derivatives/av_ccep/ 下每个 .mat 的结构（变量名、形状、类型、前几个值）写成文本，供 p202b 与作者结果逐项核对。

用法（workflows/pvb-p201-audit-gate.yaml）：
    PRJ_STEP = scripts/p201_d5_audit.py
    PRJ_ARGS = --config configs/datasets/p_d5_ds004080.yaml   [--subjects ...] [--n-pulses 40]
输出：$PRJ_OUT/runs/P201_d5_audit/<时间>_<git7>/  runs.tsv、avccep_structure.txt、report.md、log.txt、DONE
"""
from __future__ import annotations

import argparse
import glob
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

EXP_ID = "P201_d5_audit"


def audit_run(events_tsv: str, n_pulses: int) -> dict:
    meta = io_d5.read_run_meta(events_tsv)
    rd = io_d5.RawReader(meta)
    tsv_names = meta.channels["NAME"].tolist()
    r = dict(run_id=meta.run_id, subject=meta.subject,
             sfreq_json=meta.sfreq, sfreq_vhdr=rd.sfreq,
             dur_json_s=meta.duration_s, dur_vhdr_s=round(rd.n_times / rd.sfreq, 3),
             n_ch_vhdr=len(rd.names), n_ch_tsv=len(tsv_names),
             ch_only_vhdr=";".join(sorted(set(rd.names) - set(tsv_names))) or "-",
             ch_only_tsv=";".join(sorted(set(tsv_names) - set(rd.names))) or "-",
             names_same_order=rd.names == tsv_names,
             n_stim=len(meta.stim))
    st = meta.stim
    if "sample_start" in st:
        dif = pd.to_numeric(st["sample_start"], errors="coerce") - st["sample0"]
        vc = dif.value_counts().head(3)
        r["sample_start_minus_onset"] = ";".join(f"{int(k)}:{v}" for k, v in vc.items())
    # 刺激伪迹对齐：取均匀分布的 n_pulses 个脉冲
    sf = rd.sfreq
    h = int(round(0.05 * sf))
    sel = st.iloc[np.linspace(0, len(st) - 1, min(n_pulses, len(st))).astype(int)] if len(st) else st
    lat, ratio = [], []
    for _, ev in sel.iterrows():
        els = io_d5.pair_electrodes(ev["pair"])
        pk = rd.picks(els)
        if not pk:
            continue
        seg = rd.segments(pk, np.array([ev["sample0"] - h]), 2 * h)[0]
        if not np.isfinite(seg).all():
            continue
        base = np.median(seg[:, : int(0.04 * sf)], axis=1, keepdims=True)
        dev = np.abs(seg - base).max(axis=0)
        i = int(np.argmax(dev))
        lat.append((i - h) / sf * 1000)
        ratio.append(dev[i] / (np.median(np.abs(seg[:, : int(0.04 * sf)] - base)) + 1e-9))
    lat = np.array(lat)
    r.update(art_n=len(lat), art_lat_med_ms=float(np.median(lat)) if len(lat) else np.nan,
             art_lat_abs_le2ms=float(np.mean(np.abs(lat) <= 2.0)) if len(lat) else np.nan,
             art_ratio_med=float(np.median(ratio)) if len(ratio) else np.nan)
    return r


def describe(obj, name="", depth=0, max_depth=4, out=None):
    """递归描述 scipy.io.loadmat / h5py 读出的对象。"""
    out = [] if out is None else out
    pad = "  " * depth
    if depth > max_depth:
        return out
    if hasattr(obj, "_fieldnames"):                      # scipy mat_struct
        out.append(f"{pad}{name}: struct fields={obj._fieldnames}")
        for f in obj._fieldnames:
            describe(getattr(obj, f), f, depth + 1, max_depth, out)
    elif isinstance(obj, dict):
        out.append(f"{pad}{name}: dict keys={[k for k in obj if not str(k).startswith('__')][:40]}")
        for k, v in obj.items():
            if not str(k).startswith("__"):
                describe(v, str(k), depth + 1, max_depth, out)
    elif isinstance(obj, np.ndarray):
        s = f"{pad}{name}: ndarray shape={obj.shape} dtype={obj.dtype}"
        if obj.dtype == object and obj.size:
            out.append(s + f"  (object；展开第 1 个元素)")
            describe(obj.flat[0], f"{name}[0]", depth + 1, max_depth, out)
        else:
            if obj.size and obj.dtype.kind in "US":
                s += f"  前几个={list(obj.ravel()[:5])}"
            elif obj.size and obj.dtype.kind in "fiub":
                v = obj.astype(float).ravel()
                fin = v[np.isfinite(v)]
                s += f"  NaN比例={1 - fin.size / v.size:.3f}" + (f"  范围=[{fin.min():.4g}, {fin.max():.4g}]" if fin.size else "")
            out.append(s)
    else:
        out.append(f"{pad}{name}: {type(obj).__name__} = {str(obj)[:120]}")
    return out


def describe_mat(path: str) -> list[str]:
    head = [f"### {os.path.basename(path)}  ({os.path.getsize(path) / 2**20:.1f} MiB)"]
    try:
        from scipy.io import loadmat
        m = loadmat(path, squeeze_me=True, struct_as_record=False)
        return head + ["格式：MATLAB v5/v7（scipy.io.loadmat）"] + describe(m, "file")
    except NotImplementedError:
        import h5py
        lines = head + ["格式：MATLAB v7.3（HDF5，h5py）"]
        with h5py.File(path, "r") as f:
            def visit(n, o):
                if n.count("/") > 4:
                    return
                if isinstance(o, h5py.Dataset):
                    lines.append(f"  {n}: dataset shape={o.shape} dtype={o.dtype}")
                else:
                    lines.append(f"  {n}/: group")
            f.visititems(visit)
        return lines
    except Exception as e:
        return head + [f"读取失败：{e!r}"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/datasets/p_d5_ds004080.yaml")
    ap.add_argument("--subjects", default="")
    ap.add_argument("--n-pulses", type=int, default=40)
    ap.add_argument("--procs", type=int, default=0)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(CODE / a.config, encoding="utf-8"))
    subs = [s for s in a.subjects.split(",") if s] or cfg["subjects_gate"]
    root = os.path.join(os.environ["PRJ_DATA"], cfg["raw_dir"])
    d = runs.new_run_dir(Path(os.environ["PRJ_OUT"]), EXP_ID)
    runs.write_snapshot(d, {**cfg, "_args": vars(a), "_subjects": subs})
    log = runs.Log(d)
    files = [f for s in subs for f in io_d5.find_runs(root, s)]
    log(f"{EXP_ID}: {len(subs)} 名患者，{len(files)} 个 run")
    rows = []
    with ProcessPoolExecutor(max_workers=min(a.procs or cfg.get("procs", 8), max(1, len(files)))) as ex:
        futs = {ex.submit(audit_run, f, a.n_pulses): f for f in files}
        for fu in as_completed(futs):
            try:
                r = fu.result(); rows.append(r)
                log(f"OK  {r['run_id']}  sfreq={r['sfreq_vhdr']:g}  ch={r['n_ch_vhdr']}/{r['n_ch_tsv']}  "
                    f"伪迹潜伏期中位 {r['art_lat_med_ms']:.2f} ms（|≤2 ms| {r['art_lat_abs_le2ms']:.2f}）")
            except Exception as e:
                rows.append(dict(run_id=os.path.basename(futs[fu]), error=repr(e)))
                log(f"ERR {os.path.basename(futs[fu])}: {e!r}\n{traceback.format_exc()}")
    df = pd.DataFrame(rows).sort_values("run_id")
    df.to_csv(d / "runs.tsv", sep="\t", index=False)

    mats = sorted(glob.glob(os.path.join(root, "derivatives", "av_ccep", "**", "*.mat"), recursive=True))
    struct = [f"# derivatives/av_ccep 结构（{len(mats)} 个 .mat）", ""]
    for p in mats:
        struct += describe_mat(p) + [""]
    (d / "avccep_structure.txt").write_text("\n".join(struct), encoding="utf-8")
    log("\n".join(struct[:200]))

    n_err = int(df["error"].notna().sum()) if "error" in df else 0
    ok = df[df.get("error", pd.Series([np.nan] * len(df))).isna()] if n_err else df
    rep = [f"# {EXP_ID}", "",
           f"- run {len(df)} 个，失败 {n_err}",
           f"- 采样率 vhdr = json：{bool((ok.sfreq_vhdr == ok.sfreq_json).all())}；"
           f"时长差最大 {float((ok.dur_vhdr_s - ok.dur_json_s).abs().max()):.3f} s",
           f"- 通道名与 channels.tsv 完全一致（含顺序）的 run：{int(ok.names_same_order.sum())} / {len(ok)}",
           f"- sample_start − round(onset×sfreq)（样本数:次数）：{sorted(set(ok.get('sample_start_minus_onset', pd.Series(['-']))))}",
           f"- 刺激伪迹潜伏期（被刺激电极上 |偏离| 最大的时刻）：各 run 中位数 "
           f"[{ok.art_lat_med_ms.min():.2f}, {ok.art_lat_med_ms.max():.2f}] ms；|≤ 2 ms| 的比例最低 {ok.art_lat_abs_le2ms.min():.2f}",
           f"- av_ccep：{len(mats)} 个 .mat，结构见 avccep_structure.txt", "",
           ok.drop(columns=[c for c in ["error"] if c in ok]).to_string(index=False)]
    (d / "report.md").write_text("\n".join(rep) + "\n", encoding="utf-8")
    log("\n".join(rep))
    log(f"SUMMARY exp={EXP_ID} ok={len(df) - n_err}/{len(files)} mats={len(mats)}")
    if n_err == 0 and len(files):
        runs.mark_done(d)


if __name__ == "__main__":
    main()
