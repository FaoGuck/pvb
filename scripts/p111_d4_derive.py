"""P111：D4（ds004024）连续降采样派生——逐 run 下载 → 按脉冲分段抗混叠降采样 → 核对 → 存 .h5 → 删原始。

范围（00 第 14 节 2026-11-27）：spTMS 全部 348 个 run + rest 119 个 run；参数在 configs/datasets/p_d4_ds004024.yaml 的 p111
做法：
  - 读 20 kHz 原始（MNE，按段读，不整块载入）；通道类型以 channels.tsv 为准
  - **按脉冲分段**：切点 = 每个脉冲前 2 ms（对齐到 20 个样本的整数倍）；每段两端对称延拓 0.1 s 后单独 resample_poly 降采样，
    再裁回 → 降采样滤波永远不跨过脉冲，脉冲后的数据不会影响脉冲前的任何样本（00 第 6 节第 1 条；tests/test_d4_derive_leakage.py）
  - EEG 64 导与 ECG / VEOG / HEOG → 1 000 Hz，EMG 两导 → 5 000 Hz；单位 µV，float32；事件原样（20 kHz 样本号 + 秒）
  - 核对：时长、脉冲数；取中间 10 s，派生值与“整段 resample_poly”在离脉冲 ≥ 50 ms、离这段两端 ≥ 0.1 s 的样本上的相关 ≥ 0.99（每导联）
  - spTMS 另存逐脉冲质控表（derived/d4/qc/）：两导 EMG 的 MEP / 预激活 / 噪声峰峰值、刺激侧、W3 EEG 逐导联峰峰值
    （复用 D1 的 pvb/preprocess/epoch.py：60 Hz 陷波 + 1–45 Hz → 250 Hz → 平均参考，与 P110 相同）
  - 每个 run 完成后写 derived/d4/_done/<run>.txt；--done-dir 里已有标记的 run 跳过；开始新 run 的时间受 budget_min 限制
  - 下载下一个 run 与处理当前 run 同时进行（瓶颈是外网下载，不是计算）
用法：
  云上：  scripts/p111_d4_derive.py [--runs <基名,…>] [--done-dir <目录>]
  本地冒烟：--local-dir <目录>（不下载、不删除）
输出：$PRJ_DATA/derived/d4/{cont,qc,_done}/…；$PRJ_OUT/runs/P111_d4_derive/<时间>_<git7>/（runs.tsv、report.md、DONE）
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import glob
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import signal

CODE = Path(os.environ["PRJ_CODE"])
sys.path.insert(0, str(CODE))
from pvb.preprocess import epoch as ep_mod  # noqa: E402
from pvb.probing import runs  # noqa: E402

EXP_ID = "P111_d4_derive"
CFG_PATH = "configs/datasets/p_d4_ds004024.yaml"
FS_IN = 20000
SUFFIXES = ["_eeg.vhdr", "_eeg.vmrk", "_events.tsv", "_channels.tsv", "_eeg.json", "_eeg.eeg"]


# ---------- 分段降采样（纯函数，防泄漏测试直接调用） ----------
def cut_points(pulses: np.ndarray, n: int, cut_samples: int, align: int) -> np.ndarray:
    """返回段边界 [0, c1, c2, …, n_end]：c_k = 对齐到 align 的 (脉冲_k − cut_samples) 向下取整；n_end = n 向下对齐。"""
    n_end = n // align * align
    c = [(int(p) - cut_samples) // align * align for p in np.sort(pulses)]
    c = [x for x in c if 0 < x < n_end]
    return np.array(sorted(set([0] + c + [n_end])), dtype=np.int64)


def downsample_segments(x: np.ndarray, bounds: np.ndarray, q: int, pad: int) -> np.ndarray:
    """x: (n_ch, n)。每段 [b_i, b_{i+1}) 单独：对称延拓 → resample_poly(1, q) → 裁回。段长都是 q 的整数倍。"""
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        seg = np.asarray(x[:, a:b], dtype=np.float64)
        n = seg.shape[1]
        p = min(n - 1, pad) // q * q
        if p > 0:
            y = signal.resample_poly(np.pad(seg, [(0, 0), (p, p)], mode="reflect"), 1, q, axis=1, padtype="line")
            y = y[:, p // q: p // q + n // q]
        else:
            y = seg[:, ::q]
        out.append(y)
    return np.concatenate(out, axis=1)


# ---------- 下载 ----------
def fetch(base: str, src: str, dst: Path, aws: str) -> float:
    t0 = time.time()
    for suf in SUFFIXES:
        p = dst / (base + suf)
        p.parent.mkdir(parents=True, exist_ok=True)
        for k in range(3):
            r = subprocess.run([aws, "s3", "cp", "--no-sign-request", "--only-show-errors", f"{src}/{base}{suf}", str(p)])
            if r.returncode == 0 and p.is_file():
                break
            time.sleep(30)
        else:
            if suf != "_eeg.json":
                raise RuntimeError(f"下载失败：{base}{suf}")
    return time.time() - t0


def list_runs(keys_tsv: Path, cfg: dict) -> list[str]:
    k = pd.read_csv(keys_tsv, sep="\t", header=None, names=["key", "bytes"])
    eeg = k[k.key.str.endswith("_eeg.eeg")].key
    tasks = cfg["p111"]["tasks"]
    bases = [x[: -len("_eeg.eeg")] for x in eeg if any(f"_task-{t}_" in x for t in tasks)]
    order = {t: i for i, t in enumerate(tasks)}
    return sorted(bases, key=lambda b: (b.split("/")[0], b.split("/")[1], order[b.split("_task-")[1].split("_")[0]], b))


# ---------- 处理一个 run ----------
def ptp(x, t_ms, win):
    m = (t_ms >= win[0]) & (t_ms <= win[1])
    s = x[..., m]
    return s.max(axis=-1) - s.min(axis=-1)


def process_run(root: Path, base: str, cfg: dict, data: Path) -> dict:
    import h5py
    import mne
    p, p0 = cfg["p111"], cfg["p110"]
    raw = mne.io.read_raw_brainvision(str(root / (base + "_eeg.vhdr")), preload=False, verbose="ERROR")
    if int(round(raw.info["sfreq"])) != FS_IN:
        raise ValueError(f"采样率 {raw.info['sfreq']} ≠ 20000")
    names = raw.ch_names
    cht = pd.read_csv(root / (base + "_channels.tsv"), sep="\t", encoding="utf-8-sig")
    typ = dict(zip(cht["name"], cht["type"]))
    eeg_ch = [c for c in names if typ.get(c) == "EEG"]
    aux_ch = [c for c in names if typ.get(c) in ("ECG", "EOG")]
    emg_ch = [cfg["emg_channels"]["right"], cfg["emg_channels"]["left"]]
    ev = pd.read_csv(root / (base + "_events.tsv"), sep="\t", encoding="utf-8-sig")
    task = base.split("_task-")[1].split("_")[0]
    pulses = np.sort(ev.loc[ev.trial_type.astype(str).isin(cfg["pulse_types"]), "sample"].to_numpy(np.int64))
    q_eeg, q_emg = FS_IN // p["sfreq"]["eeg"], FS_IN // p["sfreq"]["emg"]
    align = int(np.lcm(q_eeg, q_emg))
    bounds = cut_points(pulses, raw.n_times, int(round(p["cut_before_pulse_ms"] * FS_IN / 1000)), align)
    pad = int(round(p["pad_s"] * FS_IN))
    idx = {c: names.index(c) for c in names}
    pk_e, pk_a, pk_m = [idx[c] for c in eeg_ch], [idx[c] for c in aux_ch], [idx[c] for c in emg_ch]

    eeg_o, aux_o, emg_o = [], [], []
    for a, b in zip(bounds[:-1], bounds[1:]):
        seg = raw.get_data(start=int(a), stop=int(b)) * 1e6            # 69 × n，µV
        bb = np.array([0, b - a])
        eeg_o.append(downsample_segments(seg[pk_e], bb, q_eeg, pad))
        aux_o.append(downsample_segments(seg[pk_a], bb, q_eeg, pad))
        emg_o.append(downsample_segments(seg[pk_m], bb, q_emg, pad))
    eeg = np.concatenate(eeg_o, 1).astype(np.float32)
    aux = np.concatenate(aux_o, 1).astype(np.float32)
    emg = np.concatenate(emg_o, 1).astype(np.float32)
    n_end = int(bounds[-1])

    # 核对：中间 10 s，派生 vs 整段 resample_poly，只比离脉冲 ≥ 50 ms 的样本
    cs = int(p["verify"]["chunk_s"] * FS_IN) // align * align
    c0 = max(0, (n_end // 2 - cs // 2) // align * align)
    ref = signal.resample_poly(raw.get_data(picks=pk_e, start=c0, stop=c0 + cs) * 1e6, 1, q_eeg, axis=1)
    der = eeg[:, c0 // q_eeg: (c0 + cs) // q_eeg]
    t_in = c0 + np.arange(der.shape[1]) * q_eeg
    far = np.ones(der.shape[1], bool)
    for pk in pulses:
        far &= np.abs(t_in - pk) >= 0.05 * FS_IN
    edge = int(0.1 * p["sfreq"]["eeg"])                             # 参考算法在这段两端补零，两端各 0.1 s 不比（冒烟测试发现）
    far[:edge] = False
    far[-edge:] = False
    corr = [np.corrcoef(ref[i, far], der[i, far])[0, 1] for i in range(len(eeg_ch)) if np.std(ref[i, far]) > 0]
    corr_min = float(np.nanmin(corr)) if corr else np.nan
    ok_len = eeg.shape[1] == n_end // q_eeg and emg.shape[1] == n_end // q_emg
    ok_corr = bool(corr_min >= p["verify"]["min_corr"])

    sub, ses = base.split("/")[0], base.split("/")[1][4:]
    out_h5 = data / p["out_dir"] / (base.split("/eeg/")[0]) / (os.path.basename(base) + ".h5")
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_h5, "w") as f:
        f.create_dataset("eeg", data=eeg)
        f.create_dataset("aux", data=aux)
        f.create_dataset("emg", data=emg)
        f.create_dataset("pulse_sample_20k", data=pulses)
        f.create_dataset("cut_bounds_20k", data=bounds)
        f.attrs.update(dict(eeg_ch=json.dumps(eeg_ch), aux_ch=json.dumps(aux_ch), emg_ch=json.dumps(emg_ch),
                            sfreq_eeg=p["sfreq"]["eeg"], sfreq_aux=p["sfreq"]["aux"], sfreq_emg=p["sfreq"]["emg"],
                            sfreq_in=FS_IN, n_times_in=int(raw.n_times), n_end_in=n_end, unit="uV",
                            source=f"{cfg['source']}/{base}_eeg.eeg", events_tsv=ev.to_csv(sep="\t", index=False),
                            git=runs.git_commit(), method="按脉冲分段（脉冲前 2 ms 切开），每段对称延拓 0.1 s 后 resample_poly"))
    sha = hashlib.sha256(out_h5.read_bytes()).hexdigest()

    info = dict(base=os.path.basename(base), subject=sub, session=ses, task=task, n_pulse=len(pulses),
                n_seg=len(bounds) - 1, dur_s=n_end / FS_IN, h5_mib=out_h5.stat().st_size / 2**20,
                ok_len=ok_len, corr_min=corr_min, ok_corr=ok_corr, sha256=sha)

    if task == "spTMS" and len(pulses):
        fe, fm = p["sfreq"]["eeg"], p["sfreq"]["emg"]
        lo, hi = int(-0.06 * fm), int(0.08 * fm)
        t_ms = np.arange(lo, hi) / fm * 1000
        seg = np.full((len(pulses), 2, hi - lo), np.nan)
        for k, pk in enumerate(pulses):
            c = int(round(pk / FS_IN * fm))
            if c + lo >= 0 and c + hi <= emg.shape[1]:
                seg[k] = emg[:, c + lo: c + hi]
        mep2 = ptp(seg, t_ms, p0["mep_window_ms"])
        med = np.nanmedian(mep2, axis=0)
        ti = int(np.nanargmax(med))
        e = p0["epoch"]
        sos = ep_mod.clean_filters(fe, e["clean"]["notch_hz"], e["clean"]["notch_q"], e["clean"]["band_hz"], e["clean"]["filter_order"])
        use = [i for i, c in enumerate(eeg_ch) if c not in cfg["eeg"]["drop_channels"]]
        spatial = ep_mod.spatial_matrix(len(use), None, average_ref=True)
        nw, off = int(round(e["window_s"] * fe)), int(round(e["end_offset_s"] * fe))
        cptp = np.full((len(pulses), len(use)), np.nan, dtype=np.float32)
        for k, pk in enumerate(pulses):
            stop = int(pk // q_eeg) - off                               # 脉冲前 10 ms（派生采样点）
            if stop - nw >= 0:
                w = ep_mod.clean_window(eeg[use, stop - nw: stop].astype(np.float64)[None], fe, int(e["clean"]["sfreq"]), sos, spatial)[0]
                cptp[k] = w.max(axis=1) - w.min(axis=1)
        qc = pd.DataFrame({"subject": sub, "session": ses, "base": os.path.basename(base), "trial": np.arange(1, len(pulses) + 1),
                           "pulse_sample_20k": pulses, "prev_gap_s": np.concatenate([[np.nan], np.diff(pulses) / FS_IN]),
                           "mep_pp_right": mep2[:, 0], "mep_pp_left": mep2[:, 1],
                           "pre_pp_right": ptp(seg[:, 0], t_ms, p0["pre_window_ms"]), "pre_pp_left": ptp(seg[:, 1], t_ms, p0["pre_window_ms"]),
                           "noise_pp_right": ptp(seg[:, 0], t_ms, p0["noise_window_ms"]), "noise_pp_left": ptp(seg[:, 1], t_ms, p0["noise_window_ms"]),
                           "target": ["EMG Right", "EMG Left"][ti]})
        for j, i in enumerate(use):
            qc[f"w3ptp_{eeg_ch[i]}"] = cptp[:, j]
        qd = data / p["qc_dir"] / sub
        qd.mkdir(parents=True, exist_ok=True)
        qc.to_parquet(qd / (os.path.basename(base) + "_qc.parquet"), index=False)
        info.update(target=["EMG Right", "EMG Left"][ti], stim_side=["L-M1", "R-M1"][ti], side_ratio=float(med[ti] / max(med[1 - ti], 1e-9)),
                    mep_med=float(med[ti]))

    dd = data / p["done_dir"]
    dd.mkdir(parents=True, exist_ok=True)
    if info["ok_len"] and info["ok_corr"]:
        (dd / (os.path.basename(base) + ".txt")).write_text("\n".join(f"{k}={v}" for k, v in info.items()) + "\n", encoding="utf-8")
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="", help="只做这些 run（逗号分隔的文件基名，如 sub-CON001_ses-async4ms_task-spTMS_run-01）")
    ap.add_argument("--done-dir", default="", help="已完成标记所在目录（只读输入）；里面有的 run 跳过")
    ap.add_argument("--local-dir", help="本地冒烟：直接处理这个目录里的 run（不下载、不删除）")
    ap.add_argument("--tmp", default="/osmo/run/workspace/tmp_p111")
    ap.add_argument("--aws", default="/osmo/run/workspace/envs/s2r/bin/aws")
    ap.add_argument("--budget-min", type=float, default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(CODE / CFG_PATH, encoding="utf-8"))
    data = Path(os.environ["PRJ_DATA"])
    budget = (a.budget_min if a.budget_min is not None else cfg["p111"]["budget_min"]) * 60

    if a.local_dir:
        root = Path(a.local_dir)
        bases = sorted(str(Path(x).relative_to(root)).replace(os.sep, "/")[: -len("_eeg.vhdr")]
                       for x in glob.glob(str(root / "sub-*" / "ses-*" / "eeg" / "*_eeg.vhdr")))
    else:
        root = Path(a.tmp); root.mkdir(parents=True, exist_ok=True)
        bases = list_runs(data / cfg["manifest"], cfg)
    if a.runs:
        want = set(a.runs.split(","))
        bases = [b for b in bases if os.path.basename(b) in want]
    done = {Path(x).stem for x in glob.glob(os.path.join(a.done_dir, "*.txt"))} if a.done_dir else set()
    todo = [b for b in bases if os.path.basename(b) not in done]

    d = runs.new_run_dir(Path(os.environ["PRJ_OUT"]), EXP_ID)
    runs.write_snapshot(d, {**cfg, "_resolved": {"n_all": len(bases), "n_done_before": len(done), "n_todo": len(todo),
                                                 "runs_arg": a.runs, "git": runs.git_commit(), "run_id": os.environ.get("PRJ_RUN_ID")}})
    log = runs.Log(d)
    log(f"[P111] run_dir={d} 全部 {len(bases)}，之前已完成 {len(done)}，本次待做 {len(todo)}，时间预算 {budget / 60:.0f} min")
    t_start = time.time()
    infos = []
    pool = cf.ThreadPoolExecutor(max_workers=1)
    fut = None
    if todo and not a.local_dir:
        fut = pool.submit(fetch, todo[0], cfg["source"], root, a.aws)
    for i, b in enumerate(todo):
        t0 = time.time()
        try:
            dl = fut.result() if fut is not None else 0.0
            fut = None
            # 预取下一个（只在时间预算内开始）
            if not a.local_dir and i + 1 < len(todo) and time.time() - t_start < budget:
                fut = pool.submit(fetch, todo[i + 1], cfg["source"], root, a.aws)
            info = process_run(root, b, cfg, data)
            info.update(download_s=dl, process_s=time.time() - t0)
            infos.append(info)
            log(f"[P111] {i + 1}/{len(todo)} {info['base']} {'OK' if info['ok_len'] and info['ok_corr'] else 'VERIFY_FAIL'} "
                f"脉冲 {info['n_pulse']} 段 {info['n_seg']} 时长 {info['dur_s']:.0f}s h5 {info['h5_mib']:.0f} MiB "
                f"相关最小 {info['corr_min']:.4f} " + (f"{info.get('stim_side', '')} 两侧比 {info.get('side_ratio', np.nan):.1f} " if 'stim_side' in info else "")
                + f"[下载 {dl:.0f}s 处理 {info['process_s']:.0f}s，已用 {(time.time() - t_start) / 60:.0f} min]")
        except Exception as e:  # noqa: BLE001
            infos.append(dict(base=os.path.basename(b), error=repr(e)))
            log(f"[P111] {i + 1}/{len(todo)} {os.path.basename(b)} FAIL {e!r}"); log(traceback.format_exc())
        finally:
            if not a.local_dir:
                for f in glob.glob(str(root / (b + "*"))):
                    os.remove(f)                                    # 原始文件不保留（共享规则第 5 条）
        if not a.local_dir and fut is None:                         # 没有预取下一个 = 做完了或时间预算用完
            if i + 1 < len(todo):
                log(f"[P111] 时间预算用完（{(time.time() - t_start) / 60:.0f} min），剩 {len(todo) - i - 1} 个留给下一批")
            break
    pool.shutdown(wait=True)
    cols = ["base", "ok_len", "ok_corr", "corr_min", "h5_mib", "error"]
    ri = pd.DataFrame(infos)
    for c in cols:                                                  # 本次没有要做的 run 时表是空的（冒烟测试发现）
        if c not in ri:
            ri[c] = np.nan
    ri.to_csv(d / "runs.tsv", sep="\t", index=False)
    ok = ri[ri.error.isna() & ri.ok_len.fillna(False).astype(bool) & ri.ok_corr.fillna(False).astype(bool)]
    n_left = len(todo) - len(ok)
    rep = [f"# {EXP_ID}", "", f"- 全部 {len(bases)} 个 run；之前已完成 {len(done)}；本次尝试 {len(ri)}，成功 {len(ok)}；**还剩 {n_left + 0} 个**",
           f"- 派生合计 {ok.h5_mib.sum() / 1024:.2f} GiB；核对相关最小值 {ok.corr_min.min() if len(ok) else float('nan'):.4f}",
           "", ri.round(3).to_string(index=False) if len(ri) else "（本次没有要做的 run）"]
    (d / "report.md").write_text("\n".join(rep) + "\n", encoding="utf-8")
    log("\n".join(rep[:4]))
    log(f"SUMMARY exp={EXP_ID} tried={len(ri)} ok={len(ok)} fail={len(ri) - len(ok)} done_total={len(done) + len(ok)}/{len(bases)} "
        f"remaining={len(bases) - len(done) - len(ok)} gib={ok.h5_mib.sum() / 1024 if len(ok) else 0:.2f} run_dir={d}")
    runs.mark_done(d)


if __name__ == "__main__":
    main()
