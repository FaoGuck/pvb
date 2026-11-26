"""P110 第 3 步 / V5e：D4（ds004024）MEP 审计——每人 2 个 run，估计刺激侧、MEP 分布、可用试验比例（G0-T1 ②）。

设计在运行前写死（00 第 14 节 2026-11-27；参数全在 configs/datasets/p_d4_ds004024.yaml 的 p110）：
  挑 run：每人第一个会话（async4ms → 9ms → 14ms）的 spTMS run-01、run-02（100% rMT），共 26 个
  刺激侧：两导 EMG 在 20–40 ms 峰峰值中位数较大的一导 = 目标肌（EMG Right → 左 M1）
  每个脉冲：MEP（目标肌 20–40 ms 峰峰值）、预激活（−25 → −5 ms > 50 µV）、噪声（−60 → −40 ms）、
           EEG 质控（W3 = 脉冲前 3.01 → 0.01 s，复用 D1 的 pvb/preprocess/epoch.py：去均值 + 抗混叠到 1 000 Hz →
           60 Hz 陷波 + 1–45 Hz → 250 Hz → 平均参考；除 Fp1 / Fp2 外任一导峰峰值 > 150 µV 剔除；TP9 / TP10 不用，与 D1 相同）、
           guard（窗口起点距上一个脉冲 < 0.9 s 或早于记录起点）、MEP 极端值（run 内 ln MEP 去 51 试验滑动中位数后 MAD 规则）
  可用试验 = 不触发任一条（与 D1 keep_W3 相同，不要求 MEP 出现）；MEP 出现率（≥ 50 µV）另报
防泄漏：本步只做审计，EEG 只读脉冲前到 −10 ms 为止（epoch.py 有断言）；MEP 只用于描述，不参与任何特征

三种用法：
  云上（默认）：从 $PRJ_DATA/raw/_manifest/ds004024.keys.tsv 挑 run → 逐个 aws s3 cp（--no-sign-request）到 --tmp → 处理 → 删除信号文件
  本地冒烟：--local-dir <目录>（目录下按 BIDS 放好 vhdr / vmrk / eeg / events / channels，不下载、不删除）
  本地估计：--estimate <本步的运行目录>（读 trials.parquet + out/summary/p110_d4_subjects.tsv，写 estimate.md）
输出：$PRJ_OUT/runs/P110_d4_mep/<时间>_<git7>/  trials.parquet（每个脉冲一行）、runs.tsv（每个 run 一行）、
      waves.npz（平均 EMG 波形）、figures/、report.md、DONE；最后一行 SUMMARY
"""
from __future__ import annotations

import argparse
import fnmatch
import glob
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CODE = Path(os.environ["PRJ_CODE"])
sys.path.insert(0, str(CODE))
from pvb.preprocess import epoch as ep_mod  # noqa: E402
from pvb.probing import runs  # noqa: E402
from pvb.targets.mep import block_running_median  # noqa: E402

EXP_ID = "P110_d4_mep"
CFG_PATH = "configs/datasets/p_d4_ds004024.yaml"


def select_runs(keys_tsv: Path, cfg: dict) -> list[str]:
    """按写死的规则挑 run，返回 BIDS 基名（不含 _eeg.eeg），例如 sub-CON001/ses-async4ms/eeg/sub-CON001_ses-async4ms_task-spTMS_run-01。"""
    k = pd.read_csv(keys_tsv, sep="\t", header=None, names=["key", "bytes"])
    eeg = k[k.key.str.contains("task-spTMS") & k.key.str.endswith("_eeg.eeg")].key
    have = {}
    for key in eeg:
        sub, ses = key.split("/")[0], key.split("/")[1][4:]
        run = key.split("_run-")[1][:2]
        have.setdefault(sub, {}).setdefault(ses, set()).add(run)
    out = []
    for sub in sorted(have):
        ses = next(s for s in cfg["p110"]["session_order"] if s in have[sub])
        for r in cfg["p110"]["runs"]:
            if r in have[sub][ses]:
                out.append(f"{sub}/ses-{ses}/eeg/{sub}_ses-{ses}_task-spTMS_run-{r}")
    return out


def fetch(base: str, src: str, dst: Path, aws: str, log) -> None:
    for suf in ["_eeg.vhdr", "_eeg.vmrk", "_events.tsv", "_channels.tsv", "_eeg.eeg"]:
        p = dst / (base + suf)
        p.parent.mkdir(parents=True, exist_ok=True)
        for k in range(3):
            r = subprocess.run([aws, "s3", "cp", "--no-sign-request", "--only-show-errors", f"{src}/{base}{suf}", str(p)])
            if r.returncode == 0 and p.is_file():
                break
            log(f"  下载失败（第 {k + 1} 次）{base}{suf}，30 秒后重试"); time.sleep(30)
        else:
            raise RuntimeError(f"下载失败：{base}{suf}")


def ptp(x: np.ndarray, t_ms: np.ndarray, win) -> np.ndarray:
    m = (t_ms >= win[0]) & (t_ms <= win[1])
    s = x[..., m]
    return s.max(axis=-1) - s.min(axis=-1)


def process_run(root: Path, base: str, cfg: dict) -> tuple[pd.DataFrame, dict, np.ndarray]:
    import mne
    p = cfg["p110"]
    raw = mne.io.read_raw_brainvision(str(root / (base + "_eeg.vhdr")), preload=False, verbose="ERROR")
    fs = int(round(raw.info["sfreq"]))
    names = raw.ch_names
    ev = pd.read_csv(root / (base + "_events.tsv"), sep="\t", encoding="utf-8-sig")
    pulses = np.sort(ev.loc[ev.trial_type.astype(str).isin(cfg["pulse_types"]), "sample"].to_numpy(int))
    sub, ses = base.split("/")[0], base.split("/")[1][4:]
    run = base.split("_run-")[1][:2]
    emg_i = [names.index(cfg["emg_channels"]["right"]), names.index(cfg["emg_channels"]["left"])]
    # 通道类型以 channels.tsv 为准（MNE 读 BrainVision 时把 EMG / ECG / EOG 也当成 eeg，冒烟测试发现）
    cht = pd.read_csv(root / (base + "_channels.tsv"), sep="\t", encoding="utf-8-sig")
    is_eeg = dict(zip(cht["name"], (cht["type"] == "EEG") & (cht["status"] == "good")))
    eeg_names = [c for c in names if is_eeg.get(c, False) and c not in cfg["eeg"]["drop_channels"]]
    eeg_i = [names.index(c) for c in eeg_names]
    judge = np.array([c not in p["qc"]["eeg_ignore_channels"] for c in eeg_names])
    blink = ~judge

    # ---- EMG：每个脉冲 −60 → +80 ms（µV）----
    lo, hi = int(round(-0.06 * fs)), int(round(0.08 * fs))
    t_ms = np.arange(lo, hi) / fs * 1000
    emg = np.full((len(pulses), 2, hi - lo), np.nan)
    for k, pk in enumerate(pulses):
        if pk + lo >= 0 and pk + hi <= raw.n_times:
            emg[k] = raw.get_data(picks=emg_i, start=pk + lo, stop=pk + hi) * 1e6
    mep_both = ptp(emg, t_ms, p["side_window_ms"])                  # 脉冲 × 2（右、左）
    med = np.nanmedian(mep_both, axis=0)
    ti = int(np.nanargmax(med))                                     # 0 = EMG Right（左 M1），1 = EMG Left（右 M1）
    target = ["EMG Right", "EMG Left"][ti]

    # ---- EEG 质控（W3，复用 D1 的 epoch.py）----
    e = p["epoch"]
    ep_cfg = {"end_offset_s": e["end_offset_s"], "master_window_s": e["window_s"], "master_sfreq": e["master_sfreq"],
              "windows_s": [e["window_s"]], "guard_s": e["guard_s"], "clean": {
                  "sfreq": e["clean"]["sfreq"], "notch_hz": e["clean"]["notch_hz"], "notch_q": e["clean"]["notch_q"],
                  "band_hz": e["clean"]["band_hz"], "filter_order": e["clean"]["filter_order"]}}
    spatial = ep_mod.spatial_matrix(len(eeg_i), None, average_ref=True)
    read = lambda a, b: raw.get_data(picks=eeg_i, start=a, stop=b) * 1e6  # noqa: E731
    o = ep_mod.epoch_recording(read, raw.n_times, pulses, fs, ep_cfg, spatial, len(eeg_i))
    w = ep_mod.wname(e["window_s"])
    cw = o[f"clean_{w}"]
    cptp = cw.max(axis=-1) - cw.min(axis=-1)                        # 脉冲 × 导联
    eeg_max = np.where(o["ok_read"], np.nanmax(np.where(judge, cptp, -np.inf), axis=1), np.nan)
    blink_ptp = np.where(o["ok_read"], np.nanmax(np.where(blink, cptp, -np.inf), axis=1), np.nan) if blink.any() else np.nan

    q = p["qc"]
    mep = mep_both[:, ti]
    ln = np.log(np.maximum(mep, 1e-3))
    trend = block_running_median(ln, np.zeros(len(ln), int), q["detrend_window"])
    resid = ln - trend
    ok_e = np.isfinite(resid)
    dev = np.abs(resid - np.nanmedian(resid[ok_e]))
    mad = np.nanmedian(np.abs(resid[ok_e] - np.nanmedian(resid[ok_e])))
    prev_gap = np.concatenate([[np.nan], np.diff(pulses) / fs])
    df = pd.DataFrame({
        "subject": sub, "session": ses, "run": run, "intensity_rmt": cfg["intensity_rmt"][run], "target": target,
        "trial": np.arange(1, len(pulses) + 1), "pulse_sample": pulses, "onset_s": pulses / fs, "prev_gap_s": prev_gap,
        "mep_pp_right": mep_both[:, 0], "mep_pp_left": mep_both[:, 1], "mep_pp": mep,
        "pre_pp": ptp(emg[:, ti], t_ms, p["pre_window_ms"]), "noise_pp": ptp(emg[:, ti], t_ms, p["noise_window_ms"]),
        "ln_mep": ln, "mep_resid": resid, "eeg_ptp_max_judged": eeg_max, "blink_ptp": blink_ptp,
        "guard_ok": o[f"guard_ok_{w}"],
    })
    df["rej_guard"] = ~df.guard_ok
    df["rej_pre"] = df.pre_pp > q["pre_pp_max_uv"]
    df["rej_mad"] = dev > q["mep_mad"]["k"] * q["mep_mad"]["scale"] * mad
    df["rej_eeg"] = ~(df.eeg_ptp_max_judged <= q["eeg_ptp_max_uv"])
    df["keep"] = ~(df.rej_guard | df.rej_pre | df.rej_mad | df.rej_eeg)
    df["mep_present"] = df.mep_pp >= p["presence_uv"]
    info = dict(subject=sub, session=ses, run=run, target=target, stim_side={"EMG Right": "左 M1", "EMG Left": "右 M1"}[target],
                n_pulse=len(pulses), sfreq=fs, n_eeg_used=len(eeg_i),
                med_mep_right=float(med[0]), med_mep_left=float(med[1]),
                side_ratio=float(med[ti] / max(med[1 - ti], 1e-9)),
                mep_med=float(np.nanmedian(mep)), mep_q25=float(np.nanpercentile(mep, 25)), mep_q75=float(np.nanpercentile(mep, 75)),
                presence=float(df.mep_present.mean()), noise_med=float(df.noise_pp.median()),
                rej_guard=int(df.rej_guard.sum()), rej_pre=int(df.rej_pre.sum()), rej_mad=int(df.rej_mad.sum()),
                rej_eeg=int(df.rej_eeg.sum()), n_keep=int(df.keep.sum()), keep_rate=float(df.keep.mean()),
                presence_in_keep=float(df.loc[df.keep, "mep_present"].mean()) if df.keep.any() else np.nan)
    wv = np.nanmean(emg, axis=0)                                    # 2 × T 平均波形
    wm = (t_ms >= p["wave_ms"][0]) & (t_ms <= p["wave_ms"][1])
    return df, info, wv[:, wm][:, ::4].astype(np.float32)           # 20 kHz → 5 kHz 存图用


def plot(d: Path, rinfo: pd.DataFrame, waves: dict, trials: pd.DataFrame, wave_ms):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print("没有 matplotlib，跳过作图：", e); return
    (d / "figures").mkdir(exist_ok=True)
    n = len(rinfo); nc = 4; nr = int(np.ceil(n / nc))
    fig, ax = plt.subplots(nr, nc, figsize=(4 * nc, 2.4 * nr), squeeze=False)
    for a, (_, r) in zip(ax.flat, rinfo.iterrows()):
        key = f"{r.subject}_{r.session}_run-{r.run}"
        if key in waves:
            wv = waves[key]; t = np.linspace(wave_ms[0], wave_ms[1], wv.shape[1])
            a.plot(t, wv[0], lw=0.8, label="EMG Right"); a.plot(t, wv[1], lw=0.8, label="EMG Left")
            a.axvspan(20, 40, color="0.9", zorder=0)
            m = (t < -1) | (t > 8)                                  # y 轴范围不看 0–8 ms 的 TMS 伪迹
            lim = np.nanmax(np.abs(wv[:, m])) * 1.1
            if np.isfinite(lim) and lim > 0:
                a.set_ylim(-lim, lim)
        side = {"左 M1": "L-M1", "右 M1": "R-M1"}.get(r.stim_side, "?")   # 图里不用中文（容器没有中文字体）
        a.set_title(f"{r.subject[4:]} {r.session} r{r.run} -> {side}", fontsize=8); a.tick_params(labelsize=6)
    for a in list(ax.flat)[n:]:
        a.axis("off")
    ax.flat[0].legend(fontsize=6)
    fig.suptitle("Average EMG (uV; 0 = pulse; grey = MEP window 20-40 ms; y-range ignores 0-8 ms artifact)", fontsize=9)
    fig.tight_layout(); fig.savefig(d / "figures" / "avg_emg.png", dpi=120); plt.close(fig)
    fig, ax = plt.subplots(nr, nc, figsize=(4 * nc, 2.2 * nr), squeeze=False)
    for a, (_, r) in zip(ax.flat, rinfo.iterrows()):
        t = trials[(trials.subject == r.subject) & (trials.session == r.session) & (trials.run == r.run)]
        a.hist(np.log10(t.mep_pp.clip(lower=1)), bins=30, color="C0", alpha=0.8, label="MEP 20-40 ms")
        a.hist(np.log10(t.noise_pp.clip(lower=1)), bins=30, color="C3", alpha=0.5, label="noise -60..-40 ms")
        a.axvline(np.log10(50), color="k", lw=0.6)
        a.set_title(f"{r.subject[4:]} r{r.run} present {r.presence:.2f} keep {r.keep_rate:.2f}", fontsize=8); a.tick_params(labelsize=6)
    for a in list(ax.flat)[n:]:
        a.axis("off")
    ax.flat[0].legend(fontsize=6)
    fig.suptitle("Target-muscle peak-to-peak, log10(uV); line = 50 uV", fontsize=9)
    fig.tight_layout(); fig.savefig(d / "figures" / "mep_hist.png", dpi=120); plt.close(fig)


def estimate(run_dir: Path, summary_dir: Path, cfg: dict):
    """本地：每人每侧可用数 ≈ 该侧 run 的保留率 × 该侧脉冲上限（元数据审计），判 G0-T1 ② 的预计结果。"""
    ri = pd.read_csv(run_dir / "runs.tsv", sep="\t", dtype={"run": str})
    ri = ri[ri.error.isna()] if "error" in ri else ri
    sb = pd.read_csv(summary_dir / "p110_d4_subjects.tsv", sep="\t", index_col=0)
    g = cfg["p110"]["g0_t1_2"]
    rows = []
    for sub in sb.index:
        r = ri[ri.subject == sub]
        for side in ["左 M1", "右 M1"]:
            rs = r[r.stim_side == side]
            kr = float(rs.n_keep.sum() / rs.n_pulse.sum()) if len(rs) else np.nan
            rows.append(dict(subject=sub, side=side, runs_audited=len(rs), keep_rate=kr,
                             presence=float(np.average(rs.presence, weights=rs.n_pulse)) if len(rs) else np.nan,
                             upper_100=int(sb.loc[sub, "one_side_100_upper"]), upper_all=int(sb.loc[sub, "one_side_all_upper"]),
                             est_100=kr * sb.loc[sub, "one_side_100_upper"] if len(rs) else np.nan,
                             est_all=kr * sb.loc[sub, "one_side_all_upper"] if len(rs) else np.nan))
    e = pd.DataFrame(rows)
    best = e.sort_values("est_100", ascending=False).groupby("subject").head(1).set_index("subject").sort_index()
    n100 = int((best.est_100 >= g["min_trials"]).sum()); nall = int((best.est_all >= g["min_trials"]).sum())
    ok = n100 >= g["min_subjects"]
    L = ["# P110 第 3 步：G0-T1 ② 的估计（本地；运行目录 " + run_dir.name + "）", "",
         f"- 规则（00 第 14 节 2026-11-27，运行前写死）：每人每侧可用数 ≈ 该侧审计 run 的保留率 × 该侧 100% rMT 脉冲上限；每人取较大的一侧；"
         f"≥ {g['min_trials']} 的被试 ≥ {g['min_subjects']} 人 → 预计通过",
         f"- **只用 100% rMT：{n100} / {len(best)} 人 ≥ {g['min_trials']} → G0-T1 ② 预计{'通过' if ok else '不通过'}**"
         f"；加上 110% rMT：{nall} / {len(best)} 人（另报）",
         "- 最终数字在 P111 / P112 全量派生后确认", "", "## 每人每侧", "", e.round(3).to_string(index=False), "",
         "## 每人取较大一侧", "", best.round(3).to_string()]
    (run_dir / "estimate.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    e.to_csv(run_dir / "estimate.tsv", sep="\t", index=False)
    print("\n".join(L))
    print(f"SUMMARY exp={EXP_ID}_estimate n_ge200_100={n100}/{len(best)} n_ge200_all={nall}/{len(best)} G0_T1_2_est={'PASS' if ok else 'FAIL'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--local-dir", help="本地冒烟：直接处理这个目录里的 run（不下载、不删除）")
    ap.add_argument("--estimate", help="本地估计：本步的运行目录")
    ap.add_argument("--tmp", default="/osmo/run/workspace/tmp_p110", help="云上临时目录（信号文件处理完即删）")
    ap.add_argument("--aws", default="/osmo/run/workspace/envs/s2r/bin/aws")
    ap.add_argument("--max-runs", type=int, default=0, help="只处理前 N 个（冒烟用）")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(CODE / CFG_PATH, encoding="utf-8"))
    if a.estimate:
        estimate(Path(a.estimate), Path(os.environ["PRJ_OUT"]) / "summary", cfg); return

    data = Path(os.environ["PRJ_DATA"])
    if a.local_dir:
        root = Path(a.local_dir)
        bases = sorted(str(Path(p).relative_to(root)).replace(os.sep, "/")[: -len("_eeg.vhdr")]
                       for p in glob.glob(str(root / "sub-*" / "ses-*" / "eeg" / "*task-spTMS*_eeg.vhdr")))
    else:
        root = Path(a.tmp); root.mkdir(parents=True, exist_ok=True)
        bases = select_runs(data / cfg["manifest"], cfg)
    if a.max_runs:
        bases = bases[: a.max_runs]
    d = runs.new_run_dir(Path(os.environ["PRJ_OUT"]), EXP_ID)
    runs.write_snapshot(d, {**cfg, "_resolved": {"runs": bases, "local_dir": a.local_dir, "git": runs.git_commit(),
                                                 "run_id": os.environ.get("PRJ_RUN_ID")}})
    log = runs.Log(d)
    log(f"[P110] run_dir={d} runs={len(bases)} mode={'local' if a.local_dir else 'cloud'}")
    trials, infos, waves = [], [], {}
    for i, b in enumerate(bases, 1):
        t0 = time.time()
        try:
            if not a.local_dir:
                fetch(b, cfg["source"], root, a.aws, log)
            df, info, wv = process_run(root, b, cfg)
            trials.append(df); infos.append(info); waves[f"{info['subject']}_{info['session']}_run-{info['run']}"] = wv
            log(f"[P110] {i}/{len(bases)} {b.split('/')[-1]} OK n={info['n_pulse']} 目标 {info['target']}（{info['stim_side']}，"
                f"中位 {info['mep_med']:.0f} µV，两侧比 {info['side_ratio']:.1f}）出现率 {info['presence']:.2f} "
                f"保留 {info['n_keep']}/{info['n_pulse']}（guard {info['rej_guard']} 预激活 {info['rej_pre']} MAD {info['rej_mad']} "
                f"EEG {info['rej_eeg']}）噪声中位 {info['noise_med']:.1f} µV [{time.time() - t0:.0f}s]")
        except Exception as e:  # noqa: BLE001  逐 run 容错，与 S2R 约定一致
            sub = b.split("/")[0]
            infos.append(dict(subject=sub, session=b.split("/")[1][4:], run=b.split("_run-")[1][:2], error=repr(e)))
            log(f"[P110] {i}/{len(bases)} {b.split('/')[-1]} FAIL {e!r}"); log(traceback.format_exc())
        finally:
            if not a.local_dir:
                for f in glob.glob(str(root / (b + "_eeg.eeg"))):
                    os.remove(f)                                    # 原始信号不保留（共享规则第 5 条）
    ri = pd.DataFrame(infos)
    if "error" not in ri:
        ri["error"] = np.nan
    ri.to_csv(d / "runs.tsv", sep="\t", index=False)
    if trials:
        tr = pd.concat(trials, ignore_index=True)
        tr.to_parquet(d / "trials.parquet", index=False)
        np.savez_compressed(d / "waves.npz", **waves)
        plot(d, ri[ri.error.isna()], waves, tr, cfg["p110"]["wave_ms"])
    ok = ri.error.isna()
    good = ri[ok]
    rep = [f"# {EXP_ID}：D4 MEP 审计（每人 2 个 run）", "",
           f"- 成功 {int(ok.sum())} / {len(ri)} 个 run；设计见 00 第 14 节 2026-11-27",
           f"- 目标侧：{good.stim_side.value_counts().to_dict() if len(good) else {}}；同一人两个 run 同侧的："
           f"{int((good.groupby('subject').stim_side.nunique() == 1).sum()) if len(good) else 0} 人",
           f"- 保留率中位 {good.keep_rate.median():.3f}；MEP 出现率（≥ 50 µV）中位 {good.presence.median():.3f}；噪声中位 {good.noise_med.median():.1f} µV"
           if len(good) else "- 无成功的 run", "",
           (good.drop(columns=["error"]).round(3).to_string(index=False) if len(good) else ""), "",
           ("## 失败\n\n" + ri[~ok][["subject", "session", "run", "error"]].to_string(index=False)) if (~ok).any() else "",
           "", "G0-T1 ② 的估计在本地跑：`scripts/p110_d4_mep_audit.py --estimate <本目录>`（需要 out/summary/p110_d4_subjects.tsv）"]
    (d / "report.md").write_text("\n".join(rep) + "\n", encoding="utf-8")
    log("\n".join(rep))
    log(f"SUMMARY exp={EXP_ID} runs_ok={int(ok.sum())}/{len(ri)} keep_med={good.keep_rate.median() if len(good) else float('nan'):.3f} "
        f"presence_med={good.presence.median() if len(good) else float('nan'):.3f} run_dir={d}")
    if not a.local_dir:
        shutil.rmtree(root, ignore_errors=True)
    runs.mark_done(d)


if __name__ == "__main__":
    main()
