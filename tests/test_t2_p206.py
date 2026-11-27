"""P206 框架（pvb/t2/fm.py、scripts/p206_extract.py、scripts/p206_eval.py）的防泄漏检查与端到端冒烟（2026-11-27）。

提取（00 第 6 节第 2 条的两项替换测试，用确定性的 spec_test 编码器，数据与 tests/test_t2_observation.py 相同）：
  1. 把每个脉冲后 T_tail 以内（及下一个脉冲前 pre_gap 以内）换成噪声 → 全部表征累加量逐位不变
  2. 把留出刺激对 X 的整段刺激期换成噪声 → 其他 owner 的累加量逐位不变；不含 X 的观测池逐位不变；X 自己的变了
  3. owner 的刺激电极在它自己的段里被屏蔽（计数为 0）
评估（合成的标签 + 表征文件；幅度 = 距离项 + 记录电极的一个“表征可见”的属性 + 噪声）：
  4. 改掉测试折自己单位的目标 → 测试折的预测逐位不变（其他折变了 = 测试灵敏）
  5. 改掉测试折 owner 的表征累加量 → 测试折的预测逐位不变（表征只来自可用折的 owner）
端到端：
  6. scripts/p206_extract.py 在 tests/sim_d5.py 的模拟 BrainVision run 上跑通（spec_test），输出形状正确
  7. scripts/p206_eval.py 跑通；基线 B0 / B3 的 Spearman 与 scripts/p205b_amplitude.py 逐位相同；B0+M 明显好于 B0
运行：python tests/test_t2_p206.py（约 1–2 分钟）
"""
from __future__ import annotations

import glob
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE)); sys.path.insert(0, str(CODE / "tests"))
import test_t2_observation as TO  # noqa: E402
from test_t2_folds import grid  # noqa: E402
from pvb.t2 import baselines as bl, encoders as encs, fm, folds as fo, io_d5, observation as obs  # noqa: E402

ENC = {"spec": encs.SpecTest(sfreq=256)}
SUB = "sub-ccepAgeSIM06"


def emb_of(x, stim):
    segs = obs.segment_table(stim, TO.SF, x.shape[1], TO.CFG)
    return fm.owner_embed_accumulators(TO.FakeReader(x), TO.NAMES, list(range(len(TO.NAMES))), segs, TO.SF, TO.CFG, ENC)["spec"]


def same(a, b):
    return all(np.array_equal(a[k], b[k]) for k in ("emb_sums", "emb_counts", "sim_sums", "sim_counts"))


def test_extract_tail_noise():
    x, stim = TO.make()
    a0 = emb_of(x, stim)
    assert a0["dim"] == 5 and a0["emb_counts"].sum() > 0 and a0["sim_counts"].sum() > 0
    rng = np.random.default_rng(1)
    y = x.copy()
    g = int(round(TO.CFG["pre_gap"] * TO.SF))
    for s in stim.sample0.to_numpy():
        e = s + int(round(TO.CFG["t_tail"] * TO.SF))
        y[:, s:e] = rng.normal(0, 500, (len(TO.NAMES), e - s))
        y[:, s - g:s] = rng.normal(0, 500, (len(TO.NAMES), g))
    assert same(a0, emb_of(y, stim))


def test_extract_heldout_period():
    x, stim = TO.make()
    a0 = emb_of(x, stim)
    X = io_d5.pair_name("E04-E03")
    first = stim[stim.pair == X].sample0.min()
    nxt = stim[stim.sample0 > stim[stim.pair == X].sample0.max()].sample0.min()
    y = x.copy()
    y[:, first:nxt] = np.random.default_rng(2).normal(0, 500, (len(TO.NAMES), nxt - first))
    a1 = emb_of(y, stim)
    k = list(a0["owners"]).index(X)
    oth = [i for i in range(len(a0["owners"])) if i != k]
    for key in ("emb_sums", "sim_sums"):
        assert np.array_equal(a0[key][oth], a1[key][oth])
    assert not np.array_equal(a0["emb_sums"][k], a1["emb_sums"][k])          # 测试是灵敏的
    p0, p1 = fm.EmbedPool([a0]), fm.EmbedPool([a1])
    allowed = set(p0.owners) - {X}
    E0, S0 = p0.mean(p0.owner_mask(allowed)); E1, S1 = p1.mean(p1.owner_mask(allowed))
    assert np.array_equal(E0, E1, equal_nan=True) and np.array_equal(S0, S1, equal_nan=True)


def test_laplacian_nearest2():
    """BrainBERT 的重参考：一条 10 mm 间距的电极条，中间电极 = 自己 − 两侧均值；缺坐标的为 NaN。"""
    xyz = np.array([[0, 0, 0], [10, 0, 0], [20, 0, 0], [30, 0, 0], [np.nan] * 3], float)
    x = np.arange(5 * 4, dtype=float).reshape(5, 4) ** 2
    y = encs.laplacian_nearest2(x, xyz)
    assert np.allclose(y[1], x[1] - 0.5 * (x[0] + x[2])) and np.allclose(y[2], x[2] - 0.5 * (x[1] + x[3]))
    assert np.isnan(y[4]).all() and np.isfinite(y[:4]).all()


def test_extract_masks_own_stim():
    x, stim = TO.make()
    a0 = emb_of(x, stim)
    k = list(a0["owners"]).index("E01-E02")
    ci = {c: i for i, c in enumerate(TO.NAMES)}
    assert a0["emb_counts"][k][ci["E01"]] == 0 and a0["emb_counts"][k][ci["E02"]] == 0
    assert a0["emb_counts"][k][ci["E05"]] > 0


# ---------------- 评估 ----------------

def make_eval(data: Path, seed: int = 0, D: int = 6):
    """标签 + 电极 + 观测累加量（与 test_t2_p205b_smoke 同一种合成）+ 表征累加量。"""
    el, pairs = grid()
    rng = np.random.default_rng(seed)
    chans = list(el.index); n = len(chans); ci = {c: i for i, c in enumerate(chans)}
    V = rng.normal(0, 1, (n, D))                         # 每个电极的“真实表征”
    conn = rng.normal(0, 1, (n, n)); conn = (conn + conn.T) / 2
    rows = []
    for p in pairs:
        e1, e2 = p.split("-")
        mid = el.loc[[e1, e2]].to_numpy(float).mean(axis=0)
        for ch in chans:
            if ch in (e1, e2):
                continue
            d = float(np.linalg.norm(el.loc[ch].to_numpy(float) - mid))
            yv = rng.random() < 1 / (1 + np.exp(-(-d / 12 + 2.5)))
            amp = np.exp(5.5 - d / 40 + 0.6 * V[ci[ch], 0]) if yv else 30.0      # 幅度 = 距离 + 记录电极的表征属性
            a = -amp * np.exp(rng.normal(0, 0.15))
            rows.append(dict(run_id=f"{SUB}_ses-1_run-011200", subject=SUB, stim_pair=p, channel=ch, n_trials=10,
                             n1_detected=bool(yv), dist_mm=d, winmin_uv=a,
                             winmin_odd_uv=a * np.exp(rng.normal(0, 0.15)), winmin_even_uv=a * np.exp(rng.normal(0, 0.15)),
                             n1_detected_odd=bool(yv), n1_detected_even=bool(yv)))
    lab = pd.DataFrame(rows)
    (data / "t2/d5/ccep_labels").mkdir(parents=True); (data / "t2/d5/observation" / SUB).mkdir(parents=True)
    lab.to_parquet(data / "t2/d5/ccep_labels" / f"{SUB}_labels.parquet", index=False)
    el.reset_index().to_parquet(data / "t2/d5/observation" / f"{SUB}_electrodes.parquet", index=False)
    sums = np.stack([np.stack([conn + rng.normal(0, 0.3, (n, n)), rng.normal(0, 1, (n, n))]) for _ in pairs]) * 10
    acc = dict(owners=np.array(pairs), measures=np.array(["corr_bb", "noise"]), channels=np.array(chans),
               sums=sums.astype(np.float32), counts=np.full((len(pairs), n, n), 10, np.int32),
               n_segments=np.full(len(pairs), 10, np.int32))
    np.savez_compressed(data / "t2/d5/observation" / SUB / f"{SUB}_ses-1_run-011200_obs.npz", **acc)
    emb = np.stack([(V + rng.normal(0, 0.3, (n, D))) * 10 for _ in pairs]).astype(np.float32)
    ea = dict(owners=np.array(pairs), channels=np.array(chans), emb_sums=emb,
              emb_counts=np.full((len(pairs), n), 10, np.int32), sim_sums=np.zeros((len(pairs), n, n), np.float32),
              sim_counts=np.zeros((len(pairs), n, n), np.int32), n_segments=np.full(len(pairs), 10, np.int32),
              dim=np.int64(D), seconds_model=np.float64(0), n_chseg=np.int64(0), seconds_read=np.float64(0))
    (data / "t2/d5/fm/synth" / SUB).mkdir(parents=True)
    np.savez_compressed(data / "t2/d5/fm/synth" / SUB / f"{SUB}_ses-1_run-011200_emb.npz", **ea)
    return lab, el, pairs, acc, ea


def test_eval_no_leak(lab, el, pairs, acc, ea, cfg):
    thr = fo.adjacency_threshold(el[["x", "y", "z"]].to_numpy(float), cfg["folds"]["adjacent_factor"])
    u = lab[fo.eval_mask(lab, el, thr)].reset_index(drop=True)
    fold_of = fo.assign_folds(pairs, cfg["folds"]["k"], cfg["folds"]["seed"])
    pool = bl.ObsPool([acc])
    cb = {**cfg["baselines"], **cfg["amplitude"]}
    FS = {"B0+M": ["B0", "Mh", "Ms"], "B3+M": ["B0", "B1", "B2", "Mh", "Ms"]}

    def run(uu, eacc):
        r = uu[uu.n1_detected].reset_index(drop=True)
        r = r.assign(amp_log=np.log(np.maximum(-r.winmin_uv.to_numpy(), 1.0)))
        kw = dict(feature_sets=FS, extra=fm.make_extra(fm.EmbedPool([eacc])), pca_blocks=("Mh",), n_pc=8)
        p, _ = bl.cv_predict(r, fold_of, pool, list(FS), cb, target="amp_log", task="reg", source=uu, **kw)
        return r, p

    r0, p0 = run(u, ea)
    f0 = r0.stim_pair.map(fold_of).to_numpy() == 0
    assert f0.sum() > 10 and all(np.isfinite(p0[s]).all() for s in p0)
    # 4. 改测试折自己的目标
    u2 = u.copy()
    m = u2.stim_pair.map(fold_of).to_numpy() == 0
    u2.loc[m, "winmin_uv"] = -np.random.default_rng(1).uniform(100, 3000, m.sum())
    _, p2 = run(u2, ea)
    for s in p0:
        assert np.array_equal(p0[s][f0], p2[s][f0]), f"{s}：测试折的预测用到了自己的目标"
        assert not np.array_equal(p0[s][~f0], p2[s][~f0]), f"{s}：其他折没变，测试不灵敏"
    # 5. 改测试折 owner 的表征
    e3 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in ea.items()}
    own0 = np.array([fold_of[o] == 0 for o in ea["owners"]])
    e3["emb_sums"][own0] = np.random.default_rng(3).normal(0, 50, e3["emb_sums"][own0].shape).astype(np.float32)
    _, p3 = run(u, e3)
    for s in p0:
        assert np.array_equal(p0[s][f0], p3[s][f0]), f"{s}：测试折的预测用到了测试折刺激对的观测段表征"
        assert not np.array_equal(p0[s][~f0], p3[s][~f0]), f"{s}：其他折没变，测试不灵敏"
    print("eval no-leak OK", int(f0.sum()), "个测试折单位")


def env_for(data, out):
    return {**os.environ, "PRJ_CODE": str(CODE), "PRJ_DATA": str(data), "PRJ_OUT": str(out), "PRJ_GIT": "simtest",
            "PRJ_RUN_ID": "sim", "MKL_NUM_THREADS": "1"}


def run_script(args, env):
    r = subprocess.run([sys.executable, "-X", "utf8"] + args, cwd=CODE, env=env, capture_output=True, text=True,
                       encoding="utf-8")
    print(r.stdout[-2500:]); print(r.stderr[-2500:])
    assert r.returncode == 0, args
    return r


def test_extract_e2e(tmp: Path):
    import sim_d5
    data, out = tmp / "ext/data", tmp / "ext/out"
    sim_d5.make_sim(str(data / "raw/ds004080"))
    for workers in (1, 2):                             # 2 = 子进程分片（第二次全部已存在 → 跳过）
        run_script(["scripts/p206_extract.py", "--models", "spec_test", "--subjects", sim_d5.SUB,
                    "--workers", str(workers)], env_for(data, out / f"w{workers}"))   # 运行目录精确到分钟，分开放
    f = glob.glob(str(data / f"t2/d5/fm/spec_test/{sim_d5.SUB}/*_emb.npz"))
    assert len(f) == 1
    a = dict(np.load(f[0]))
    ch = list(a["channels"])
    assert "PT06" not in ch and len(ch) == 7                       # bad 通道不读
    assert a["dim"] == 5 and a["emb_sums"].shape == (len(a["owners"]), 7, 5)
    k = list(a["owners"]).index("PT01-PT02")
    assert a["emb_counts"][k][ch.index("PT01")] == 0 and a["emb_counts"][k][ch.index("PT03")] > 0
    l1, l2 = (open(glob.glob(str(out / f"w{w}/runs/P206_extract_d5/*/log.txt"))[0], encoding="utf-8").read() for w in (1, 2))
    assert "new=1 skip=0 err=0" in l1 and "new=0 skip=1 err=0" in l2, (l1[-300:], l2[-300:])
    print("extract e2e OK", a["emb_counts"].sum(), "个通道段")


def test_eval_e2e(tmp: Path, cfg: dict):
    data, out = tmp / "ev/data", tmp / "ev/out"
    make_eval(data)
    cfg = {**cfg, "baselines": {**cfg["baselines"], "n_perm": 2}}
    cfgp = tmp / "cfg.yaml"
    yaml.safe_dump(cfg, open(cfgp, "w", encoding="utf-8"), allow_unicode=True)
    env = env_for(data, out)
    run_script(["scripts/p206_eval.py", "--config", str(cfgp), "--models", "synth,missing", "--subjects", SUB], env)
    run_script(["scripts/p205b_amplitude.py", "--config", str(cfgp), "--subjects", SUB], env)
    met = pd.read_csv(glob.glob(str(out / "runs/P206_eval_d5/*/metrics.csv"))[0])
    v = {(r.model, r.feature_set, r.metric): r.value for r in met.itertuples()}
    amp = pd.read_csv(glob.glob(str(out / "runs/P205b_amplitude_d5/*/amp.tsv"))[0], sep="\t").iloc[0]
    M = "spearman_amp_resp"
    for s in ("B0", "B0+B2", "B3"):
        assert v[("baseline", s, M)] == amp[f"rho_{s}"], (s, v[("baseline", s, M)], amp[f"rho_{s}"])
    assert v[("synth", "B0+M", M)] > v[("baseline", "B0", M)] + 0.05, v
    assert np.isfinite(v[("synth", "B0+M", "spearman_amp_within_pair")]) and np.isfinite(v[("synth", "B0+M", "auroc")])
    log = open(glob.glob(str(out / "runs/P206_eval_d5/*/log.txt"))[0], encoding="utf-8").read()
    assert "SUMMARY exp=P206_eval_d5 subjects=1/1" in log and "missing: 没有表征文件" in log
    assert glob.glob(str(out / "runs/P206_eval_d5/*/DONE"))
    print("eval e2e OK", {k: round(v[k], 3) for k in [("baseline", "B0", M), ("baseline", "B3", M), ("synth", "B0+M", M),
                                                       ("synth", "B3+M", M)]})


def main():
    for f in [test_extract_tail_noise, test_extract_heldout_period, test_extract_masks_own_stim, test_laplacian_nearest2]:
        f(); print("OK", f.__name__)
    tmp = Path(tempfile.mkdtemp(prefix="pvb_p206_"))
    cfg = yaml.safe_load(open(CODE / "configs/datasets/p_d5_ds004080.yaml", encoding="utf-8"))
    lab, el, pairs, acc, ea = make_eval(tmp / "nl/data")
    test_eval_no_leak(lab, el.copy(), pairs, acc, ea, cfg)
    test_extract_e2e(tmp)
    test_eval_e2e(tmp, cfg)
    print("ALL OK", tmp)


if __name__ == "__main__":
    main()
