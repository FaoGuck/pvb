"""P205b（响应电极幅度基线）的防泄漏检查 + 端到端冒烟（临时目录，不碰项目的 data / out）。

合成数据：6 条电极条（tests/test_t2_folds.grid）；N1 概率随距离下降；有 N1 的单位的幅度 = 距离项 + 连接度项 + 噪声，
观测池里的第 1 个连接度带着同一个“真实连接度”，所以 B0+B2 / B3 的 Spearman 应高于 B0+B1，置换增量 > 0。
1. 防泄漏：把测试折自己单位的幅度（目标）和窗内最小值（B1 的来源）全部改掉，测试折的预测必须逐位不变
2. 端到端：跑 scripts/p205b_amplitude.py，检查输出与 SUMMARY
运行：python tests/test_t2_p205b_smoke.py（约 1 分钟）
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
from test_t2_folds import grid  # noqa: E402
from pvb.t2 import baselines as bl, folds as fo  # noqa: E402

SUB = "sub-ccepAgeSIM03"


def make(data: Path, seed: int = 0):
    el, pairs = grid()
    rng = np.random.default_rng(seed)
    chans = list(el.index); n = len(chans); ci = {c: i for i, c in enumerate(chans)}
    conn = rng.normal(0, 1, (n, n)); conn = (conn + conn.T) / 2
    rows = []
    for p in pairs:
        e1, e2 = p.split("-")
        mid = el.loc[[e1, e2]].to_numpy(float).mean(axis=0)
        for ch in chans:
            if ch in (e1, e2):
                continue
            d = float(np.linalg.norm(el.loc[ch].to_numpy(float) - mid))
            c = (conn[ci[e1], ci[ch]] + conn[ci[e2], ci[ch]]) / 2
            y = rng.random() < 1 / (1 + np.exp(-(-d / 12 + 2.5)))          # 有无只看距离（让响应单位够多）
            amp = np.exp(5.5 - d / 40 + 0.5 * c) if y else 30.0            # 幅度 = 距离 + 连接度
            a = -amp * np.exp(rng.normal(0, 0.15))
            rows.append(dict(run_id=f"{SUB}_ses-1_run-011200", subject=SUB, stim_pair=p, channel=ch, n_trials=10,
                             n1_detected=bool(y), dist_mm=d, winmin_uv=a,
                             winmin_odd_uv=a * np.exp(rng.normal(0, 0.15)), winmin_even_uv=a * np.exp(rng.normal(0, 0.15)),
                             n1_detected_odd=bool(y), n1_detected_even=bool(y)))
    lab = pd.DataFrame(rows)
    (data / "t2/d5/ccep_labels").mkdir(parents=True); (data / "t2/d5/observation" / SUB).mkdir(parents=True)
    lab.to_parquet(data / "t2/d5/ccep_labels" / f"{SUB}_labels.parquet", index=False)
    el.reset_index().to_parquet(data / "t2/d5/observation" / f"{SUB}_electrodes.parquet", index=False)
    sums = np.stack([np.stack([conn + rng.normal(0, 0.3, (n, n)), rng.normal(0, 1, (n, n))]) for _ in pairs]) * 10
    acc = dict(owners=np.array(pairs), measures=np.array(["corr_bb", "noise"]), channels=np.array(chans),
               sums=sums.astype(np.float32), counts=np.full((len(pairs), n, n), 10, np.int32),
               n_segments=np.full(len(pairs), 10, np.int32))
    np.savez_compressed(data / "t2/d5/observation" / SUB / f"{SUB}_ses-1_run-011200_obs.npz", **acc)
    return lab, el, pairs, acc


def test_no_leak(lab, el, pairs, acc, cfg):
    thr = fo.adjacency_threshold(el[["x", "y", "z"]].to_numpy(float), cfg["folds"]["adjacent_factor"])
    u = lab[fo.eval_mask(lab, el, thr)].reset_index(drop=True)
    fold_of = fo.assign_folds(pairs, cfg["folds"]["k"], cfg["folds"]["seed"])
    pool = bl.ObsPool([acc])
    cb = {**cfg["baselines"], **cfg["amplitude"]}

    def run(uu):
        r = uu[uu.n1_detected].reset_index(drop=True)
        r = r.assign(amp_log=np.log(np.maximum(-r.winmin_uv.to_numpy(), 1.0)))
        p, _ = bl.cv_predict(r, fold_of, pool, ["B0+B1", "B3"], cb, target="amp_log", task="reg", source=uu)
        return r, p

    r0, p0 = run(u)
    f0 = r0.stim_pair.map(fold_of).to_numpy() == 0
    assert f0.sum() > 10 and np.isfinite(p0["B3"]).all()
    u2 = u.copy()
    m = u2.stim_pair.map(fold_of).to_numpy() == 0
    u2.loc[m, "winmin_uv"] = -np.random.default_rng(1).uniform(100, 3000, m.sum())     # 改掉测试折自己的目标
    r2, p2 = run(u2)
    assert (r2.stim_pair.to_numpy() == r0.stim_pair.to_numpy()).all()
    for s in p0:
        assert np.array_equal(p0[s][f0], p2[s][f0]), f"{s}：测试折的预测用到了自己的幅度"
        assert not np.array_equal(p0[s][~f0], p2[s][~f0]), f"{s}：其他折的预测没变，测试不灵敏"
    print("no-leak OK", int(f0.sum()), "个测试折单位")


def main():
    tmp = tempfile.mkdtemp(prefix="pvb_p205b_")
    data, out = Path(tmp) / "data", Path(tmp) / "out"
    lab, el, pairs, acc = make(data)
    cfg = yaml.safe_load(open(CODE / "configs/datasets/p_d5_ds004080.yaml", encoding="utf-8"))
    cfg["baselines"]["n_perm"] = 10
    test_no_leak(lab, el.copy(), pairs, acc, cfg)
    cfgp = Path(tmp) / "cfg.yaml"
    yaml.safe_dump(cfg, open(cfgp, "w", encoding="utf-8"), allow_unicode=True)
    env = {**os.environ, "PRJ_CODE": str(CODE), "PRJ_DATA": str(data), "PRJ_OUT": str(out), "PRJ_GIT": "simtest",
           "PRJ_RUN_ID": "sim"}
    r = subprocess.run([sys.executable, "-X", "utf8", "scripts/p205b_amplitude.py", "--config", str(cfgp),
                        "--subjects", SUB], cwd=CODE, env=env, capture_output=True, text=True, encoding="utf-8")
    print(r.stdout[-3000:]); print(r.stderr[-3000:])
    assert r.returncode == 0
    res = pd.read_csv(glob.glob(str(out / "runs/P205b_amplitude_d5/*/amp.tsv"))[0], sep="\t").iloc[0]
    assert res.n_resp >= 100 and res.ceiling_rho > 0.8, res
    assert res.rho_B0 > 0.2 and res["rho_B0+B2"] > res["rho_B0+B1"] + 0.05 and res.increment > 0, res
    assert glob.glob(str(out / "runs/P205b_amplitude_d5/*/DONE"))
    log = open(glob.glob(str(out / "runs/P205b_amplitude_d5/*/log.txt"))[0], encoding="utf-8").read()
    assert "SUMMARY exp=P205b_amplitude_d5 subjects=1/1" in log
    print("ALL OK", tmp)


if __name__ == "__main__":
    main()
