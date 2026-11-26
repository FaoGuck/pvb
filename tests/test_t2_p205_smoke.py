"""P205 端到端冒烟：在合成的长表 / 电极表 / 观测累加量（临时目录）上跑 scripts/p205_baselines.py。

合成数据：6 条电极条（tests/test_t2_folds.grid），N1 概率随距离下降；B2 的第 1 个连接度对“有 N1”的单位植入信号
（观测池里 owner 之外的段也带着这个结构），所以 B3 应该比 B0+B1 好，置换后的增量应该 > 0。
运行：python tests/test_t2_p205_smoke.py（约 1–2 分钟）
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

SUB = "sub-ccepAgeSIM02"


def main():
    tmp = tempfile.mkdtemp(prefix="pvb_p205_")
    data, out = Path(tmp) / "data", Path(tmp) / "out"
    el, pairs = grid()
    rng = np.random.default_rng(0)
    chans = list(el.index)
    n = len(chans)
    ci = {c: i for i, c in enumerate(chans)}
    conn = rng.normal(0, 1, (n, n)); conn = (conn + conn.T) / 2          # 真实的“连接度”
    rows = []
    for p in pairs:
        e1, e2 = p.split("-")
        mid = el.loc[[e1, e2]].to_numpy(float).mean(axis=0)
        for ch in chans:
            if ch in (e1, e2):
                continue
            d = float(np.linalg.norm(el.loc[ch].to_numpy(float) - mid))
            z = -d / 12 + 1.5 * (conn[ci[e1], ci[ch]] + conn[ci[e2], ci[ch]]) / 2
            y = rng.random() < 1 / (1 + np.exp(-(z + 1)))
            a = -300.0 * y + rng.normal(0, 40)
            rows.append(dict(run_id=f"{SUB}_ses-1_run-011200", subject=SUB, stim_pair=p, channel=ch, n_trials=10,
                             n1_detected=bool(y), dist_mm=d, winmin_uv=a,
                             winmin_odd_uv=a + rng.normal(0, 40), winmin_even_uv=a + rng.normal(0, 40),
                             n1_detected_odd=bool(y), n1_detected_even=bool(y)))
    lab = pd.DataFrame(rows)
    (data / "t2/d5/ccep_labels").mkdir(parents=True); (data / "t2/d5/observation" / SUB).mkdir(parents=True)
    lab.to_parquet(data / "t2/d5/ccep_labels" / f"{SUB}_labels.parquet", index=False)
    el.reset_index().to_parquet(data / "t2/d5/observation" / f"{SUB}_electrodes.parquet", index=False)
    sums = np.stack([np.stack([conn + rng.normal(0, 0.3, (n, n)), rng.normal(0, 1, (n, n))]) for _ in pairs]) * 10
    np.savez_compressed(data / "t2/d5/observation" / SUB / f"{SUB}_ses-1_run-011200_obs.npz",
                        owners=np.array(pairs), measures=np.array(["corr_bb", "noise"]), channels=np.array(chans),
                        sums=sums.astype(np.float32), counts=np.full((len(pairs), n, n), 10, np.int32),
                        n_segments=np.full(len(pairs), 10, np.int32))
    cfg = yaml.safe_load(open(CODE / "configs/datasets/p_d5_ds004080.yaml", encoding="utf-8"))
    cfg["baselines"]["n_perm"] = 10
    cfgp = Path(tmp) / "cfg.yaml"
    yaml.safe_dump(cfg, open(cfgp, "w", encoding="utf-8"), allow_unicode=True)
    env = {**os.environ, "PRJ_CODE": str(CODE), "PRJ_DATA": str(data), "PRJ_OUT": str(out), "PRJ_GIT": "simtest",
           "PRJ_RUN_ID": "sim"}
    r = subprocess.run([sys.executable, "-X", "utf8", "scripts/p205_baselines.py", "--config", str(cfgp),
                        "--subjects", SUB], cwd=CODE, env=env, capture_output=True, text=True, encoding="utf-8")
    print(r.stdout[-3000:]); print(r.stderr[-3000:])
    assert r.returncode == 0
    res = pd.read_csv(glob.glob(str(out / "runs/P205_baselines_d5/*/auroc.tsv"))[0], sep="\t").iloc[0]
    assert res.auc_B0 > 0.65, res
    assert res.auc_B3 > res["auc_B0+B1"] and res.increment > 0, res
    print("ALL OK", tmp)


if __name__ == "__main__":
    main()
