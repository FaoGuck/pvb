"""P201 / P202 的冒烟测试：在模拟 run（tests/sim_d5.py）上跑两个脚本，检查找回了植入的 N1。

本地运行（约 10 秒）：
  python tests/test_t2_ccep_sim.py            （用临时目录，不碰项目的 data / out）
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

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "tests"))
import sim_d5  # noqa: E402


def run(step: str, env: dict, *args):
    r = subprocess.run([sys.executable, "-X", "utf8", step, "--subjects", sim_d5.SUB, "--procs", "1", *args],
                       cwd=CODE, env=env, capture_output=True, text=True, encoding="utf-8")
    print(r.stdout[-3000:]); print(r.stderr[-3000:])
    assert r.returncode == 0, step


def main(sf: int = 2048):
    tmp = tempfile.mkdtemp(prefix="pvb_sim_")
    data, out = os.path.join(tmp, "data"), os.path.join(tmp, "out")
    sim_d5.make_sim(os.path.join(data, "raw", "ds004080"), SF=sf)
    env = {**os.environ, "PRJ_CODE": str(CODE), "PRJ_DATA": data, "PRJ_OUT": out, "PRJ_GIT": "simtest", "PRJ_RUN_ID": "sim"}

    run("scripts/p201_d5_audit.py", env)
    a = pd.read_csv(glob.glob(os.path.join(out, "runs", "P201_d5_audit", "*", "runs.tsv"))[0], sep="\t").iloc[0]
    assert a.sfreq_vhdr == sf and a.names_same_order, a
    assert abs(a.art_lat_med_ms) <= 2.0, f"伪迹没对齐：{a.art_lat_med_ms}"
    assert a.sample_start_minus_onset.startswith("1:"), a.sample_start_minus_onset
    st = open(glob.glob(os.path.join(out, "runs", "P201_d5_audit", "*", "avccep_structure.txt"))[0], encoding="utf-8").read()
    assert "n1_peak_sample" in st and "stimpnames" in st

    run("scripts/p201b_timing.py", env)
    b = pd.read_csv(glob.glob(os.path.join(out, "runs", "P201b_d5_timing", "*", "runs.tsv"))[0], sep="\t").iloc[0]
    assert abs(b.edge_lat_med_ms) <= 1000 / sf + 1e-6, f"邻近电极的伪迹边沿不在 0：{b.edge_lat_med_ms}"
    assert glob.glob(os.path.join(out, "runs", "P201b_d5_timing", "*", "figures", "*.png"))

    run("scripts/p202_ccep_labels.py", env)
    lab = pd.read_parquet(os.path.join(data, "t2", "d5", "ccep_labels", f"{sim_d5.SUB}_labels.parquet"))
    pairs = sorted(lab.stim_pair.unique())
    assert pairs == ["PT01-PT02", "PT04-PT05"], pairs                 # 极性合并；4 次的那对被丢弃
    assert (lab.n_trials == 10).all()
    assert "PT06" not in set(lab.channel)                              # bad 通道不评估
    det = lab.set_index(["stim_pair", "channel"]).n1_detected
    # 植入幅度 -900·exp(-d/12)：d = 15 mm → -258 µV（≥ 170，应检出）；d = 25 mm → -112 µV（应检不出）
    assert det[("PT01-PT02", "PT03")] and not det[("PT01-PT02", "PT04")]
    # PT04-PT05 的中点在 35 mm：PT03 距 15 mm（检出），PT07 距 25 mm、PT08 距 35 mm（都检不出）
    assert det[("PT04-PT05", "PT03")] and not det[("PT04-PT05", "PT07")] and not det[("PT04-PT05", "PT08")]
    assert int(lab.n1_detected.sum()) == 2
    lat = lab[lab.n1_detected].n1_latency_ms
    assert lat.between(18, 22).all(), lat.tolist()
    z = np.load(glob.glob(os.path.join(data, "t2", "d5", "ccep_labels", sim_d5.SUB, "*_ccep.npz"))[0])
    assert z["avg"].shape[0] == 2 and z["avg"].shape == z["avg_odd"].shape and int(z["sfreq_store"]) == 512
    tail = lab[lab.n1_detected].tail_s
    assert tail.between(0.3, 1.0).all(), tail.tolist()                  # 晚成分到约 0.6 s
    assert glob.glob(os.path.join(out, "runs", "P202_ccep_labels_d5", "*", "DONE"))
    sat = pd.read_parquet(os.path.join(data, "t2", "d5", "ccep_labels", f"{sim_d5.SUB}_stim_saturation.parquet"))
    assert len(sat) == 4 and sat.recover_med_s.between(0, 0.003).all(), sat     # 模拟的伪迹 5000 µV 持续 2 ms
    assert {"sat_frac_n1win", "sat_frac_base"} <= set(lab.columns)
    print("ALL OK", sf, tmp)


if __name__ == "__main__":
    for sf in (2048, 512):
        main(sf)
