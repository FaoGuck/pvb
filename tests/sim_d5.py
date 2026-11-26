"""生成一个模拟的 ds004080 式 run（BIDS sidecar + BrainVision float32），给 P201 / P202 的本地冒烟和测试用。

8 个电极 PT01–PT08 排成一条线、间隔 10 mm；2048 Hz；脉冲每 5 s 一次。
刺激对：PT01-PT02 10 次（其中 5 次写成 PT02-PT01，检验极性合并）、PT04-PT05 10 次、PT07-PT08 4 次（< 5 次，应被丢弃）。
PT06 标成 bad。响应：在 20 ms 处的负峰，幅度 = -900 × exp(-距离 / 12 mm) µV（近的有 N1、远的没有）；
300 ms 处一个正的慢波（晚成分，到约 0.6 s 回到基线）；被刺激电极在 0–2 ms 有 5000 µV 的伪迹；白噪声 15 µV。
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

SUB, SES, RUN = "sub-ccepAgeSIM01", "ses-1", "011200"
SF = 2048
NAMES = [f"PT{i:02d}" for i in range(1, 9)]


def _bv(base: str, data_uv: np.ndarray, SF: int = SF):
    stem = os.path.basename(base)
    data_uv.T.astype("<f4").tofile(base + "_ieeg.eeg")          # MULTIPLEXED：样本优先
    with open(base + "_ieeg.vhdr", "w", encoding="utf-8") as f:
        f.write("Brain Vision Data Exchange Header File Version 1.0\n\n[Common Infos]\nCodepage=UTF-8\n"
                f"DataFile={stem}_ieeg.eeg\nMarkerFile={stem}_ieeg.vmrk\nDataFormat=BINARY\nDataOrientation=MULTIPLEXED\n"
                f"NumberOfChannels={len(NAMES)}\nSamplingInterval={1e6 / SF}\n\n[Binary Infos]\nBinaryFormat=IEEE_FLOAT_32\n\n"
                "[Channel Infos]\n" + "".join(f"Ch{i + 1}={n},,1,µV\n" for i, n in enumerate(NAMES)))
    with open(base + "_ieeg.vmrk", "w", encoding="utf-8") as f:
        f.write("Brain Vision Data Exchange Marker File, Version 1.0\n\n[Common Infos]\nCodepage=UTF-8\n"
                f"DataFile={stem}_ieeg.eeg\n\n[Marker Infos]\nMk1=New Segment,,1,1,0\n")


def make_sim(root: str, seed: int = 0, SF: int = SF) -> str:
    rng = np.random.default_rng(seed)
    d = os.path.join(root, SUB, SES, "ieeg")
    os.makedirs(d, exist_ok=True)
    base = os.path.join(d, f"{SUB}_{SES}_task-SPESclin_run-{RUN}")
    xyz = np.array([[10.0 * i, 0.0, 0.0] for i in range(len(NAMES))])
    plan = [("PT01-PT02", 5), ("PT02-PT01", 5), ("PT04-PT05", 10), ("PT07-PT08", 4)]
    onsets, sites = [], []
    t = 3.0
    for site, n in plan:
        for _ in range(n):
            onsets.append(t); sites.append(site); t += 5.0
    dur = t + 3.0
    n = int(dur * SF)
    x = rng.normal(0, 15, (len(NAMES), n))
    tt = np.arange(int(1.2 * SF)) / SF
    for on, site in zip(onsets, sites):
        a, b = site.split("-")
        ia, ib = NAMES.index(a), NAMES.index(b)
        mid = (xyz[ia] + xyz[ib]) / 2
        s0 = int(round(on * SF))
        for c in range(len(NAMES)):
            if c in (ia, ib):
                x[c, s0:s0 + int(0.002 * SF)] += 5000
                continue
            dist = np.linalg.norm(xyz[c] - mid)
            resp = -900 * np.exp(-dist / 12) * np.exp(-0.5 * ((tt - 0.020) / 0.005) ** 2)
            resp += 150 * np.exp(-dist / 12) * np.exp(-0.5 * ((tt - 0.300) / 0.08) ** 2)
            x[c, s0:s0 + tt.size] += resp
            x[c, s0:s0 + max(1, int(0.001 * SF))] += 2000 * np.exp(-dist / 20)   # 传导到邻近电极的刺激伪迹（0–1 ms）
    _bv(base, x, SF)
    json.dump({"TaskName": "task-SPESclin", "SamplingFrequency": SF, "RecordingDuration": n / SF, "ECOGChannelCount": 8},
              open(base + "_ieeg.json", "w"))
    pd.DataFrame({"name": NAMES, "type": "ECOG", "units": "µV", "sampling_frequency": SF,
                  "status": ["bad" if nm == "PT06" else "good" for nm in NAMES]}).to_csv(base + "_channels.tsv", sep="\t", index=False)
    pd.DataFrame({"onset": onsets, "duration": 0.001, "trial_type": "electrical_stimulation", "sub_type": "SPESclin",
                  "sample_start": [int(round(o * SF)) + 1 for o in onsets],
                  "electrical_stimulation_type": "monophasic", "electrical_stimulation_site": sites,
                  "electrical_stimulation_current": 0.008, "electrical_stimulation_frequency": 0.2}
                 ).to_csv(base + "_events.tsv", sep="\t", index=False)
    pd.DataFrame({"name": NAMES, "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}).to_csv(
        os.path.join(d, f"{SUB}_{SES}_electrodes.tsv"), sep="\t", index=False)
    # 假的 av_ccep，检验 .mat 结构打印
    from scipy.io import savemat
    dv = os.path.join(root, "derivatives", "av_ccep")
    os.makedirs(dv, exist_ok=True)
    savemat(os.path.join(dv, "ccepData_sim.mat"), {"dataBase": {"sub_label": SUB, "n1_peak_sample": np.full((8, 2), np.nan),
                                                                 "stimpnames": np.array(["PT01-PT02", "PT04-PT05"], dtype=object)}})
    return base
