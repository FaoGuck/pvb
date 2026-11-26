"""00 第 6 节第 2 条的两项替换测试（T2 观测段防泄漏）+ 观测段边界检查。

  1. 把每个脉冲后 T_tail 以内（以及下一个脉冲前 pre_gap 以内）的数据换成噪声 → 全部 owner 的累加量逐位不变
  2. 把留出刺激对 X 的整段刺激期（X 第一个脉冲 → 下一个刺激对第一个脉冲）换成噪声 →
     其他 owner 的累加量逐位不变，且 X 所在折的观测池（不含 X）逐位不变；X 自己的累加量确实变了（测试有效）
  3. 每个子段都在 [上一个脉冲 + T_tail, 下一个脉冲 − pre_gap) 内，且 ≥ min_len
运行：python tests/test_t2_observation.py（或 pytest）
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from pvb.t2 import baselines as bl, folds as fo, io_d5, observation as obs  # noqa: E402

SF = 512
CFG = dict(t_tail=2.0, pre_gap=0.010, min_len=1.0, sfreq=256, mask_owner_stim=True, mask_next_stim=True, sat_uv=3000,
           bands={"delta": [1, 4], "theta": [4, 8], "alpha": [8, 13], "beta": [13, 30], "gamma": [30, 70],
                  "hgamma": [70, 120]},
           envelope_bands=["beta", "hgamma"])
NAMES = [f"E{i:02d}" for i in range(1, 13)]
PAIRS = ["E01-E02", "E04-E03", "E05-E06", "E08-E07"]      # 第二、四对故意写成反方向


class FakeReader:
    def __init__(self, x):
        self.x = x

    def segments(self, picks, starts, n):
        out = np.full((len(starts), len(picks), n), np.nan, np.float32)
        for k, s in enumerate(starts):
            if 0 <= s and s + n <= self.x.shape[1]:
                out[k] = self.x[picks, s:s + n]
        return out


def make(seed=0):
    rng = np.random.default_rng(seed)
    on, site = [], []
    t = 3.0
    for p in PAIRS:
        for _ in range(5):
            on.append(t); site.append(p); t += 5.0
        t += 20.0                                           # 刺激对之间的空档
    n = int((t + 5) * SF)
    common = rng.normal(0, 20, n)
    x = rng.normal(0, 20, (len(NAMES), n)) + 0.5 * common[None]
    for s in on:                                            # 刺激伪迹与晚成分（只在 T_tail 以内）
        i = int(round(s * SF))
        x[:, i:i + int(1.5 * SF)] += 300 * np.exp(-np.arange(int(1.5 * SF)) / SF / 0.3)[None]
    stim = pd.DataFrame({"onset": on, "electrical_stimulation_site": site})
    stim["pair"] = stim.electrical_stimulation_site.map(io_d5.pair_name)
    stim["sample0"] = np.round(stim.onset * SF).astype(np.int64)
    return x.astype(np.float32), stim


def acc_of(x, stim):
    segs = obs.segment_table(stim, SF, x.shape[1], CFG)
    return obs.owner_accumulators(FakeReader(x), NAMES, list(range(len(NAMES))), segs, SF, CFG), segs


def test_segments_within_bounds():
    x, stim = make()
    segs = obs.segment_table(stim, SF, x.shape[1], CFG)
    s0 = stim.sample0.to_numpy()
    for r in segs.itertuples():
        k = r.pulse_idx
        assert r.start >= s0[k] + CFG["t_tail"] * SF - 1
        if k + 1 < len(s0):
            assert r.stop <= s0[k + 1] - CFG["pre_gap"] * SF + 1
        assert r.length_s >= CFG["min_len"] - 1e-9
    assert set(segs.owner_pair) == set(stim.pair)


def test_noise_within_tail_does_not_change_features():
    x, stim = make()
    a0, _ = acc_of(x, stim)
    rng = np.random.default_rng(1)
    y = x.copy()
    s0 = stim.sample0.to_numpy()
    for k, s in enumerate(s0):
        e = s + int(round(CFG["t_tail"] * SF))
        y[:, s:e] = rng.normal(0, 500, (len(NAMES), e - s))
        g = int(round(CFG["pre_gap"] * SF))
        y[:, s - g:s] = rng.normal(0, 500, (len(NAMES), g))
    a1, _ = acc_of(y, stim)
    assert np.array_equal(a0["sums"], a1["sums"]) and np.array_equal(a0["counts"], a1["counts"])


def test_noise_in_heldout_pair_period_does_not_change_its_fold_pool():
    x, stim = make()
    a0, _ = acc_of(x, stim)
    X = io_d5.pair_name("E04-E03")
    first = stim[stim.pair == X].sample0.min()
    nxt = stim[stim.sample0 > stim[stim.pair == X].sample0.max()].sample0.min()
    y = x.copy()
    y[:, first:nxt] = np.random.default_rng(2).normal(0, 500, (len(NAMES), nxt - first))
    a1, _ = acc_of(y, stim)
    k = list(a0["owners"]).index(X)
    others = [i for i in range(len(a0["owners"])) if i != k]
    assert np.array_equal(a0["sums"][others], a1["sums"][others])
    assert not np.array_equal(a0["sums"][k], a1["sums"][k])             # 测试是灵敏的
    p0, p1 = bl.ObsPool([a0]), bl.ObsPool([a1])
    mask = np.array([o != X for o in p0.owners])
    assert np.array_equal(p0.mean(mask), p1.mean(mask), equal_nan=True)


def test_stim_electrodes_masked():
    x, stim = make()
    a0, segs = acc_of(x, stim)
    k = list(a0["owners"]).index("E01-E02")
    ci = {c: i for i, c in enumerate(NAMES)}
    # owner E01-E02 的段里 E01、E02 被屏蔽 → 与任何通道的 count 都应为 0
    assert a0["counts"][k][ci["E01"]].sum() == 0 and a0["counts"][k][:, ci["E02"]].sum() == 0
    assert a0["counts"][k][ci["E05"], ci["E09"]] > 0


if __name__ == "__main__":
    for f in [test_segments_within_bounds, test_noise_within_tail_does_not_change_features,
              test_noise_in_heldout_pair_period_does_not_change_its_fold_pool, test_stim_electrodes_masked]:
        f(); print("OK", f.__name__)
