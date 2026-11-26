# 来自 S2R 505b298 的 tests/test_no_leakage.py，2026-11-26 复制（机械改名：S2R_→PRJ_、包名 s2r→项目名）
"""总纲 6.3 第 4 条：防泄漏测试。

把某个脉冲之后的数据全部换成噪声，重新跑 S3 的片段处理，
该试验（以及之前所有试验）的 master 和各窗口 clean 必须逐样本完全相同。

运行：python tests/test_no_leakage.py   或   pytest tests/
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from pvb.preprocess import epoch  # noqa: E402


def test_no_leakage():
    ok, msg = epoch.leakage_selftest(n_trials_check=8)
    assert ok, msg


def test_detects_leakage():
    """反向检查：故意对整段连续数据做零相位滤波再切片（禁止的做法），测试必须能发现。"""
    from scipy import signal

    rng = np.random.default_rng(1)
    fs, n_ch = 5000, 2
    pulses = (np.cumsum(rng.uniform(4.3, 5.7, 8)) * fs).astype(np.int64) + 5 * fs
    n = int(pulses[-1] + 3 * fs)
    data = rng.normal(0, 10, (n_ch, n))
    sos = signal.butter(4, 45, fs=fs, output="sos")
    ep = {"end_offset_s": 0.01, "master_window_s": 4.0, "master_sfreq": 1000, "windows_s": [1.0],
          "guard_s": 0.9, "clean": {"sfreq": 250, "notch_hz": 60, "notch_q": 30, "band_hz": [1.0, 45.0],
                                    "filter_order": 4}}

    def run(d):
        filt = signal.sosfiltfilt(sos, d, axis=-1)      # 错误做法：先全局零相位滤波
        return epoch.epoch_recording(lambda s, e: filt[:, s:e], n, pulses, fs, ep, epoch.spatial_matrix(n_ch), n_ch)

    a = run(data)
    d2 = data.copy()
    d2[:, pulses[3]:] = rng.normal(0, 1000, (n_ch, n - pulses[3]))
    b = run(d2)
    assert not np.array_equal(a["master"][3], b["master"][3]), "测试没能发现全局滤波造成的泄漏"


if __name__ == "__main__":
    test_no_leakage()
    test_detects_leakage()
    print("test_no_leakage: PASS（含反向检查）")
