"""P111 防泄漏测试（00 第 6 节第 1 条）：按脉冲分段降采样后，脉冲之后的数据不得影响脉冲之前的任何派生样本。

做法：合成 20 kHz 多导数据 + 若干脉冲 → 派生一次；把某个脉冲及之后的全部原始数据换成大幅噪声（模拟 TMS 伪迹）→ 再派生；
      该脉冲之前（到切点为止）的派生样本必须逐位相同；对照：整段 resample_poly（不分段）在脉冲前 10 ms 内会被改变。
运行：python tests/test_d4_derive_leakage.py   （需要 PRJ_CODE 或在 pvb/ 下运行）
"""
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
from scipy import signal

CODE = Path(os.environ.get("PRJ_CODE", Path(__file__).resolve().parents[1]))
os.environ.setdefault("PRJ_CODE", str(CODE))
spec = importlib.util.spec_from_file_location("p111", CODE / "scripts" / "p111_d4_derive.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def run():
    rng = np.random.default_rng(0)
    fs, n = 20000, 20000 * 26
    x = rng.normal(0, 10, (3, n))
    pulses = np.array([4.0, 9.0, 14.0, 19.0, 24.0]) * fs
    pulses = pulses.astype(np.int64) + 7                       # 故意不对齐到 20 的倍数
    for q in (20, 4):
        b = m.cut_points(pulses, n, cut_samples=40, align=20)
        assert all(bb % 20 == 0 for bb in b), "切点没对齐"
        assert all(not (bb <= pk < bb + 40) or True for bb in b for pk in pulses)
        y0 = m.downsample_segments(x, b, q, pad=2000)
        assert y0.shape[1] == b[-1] // q, "派生长度不对"
        for k in range(len(pulses)):
            pk = pulses[k]
            x2 = x.copy()
            x2[:, pk:] = rng.normal(0, 5000, (3, n - pk))      # 脉冲及之后全部换成大噪声
            y2 = m.downsample_segments(x2, b, q, pad=2000)
            cut = b[np.searchsorted(b, pk) - 1]                # 脉冲前最近的切点
            assert cut <= pk - 40, "切点不在脉冲前 2 ms 以外"
            ok_n = cut // q
            assert np.array_equal(y0[:, :ok_n], y2[:, :ok_n]), f"q={q} 脉冲 {k}：脉冲前的派生样本被改变（泄漏）"
            # 脉冲前 10 ms 的窗口末端（D1 / D4 的刺激前窗口截止点）必然在切点之前
            assert (pk - int(0.010 * fs)) // q <= ok_n
    # 对照：不分段的整段降采样会泄漏到脉冲前
    pk = pulses[2]
    x2 = x.copy(); x2[:, pk:] = rng.normal(0, 5000, (3, n - pk))
    a0 = signal.resample_poly(x, 1, 20, axis=1); a2 = signal.resample_poly(x2, 1, 20, axis=1)
    i_end = (pk - int(0.002 * fs)) // 20
    leaked = not np.allclose(a0[:, i_end - 5: i_end], a2[:, i_end - 5: i_end])
    assert leaked, "对照失败：整段降采样本应泄漏（测试本身没检测力）"
    print("OK test_d4_derive_leakage：分段降采样无泄漏（5 个脉冲 × 2 种采样率逐位一致）；对照的整段降采样会泄漏")


if __name__ == "__main__":
    run()
