#!/usr/bin/env python
# 来自 S2R 505b298 的 scripts/s07_smoke.py，2026-11-26 复制（机械改名：S2R_→PRJ_、包名 s2r→项目名）
"""S7 冒烟测试：三个基础模型能否加载、输出形状/层数是否符合 configs/models/*.yaml、token 排列是否正确、是否确定。

在准备任务（workflows/s2r-fm-setup.yaml）里用新环境运行，不需要 D1 数据（用随机输入）：
    python scripts/s07_smoke.py --models-dir $PRJ_DATA/models --report $PRJ_DATA/models/_smoke_report.txt
每个模型检查：
  1. 加载（LaBraM 打印权重键的核对结果，REVE 核对 28 个电极都找到位置）
  2. W4 / W2 / W1 的层数、每层形状 (B, C, S, D)
  3. 逐层取输出的末层 = 官方 forward 的输出（REVE、CBraMod）
  4. token 排列：只改第 3 导最后 0.5 s，嵌入层变化最大的 token 应是 (导联 3, 最后一个 patch)
  5. 同一输入两次结果一致（确定性）
  6. 随机初始化（seed 0）能构造、输出与预训练不同
退出码 0 = 全部通过。
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

CODE = Path(os.environ.get("PRJ_CODE", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(CODE))
from pvb.embeddings import common, models  # noqa: E402

CH_D1 = ["FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8", "T7", "T8", "P7", "P8",
         "FZ", "CZ", "PZ", "IZ", "FC1", "FC2", "CP1", "CP2", "FC5", "FC6", "CP5", "CP6"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--models", nargs="*", default=["reve", "labram", "cbramod"])
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lines, ok_all = [f"device={dev} torch={torch.__version__}"], True

    def log(s):
        print(s, flush=True)
        lines.append(s)

    # 0. 预处理（假 master：2 个试验、28 导、4 s、1000 Hz）
    rng = np.random.default_rng(0)
    master = rng.normal(0, 20, (2, 28, 4000))
    for name in a.models:
        pp = models.load_model_cfg(CODE, name)["preprocess"]
        log(f"[{name}] preprocess " + " ".join(f"W{w}={common.preprocess(master, 1000, w, pp).shape}" for w in (4, 2, 1)))

    for name in a.models:
        try:
            cfg = models.load_model_cfg(CODE, name)
            w = models.get_wrapper(name, CODE, Path(a.models_dir), dev, pretrained=True)
            log(f"[{name}] 加载 OK，参数 {w.n_params() / 1e6:.2f} M")
            if name == "labram":
                r = w.load_report
                log(f"[{name}] 构造参数 {w.build_kw}；权重键 {r['n_ckpt']}，模型键 {r['n_model']}；"
                    f"缺 {len(r['missing'])}: {r['missing'][:8]}；多 {len(r['unexpected'])}: {r['unexpected'][:8]}")
                log(f"[{name}] standard_1020 名表 {len(w.ch_list)} 个；D1 不在表里的: {[c for c in CH_D1 if c not in w.ch_list]}")
            if name == "reve":
                log(f"[{name}] 位置 {tuple(w.positions(CH_D1).shape)}")
            for ws in (4, 2, 1):
                x = torch.as_tensor(rng.normal(0, 0.2, (2, 28, 200 * ws)), dtype=torch.float32, device=dev)
                L = w.layers(x, CH_D1)
                shapes = {tuple(t.shape) for t in L}
                ok = len(L) == cfg["n_layers"] and all(s[1] == 28 and s[3] == cfg["dim"] for s in shapes)
                ok_all &= ok
                log(f"[{name}] W{ws}: {len(L)} 层，形状 {sorted(shapes)} {'OK' if ok else '不符合配置!'}")
                if ws == 4:
                    if hasattr(w, "final"):
                        d = float((L[-1] - w.final(x, CH_D1)).abs().max())
                        ok_all &= d < 1e-4
                        log(f"[{name}] 逐层末层 vs 官方 forward 最大差 {d:.2e} {'OK' if d < 1e-4 else '不一致!'}")
                    x2 = x.clone()
                    x2[0, 3, -100:] += 1.0
                    diff = (w.layers(x2, CH_D1)[0][0] - L[0][0]).abs().sum(-1)        # (C, S)
                    c, s = np.unravel_index(int(diff.argmax()), tuple(diff.shape))
                    ok = (c, s) == (3, diff.shape[1] - 1)
                    ok_all &= ok
                    log(f"[{name}] token 排列：改第 3 导最后 0.5 s → 嵌入层变化最大在 (导联 {c}, patch {s}/{diff.shape[1] - 1}) "
                        f"{'OK' if ok else '排列可能错!'}")
                    d2 = max(float((p - q).abs().max()) for p, q in zip(L, w.layers(x, CH_D1)))
                    log(f"[{name}] 两次运行最大差 {d2:.2e} {'OK' if d2 < 1e-5 else '不确定!'}")
                    ok_all &= d2 < 1e-5
                    pools = common.pool_layers(L, CH_D1)
                    log(f"[{name}] 池化 " + " ".join(f"{k}={v.shape}" for k, v in pools.items()))
                    w0 = models.get_wrapper(name, CODE, Path(a.models_dir), dev, pretrained=False, seed=0)
                    d3 = float((w0.layers(x, CH_D1)[-1] - L[-1]).abs().mean())
                    log(f"[{name}] 随机初始化 seed0 构造 OK，与预训练末层平均差 {d3:.3g} {'OK' if d3 > 1e-3 else '与预训练相同?!'}")
                    ok_all &= d3 > 1e-3
                    del w0
            del w
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            ok_all = False
            log(f"[{name}] FAIL\n{traceback.format_exc()}")
    log(f"SMOKE {'PASS' if ok_all else 'FAIL'}")
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
