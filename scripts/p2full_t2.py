"""P206 准备 B：把 T2 的标签、上限、观测段、基线从闸门 5 人扩到 ds004080 全部 74 人（00 第 14 节 2026-11-27）。

不改任何算法：按阶段依次调用现有脚本，只把患者名单换成全部患者（闸门 5 人包含在内，结果应与 11-26 相同）：
  --stage raw ：scripts/p202_ccep_labels.py → p203_split_half.py → p204_observation.py   （要读原始信号 data/raw/ds004080/）
  --stage base：scripts/p205_baselines.py → p205b_amplitude.py                           （只读 data/t2/d5/，即 raw 阶段的产出）
患者名单：raw 阶段 = data/raw/ds004080/ 下全部 sub-*；base 阶段 = data/t2/d5/ccep_labels/ 下有标签长表的患者
每一步的 SUMMARY 行原样打印；任何一步返回非 0 都记下来，但继续做后面的步（base 阶段要的是 raw 阶段尽量完整的产出）
用法（云上，workflows/pvb-p2full-t2.yaml）：python scripts/p2full_t2.py --stage raw --procs 8
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

CODE = Path(os.environ["PRJ_CODE"])
CFG = "configs/datasets/p_d5_ds004080.yaml"


def run(step: str, args: list[str]) -> int:
    t0 = time.time()
    print(f"\n===== {step} {' '.join(a if len(a) < 80 else a[:60] + '…' for a in args)}", flush=True)
    p = subprocess.run([sys.executable, str(CODE / step)] + args, cwd=str(CODE))
    print(f"===== {step} 结束 rc={p.returncode} 用时 {(time.time() - t0) / 60:.1f} min", flush=True)
    return p.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["raw", "base"], required=True)
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(CODE / CFG, encoding="utf-8"))
    data = Path(os.environ["PRJ_DATA"])
    if a.stage == "raw":
        subs = sorted(p.name for p in (data / cfg["raw_dir"]).glob("sub-*") if p.is_dir())
        steps = [("scripts/p202_ccep_labels.py", ["--procs", str(a.procs)]),
                 ("scripts/p203_split_half.py", []),
                 ("scripts/p204_observation.py", ["--procs", str(a.procs)])]
    else:
        subs = sorted(p.name[: -len("_labels.parquet")] for p in (data / cfg["labels_dir"]).glob("sub-*_labels.parquet"))
        steps = [("scripts/p205_baselines.py", ["--procs", str(a.procs)]),
                 ("scripts/p205b_amplitude.py", ["--procs", str(a.procs)])]
    gate = set(cfg["subjects_gate"])
    print(f"[P2FULL] stage={a.stage} 患者 {len(subs)} 名（含闸门 {len(gate & set(subs))} / {len(gate)}）：{subs[0]} … {subs[-1]}", flush=True)
    if not subs:
        print("SUMMARY_P2FULL stage=%s subjects=0 FAIL" % a.stage); return
    rcs = {}
    for step, extra in steps:
        rcs[Path(step).stem] = run(step, ["--config", CFG, "--subjects", ",".join(subs)] + extra)
    print("SUMMARY_P2FULL stage=%s subjects=%d " % (a.stage, len(subs)) + " ".join(f"{k}=rc{v}" for k, v in rcs.items()))


if __name__ == "__main__":
    main()
