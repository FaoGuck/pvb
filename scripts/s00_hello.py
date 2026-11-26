#!/usr/bin/env python
# 来自 S2R 505b298 的 scripts/s00_hello.py，2026-11-26 复制（机械改名：S2R_→PRJ_、包名 s2r→项目名）
"""运行器自检：确认环境变量、代码、数据输入、结果回传都对。

用法: python scripts/s00_hello.py
输出: $PRJ_OUT/runs/_debug/<run_id>/hello.txt
"""
import os
import sys
import platform
from pathlib import Path

run_id = os.environ.get("PRJ_RUN_ID", "local")
code, data, out = (Path(os.environ[k]) for k in ("PRJ_CODE", "PRJ_DATA", "PRJ_OUT"))

lines = [f"run_id = {run_id}", f"python = {sys.version.split()[0]}  {platform.node()}",
         f"git = {os.environ.get('PRJ_GIT', '?')}  dirty = {os.environ.get('PRJ_GIT_DIRTY', '?')}"]
for k, p in (("PRJ_CODE", code), ("PRJ_DATA", data), ("PRJ_OUT", out)):
    lines.append(f"{k} = {p}  exists={p.exists()}")

try:
    import torch
    lines.append(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}")
except Exception as ex:  # noqa: BLE001
    lines.append(f"torch import failed: {ex}")
import mne  # noqa: E402
lines.append(f"mne {mne.__version__}")

lines.append("data 目录（前 3 层，含软链接）:")
for p in sorted(data.rglob("*")):
    rel = p.relative_to(data)
    if len(rel.parts) <= 3:
        lines.append(f"  {rel}{'/' if p.is_dir() else ''}")

dst = out / "runs" / "_debug" / run_id
dst.mkdir(parents=True, exist_ok=True)
(dst / "hello.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
print(f"wrote {dst / 'hello.txt'}")
