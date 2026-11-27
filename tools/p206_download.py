#!/usr/bin/env python
"""P206 准备 A：下载候选模型权重到 <out>/<目录名>/（Hugging Face 快照），每个目录写 _SOURCE.txt（仓库、提交号、时间、每个文件的大小与 sha256）。

模型清单（00 第 14 节 2026-11-27 锁定）：BrainBERT、PopT、MVPFormer-S、MVPFormer-M（braindecode 重新托管版）、MOMENT、Chronos
在云上准备任务（workflows/pvb-p206a-env.yaml → tools/p206_env_setup.sh）里用新环境的 python 运行：
    python tools/p206_download.py --out $PRJ_DATA/models
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
from pathlib import Path

REPOS = {
    "brainbert": "braindecode/brainbert-pretrained",
    "popt": "braindecode/popt-pretrained",
    "mvpformer_s": "braindecode/mvpformer-s-pretrained",
    "mvpformer_m": "braindecode/mvpformer-m-pretrained",
    "moment_large": "AutonLab/MOMENT-1-large",
    "chronos_t5_large": "amazon/chronos-t5-large",
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="", help="逗号分隔的目录名，只下这些")
    a = ap.parse_args()
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi()
    out = Path(a.out)
    ok_all = True
    for name, repo in REPOS.items():
        if a.only and name not in a.only.split(","):
            continue
        d = out / name
        try:
            info = api.model_info(repo)
            rev = info.sha
            snapshot_download(repo_id=repo, revision=rev, local_dir=str(d))
            files = sorted(p for p in d.rglob("*") if p.is_file() and ".cache" not in p.parts and p.name != "_SOURCE.txt")
            lines = [f"repo: https://huggingface.co/{repo}", f"revision: {rev}",
                     f"license (card): {getattr(info, 'card_data', None) and info.card_data.get('license')}",
                     f"downloaded_at_utc: {dt.datetime.utcnow().isoformat(timespec='seconds')}Z", "files:"]
            lines += [f"  {p.relative_to(d)}\t{p.stat().st_size}\t{sha256(p)}" for p in files]
            (d / "_SOURCE.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            tot = sum(p.stat().st_size for p in files) / 2**30
            print(f"OK   {name:18s} {repo}  rev={rev[:10]}  {len(files)} 个文件 {tot:.2f} GiB")
        except Exception as e:  # noqa: BLE001  一个失败不影响其他
            ok_all = False
            print(f"FAIL {name:18s} {repo}  {e!r}"[:300])
    raise SystemExit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
