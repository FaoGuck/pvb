#!/usr/bin/env python
# 来自 S2R 505b298 的 tools/fm_download.py，2026-11-26 复制（机械改名：S2R_→PRJ_、包名 s2r→项目名）
"""下载三个基础模型的官方权重与源码到 <models_dir>/<model>/，固定到当时的提交，记录来源与 sha256（总纲第 5 节规定 3）。

在云上准备任务（workflows/s2r-fm-setup.yaml）里用新环境的 python 运行：
    python tools/fm_download.py --out $PRJ_DATA/models
结果：
    reve/reve-base/、reve/reve-positions/   Hugging Face 快照（brain-bzh/*）
    labram/src/                             GitHub 935963004/LaBraM 固定提交的源码压缩包解压（含 checkpoints/labram-base.pth）
    cbramod/src/                            GitHub wjq-learning/CBraMod 固定提交的源码
    cbramod/pretrained_weights.pth          HF weighting666/CBraMod
    <model>/_SOURCE.txt                     地址、提交号、下载时间、每个文件的大小和 sha256
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def get(url: str, timeout=600) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "s2r-research-client"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def github_sha(repo: str) -> str:
    return json.loads(get(f"https://api.github.com/repos/{repo}/commits/HEAD"))["sha"]


def github_src(repo: str, dest: Path, keep: list[str]) -> str:
    """下载固定提交的源码压缩包，只解出 keep 里列出的文件（相对仓库根目录；以 / 结尾表示该目录下的 .py）。
    11-23 两次准备任务都在回传时中断，疑似整仓库里有平台回传处理不了的文件，所以只留推理需要的。"""
    sha = github_sha(repo)
    z = zipfile.ZipFile(io.BytesIO(get(f"https://codeload.github.com/{repo}/zip/{sha}", timeout=1800)))
    top = z.namelist()[0].split("/")[0]
    if dest.exists():
        shutil.rmtree(dest)
    n_all, kept = 0, []
    for n in z.namelist():
        rel = n[len(top) + 1:]
        if not rel or n.endswith("/"):
            continue
        n_all += 1
        ok = any(rel == k or (k.endswith("/") and rel.startswith(k) and rel.endswith(".py")
                              and "/" not in rel[len(k):]) for k in keep)
        if not ok:
            continue
        p = dest / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(z.read(n))
        kept.append(rel)
    missing = [k for k in keep if not k.endswith("/") and k not in kept]
    print(f"[{repo}] 压缩包 {n_all} 个文件，保留 {len(kept)} 个: {kept}", flush=True)
    if missing:
        raise FileNotFoundError(f"{repo} 里没有: {missing}")
    return sha


def hf_snapshot(repo: str, dest: Path, filename: str | None = None) -> str:
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    sha = HfApi().model_info(repo).sha
    if filename:
        hf_hub_download(repo, filename, revision=sha, local_dir=str(dest))
    else:
        snapshot_download(repo, revision=sha, local_dir=str(dest))
    shutil.rmtree(dest / ".cache", ignore_errors=True)            # 下载缓存（锁文件、元数据），不需要回传
    return sha


def write_source(d: Path, entries: list[dict], files: list[Path]):
    lines = [f"downloaded={dt.datetime.now().isoformat(timespec='seconds')}"]
    lines += [" ".join(f"{k}={v}" for k, v in e.items()) for e in entries]
    lines += [f"file={f.relative_to(d)} size={f.stat().st_size} sha256={sha256(f)}" for f in files]
    (d / "_SOURCE.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:3 + len(entries)]), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="*", default=["reve", "labram", "cbramod"])
    a = ap.parse_args()
    out = Path(a.out)

    if "reve" in a.models:
        d = out / "reve"
        e = []
        for repo, sub in [("brain-bzh/reve-base", "reve-base"), ("brain-bzh/reve-positions", "reve-positions")]:
            e.append({"source": f"https://huggingface.co/{repo}", "revision": hf_snapshot(repo, d / sub)})
        write_source(d, e, sorted(p for p in d.rglob("*") if p.is_file() and ".cache" not in p.parts))

    if "labram" in a.models:
        d = out / "labram"
        sha = github_src("935963004/LaBraM", d / "src",
                         ["modeling_finetune.py", "utils.py", "LICENSE", "README.md", "checkpoints/labram-base.pth"])
        write_source(d, [{"source": "https://github.com/935963004/LaBraM", "revision": sha}],
                     sorted(p for p in (d / "src").rglob("*") if p.is_file()))

    if "cbramod" in a.models:
        d = out / "cbramod"
        sha = github_src("wjq-learning/CBraMod", d / "src", ["models/", "LICENSE", "README.md"])
        hsha = hf_snapshot("weighting666/CBraMod", d, "pretrained_weights.pth")
        write_source(d, [{"source": "https://github.com/wjq-learning/CBraMod", "revision": sha},
                         {"source": "https://huggingface.co/weighting666/CBraMod", "revision": hsha}],
                     [d / "pretrained_weights.pth"] + sorted((d / "src" / "models").glob("*.py")))


if __name__ == "__main__":
    main()
