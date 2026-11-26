#!/usr/bin/env python
# 来自 S2R 505b298 的 scripts/s07_embed.py，2026-11-26 复制（机械改名：S2R_→PRJ_、包名 s2r→项目名）
"""S7：基础模型表征提取（总纲 S7；方案见 docs/05 第 9.1 节，模型核实见 docs/02 第 4 节）。

用法（workflows/s2r-s07-embed.yaml，环境 s2r-env-fm-1）:
    PRJ_STEP = scripts/s07_embed.py
    数据：data/interim/d1/epochs_pre/（S3 h5 的 master）+ data/models/（S7 准备任务下载的权重与官方源码）
本地测试（只跑 CBraMod、1 人）：--models cbramod --subjects 283 --out <临时目录>

对每个被试：读 master（脉冲前 4 s、1000 Hz）→ 每个模型按自己的官方设置在窗口内预处理 → 逐层取 token → 池化。
输出 $PRJ_DATA/features/d1/embeddings/<模型>/（随机初始化对照在 random_init_<模型>/）：
    sub-XXX_<窗口>_<池化>.npy        (n_trials, n_layers, dim)，float16（数值超出 fp16 范围时自动改 float32）
    sub-XXX_W4_<池化>_seed<k>.npy     随机初始化（只做 W4；种子 0–4 存 meanpool，种子 0 另存 c4、last1s）
    sub-XXX_trials.parquet           行顺序：aligned_idx、block（与 S5 qc 表用 aligned_idx 对上）
    meta.json                        模型版本、权重 sha256、输入规格、层数、池化、git、运行编号
    池化：meanpool = 所有 token 平均；last1s = 最后 1 s 的 patch；c4 = C4、FC2、FC6、CP2、CP6 的 token（见 s2r/embeddings/common.py）
另写 $PRJ_DATA/features/d1/embeddings/_s07_<运行编号>_report.md：每人每模型的形状、耗时、确定性复核、导联映射表
验收（总纲 S7）：同一输入两次提取结果一致（每个模型在第一个被试上复核前 N 个试验）；导联映射表人工核对。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

CODE = Path(os.environ.get("PRJ_CODE", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(CODE))
from pvb.embeddings import common, models  # noqa: E402

RUN_ID = os.environ.get("PRJ_RUN_ID", "local")
FP16_MAX = 6.0e4


def read_master(h5_path: Path):
    with h5py.File(h5_path, "r") as f:
        ch = json.loads(f.attrs["ch_names"])
        fs = int(f.attrs["master_sfreq"])
        ai = f["aligned_idx"][:]
        rows = np.where(ai >= 0)[0]
        rows = rows[np.argsort(ai[rows])]
        M = f["master"][:][rows]                      # n × 28 × 4000（µV）
        blk = f["block"][:][rows]
        bads = json.loads(f.attrs.get("bads", "[]"))
    return M, ch, fs, ai[rows], blk, bads


@torch.no_grad()
def extract(w, x200: np.ndarray, ch: list[str], pools, batch: int, dev) -> dict[str, np.ndarray]:
    out = {p: [] for p in pools}
    for i in range(0, len(x200), batch):
        xb = torch.as_tensor(x200[i:i + batch], device=dev)
        P = common.pool_layers(w.layers(xb, ch), ch, pools)
        for p in pools:
            out[p].append(P[p])
    return {p: np.concatenate(v, 0) for p, v in out.items()}


def save(arr: np.ndarray, path: Path) -> str:
    dt = np.float16 if np.abs(arr).max() < FP16_MAX else np.float32
    np.save(path, arr.astype(dt))
    return np.dtype(dt).name


def source_info(models_dir: Path, name: str) -> dict:
    p = models_dir / models.load_model_cfg(CODE, name)["local_dir"] / "_SOURCE.txt"
    if not p.exists():
        return {}
    txt = p.read_text(encoding="utf-8")
    return {"sources": re.findall(r"source=(\S+) revision=(\S+)", txt),
            "weights_sha256": {f: h for f, _, h in re.findall(r"file=(\S+) size=(\d+) sha256=(\w+)", txt)
                               if f.endswith((".safetensors", ".pth"))}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", default=["reve", "labram", "cbramod"])
    ap.add_argument("--windows", nargs="*", type=float, default=[4.0, 2.0, 1.0])
    ap.add_argument("--random-seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--subjects", nargs="*")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", help="默认 $PRJ_DATA/features/d1/embeddings")
    a = ap.parse_args()

    data = Path(os.environ["PRJ_DATA"])
    models_dir = data / "models"
    out = Path(a.out) if a.out else data / "features" / "d1" / "embeddings"
    ep = data / "interim" / "d1" / "epochs_pre"
    subs = a.subjects or sorted(p.name[4:7] for p in ep.glob("sub-*_pre.h5"))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    lines = [f"# S7 表征提取报告 {RUN_ID}", "", f"- git: {os.environ.get('PRJ_GIT', '?')}；设备 {dev}；torch {torch.__version__}",
             f"- 模型 {a.models}；窗口 {a.windows}；随机初始化种子 {a.random_seeds}（W4）；被试 {len(subs)}", ""]
    print(f"[S7] run={RUN_ID} models={a.models} windows={a.windows} subjects={len(subs)} device={dev}", flush=True)

    # 1. 载入全部模型（预训练 + 随机初始化），整个运行只载一次
    W, R = {}, {}
    for name in a.models:
        t0 = time.time()
        W[name] = models.get_wrapper(name, CODE, models_dir, dev, pretrained=True)
        R[name] = {k: models.get_wrapper(name, CODE, models_dir, dev, pretrained=False, seed=k) for k in a.random_seeds}
        cfg = W[name].cfg
        for sub_dir, pretrained in [(name, True), (f"random_init_{name}", False)]:
            d = out / sub_dir
            d.mkdir(parents=True, exist_ok=True)
            (d / "meta.json").write_text(json.dumps({
                "model": name, "pretrained": pretrained, "random_seeds": a.random_seeds if not pretrained else None,
                "config": cfg, **source_info(models_dir, name), "pools": list(common.POOLS),
                "c4_neighborhood": common.C4_NEIGHBORHOOD, "windows": [f"W{int(w)}" for w in a.windows],
                "shape": "(n_trials, n_layers, dim); layer 0 = 进入第一个 block 之前的嵌入",
                "git": os.environ.get("PRJ_GIT"), "run_id": RUN_ID, "n_params": W[name].n_params()},
                ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[S7] {name} 载入（预训练 + {len(a.random_seeds)} 个随机初始化）{time.time() - t0:.0f}s", flush=True)

    # 2. 逐个被试
    rows, errors, checked = [], {}, set()
    mapping_done = False
    for s in subs:
        t_sub = time.time()
        try:
            M, ch, fs, ai, blk, bads = read_master(ep / f"sub-{s}_pre.h5")
            if bads:
                raise ValueError(f"有坏导 {bads}：master 未插值，S7 需要先处理（D1 目前 20 人都无坏导）")
            if not mapping_done:
                lines += ["## 导联映射（输入张量第 i 导 → 各模型看到的电极）", "", "| i | D1 名 | " +
                          " | ".join(a.models) + " |", "|---|---|" + "---|" * len(a.models)]
                for i, c in enumerate(ch):
                    cells = []
                    for name in a.models:
                        w = W[name]
                        if name == "reve":
                            cells.append(f"{w.alias.get(c, c)} {np.round(w.positions([c]).cpu().numpy()[0], 3).tolist()}")
                        elif name == "labram":
                            cells.append(f"{w.ch_list[w.input_chans([c])[1] - 1]}（表中第 {w.input_chans([c])[1] - 1} 个）")
                        else:
                            cells.append(f"第 {i} 导（只看顺序）")
                    lines.append(f"| {i} | {c} | " + " | ".join(cells) + " |")
                lines += ["", f"c4 池化用的导联下标: {[ch.index(c) for c in common.C4_NEIGHBORHOOD]} = {common.C4_NEIGHBORHOOD}", ""]
                mapping_done = True
            for name in a.models:
                pp = W[name].cfg["preprocess"]
                for wsec in a.windows:
                    wn = f"W{int(wsec)}"
                    t0 = time.time()
                    x = common.preprocess(M, fs, wsec, pp)
                    E = extract(W[name], x, ch, common.POOLS, a.batch, dev)
                    dts = {p: save(v, out / name / f"sub-{s}_{wn}_{p}.npy") for p, v in E.items()}
                    det = np.nan
                    if name not in checked:
                        # 确定性复核：用与正式提取完全相同的第一个批重跑（批大小不同时 GPU 会换算法，产生 1e-6 量级的相对差，不算不确定）
                        nb = min(a.batch, len(x))
                        E2 = extract(W[name], x[:nb], ch, common.POOLS, a.batch, dev)
                        det = max(float(np.abs(E2[p] - E[p][:nb]).max()) for p in E)
                    rows.append({"subject": f"sub-{s}", "model": name, "window": wn, "n": len(x),
                                 "shape": str(E["meanpool"].shape), "dtype": dts["meanpool"],
                                 "absmax": float(max(np.abs(v).max() for v in E.values())),
                                 "nan": int(sum(np.isnan(v).sum() for v in E.values())), "det_maxdiff": det,
                                 "sec": round(time.time() - t0, 1)})
                    if wsec == 4.0:
                        for k, wr in R[name].items():
                            pools = list(common.POOLS) if k == 0 else ["meanpool"]
                            Er = extract(wr, x, ch, pools, a.batch, dev)
                            for p, v in Er.items():
                                save(v, out / f"random_init_{name}" / f"sub-{s}_{wn}_{p}_seed{k}.npy")
                checked.add(name)
                for sub_dir in (name, f"random_init_{name}"):
                    pd.DataFrame({"aligned_idx": ai, "block": blk}).to_parquet(out / sub_dir / f"sub-{s}_trials.parquet",
                                                                               index=False)
            print(f"[S7] sub-{s} OK n={len(ai)} {time.time() - t_sub:.0f}s", flush=True)
        except Exception:  # noqa: BLE001
            errors[s] = traceback.format_exc()
            print(f"[S7] sub-{s} FAIL\n{errors[s]}", flush=True)
        finally:
            if dev.type == "cuda":
                torch.cuda.empty_cache()

    sm = pd.DataFrame(rows)
    lines.insert(5, f"- 成功 {len(subs) - len(errors)} / {len(subs)} 人；失败: {', '.join(errors) or '无'}")
    if len(sm):
        bad_nan = int(sm["nan"].sum())
        det = sm["det_maxdiff"].dropna()
        lines += ["## 验收", "", f"- NaN 总数: {bad_nan}",
                  f"- 确定性复核（每个模型在第一个被试上，用同一个批（前 {a.batch} 个试验）重跑）最大差: "
                  + ", ".join(f"{r.model}/{r.window}={r.det_maxdiff:.2e}" for r in sm.dropna(subset=['det_maxdiff']).itertuples()),
                  f"- 存储精度: {sm.groupby('model')['dtype'].agg(lambda v: ','.join(sorted(set(v)))).to_dict()}；"
                  f"各模型表征绝对值最大: {sm.groupby('model')['absmax'].max().round(1).to_dict()}", "",
                  "## 每人每模型每窗口", "", "| " + " | ".join(sm.columns) + " |", "|" + "---|" * len(sm.columns)]
        lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in sm.itertuples(index=False)]
        ok_det = bool((det < 1e-4).all()) if len(det) else False
        lines.insert(6, f"- 验收：NaN {'无' if bad_nan == 0 else bad_nan}；确定性 {'通过' if ok_det else '未通过'}")
    if errors:
        lines += ["", "## 错误", ""] + [f"### sub-{k}\n```\n{v}\n```" for k, v in errors.items()]
    out.mkdir(parents=True, exist_ok=True)
    (out / f"_s07_{RUN_ID}_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    sm.to_csv(out / f"_s07_{RUN_ID}_summary.tsv", sep="\t", index=False)
    print(f"SUMMARY_S7 subjects_ok={len(subs) - len(errors)}/{len(subs)}", flush=True)
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
