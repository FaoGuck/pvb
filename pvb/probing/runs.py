# 来自 S2R 505b298 的 s2r/probing/runs.py，2026-11-26 复制（机械改名：S2R_→PRJ_、包名 s2r→项目名）
"""实验运行目录（总纲 3.4、7.2、7.3）。

out/runs/<exp_id>/<YYYYmmdd-HHMM>_<git7>/
    config.yaml   本次实验配置的完整快照
    env.txt       已安装的包及版本
    log.txt       运行日志
    metrics.csv   结果长表（7.3）
    preds.parquet 每个试验的预测
    DONE          正常结束才生成
"""
from __future__ import annotations

import datetime as _dt
import importlib.metadata as md
import os
from pathlib import Path

import yaml

METRIC_COLUMNS = ["exp_id", "run_id", "git_commit", "dataset", "subject", "predictor", "window", "pool",
                  "layer", "target", "metric", "value", "p_perm", "n_trials", "seed"]


def git_commit() -> str:
    return os.environ.get("PRJ_GIT", "nogit")


def new_run_dir(out_root: Path, exp_id: str) -> Path:
    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M")
    d = Path(out_root) / "runs" / exp_id / f"{ts}_{git_commit()[:7]}"
    d.mkdir(parents=True, exist_ok=False)
    return d


def write_snapshot(d: Path, cfg: dict):
    (d / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    pk = sorted(f"{x.metadata['Name']}=={x.version}" for x in md.distributions() if x.metadata["Name"])
    (d / "env.txt").write_text("\n".join(dict.fromkeys(pk)) + "\n", encoding="utf-8")


class Log:
    def __init__(self, d: Path):
        self.f = open(d / "log.txt", "a", encoding="utf-8")

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        self.f.write(s + "\n")
        self.f.flush()


def mark_done(d: Path):
    (d / "DONE").write_text(_dt.datetime.now().isoformat() + "\n", encoding="utf-8")
