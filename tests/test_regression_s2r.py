"""P101 比较脚本的自检（00 第 7 节 test_regression_s2r.py）：用 S2R 的参考运行（只读）构造“PVB 输出”。

1. 原样复制 → 差为 0，应 PASS
2. 某一行 value 改 0.006（超容差）→ FAIL；改 0.004 → PASS
3. 少一行 → FAIL（键不一致）
只读 D:/proj_LLM/proj/proj_S2R/out；写临时目录。运行：python -X utf8 tests/test_regression_s2r.py（几秒）
"""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

CODE = Path(__file__).resolve().parents[1]
S2R_OUT = Path("D:/proj_LLM/proj/proj_S2R/out")
sys.path.insert(0, str(CODE / "scripts"))
os.environ.setdefault("PRJ_CODE", str(CODE))
import p101_compare as pc  # noqa: E402


def run_case(tag: str, edit=None) -> str:
    tmp = Path(tempfile.mkdtemp(prefix=f"pvb_p101_{tag}_"))
    for exp, ref in pc.S2R_REF.items():
        dst = tmp / "runs" / exp / "20990101-0000_test"
        dst.mkdir(parents=True)
        m = pd.read_csv(S2R_OUT / "runs" / exp / ref / "metrics.csv")
        if edit:
            m = edit(exp, m)
        m.to_csv(dst / "metrics.csv", index=False)
        (dst / "DONE").write_text("ok")
    env = {**os.environ, "PRJ_CODE": str(CODE), "PRJ_OUT": str(tmp / "cmp"), "PRJ_RUN_ID": "test", "PRJ_GIT": "test"}
    r = subprocess.run([sys.executable, "-X", "utf8", "scripts/p101_compare.py", "--pvb-out", str(tmp), "--s2r-out", str(S2R_OUT)],
                       cwd=CODE, env=env, capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stderr[-2000:]
    line = [x for x in r.stdout.splitlines() if x.startswith("SUMMARY")][-1]
    shutil.rmtree(tmp, ignore_errors=True)
    print(tag, "→", line)
    return line


def bump(delta):
    def f(exp, m):
        if exp == "E103_fm_d1":
            m = m.copy(); m.loc[7, "value"] = m.loc[7, "value"] + delta
        return m
    return f


def main():
    assert "G0_T1_1=PASS" in run_case("same")
    assert "G0_T1_1=FAIL" in run_case("over", bump(0.006))
    assert "G0_T1_1=PASS" in run_case("under", bump(0.004))
    assert "G0_T1_1=FAIL" in run_case("missing", lambda exp, m: m.iloc[1:] if exp == "E110_history_d1" else m)
    print("ALL OK")


if __name__ == "__main__":
    main()
