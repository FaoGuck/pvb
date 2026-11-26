"""提交前检查 workflows/*.yaml（docs/02 第 4 节检查表里能自动做的部分）。

  python tests/test_workflows.py                    检查全部
  python tests/test_workflows.py workflows/x.yaml   只检查指定文件

逐个文件检查：
  1. YAML 能解析；workflow.name 以 pvb- 开头；outputs 全是 s3://<桶>/pvb/…
  2. Jinja：去掉平台变量 {{input:N}} / {{output}} / {{workflow_id}} 后，不得再有 "{{"、"{%"、"{#"
     （bash 的 ${#数组} 也会被当成 Jinja 注释开头——2026-11-26 pvb-p201-audit-gate 因此被平台拒收；ref_s2r/10 第 5 节）；
     再用 jinja2 实际解析一遍
  3. files 里的 entry.sh 能过 bash -n
"""
from __future__ import annotations

import glob
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

CODE = Path(__file__).resolve().parents[1]
BUCKET = "s3://tenant-vww8n20rmcahx6ur-20260904-1fbu/pvb/"
PLATFORM_VARS = re.compile(r"\{\{(input:\d+|output|workflow_id)\}\}")


def check(path: Path) -> list[str]:
    errs = []
    text = path.read_text(encoding="utf-8")
    try:
        w = yaml.safe_load(text)["workflow"]
    except Exception as e:
        return [f"YAML 解析失败：{e}"]
    if not str(w.get("name", "")).startswith("pvb-"):
        errs.append(f"name 不以 pvb- 开头：{w.get('name')}")
    for t in w.get("tasks", []):
        for o in t.get("outputs", []):
            if not str(o.get("url", "")).startswith(BUCKET):
                errs.append(f"outputs 不在 pvb/ 前缀：{o.get('url')}")
    stripped = PLATFORM_VARS.sub("X", text)
    for tok in ["{{", "{%", "{#"]:
        for ln, line in enumerate(stripped.splitlines(), 1):
            if tok in line:
                errs.append(f"第 {ln} 行有 Jinja 标记 {tok!r}：{line.strip()[:100]}")
    try:
        import jinja2
        jinja2.Environment().parse(stripped)
    except ImportError:
        pass
    except Exception as e:
        errs.append(f"jinja2 解析失败：{e}")
    bash = shutil.which("bash")
    for t in w.get("tasks", []):
        for f in t.get("files", []):
            if str(f.get("path", "")).endswith(".sh") and bash:
                with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False, newline="\n", encoding="utf-8") as tf:
                    tf.write(PLATFORM_VARS.sub("X", f["contents"]))
                r = subprocess.run([bash, "-n", tf.name], capture_output=True, text=True)
                if r.returncode:
                    errs.append(f"{f['path']} bash -n 失败：{r.stderr.strip()[:300]}")
    return errs


def main(argv):
    files = [Path(a) for a in argv] or sorted(Path(p) for p in glob.glob(str(CODE / "workflows" / "*.yaml")))
    bad = 0
    for p in files:
        e = check(p)
        print(("OK   " if not e else "FAIL ") + p.name)
        for x in e:
            print("     " + x)
        bad += bool(e)
    print(f"{len(files) - bad}/{len(files)} 通过")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
