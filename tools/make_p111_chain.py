"""生成 workflows/pvb-p111-chain-d4.yaml：P111 正式派生，一个工作流里 N 个串联任务（2026-11-27）。

依据：
  - pvb-test-chain-1（11-27）：`inputs: - task: 上游` 可用；上游结束即回传，下游的 url 输入能读到；url 输入在前、task 输入在后（编号按列表顺序）
  - pvb-p111-pilot-d4-1（11-27）：每个 spTMS run 下载 ≈ 200 s、计算 ≈ 20 s（边下边算）；rest ≈ 95 s
做法：以 pvb-p111-pilot-d4.yaml 为模板，每个任务：
  - 加第 4 个输入 = Bucket pvb/data/derived/d4/_done/（已完成标记，放到 $ROOT/data/derived/d4/_done；新写的标记照常回传）
  - 脚本参数 --done-dir：跳过已完成的 run；脚本自己的时间预算 105 min（configs 的 p111.budget_min）
  - 外面再套 `timeout 150m`（不论平台的 exec_timeout 是按任务还是按整个工作流算，单个任务都不会无限跑下去）
  - 日志名 = 工作流编号_任务名（各任务的日志不互相覆盖）
  - 从第 2 个任务起依赖上一个任务（上一个结束、回传之后才开始）
用法：python tools/make_p111_chain.py [任务数，默认 14]
"""
import re
import sys
from pathlib import Path

N = int(sys.argv[1]) if len(sys.argv) > 1 else 14
WF = Path(__file__).resolve().parents[1] / "workflows"
src = (WF / "pvb-p111-pilot-d4.yaml").read_text(encoding="utf-8")
task0 = src[src.index("  - name: run\n"): src.index("  timeout:\n")]
res = src[src.index("  resources:\n"): src.index("  tasks:\n")]

rep = [
    ('      PRJ_ARGS: "--runs sub-CON001_ses-async4ms_task-spTMS_run-01,sub-CON001_ses-async4ms_task-spTMS_run-02,sub-CON001_ses-async4ms_task-rest_run-01"   # ★2',
     '      PRJ_ARGS: "--done-dir /osmo/run/workspace/proj_PVB/data/derived/d4/_done"   # ★2'),
    ("      PRJ_DATA_IN: data/raw/_manifest          # ★3（只需要 ds004024.keys.tsv）",
     "      PRJ_DATA_IN: data/raw/_manifest          # ★3（只需要 ds004024.keys.tsv）\n      PRJ_DATA_IN2: data/derived/d4/_done      # ★3b（已完成标记）"),
    ("        export PRJ_RUN_ID={{workflow_id}}", "        export PRJ_RUN_ID={{workflow_id}}_@TASK@"),
    ('        echo "data_in: $(find -L "$ROOT/$PRJ_DATA_IN" -type f | wc -l) files"\n',
     '        echo "data_in: $(find -L "$ROOT/$PRJ_DATA_IN" -type f | wc -l) files"\n'
     '        mkdir -p "$ROOT/$PRJ_DATA_IN2" && cp -as {{input:3}}/. "$ROOT/$PRJ_DATA_IN2/" || DATA_OK=FAIL\n'
     '        echo "data_in2（已完成标记）: $(find -L "$ROOT/$PRJ_DATA_IN2" -type f | wc -l) files"\n'),
    ("          PYTHONPATH=$PRJ_CODE MPLBACKEND=Agg $PY $PRJ_STEP $PRJ_ARGS",
     "          PYTHONPATH=$PRJ_CODE MPLBACKEND=Agg timeout 150m $PY $PRJ_STEP $PRJ_ARGS"),
    ("    - url: s3://tenant-vww8n20rmcahx6ur-20260904-1fbu/pvb/data/raw/_manifest/   # ★4\n",
     "    - url: s3://tenant-vww8n20rmcahx6ur-20260904-1fbu/pvb/data/raw/_manifest/   # ★4\n"
     "    - url: s3://tenant-vww8n20rmcahx6ur-20260904-1fbu/pvb/data/derived/d4/_done/   # ★4b\n@DEP@"),
]
for a, b in rep:
    assert task0.count(a) == 1, (a, task0.count(a))
    task0 = task0.replace(a, b)
assert res.count("      storage: 30Gi") == 1
res = res.replace("      storage: 30Gi          # 一次只放 1 个 run（≤ 3.1 GiB）",
                  "      storage: 40Gi          # 同时放 2 个 run（边下边算）+ 上一个任务的输出（≈ 4 GiB，串联时自动下载）+ 本任务输出")

tasks = []
for i in range(1, N + 1):
    name = f"b{i:02d}"
    dep = f"    - task: b{i - 1:02d}             # 上一个任务结束、回传之后才开始\n" if i > 1 else ""
    t = task0.replace("  - name: run\n", f"  - name: {name}\n").replace("@TASK@", name).replace("@DEP@", dep)
    tasks.append(t)

head = f"""# P111 正式派生：D4 spTMS 348 + rest 119 个 run，一个工作流里 {N} 个串联任务（由 tools/make_p111_chain.py 生成，不要手改）
# 每个任务：跳过已完成的 run，做 105 min 的新 run 后收尾回传（外面套 timeout 150m）；下一个任务读 Bucket 上最新的已完成标记继续
# 依据：pvb-test-chain-1、pvb-p111-pilot-d4-1（11-27）；设计见 scripts/p111_d4_derive.py 文件头
# 预计：spTMS ≈ 200 s / run、rest ≈ 95 s / run → 约 13 个任务做完（第 14 个是余量，没活就几分钟结束）；共 ≈ 24 h、≈ ¥75
# exec_timeout 按“整个工作流合计”最坏情况设（{N} × 2.5 h）；若平台按任务算，单个任务也有 timeout 150m 兜底
"""
body = "workflow:\n  name: pvb-p111-chain-d4\n" + res + "  tasks:\n" + "".join(tasks) + f"  timeout:\n    exec_timeout: {int(N * 2.5) + 1}h\n"
out = WF / "pvb-p111-chain-d4.yaml"
out.write_text(head + body, encoding="utf-8", newline="\n")
print(f"写好 {out}，{N} 个任务，exec_timeout {int(N * 2.5) + 1}h")
