#!/bin/bash
# P206 准备 A（workflows/pvb-p206a-env.yaml 调用）：建环境 pvb-env-1 → 下载权重 → 冒烟测试 / 接口核实（V8a–c）→ 打包环境
# 照 S2R 505b298 的 tools/fm_setup.sh 的做法（只读参考，2026-11-27）：镜像没有 ensurepip → venv --without-pip --system-site-packages，
# 借镜像的 pip；镜像自带的包全部作为约束，不升级不覆盖。恢复时必须解压回 /osmo/run/workspace/envs/pvb-1、用同一个镜像（共享规则第 3 条）
# 与 S2R 不同：braindecode、Chronos、MOMENT 分三次装，各自记录成败（一个与镜像冲突不影响其他两个）
# 产出（相对 $ROOT，由 YAML 按原路径写回 Bucket pvb/）：
#   env/pvb-env-1/   pvb-venv-1.tar.gz、pvb-venv-1.sha256、venv_freeze.txt、base_pip_list.txt、pip_*.log
#   data/models/     <模型>/（Hugging Face 快照 + _SOURCE.txt）、_smoke_report.txt
set -u
ROOT=/osmo/run/workspace/proj_PVB
CODE=$ROOT/pvb
ENVNAME=pvb-env-1
VENV=/osmo/run/workspace/envs/pvb-1
EOUT=$ROOT/env/$ENVNAME
MODELS=$ROOT/data/models
mkdir -p "$EOUT" "$MODELS"
S_VENV=FAIL; S_BD=FAIL; S_CH=FAIL; S_MO=FAIL; S_DL=FAIL; S_SMOKE=FAIL; S_PACK=FAIL

echo "===== 1. venv"
python3 --version
python3 -m pip list --format=freeze > "$EOUT/base_pip_list.txt"
grep -v -iE '^(pip|setuptools|wheel)==' "$EOUT/base_pip_list.txt" > /tmp/constraints.txt
grep -iE '^(torch|numpy|scipy|mne|pandas|transformers|huggingface.hub)==' "$EOUT/base_pip_list.txt"
python3 -m venv --without-pip --system-site-packages "$VENV" && S_VENV=OK
PY=$VENV/bin/python

pipi() {  # pipi <日志名> <包…>：带约束安装；失败时再看不带约束会要什么版本（只做 --dry-run，不装）
  local log="$EOUT/pip_$1.log"; shift
  $PY -m pip install --no-cache-dir -c /tmp/constraints.txt "$@" > "$log" 2>&1
  local rc=$?
  tail -15 "$log"
  if [ $rc != 0 ]; then
    echo "--- 带约束安装失败；不带约束时 pip 会选的版本（--dry-run，不安装）："
    $PY -m pip install --no-cache-dir --dry-run "$@" 2>&1 | tail -8 | tee -a "$log"
  fi
  return $rc
}

echo "===== 2a. braindecode（BrainBERT、PopT、MVPFormer）+ huggingface_hub、h5py、pyarrow"
pipi braindecode braindecode "huggingface_hub>=0.34" h5py pyarrow && S_BD=OK
echo "===== 2b. Chronos"
pipi chronos chronos-forecasting && S_CH=OK
echo "===== 2c. MOMENT"
pipi moment momentfm && S_MO=OK

echo "===== 3. import"
$PY - <<'EOF'
import importlib
for m in ["torch", "numpy", "scipy", "mne", "braindecode", "huggingface_hub", "h5py", "pyarrow", "chronos", "momentfm", "transformers"]:
    try:
        mod = importlib.import_module(m); print(f"{m:16s} {getattr(mod, '__version__', '?')}")
    except Exception as e:  # noqa: BLE001
        print(f"{m:16s} IMPORT_FAIL {e!r}"[:200])
import torch; print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
EOF

echo "===== 4. 下载权重（Hugging Face 快照，记录提交号与 sha256）"
HF_HUB_DISABLE_TELEMETRY=1 $PY "$CODE/tools/p206_download.py" --out "$MODELS" && S_DL=OK
du -sh "$MODELS"/*/ 2>/dev/null

echo "===== 5. 冒烟测试 / 接口核实（离线）"
cd "$CODE" && PYTHONPATH=$CODE HF_HUB_OFFLINE=1 $PY scripts/p206a_smoke.py --models-dir "$MODELS" \
  --report "$MODELS/_smoke_report.txt" && S_SMOKE=OK

echo "===== 6. 打包环境（冒烟测试失败也打包，便于复用）"
$PY -m pip freeze --local > "$EOUT/venv_freeze.txt" 2>/dev/null
find "$VENV" -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null
tar -czf "$EOUT/pvb-venv-1.tar.gz" -C /osmo/run/workspace/envs pvb-1 && \
  (cd "$EOUT" && sha256sum pvb-venv-1.tar.gz > pvb-venv-1.sha256) && S_PACK=OK
cat "$EOUT/pvb-venv-1.sha256" 2>/dev/null; du -sh "$VENV" "$EOUT/pvb-venv-1.tar.gz" 2>/dev/null

echo "SUMMARY_P206A venv=$S_VENV braindecode=$S_BD chronos=$S_CH moment=$S_MO download=$S_DL smoke=$S_SMOKE pack=$S_PACK"
