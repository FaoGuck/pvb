#!/bin/bash
# P206 准备 A2（workflows/pvb-p206a2-env.yaml 调用；2026-11-27）
# 为什么要 A2：pvb-p206a-env-1 发现 braindecode 2.1.0 要求 Python ≥ 3.11、torch ≥ 2.6、mne ≥ 1.13，而唯一可用的镜像
#   nvcr.io/nvidia/pytorch:24.10-py3 是 Python 3.10、torch 2.5 → 不能再“借镜像的包”；momentfm 把 numpy 锁在 1.25.2、
#   transformers 锁在 4.33.3，与 braindecode 装不到一起。
# 做法：用 uv 装独立的 Python（放在 /osmo/run/workspace/envs/pvb-py/，与环境一起打包，恢复时路径不变——共享规则第 3 条），
#   建两个不借镜像包的 venv，torch 用官方 cu126 轮子：
#   pvb-env-2       Python 3.12，venv /osmo/run/workspace/envs/pvb-2      ：braindecode[hub] 2.1.0 + Chronos（BrainBERT、PopT、MVPFormer、Chronos）
#   pvb-env-moment-1 Python 3.11，venv /osmo/run/workspace/envs/pvb-moment ：momentfm 0.1.4（MOMENT）
# 两个环境都装 mne、h5py、pyarrow、pyyaml、scipy、pandas、scikit-learn（P206 提取时要读 ds004080、跑 pvb/t2）
# 权重不重下：读 A 已下好的 Bucket pvb/data/models/（YAML 的第 2 个输入，放到 $ROOT/data/models）
# 产出（相对 $ROOT，写回 Bucket pvb/）：env/pvb-env-2/、env/pvb-env-moment-1/（tar.gz + sha256 + freeze + 日志）、data/models/_smoke_report_{env2,moment}.txt
set -u
ROOT=/osmo/run/workspace/proj_PVB
CODE=$ROOT/pvb
ENVS=/osmo/run/workspace/envs
MODELS=$ROOT/data/models
export UV_PYTHON_INSTALL_DIR=$ENVS/pvb-py UV_CACHE_DIR=/tmp/uvcache UV_LINK_MODE=copy
TORCH_IDX=https://download.pytorch.org/whl/cu126
S_UV=FAIL; S_E2=FAIL; S_EM=FAIL; S_SM2=FAIL; S_SMM=FAIL; S_P2=FAIL; S_PM=FAIL
mkdir -p $ROOT/env/pvb-env-2 $ROOT/env/pvb-env-moment-1

echo "===== 0. uv + 独立 Python 3.12 / 3.11"
python3 -m pip install --no-cache-dir --target /tmp/uvpkg uv > /tmp/uv_install.log 2>&1; tail -2 /tmp/uv_install.log
UV=/tmp/uvpkg/bin/uv
$UV --version && $UV python install 3.12 3.11 && S_UV=OK
ls $ENVS/pvb-py

build() {  # build <venv 目录> <python 版本> <日志目录> <包…>
  local V=$1 PYV=$2 LOGD=$3; shift 3
  $UV venv $V --python $PYV || return 1
  $UV pip install --python $V/bin/python torch==2.7.1 torchaudio==2.7.1 --index-url $TORCH_IDX > $LOGD/pip_torch.log 2>&1 \
    || { tail -20 $LOGD/pip_torch.log; return 1; }
  $UV pip install --python $V/bin/python "$@" > $LOGD/pip_main.log 2>&1 || { tail -30 $LOGD/pip_main.log; return 1; }
  tail -5 $LOGD/pip_main.log
  $UV pip freeze --python $V/bin/python > $LOGD/venv_freeze.txt
}

echo "===== 1. pvb-env-2（Python 3.12：braindecode 2.1.0 + Chronos）"
[ $S_UV = OK ] && build $ENVS/pvb-2 3.12 $ROOT/env/pvb-env-2 "braindecode[hub]==2.1.0" chronos-forecasting \
  "mne>=1.13" h5py pyarrow pyyaml scipy pandas scikit-learn && S_E2=OK
echo "===== 2. pvb-env-moment-1（Python 3.11：momentfm 0.1.4）"
[ $S_UV = OK ] && build $ENVS/pvb-moment 3.11 $ROOT/env/pvb-env-moment-1 "momentfm==0.1.4" \
  "mne<1.10" h5py pyarrow pyyaml scipy pandas scikit-learn && S_EM=OK

echo "===== 3. import"
for V in $ENVS/pvb-2 $ENVS/pvb-moment; do
  echo "--- $V"
  $V/bin/python - <<'EOF'
import importlib, sys
print("python", sys.version.split()[0])
for m in ["torch", "numpy", "scipy", "pandas", "mne", "h5py", "pyarrow", "sklearn", "braindecode", "chronos", "momentfm", "transformers", "huggingface_hub"]:
    try:
        mod = importlib.import_module(m); print(f"  {m:16s} {getattr(mod, '__version__', '?')}")
    except Exception as e:  # noqa: BLE001
        print(f"  {m:16s} -（{type(e).__name__}）")
import torch; print("  cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
EOF
done

echo "===== 4. 冒烟测试 / 接口核实（离线，读 A 下好的权重）"
cd $CODE
[ $S_E2 = OK ] && PYTHONPATH=$CODE HF_HUB_OFFLINE=1 $ENVS/pvb-2/bin/python scripts/p206a_smoke.py --models-dir $MODELS \
  --only BrainBERT,MVPFormer-S,MVPFormer-M,PopT,Chronos --report $MODELS/_smoke_report_env2.txt && S_SM2=OK
[ $S_EM = OK ] && PYTHONPATH=$CODE HF_HUB_OFFLINE=1 $ENVS/pvb-moment/bin/python scripts/p206a_smoke.py --models-dir $MODELS \
  --only MOMENT --report $MODELS/_smoke_report_moment.txt && S_SMM=OK

echo "===== 5. 打包（冒烟失败也打包，便于复用；gzip -1 求快）"
find $ENVS/pvb-2 $ENVS/pvb-moment $ENVS/pvb-py -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null
pack() {  # pack <输出目录> <包名> <目录…（相对 $ENVS）>
  local O=$1 N=$2; shift 2
  (cd $ENVS && tar -cf - "$@" | gzip -1 > $O/$N.tar.gz) && (cd $O && sha256sum $N.tar.gz > $N.sha256) && cat $O/$N.sha256 && du -sh $O/$N.tar.gz
}
[ $S_E2 = OK ] && pack $ROOT/env/pvb-env-2 pvb-venv-2 pvb-2 pvb-py && S_P2=OK
[ $S_EM = OK ] && pack $ROOT/env/pvb-env-moment-1 pvb-venv-moment-1 pvb-moment pvb-py && S_PM=OK
du -sh $ENVS/pvb-2 $ENVS/pvb-moment $ENVS/pvb-py 2>/dev/null

echo "SUMMARY_P206A2 uv=$S_UV env2=$S_E2 moment=$S_EM smoke_env2=$S_SM2 smoke_moment=$S_SMM pack_env2=$S_P2 pack_moment=$S_PM"
