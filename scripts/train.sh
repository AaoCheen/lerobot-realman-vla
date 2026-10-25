#!/usr/bin/env bash
# 本分支修改：增加π0.5四卡FSDP准备/训练/恢复入口，保留上游其他策略入口。
# π0.5：bash scripts/train.sh pi05 prepare  # 只准备划分/配置，不占GPU
#       PI05_OUTPUT_DIR=/大容量盘/新实验 bash scripts/train.sh pi05
#       PI05_OUTPUT_DIR=/同一实验目录 bash scripts/train.sh pi05 resume
# 可选 PI05_STEPS=10 做真实模型短跑；PYTHON 指向独立训练环境解释器。
# 其他策略仍用 bash scripts/train.sh act|diffusion|vqbet|smolvla|pi0 [resume]
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
POLICY="${1:-act}"
MODE="${2:-train}"
PYTHON="${PYTHON:-python}"
if [[ "$MODE" != train && "$MODE" != prepare && "$MODE" != resume ]]; then
    echo "第二参数只能是 prepare、resume 或省略" >&2; exit 2
fi
CONFIG_FILE="configs/${POLICY}_realman.yaml"
[[ -f "$CONFIG_FILE" ]] || { echo "配置不存在：$CONFIG_FILE" >&2; exit 2; }
if [[ "$POLICY" == pi05 ]]; then
    # CUDA_VISIBLE_DEVICES是物理卡号；默认0、1、2、7号卡；可显式设置PI05_GPUS，禁止自动抢占其他卡。
    export CUDA_VISIBLE_DEVICES="${PI05_GPUS:-0,1,2,7}"
    export TOKENIZERS_PARALLELISM=false
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    OUTPUT="${PI05_OUTPUT_DIR:-$HOME/vla-runs/pi05_realman_full}"
    RUNTIME="${PI05_RUNTIME_DIR:-$REPO_DIR/.train/pi05}"
    prep=(scripts/prepare_pi05.py --output-dir "$OUTPUT" --runtime-dir "$RUNTIME")
    [[ -n "${PI05_STEPS:-}" ]] && prep+=(--steps "$PI05_STEPS")
    [[ "$MODE" == prepare ]] && prep+=(--prepare-only)
    [[ "$MODE" == resume ]] && prep+=(--resume)
    # prepare可用轻量分析环境，不要求给它安装整个训练栈。
    "${PREP_PYTHON:-$PYTHON}" "${prep[@]}"
    [[ "$MODE" == prepare ]] && exit 0
    cmd=("$PYTHON" -m accelerate.commands.launch
         --config_file configs/accelerate_pi05_fsdp.yaml
         --num_processes 4 --num_machines 1
         --main_process_port "${PI05_PORT:-29517}"
         scripts/train_pi05_fsdp.py --config_path "$RUNTIME/train.json")
    [[ "$MODE" == resume ]] && cmd+=(--resume)
else
    [[ "$MODE" != prepare ]] || { echo 'prepare仅用于pi05' >&2; exit 2; }
    cmd=("$PYTHON" -m lerobot.scripts.lerobot_train --config_path "$CONFIG_FILE")
    if [[ "$MODE" == resume ]]; then
        OUTPUT="$($PYTHON -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["output_dir"])' "$CONFIG_FILE")"
        LAST="$OUTPUT/checkpoints/last/pretrained_model/train_config.json"
        [[ -f "$LAST" ]] || { echo "检查点不存在，不会从头重训：$LAST" >&2; exit 2; }
        cmd=("$PYTHON" -m lerobot.scripts.lerobot_train --config_path "$LAST" --resume=true)
    fi
fi
printf '执行：'; printf ' %q' "${cmd[@]}"; printf '\n'
exec "${cmd[@]}"
