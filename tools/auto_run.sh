#!/bin/bash
# =============================================================================
# auto_run.sh - GPU空闲时自动运行实验
#
# 功能：
#   1. 每60秒检测一次GPU使用情况
#   2. 当所有GPU内存占用低于阈值且无训练进程时，自动启动实验
#   3. 支持实验队列，按顺序执行
#   4. 日志记录到 work_dirs/auto_run.log
#
# 用法：
#   chmod +x tools/auto_run.sh
#   nohup bash tools/auto_run.sh > tools/auto_run.log 2>&1 &
#
# 注意：
#   - 确保在 my_diff/ 目录下运行
#   - 确保 conda 环境已激活
#   - 修改下方 CONFIG 区域的路径
# =============================================================================

# ======================== CONFIG 区域 ========================
# GPU数量
NUM_GPUS=8
# GPU内存空闲阈值（MB），低于此值认为该卡空闲
MEM_THRESHOLD=500
# 检测间隔（秒）
CHECK_INTERVAL=60
# 最多等待多少次检测（0=无限等待）
MAX_WAIT=0

# 数据集路径（根据远程服务器实际路径修改）
COCO_ANN="/path/to/data/coco/annotations/instances_train2017.json"
COCO_IMG="/path/to/data/coco/train2017"
RUOD_IMG="/path/to/data/RUOD_pic/"
RUOD_TRAIN_ANN="/path/to/data/RUOD_ANN/instances_train.json"
RUOD_TEST_ANN="/path/to/data/RUOD_ANN/instances_test.json"
UWNR_DIR="/path/to/UWNR"
UWNR_MODEL="/path/to/uwnr_epoch200.pth"
COCO_UWNR_DIR="/path/to/data/coco_uwnr"

# 工作目录
WORK_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_FILE="$WORK_DIR/tools/auto_run.log"
# ======================== END CONFIG ========================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# 检查所有GPU是否空闲
check_gpus_free() {
    # 检查是否有训练进程在跑
    if pgrep -f "tools/train.py" > /dev/null 2>&1; then
        log "检测到 train.py 进程仍在运行，跳过"
        return 1
    fi

    # 检查GPU内存占用
    local free_count=0
    for i in $(seq 0 $((NUM_GPUS - 1))); do
        # 获取GPU显存使用（MB）
        local mem_used
        mem_used=$(nvidia-smi --id=$i --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
        if [ -z "$mem_used" ]; then
            log "无法读取GPU $i 信息"
            return 1
        fi

        if [ "$mem_used" -lt "$MEM_THRESHOLD" ]; then
            free_count=$((free_count + 1))
        else
            log "GPU $i 显存占用 ${mem_used}MB > 阈值 ${MEM_THRESHOLD}MB"
        fi
    done

    if [ "$free_count" -eq "$NUM_GPUS" ]; then
        return 0
    else
        log "只有 ${free_count}/${NUM_GPUS} 张GPU空闲"
        return 1
    fi
}

# 运行训练
run_train() {
    local config=$1
    local stage_name=$2
    log "========== 开始运行: $stage_name =========="
    log "Config: $config"

    cd "$WORK_DIR" || exit 1
    CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS - 1))) \
    python -m torch.distributed.launch \
        --nproc_per_node=$NUM_GPUS \
        --master_port=29500 \
        tools/train.py "$config" \
        2>&1 | tee -a "$LOG_FILE"

    local exit_code=${PIPESTATUS[0]}
    if [ $exit_code -eq 0 ]; then
        log "$stage_name 完成 ✓"
    else
        log "$stage_name 失败 (exit code: $exit_code) ✗"
        log "终止后续实验"
        exit 1
    fi
}

# 运行测试
run_test() {
    local config=$1
    local checkpoint=$2
    local stage_name=$3
    log "========== 开始测试: $stage_name =========="

    cd "$WORK_DIR" || exit 1
    CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS - 1))) \
    python tools/test.py "$config" "$checkpoint" \
        --eval bbox \
        --cfg-options model.init_cfg=None \
        2>&1 | tee -a "$LOG_FILE"

    log "$stage_name 测试完成"
}

# 等待GPU空闲
wait_for_gpu() {
    local attempt=0
    while ! check_gpus_free; do
        attempt=$((attempt + 1))
        if [ "$MAX_WAIT" -gt 0 ] && [ "$attempt" -ge "$MAX_WAIT" ]; then
            log "已等待 $attempt 次，超过最大等待次数 $MAX_WAIT，退出"
            exit 1
        fi
        log "等待GPU空闲... (第 ${attempt} 次检测)"
        sleep $CHECK_INTERVAL
    done
    log "GPU已空闲，准备开始"
}

# ================================================================
#  实验队列（按顺序执行）
# ================================================================
main() {
    log "=========================================="
    log "自动实验脚本启动"
    log "GPU数量: $NUM_GPUS"
    log "显存阈值: ${MEM_THRESHOLD}MB"
    log "检测间隔: ${CHECK_INTERVAL}s"
    log "=========================================="

    cd "$WORK_DIR" || exit 1

    # ========== 实验A：ImageNet → RUOD ==========
    wait_for_gpu
    run_train "config/cascade_r50_ruod_baseline.py" "实验A: ImageNet+RUOD"

    wait_for_gpu
    run_test "config/cascade_r50_ruod_baseline.py" \
        "work_dirs/cascade_r50_ruod_baseline/latest.pth" \
        "实验A"

    # ========== 实验B：COCO+UWNR → RUOD ==========

    # B-1: 抽样50k COCO（CPU操作，直接执行）
    log "========== B-1: 抽样50k COCO =========="
    python tools/sample_coco.py \
        --ann "$COCO_ANN" \
        --img-dir "$COCO_IMG" \
        --output-dir "$COCO_UWNR_DIR" \
        --num 50000 \
        2>&1 | tee -a "$LOG_FILE"

    # B-2: UWNR转水下（需要GPU）
    wait_for_gpu
    log "========== B-2: UWNR转换 =========="
    CUDA_VISIBLE_DEVICES=0 \
    python tools/convert_coco_uwnr.py \
        --ann "$COCO_UWNR_DIR/annotations/instances_train50k.json" \
        --img-dir "$COCO_UWNR_DIR/images" \
        --output-dir "$COCO_UWNR_DIR" \
        --uwnr-dir "$UWNR_DIR" \
        --uwnr-model "$UWNR_MODEL" \
        2>&1 | tee -a "$LOG_FILE"

    # B-3: 在增强COCO上训练
    wait_for_gpu
    run_train "config/cascade_r50_coco_uwnr.py" "B-3: COCO-UWNR预训练"

    # B-4: 提取backbone权重
    log "========== B-4: 提取backbone权重 =========="
    python tools/extract_backbone.py \
        --checkpoint "work_dirs/cascade_r50_coco_uwnr/epoch_24.pth" \
        --output "work_dirs/cascade_r50_coco_uwnr/backbone_only.pth"

    # B-5: 微调RUOD
    wait_for_gpu
    run_train "config/cascade_r50_ruod_uwnr_pretrain.py" "B-5: COCO-UWNR+RUOD"

    wait_for_gpu
    run_test "config/cascade_r50_ruod_uwnr_pretrain.py" \
        "work_dirs/cascade_r50_ruod_uwnr_pretrain/latest.pth" \
        "实验B"

    log "=========================================="
    log "所有实验完成！"
    log "=========================================="
}

main "$@"
