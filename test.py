#!/usr/bin/env python3
"""
GPU-Monitor-LOCAL  (无通知、精简版)
一次性占用本机真正空闲的全部 GPU。
空闲判定：显存空闲 ≥80 % 且 GPU-util ≤5 %，连续 10 秒。
所有日志统一写到 /home/fuping/xcx/logs/。
连续“无空闲 GPU”信息每 30 次仅记录一次。
"""
import logging, subprocess, time, sys, os, atexit, shlex
import random
from pathlib import Path
from collections import deque
from logging.handlers import TimedRotatingFileHandler

# -------------------------------------------------
# 手动假 GPU 模式：True = 人工输入，False = 真实 nvidia-smi
FAKE_GPU_MODE = True
# -------------------------------------------------

# ------------------ 本地配置 ------------------ #
WORK_DIR_PLACEHOLDER = Path("/home/fuping/xcx/my_diff")   # 占位任务目录
WORK_DIR_TASK        = Path("/home/fuping/xcx/Detector/mmdetection")  # 排队任务目录
LOG_DIR = Path("/home/fuping/xcx/logs/")
LOG_FILE = LOG_DIR / "gpu_monitor.log"
INTERVAL = 2                            # 检测间隔（秒）
BUSY_REPEAT = 30                        # 连续无空闲日志去重阈值

ENV_PLACEHOLDER = "endiff"
BASE_CMD_PLACEHOLDER = (
    "torchrun --nproc_per_node={num_gpu} "
    "tools/train_diff.py config/EnDiff_r50_diff0.py "
    "--train-mode diff --test-num 1.0 --work-dir work_dirs/occupied "
    "--launcher pytorch --auto-resume --auto-scale-lr"
)

TASK_COMMAND_QUEUE = deque([
    (
        "PORT={port} ./tools/dist_train.sh configs/cascade_rcnn/cascade-rcnn_r50_fpn_1x_coco_ENHANCED.py 4 "
        "--work-dir ./work_dirs/cascade-rcnn_r50_fpn_1x_coco_ENHANCED ",
        "detector"
    ),
    (
        "PORT={port} ./tools/dist_train.sh configs/dino/dino-4scale_r50_8xb2-12e_coco_ENHANCED.py 4 "
        "--work-dir ./work_dirs/CO-DETR_ENHANCED ",
        "detector"
    ),
])

# ------------------ 日志 ------------------ #
def setup_logger():
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("gpu_monitor")
    logger.setLevel(logging.INFO)
    file_hdl = TimedRotatingFileHandler(LOG_FILE, when='midnight', interval=1, backupCount=30, encoding='utf-8')
    file_hdl.setFormatter(fmt)
    console_hdl = logging.StreamHandler(sys.stdout)
    console_hdl.setFormatter(fmt)
    logger.addHandler(file_hdl)
    logger.addHandler(console_hdl)
    return logger
logger = setup_logger()

CHILD_PROCS = []
atexit.register(lambda: [p.terminate() for p in CHILD_PROCS])

# ------------------ 空闲判定 ------------------ #
def gpu_is_idle(idx: str, mem_ratio: float = 0.8, util_max: int = 5, duration: int = 10) -> bool:
    cmd = ["nvidia-smi", "--query-gpu=memory.free,memory.total,utilization.gpu",
           "--format=csv,noheader,nounits", "-i", idx]
    for _ in range(duration):
        try:
            out = subprocess.check_output(cmd, text=True).strip().split(", ")
            free, total, util = map(int, out)
            if (free / total < mem_ratio) or (util > util_max):
                return False
        except Exception:
            return False
        time.sleep(1)
    return True

def get_idle_gpus() -> list[str]:
    if not FAKE_GPU_MODE:
        # 原始真实逻辑
        try:
            all_idx = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
                text=True).strip().split("\n")
            idle = [idx for idx in all_idx if gpu_is_idle(idx)]
            return idle
        except Exception as e:
            logger.error(f"获取 GPU 信息失败: {e}")
            return []
    else:
        # 人工输入模式
        raw = input(">>> 手动输入空闲 GPU（空格分隔，如 0 2 3；回车=无空闲，q=退出脚本）：").strip()
        if raw.lower() == 'q':
            logger.info("用户手动退出")
            os._exit(0)
        return raw.split()

# ------------------ 任务执行 ------------------ #
def run_command(gpus, cmd_template, env, task_type="placeholder"):
    num_gpu = len(gpus)
    gpu_str = ",".join(gpus)
    random.seed()
    port = random.randint(10000, 30000)

    if task_type == "placeholder":
        # 占位任务：使用 BASE_CMD_PLACEHOLDER
        full_cmd = BASE_CMD_PLACEHOLDER.format(num_gpu=num_gpu, port=port)
        cwd = WORK_DIR_PLACEHOLDER
    else:
        # 排队任务：使用 cmd_template
        full_cmd = cmd_template.format(num_gpu=num_gpu, port=port)
        cwd = WORK_DIR_TASK

    log_file = LOG_DIR / f"{task_type}_{gpu_str.replace(',', '-')}.log"

    # 组装最终命令
    cmd = [
        "/home/fuping/miniconda3/bin/conda", "run", "-n", env,
        "bash", "-c", full_cmd  # 用 bash -c 来执行复杂命令
    ]

    shell_cmd = " ".join(shlex.quote(str(x)) for x in cmd)
    logger.info(f"[拼接命令] {shell_cmd}")
    try:
        logger.info(f"启动 {task_type}，GPU={gpu_str}，日志={log_file}")
        with open(log_file, "w", buffering=1) as f:
            proc = subprocess.Popen(
                cmd,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu_str,
                     "PATH": "/home/fuping/miniconda3/bin:" + os.environ.get("PATH", "")},
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=str(cwd),
                text=True,
                bufsize=1
            )
            CHILD_PROCS.append(proc)
            if task_type == "placeholder":
                PLACEHOLDER_PROCS[proc.pid] = gpus
            return proc
    except Exception:
        logger.exception("启动任务失败")
        return None
# 在主循环开始前增加一个映射：占位进程 → 占的 GPU
PLACEHOLDER_PROCS = {}      # pid -> [gpu_list]

def main():
    logger.info("=" * 25 + " GPU 本地监控脚本启动 " + "=" * 25)
    occupied = set()
    busy_counter = 0
    last_full_report = time.time()

    while True:
        try:
            free_now = get_idle_gpus()
            # 空闲 = 真正空闲 + 占位任务占的
            placeholder_gpus = {g for p in PLACEHOLDER_PROCS.values() for g in p}
            candidate = [g for g in free_now if g not in occupied - placeholder_gpus]

            # -------- 1 小时整点报告 --------
            if time.time() - last_full_report >= 3600:
                current_placeholder = {g for gpus in PLACEHOLDER_PROCS.values() for g in gpus}
                all_idx = get_all_gpus_index()  # 见下方
                idle_idx = set(free_now)
                used_idx = set(all_idx) - idle_idx
                logger.info(
                    f"[整点报告] 总 GPU: {sorted(all_idx)} | "
                    f"空闲: {sorted(idle_idx)} | "
                    f"占位: {sorted(current_placeholder)} | "
                    f"已用: {sorted(used_idx)}"
                )
                last_full_report = time.time()

            # 1. 先处理排队任务：空闲+占位 >=4
            while len(candidate) >= 4 and TASK_COMMAND_QUEUE:
                need = 4
                # 如果真正空闲不足4，则杀掉占位任务补位
                if len([g for g in candidate if g not in occupied]) < 4:
                    to_kill = []
                    for pid, gpus in list(PLACEHOLDER_PROCS.items()):
                        if len(candidate) >= 4:
                            break
                        to_kill.append(pid)
                        candidate.extend(gpus)
                        occupied.difference_update(gpus)
                    for pid in to_kill:
                        proc = next(p for p in CHILD_PROCS if p.pid == pid)
                        proc.terminate()
                        CHILD_PROCS.remove(proc)
                        del PLACEHOLDER_PROCS[pid]

                # 取前 4 张
                gpus = candidate[:4]
                candidate = candidate[4:]
                cmd, env = TASK_COMMAND_QUEUE.popleft()
                run_command(gpus, cmd, env, "task")
                occupied.update(gpus)

            # 2. 剩余真正空闲 GPU 继续占位
            real_free = [g for g in free_now if g not in occupied]
            if real_free:
                proc = run_command(real_free, BASE_CMD_PLACEHOLDER, ENV_PLACEHOLDER, "placeholder")
                if proc is not None:  # 防止返回 None
                    PLACEHOLDER_PROCS[proc.pid] = real_free
                    occupied.update(real_free)

            # 3. 日志去重
            if not real_free:
                busy_counter += 1
                if busy_counter == 1:
                    logger.info("当前无空闲 GPU，继续等待...")
                if busy_counter >= BUSY_REPEAT:
                    busy_counter = 0
            else:
                busy_counter = 0

            time.sleep(INTERVAL)

        except Exception as e:
            logger.exception("主循环异常，60 秒后重试")
            time.sleep(60)

# 辅助：一次性拿到全部 GPU 索引
# def get_all_gpus_index() -> list[str]:
#     return subprocess.check_output(
#         ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
#         text=True
#     ).strip().split("\n")
def get_all_gpus_index() -> list[str]:
    return [str(i) for i in range(8)]  # 假设 8 卡机器

if __name__ == "__main__":
    main()
