# -*- coding: utf-8 -*-
"""
Group config: Pilot-SFT baseline
训练集 4186 条，纯 SFT，无 KD/DPO/Rule Loss
"""
from __future__ import annotations
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

# ── Pilot 数据路径 ──────────────────────────────────────────
BASE_DIR = Path(os.environ.get("TCM_DATA_DIR", "/path/to/mini_42_pilot"))
DATA_DIR = BASE_DIR

# ── Pilot 输出路径 ──────────────────────────────────────────
MODEL_DIR = Path(os.environ.get("TCM_MODEL_DIR", "/path/to/output"))
LOG_DIR = Path(os.environ.get("TCM_LOG_DIR", "/path/to/logs"))
CHECKPOINT_DIR = Path(os.environ.get("TCM_CHECKPOINT_DIR", "/path/to/checkpoints"))

for _d in (MODEL_DIR, LOG_DIR, CHECKPOINT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── 数据文件 ────────────────────────────────────────────────
TRAIN_FILE = DATA_DIR / "train.json"
DEV_FILE = DATA_DIR / "dev.json"
TEST_FILE = DATA_DIR / "test.json"
LABEL_LIST_FILE = DATA_DIR / "syndrome_label_list.json"
RULE_DICT_FILE = DATA_DIR / "rule_dict_42.json"
SYNDROME_CONSTRAINTS_FILE = DATA_DIR / "syndrome_constraints_42.json"

# ── 训练模式（纯 SFT）────────────────────────────────────────
USE_RULE_LOSS = False
USE_KD = False
USE_DPO = False

# ── 超参数 ─────────────────────────────────────────────────
STAGE1_EPOCHS = 3
STAGE2_EPOCHS = 2

# 输出路径
SFT_BASELINE_PATH = MODEL_DIR / "models" / "round-0"
BEST_MODEL_PATH = MODEL_DIR / "models" / "pilot_sft_clean_v3_lora_adapter"

# group name
GROUP_NAME = "pilot_sft_baseline"
