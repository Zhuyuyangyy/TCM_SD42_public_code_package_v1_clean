# -*- coding: utf-8 -*-
"""
Public global config for the TCM-SD-42 LoRA-SFT manuscript package.

This file keeps only public-path defaults and manuscript-aligned runtime settings.
Legacy KD/DPO/Rule-Loss paths are disabled for the reported experiments.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch


# ============================================================
# Runtime mode
# ============================================================
LOCAL_MODE = os.environ.get("TCM_LOCAL_MODE", "0") == "1"


# ============================================================
# Paths — TCM-SD-42 balanced inputv2
# ============================================================
BASE_DIR = Path(os.environ.get("TCM_DATA_DIR", "/path/to/processed_v4_top50_balanced_cap1000_inputv2"))
DATA_DIR = BASE_DIR
MODEL_DIR = Path(os.environ.get("TCM_MODEL_DIR", "/path/to/output/models"))
LOG_DIR = Path(os.environ.get("TCM_LOG_DIR", "/path/to/output/logs"))
CHECKPOINT_DIR = Path(os.environ.get("TCM_CHECKPOINT_DIR", "/path/to/output/checkpoints"))

for _directory in (MODEL_DIR, LOG_DIR, CHECKPOINT_DIR):
    _directory.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = DATA_DIR / "train.json"
DEV_FILE = DATA_DIR / "dev.json"
TEST_FILE = DATA_DIR / "test.json"
VAL_UNSEEN_FILE = DATA_DIR / "tcm_sd_val_unseen.json"
LABEL_LIST_FILE = DATA_DIR / "syndrome_label_list.json"
RULE_DICT_FILE = DATA_DIR / "rule_dict_42.json"
SYNDROME_CONSTRAINTS_FILE = DATA_DIR / "syndrome_constraints_42.json"

OPTIMIZATION_LOG = LOG_DIR / "optimization_log.txt"
DATASET_LOG = LOG_DIR / "dataset_log.txt"
TRAINING_LOG = LOG_DIR / "training_log.txt"
EXPERIMENT_CSV = BASE_DIR / "experiment_results.csv"
EXPERIMENT_MD = BASE_DIR / "experiment_table.md"
SUMMARY_MD = BASE_DIR / "training_summary.md"


# ============================================================
# Model / device
# ============================================================
MODEL_NAME = os.environ.get("TCM_MODEL_NAME", "/path/to/Qwen3-8B")
TEACHER_MODEL_NAME = MODEL_NAME  # no separate teacher model is used in the reported manuscript experiments

if torch.cuda.is_available():
    DEVICE = "cuda"
    GPU_NAME = torch.cuda.get_device_name(0)
    VRAM_GB = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
else:
    DEVICE = "cpu"
    GPU_NAME = "CPU"
    VRAM_GB = 0.0

MODE_DESC = f"{DEVICE} | {GPU_NAME} | model={MODEL_NAME}"


# ============================================================
# Training hyperparameters
# ============================================================
QUANT_BITS = 4 if DEVICE == "cuda" else 0
TEACHER_QUANT_BITS = 4 if DEVICE == "cuda" else 0
LORA_R = 16 if DEVICE == "cuda" else 4
LORA_ALPHA = 32 if DEVICE == "cuda" else 8
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

TRAIN_BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 16
USE_GRADIENT_CHECKPOINT = True
LEARNING_RATE = 1e-5
NUM_EPOCHS_PER_ROUND = 1
WARMUP_STEPS = 10
WARMUP_RATIO = 0.03
MAX_SEQ_LENGTH = 512
SAVE_STEP_CHECKPOINTS = False  # 关闭每100步存checkpoint，防止磁盘爆炸
SAVE_STEPS = 999999

TARGET_ACCURACY = None  # no deployment target is claimed
MAX_ROUNDS = 1
EARLY_STOP_DELTA = 0.0
PATIENCE_ROUNDS = 0
VAL_SPLIT = 0.15


# ============================================================
# Three-stage progressive training
# ============================================================
STAGE1_FROZEN_LAYERS = list(range(0, 23))
STAGE1_OPEN_LAYERS = list(range(23, 32))
STAGE1_LR = 8e-5
STAGE1_EPOCHS = 3
STAGE1_LOSS_KL_W = 1.2
STAGE1_TEMPERATURE = 2.0

STAGE2_FROZEN_LAYERS = list(range(0, 12))
STAGE2_OPEN_LAYERS = list(range(12, 32))
STAGE2_LR = 1e-5
STAGE2_EPOCHS = 2
STAGE2_LOSS_KL_W = 1.0
STAGE2_LOSS_RULE_W = 0.4
STAGE2_LOSS_SMOOTH_W = 0.1

STAGE3_LORA_R = 64
STAGE3_LORA_ALPHA = 128


# ============================================================
# Manuscript loss switches
# ============================================================
# The reported manuscript experiments use pure SFT. KD, Rule Loss, and DPO
# were not included in the formal results. These constants are retained at
# zero only so legacy utility code can import the config without failing.
USE_KD = False
USE_RULE_LOSS = False
USE_DPO = False
DISTILL_TEMP_MAX = 1.0
DISTILL_TEMP_MIN = 1.0
DISTILL_TEMPERATURE = 1.0
LABEL_SMOOTHING = 0.0

LOSS_LAMBDA_SFT = 1.0
LOSS_LAMBDA_KL = 0.0
LOSS_LAMBDA_RULE = 0.0
LOSS_LAMBDA_SMOOTH = 0.0


# ============================================================
# Dataset info — TCM-SD-42
# ============================================================
DATASET_TOTAL_TRAIN = 19125
DATASET_TOTAL_DEV = 4411
DATASET_TOTAL_TEST = 4452
N_LABELS = 42
INPUT_VERSION = "disease_chief_detection_desc500_v1"


# ============================================================
# DPO (disabled in manuscript experiments)
# ============================================================
DPO_BETA = 0.0
DPO_LEARNING_RATE = 0.0
DPO_NUM_EPOCHS = 0.0
DPO_BATCH_SIZE = 1
DPO_GRAD_ACCUM_STEPS = 1
DPO_MAX_PROMPT_LENGTH = 128
DPO_MAX_LENGTH = 512


# ============================================================
# Output model dirs
# ============================================================
SFT_BASELINE_PATH = MODEL_DIR / "round-0"
BEST_MODEL_PATH = MODEL_DIR / "pilot_sft_clean_v3_lora_adapter"
CHECKPOINT_PREFIX = CHECKPOINT_DIR / "round"


# ============================================================
# Rule dict — loaded dynamically from 42-class knowledge base
# ============================================================

def _load_json(path):
    import json as _json
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return _json.load(f)
    return {}

RULE_DICT = {}  # Will be loaded from rule_dict_42.json at runtime
SYNDROME_PRESCRIPTION_CONSTRAINTS = {}  # Will be loaded from syndrome_constraints_42.json at runtime

def _init_rules():
    """Load 42-class rules from generated files."""
    global RULE_DICT, SYNDROME_PRESCRIPTION_CONSTRAINTS
    if not RULE_DICT:
        rd = _load_json(RULE_DICT_FILE)
        # rule_dict_42.json uses "formulas" key
        for sx, info in rd.items():
            RULE_DICT[sx] = info.get("formulas", info.get("方剂", []))
    if not SYNDROME_PRESCRIPTION_CONSTRAINTS:
        SYNDROME_PRESCRIPTION_CONSTRAINTS = _load_json(SYNDROME_CONSTRAINTS_FILE)

# Auto-load on import (commented out for public use — call _init_rules() manually or set paths first)
# _init_rules()

RULE_TOKEN_DICT: dict[str, list[int]] = {}
_RULE_TOKEN_DICT_BUILT = False

MAX_RETRIES = 2
DISCLAIMER = ""  # Not needed for TCM-SD real data pipeline


def print_config() -> None:
    print("=" * 60)
    print("TCM-SD-42 Config")
    print("=" * 60)
    print(f"DEVICE: {DEVICE}")
    print(f"GPU_NAME: {GPU_NAME}")
    print(f"MODEL_NAME: {MODEL_NAME}")
    print(f"DATA_DIR: {DATA_DIR}")
    print(f"N_LABELS: {N_LABELS}")
    print(f"TRAIN: {DATASET_TOTAL_TRAIN}, DEV: {DATASET_TOTAL_DEV}, TEST: {DATASET_TOTAL_TEST}")
    print(f"INPUT_VERSION: {INPUT_VERSION}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
