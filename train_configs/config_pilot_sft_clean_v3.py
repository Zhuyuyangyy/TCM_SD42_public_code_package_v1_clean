# -*- coding: utf-8 -*-
"""
Pilot SFT Clean v3: 纯 SFT，inputv2 格式，无 rule/KD/DPO。
目标：干净格式训练；正式评估提取单一闭集证型标签。
"""
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import *

MAX_SEQ_LENGTH = 768

# ── Pilot data ────────────────────────────────────────────────
PILOT_DATA_DIR = Path(os.environ.get("TCM_DATA_DIR", "/path/to/mini_42_pilot"))
TRAIN_FILE = PILOT_DATA_DIR / "train.json"
DEV_FILE   = PILOT_DATA_DIR / "dev.json"
TEST_FILE  = PILOT_DATA_DIR / "test.json"
LABEL_LIST_FILE = PILOT_DATA_DIR / "syndrome_label_list.json"
RULE_DICT_FILE  = PILOT_DATA_DIR / "rule_dict_42.json"
SYNDROME_CONSTRAINTS_FILE = PILOT_DATA_DIR / "syndrome_constraints_42.json"

BASE_DIR = PILOT_DATA_DIR

# ── Clean v3 output ─────────────────────────────────────────
OUTPUT_ROOT = Path(os.environ.get("TCM_OUTPUT_ROOT", "/path/to/output"))
MODEL_DIR = OUTPUT_ROOT / "models"
LOG_DIR = OUTPUT_ROOT / "logs"
CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoints"
BEST_MODEL_PATH = MODEL_DIR / "pilot_sft_clean_v3_lora_adapter"

for _d in (MODEL_DIR, LOG_DIR, CHECKPOINT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

GROUP_NAME = "pilot_sft_clean_v3"

# ── Training mode: pure SFT ────────────────────────────────────
USE_RULE_LOSS = False
USE_KD = False
USE_DPO = False

LOSS_LAMBDA_SFT = 1.0
LOSS_LAMBDA_KL = 0.0
LOSS_LAMBDA_RULE = 0.0
LOSS_LAMBDA_SMOOTH = 0.0

# ── Experiment result paths ──────────────────────────────────────
EXPERIMENT_CSV = OUTPUT_ROOT / "experiment_results.csv"
EXPERIMENT_MD = OUTPUT_ROOT / "experiment_table.md"
SUMMARY_MD = OUTPUT_ROOT / "training_summary.md"