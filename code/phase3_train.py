# -*- coding: utf-8 -*-
"""
Phase 3: LoRA supervised fine-tuning reference script for the TCM-SD-42 manuscript package.

The public clean-v3 config disables KD, DPO, and Rule Loss; the reported manuscript
experiments use pure SFT.
"""

from __future__ import annotations

import gc
import json
import random
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).parent))
# NOTE: path configuration is delegated to train_configs/config_pilot_sft_clean_v3.py
# which must define DATA_DIR, MODEL_DIR, TRAIN_FILE, etc. as Path objects.
import config as cfg

# 预构建 RULE_TOKEN_DICT（避免训练时动态构建失败）
if not cfg.RULE_TOKEN_DICT:
    try:
        from transformers import AutoTokenizer as _AT
        _tok = _AT.from_pretrained(cfg.MODEL_NAME, trust_remote_code=True)
        _rtd = {}
        for _syn, _herbs in cfg.RULE_DICT.items():
            _ids = []
            for _h in _herbs:
                _ids.extend(_tok.encode(_h.replace(chr(22823), chr(37327)), add_special_tokens=False))
            _rtd[_syn] = _ids
        cfg.RULE_TOKEN_DICT = _rtd
        print(f'[RULE_TOKEN_DICT] 预构建完成，{len(_rtd)} 个证型')
    except Exception as e:
        print(f'[WARN] 预构建 RULE_TOKEN_DICT 失败: {e}')

def write_log(message: str) -> None:
    log_file = cfg.LOG_DIR / "phase3_train.log"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] [Phase3] {message}"
    print(line)
    with open(log_file, "a", encoding="utf-8") as file:
        file.write(line + "\n")


# ============================================================
# 温度退火（专利 S2）: T(t)=T_min+0.5*(T_max-T_min)*(1+cos(π*t/t_max))
# ============================================================
import math

def cosine_temperature(step: int, total_steps: int) -> float:
    """余弦退火温度，从 DISTILL_TEMP_MAX 衰减到 DISTILL_TEMP_MIN"""
    t_max = cfg.DISTILL_TEMP_MAX
    t_min = cfg.DISTILL_TEMP_MIN
    ratio = math.cos(math.pi * step / max(total_steps, 1))
    return t_min + 0.5 * (t_max - t_min) * (1.0 + ratio)


def _convert_inputv2_sample(sample: dict) -> dict:
    """Convert an inputv2-format sample to the internal format for build_training_sample().
    Inputv2 uses: input_text, syndrome_canonical, formula_candidates, treatment_candidates.
    """
    # 直接使用 inputv2 的 input_text 作为输入
    input_text = sample.get("input_text", "")
    if not input_text:
        # fallback: 重新拼接
        disease = sample.get("lcd_name", "")
        chief   = sample.get("chief_complaint", "")
        det     = sample.get("detection", "")
        desc    = sample.get("description", "")[:500]
        input_text = f"疾病名称：{disease}\n主诉：{chief}\n四诊信息：{det}\n现病描述摘要：{desc}"

    # 治法（从 knowledge base）
    treatment = sample.get("treatment_candidates", "")
    if isinstance(treatment, list):
        treatment = "、".join(str(t) for t in treatment if t)

    # 方剂候选（辅助，不进主监督）
    formulas = sample.get("formula_candidates", [])
    if isinstance(formulas, list):
        formula_str = "、".join(str(f) for f in formulas if f)
    else:
        formula_str = str(formulas) if formulas else ""

    return {
        "input_text": input_text,
        "syndrome": sample.get("syndrome_canonical", ""),
        "treatment": treatment,
        "formulas": formula_str,
    }


def get_or_create_train_val_split():
    """Load train/dev from TCM-SD-42 inputv2 (already split by original dataset)."""
    # Direct load from inputv2 — no need to re-split
    with open(cfg.TRAIN_FILE, "r", encoding="utf-8") as file:
        train_raw = json.load(file)
    with open(cfg.DEV_FILE, "r", encoding="utf-8") as file:
        val_raw = json.load(file)

    # Convert inputv2 samples to internal format if needed
    train_data = [_convert_inputv2_sample(s) for s in train_raw]
    val_data   = [_convert_inputv2_sample(s) for s in val_raw]

    write_log(f"Loaded from inputv2: train={len(train_data)}, val={len(val_data)}")
    write_log(f"INPUT_VERSION={getattr(cfg, 'INPUT_VERSION', 'unknown')}")

    # Shuffle train (already shuffled by prepare_42class.py, but ensure)
    random.seed(42)
    random.shuffle(train_data)

    # Smoke 模式：截断数据，加速验证
    smoke_max = getattr(cfg, "SMOKE_MAX_STEPS", None)
    if smoke_max:
        train_data = train_data[:128]
        val_data = val_data[:32]
        write_log(f"[Smoke] 数据截断: train={len(train_data)}, val={len(val_data)}")

    # 诊断：检查 tokenize 后有效 token 比例
    try:
        from transformers import AutoTokenizer as _DiagTok
        _diag_tok = _DiagTok.from_pretrained(cfg.MODEL_NAME, trust_remote_code=True, use_fast=False)
        over_limit = 0
        for _s in (train_data[:50] if smoke_max else train_data[:200]):
            _sample = build_training_sample(_s)
            _ids = _diag_tok.encode(_sample["prompt"] + _sample["answer"], add_special_tokens=True)
            if len(_ids) > cfg.MAX_SEQ_LENGTH:
                over_limit += 1
        _total = min(50, len(train_data)) if smoke_max else min(200, len(train_data))
        write_log(f"[诊断] MAX_SEQ_LENGTH={cfg.MAX_SEQ_LENGTH}，超过限制的样本: {over_limit}/{_total}")
        if over_limit > _total * 0.3:
            write_log(f"[警告] 超过 30% 样本会截断！建议增大 MAX_SEQ_LENGTH")
    except Exception as _e:
        write_log(f"[诊断] 跳过长度检查: {_e}")

    return train_data, val_data


def normalize_formula(value) -> str:
    if isinstance(value, list):
        return "、".join(str(item).strip() for item in value if item)
    if value is None:
        return ""
    return str(value).strip()


def legacy_build_thought_chain(entry: dict) -> str:
    value = entry.get("思维链")
    if isinstance(value, str) and value.strip():
        return value.strip()

    value = entry.get("决策推理链")
    if isinstance(value, list):
        merged = "；".join(str(item).strip() for item in value if str(item).strip())
        if merged:
            return merged

    return (
        f"根据{entry.get('症状', '')}，"
        f"判定病位在{entry.get('病位', '')}，"
        f"性质属待结合症状进一步判断，"
        f"故辨为{entry.get('证型', '')}，"
        f"治以{entry.get('治法', '')}，"
        f"方用{normalize_formula(entry.get('方剂', ''))}"
    )


def legacy_build_training_sample(entry: dict) -> dict:
    prompt = (
        f"<|im_start|>system\n你是一位资深中医专家，必须按照固定的八字段格式输出辨证论治结果。<|im_end|>\n"
        f"<|im_start|>user\n请用中文回答，并完整输出【症状】【病位】【证型】【治法】【方剂】【依据条文】【辨证解释】【思维链】八个字段。\n症状：{entry.get('症状', '')}\n病位：{entry.get('病位', '')}\n请给出辨证论治。<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    answer = (
        f"症状：{entry.get('症状', '')}\n"
        f"病位：{entry.get('病位', '')}\n"
        f"证型：{entry.get('证型', '')}\n"
        f"治法：{entry.get('治法', '')}\n"
        f"方剂：{normalize_formula(entry.get('方剂', ''))}\n"
        f"依据条文：{entry.get('依据条文', '')}\n"
        f"辨证解释：{entry.get('辨证解释', '')}\n"
        f"思维链：{build_thought_chain(entry)}<|im_end|>"
    )
    return {
        "prompt": prompt,
        "answer": answer,
        "syndrome": entry.get("证型", ""),
    }


def safe_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        return "，".join(cleaned)
    return str(value).strip()


def build_thought_chain(entry: dict) -> str:
    value = entry.get("thought_chain")
    if isinstance(value, str) and value.strip():
        return value.strip()

    syndrome = safe_text(
        entry.get("syndrome")
        or entry.get("syndrome_canonical")
        or entry.get("证型")
        or entry.get("璇佸瀷")
    )
    treatment = safe_text(entry.get("treatment") or entry.get("治法") or entry.get("娌绘硶"))

    evidence_parts = ["根据主诉、四诊信息及现病描述综合判断"]
    if syndrome:
        evidence_parts.append(f"当前表现与{syndrome}的病机特征更为吻合")
    # 删除治法和方剂：只保留证型相关辨证依据
    # if treatment:
    #     evidence_parts.append(f"治宜{treatment}")
    # 不输出方剂相关片段
    return "，".join(part for part in evidence_parts if part) + "。"


def build_training_sample(entry: dict) -> dict:
    input_text = safe_text(entry.get("input_text"))
    syndrome = safe_text(
        entry.get("syndrome")
        or entry.get("syndrome_canonical")
        or entry.get("证型")
        or entry.get("璇佸瀷")
    )
    if not input_text:
        raise ValueError("Empty input_text in training sample.")
    if not syndrome:
        raise ValueError("Empty syndrome in training sample.")

    prompt = (
        "<|im_start|>system\n"
        "你是一位资深中医专家。请根据真实临床资料判断最符合的中医证型，"
        "并严格按照“证型”和“辨证依据”两个字段作答。<|im_end|>\n"
        "<|im_start|>user\n"
        "请根据以下真实临床资料进行中医辨证。\n"
        "要求：\n"
        "1. 只输出一个最可能的证型。\n"
        "2. 证型名称尽量使用标准证型名称。\n"
        "3. 不要输出治法、方剂、病位或其他额外字段。\n\n"
        f"临床资料：\n{input_text}\n\n"
        "输出格式：\n"
        "证型：\n"
        "辨证依据：<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    answer = (
        f"证型：{syndrome}\n"
        f"辨证依据：{build_thought_chain(entry)}<|im_end|>"
    )
    return {
        "prompt": prompt,
        "answer": answer,
        "response": answer,
        "text": prompt + answer,
        "syndrome": syndrome,
        "input_text": input_text,
    }


def build_quant_config(bits: int):
    from transformers import BitsAndBytesConfig

    if bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
    if bits == 8:
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


class DynamicLabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing: float = 0.1, ignore_index: int = -100):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        valid_mask = labels.ne(self.ignore_index)
        safe_labels = labels.masked_fill(~valid_mask, 0)
        nll_loss = -log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
        # smooth loss 只在有效 token 上计算，避免 padding 拉高数值
        smooth_loss = -log_probs.mean(dim=-1)
        smooth_loss = smooth_loss * valid_mask
        mixed = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        mixed = mixed * valid_mask
        return mixed.sum() / valid_mask.sum().clamp_min(1)


def extract_field(text: str, field_name: str) -> str:
    pattern = rf"{re.escape(field_name)}[:：]\s*([^\n]+)"
    match = re.search(pattern, text)
    return match.group(1).strip() if match else ""


def violates_rule(text: str, syndrome_hint: str) -> bool:
    syndrome = extract_field(text, "证型") or syndrome_hint
    formula_text = extract_field(text, "方剂")
    taboo_list = cfg.RULE_DICT.get(syndrome, [])
    return any(taboo in formula_text for taboo in taboo_list)


def extract_field(text: str, field_name: str) -> str:
    pattern = rf"{re.escape(field_name)}[:：]\s*([^\n]+)"
    match = re.search(pattern, text)
    return match.group(1).strip() if match else ""


def violates_rule(text: str, syndrome_hint: str) -> bool:
    syndrome = extract_field(text, "证型") or syndrome_hint
    if not syndrome:
        return False

    evidence_text = extract_field(text, "辨证依据")
    formula_text = extract_field(text, "方剂")
    check_text = " ".join(part for part in [formula_text, evidence_text] if part).strip()
    if not check_text:
        return False

    taboo_list = cfg.RULE_DICT.get(syndrome, [])
    return any(taboo and taboo in check_text for taboo in taboo_list)


def compute_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = cfg.DISTILL_TEMPERATURE,
) -> torch.Tensor:

    valid_mask = labels.ne(-100)

    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

    # 🔥 置信度mask（关键提升）
    confidence = teacher_probs.max(dim=-1).values
    conf_mask = (confidence > 0.6).float()

    token_kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)

    token_kl = token_kl * valid_mask * conf_mask

    denom = (valid_mask * conf_mask).sum().clamp_min(1)

    token_kl = token_kl.sum() / denom

    return token_kl * (temperature ** 2)


def compute_rule_loss_logits(student_logits, syndrome_list):
    """
    可导 rule loss（logits级）
    惩罚 forbidden token 概率
    """
    # Gracefully handle missing RULE_TOKEN_DICT
    if not hasattr(cfg, 'RULE_TOKEN_DICT') or not cfg.RULE_TOKEN_DICT:
        # 动态构建 RULE_TOKEN_DICT
        if not hasattr(cfg, '_RULE_TOKEN_DICT_BUILT'):
            try:
                from transformers import AutoTokenizer
                tok = AutoTokenizer.from_pretrained(cfg.MODEL_NAME, trust_remote_code=True)
                rtd = {}
                for syn, herbs in cfg.RULE_DICT.items():
                    ids = []
                    for h in herbs:
                        ids.extend(tok.encode(h.replace('大量',''), add_special_tokens=False))
                    rtd[syn] = ids
                cfg.RULE_TOKEN_DICT = rtd
                cfg._RULE_TOKEN_DICT_BUILT = True
                print(f'[RULE_TOKEN_DICT] 动态构建完成，{len(rtd)} 个证型')
            except Exception as e:
                print(f'[WARN] 构建 RULE_TOKEN_DICT 失败: {e}')
                return student_logits.new_tensor(0.0)
        else:
            return student_logits.new_tensor(0.0)
    
    device = student_logits.device
    total_penalty = 0.0
    penalty_count = 0

    for i, syndrome in enumerate(syndrome_list):
        taboo_tokens = cfg.RULE_TOKEN_DICT.get(syndrome, [])
        if not taboo_tokens:
            continue

        probs = F.softmax(student_logits[i], dim=-1)

        taboo_ids = torch.tensor(taboo_tokens, device=device, dtype=torch.long)

        taboo_prob = probs[..., taboo_ids].sum(dim=-1)  # [seq]
        total_penalty += taboo_prob.mean()
        penalty_count += 1

    if penalty_count == 0:
        return student_logits.new_tensor(0.0)
    return (total_penalty / penalty_count).to(device=device, dtype=student_logits.dtype)


class TCMSFTDataset(Dataset):
    def __init__(self, samples: list[dict], tokenizer):
        self.samples = samples
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]

        # 1. 分开编码
        prompt_encoded = self.tokenizer(sample["prompt"], add_special_tokens=True)
        answer_encoded = self.tokenizer(sample["answer"], add_special_tokens=False)

        input_ids = prompt_encoded["input_ids"] + answer_encoded["input_ids"]
        labels = [-100] * len(prompt_encoded["input_ids"]) + answer_encoded["input_ids"]

        # 2. 截断或 Padding 到 MAX_SEQ_LENGTH
        if len(input_ids) > cfg.MAX_SEQ_LENGTH:
            # 截断时保留 prompt 部分 + answer 的剩余空间
            prompt_len = len(prompt_encoded["input_ids"])
            if prompt_len >= cfg.MAX_SEQ_LENGTH:
                # prompt 本身就超长，全部截断后 labels 全是 -100 → 会导致 NaN
                input_ids = input_ids[:cfg.MAX_SEQ_LENGTH]
                labels = [-100] * cfg.MAX_SEQ_LENGTH
            else:
                # 保留完整 prompt，截断 answer
                keep_answer = cfg.MAX_SEQ_LENGTH - prompt_len
                input_ids = prompt_encoded["input_ids"] + answer_encoded["input_ids"][:keep_answer]
                labels = [-100] * prompt_len + answer_encoded["input_ids"][:keep_answer]
        else:
            padding_len = cfg.MAX_SEQ_LENGTH - len(input_ids)
            input_ids += [self.tokenizer.pad_token_id] * padding_len
            labels += [-100] * padding_len

        # 3. 检查：如果 labels 全是 -100（无有效 token），用相邻样本替代
        valid_count = sum(1 for l in labels if l != -100)
        if valid_count == 0:
            # 递归用下一个样本替代，避免无限循环
            alt_index = (index + 1) % len(self.samples)
            if alt_index != index:
                return self.__getitem__(alt_index)

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor([1 if x != self.tokenizer.pad_token_id else 0 for x in input_ids]),
            "labels": torch.tensor(labels),
            "syndrome": sample["syndrome"],
        }


def collate_fn(features: list[dict]) -> dict:
    return {
        "input_ids": torch.stack([item["input_ids"] for item in features]),
        "attention_mask": torch.stack([item["attention_mask"] for item in features]),
        "labels": torch.stack([item["labels"] for item in features]),
        "syndrome": [item["syndrome"] for item in features],
    }


class PatentTrainer:
    def __init__(self, student_model, teacher_model, tokenizer, optimizer, total_steps: int = 1):
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.smooth_loss_fn = DynamicLabelSmoothingLoss(cfg.LABEL_SMOOTHING)
        self.total_steps = total_steps   # 用于温度退火计算
        self.current_step = 0            # 跨 epoch 全局步数

    def compute_loss(self, batch: dict, temperature: float = cfg.DISTILL_TEMPERATURE) -> dict:
        student_device = next(self.student_model.parameters()).device

        syndrome_list = batch.pop("syndrome")

        input_ids = batch["input_ids"].to(student_device)
        attention_mask = batch["attention_mask"].to(student_device)
        labels = batch["labels"].to(student_device)

        # ───────── student forward（有梯度）─────────
        student_outputs = self.student_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        student_logits = student_outputs.logits
        l_sft = student_outputs.loss

        l_smooth = self.smooth_loss_fn(student_logits[:, :-1, :], labels[:, 1:])

        # ───────── KL loss（仅当 USE_KD=True 时计算）─────────
        use_kd = bool(getattr(cfg, "USE_KD", False))
        if use_kd:
            teacher_device = next(self.teacher_model.parameters()).device
            with torch.no_grad():
                teacher_outputs = self.teacher_model(
                    input_ids=batch["input_ids"].to(teacher_device),
                    attention_mask=batch["attention_mask"].to(teacher_device),
                )
                teacher_logits = teacher_outputs.logits.to(student_logits.device)
            l_kl = compute_kl_loss(student_logits, teacher_logits, labels, temperature)
            l_kl = torch.clamp(l_kl, min=0.0, max=2.0)
        else:
            l_kl = l_sft.new_tensor(0.0)

        # ───────── rule loss（仅当 USE_RULE_LOSS=True 时计算）─────────
        use_rule = bool(getattr(cfg, "USE_RULE_LOSS", False))
        if use_rule:
            l_rule = compute_rule_loss_logits(student_logits, syndrome_list)
        else:
            l_rule = l_sft.new_tensor(0.0)

        # ───────── 返回原始 loss，不在这里加权 ─────────
        return {
            "l_sft": l_sft,
            "l_kl": l_kl,
            "l_rule": l_rule,
            "l_smooth": l_smooth,
        }
    def train_epoch(
        self,
        dataloader,
        epoch_index: int,
        use_annealing: bool = False,
        fixed_temperature: float | None = None,
        kl_weight: float = cfg.LOSS_LAMBDA_KL,
    ) -> float:
        
        # 🔴 DEBUG: 确认参数是否正确传入
        write_log(f"[DEBUG train_epoch] use_annealing={use_annealing}, fixed_temperature={fixed_temperature}, kl_weight={kl_weight}")

        self.student_model.train()
        self.optimizer.zero_grad(set_to_none=True)

        loss_total = 0.0
        step_count = 0

        for batch_index, batch in enumerate(dataloader, start=1):
            self.current_step += 1

            # Smoke test: 限制每 epoch 最多 N 步
            max_steps = getattr(cfg, 'SMOKE_MAX_STEPS', None)
            if max_steps and batch_index > max_steps:
                write_log(f"[Smoke] 达到 max_steps={max_steps}，提前结束本 epoch")
                break

            # ─────────────── 温度调度 ───────────────
            if fixed_temperature is not None:
                temperature = fixed_temperature
            elif use_annealing:
                temperature = cosine_temperature(self.current_step, self.total_steps)
            else:
                temperature = cfg.DISTILL_TEMPERATURE

            # ─────────────── loss 计算 ───────────────
            loss_dict = self.compute_loss(batch, temperature=temperature)

            # ─────────────── KL 权重 ───────────────
            use_kd = bool(getattr(cfg, "USE_KD", False))
            if not use_kd:
                effective_kl_w = 0.0
            elif use_annealing:
                t_norm = (temperature - cfg.DISTILL_TEMP_MIN) / max(
                    cfg.DISTILL_TEMP_MAX - cfg.DISTILL_TEMP_MIN, 1e-6
                )
                effective_kl_w = kl_weight * t_norm
            else:
                effective_kl_w = kl_weight

            # ─────────────── 统一加权 loss ───────────────
            total_loss = (
                cfg.LOSS_LAMBDA_SFT * loss_dict["l_sft"]
                + effective_kl_w * loss_dict["l_kl"]
                + cfg.LOSS_LAMBDA_RULE * loss_dict["l_rule"]
                + cfg.LOSS_LAMBDA_SMOOTH * loss_dict["l_smooth"]
            )

            loss = total_loss / cfg.GRAD_ACCUM_STEPS

            # ─────────────── 数值安全检查 ───────────────
            if not torch.isfinite(loss):
                write_log(
                    f"[WARN] 非法 loss，跳过 batch {batch_index} | "
                    f"sft={loss_dict['l_sft'].item():.4f} "
                    f"kl={loss_dict['l_kl'].item():.4f} "
                    f"rule={loss_dict['l_rule'].item():.4f} "
                    f"smooth={loss_dict['l_smooth'].item():.4f}"
                )
                self.optimizer.zero_grad(set_to_none=True)
                continue

            loss.backward()

            # ─────────────── 梯度更新 ───────────────
            is_update_step = (
                batch_index % cfg.GRAD_ACCUM_STEPS == 0
                or batch_index == len(dataloader)  # 🔥 修复最后一批
            )

            if is_update_step:
                torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            # ─────────────── 日志（真实 loss） ───────────────
            loss_total += float(total_loss.item())
            step_count += 1

            if batch_index % 10 == 0:
                write_log(
                    f"[E{epoch_index}][S{batch_index}] "
                    f"T={temperature:.2f} KLw={effective_kl_w:.3f} "
                    f"total={total_loss.item():.4f} "
                    f"sft={loss_dict['l_sft'].item():.4f} "
                    f"kl={loss_dict['l_kl'].item():.4f} "
                    f"rule={loss_dict['l_rule'].item():.4f} "
                    f"smooth={loss_dict['l_smooth'].item():.4f}"
                )

            # ─────────────── checkpoint（更安全） ───────────────
            if False and batch_index % 100 == 0:
                ckpt_dir = cfg.SFT_BASELINE_PATH / f"checkpoint-{epoch_index}-{batch_index}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                self.student_model.save_pretrained(ckpt_dir)
                self.tokenizer.save_pretrained(ckpt_dir)

        return loss_total / max(step_count, 1)

    def evaluate_epoch(self, dataloader) -> float:
        self.student_model.eval()
        loss_total = 0.0
        step_count = 0
        with torch.no_grad():
            for batch in dataloader:
                loss_dict = self.compute_loss(batch)
                # 计算带权数的总 loss
                batch_loss = (
                    cfg.LOSS_LAMBDA_SFT * loss_dict["l_sft"]
                    + cfg.LOSS_LAMBDA_KL * loss_dict["l_kl"]
                    + cfg.LOSS_LAMBDA_RULE * loss_dict["l_rule"]
                    + cfg.LOSS_LAMBDA_SMOOTH * loss_dict["l_smooth"]
                )
                loss_total += float(batch_loss.item())
                step_count += 1
        return loss_total / max(step_count, 1)


def freeze_layers(model, frozen_layer_indices: list[int]) -> None:
    """冻结指定 Transformer Block 层的所有参数（专利 S4 三阶段解锁）"""
    for name, param in model.named_parameters():
        for layer_idx in frozen_layer_indices:
            # 匹配 base_model.model.model.layers.N 或 model.layers.N
            if f".layers.{layer_idx}." in name:
                param.requires_grad_(False)
                break


def unfreeze_layers(model, open_layer_indices: list[int]) -> None:
    """解冻指定 Transformer Block 层（LoRA 参数除外，由 PEFT 控制）"""
    for name, param in model.named_parameters():
        for layer_idx in open_layer_indices:
            if f".layers.{layer_idx}." in name:
                if param.is_floating_point():
                    param.requires_grad_(True)
                break


def load_teacher_student_models():
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    student_kwargs = {"trust_remote_code": True}
    teacher_kwargs = {"trust_remote_code": True}
    if cfg.DEVICE == "cuda":
        student_kwargs["quantization_config"] = build_quant_config(cfg.QUANT_BITS)
        student_kwargs["device_map"] = "auto"
        teacher_kwargs["quantization_config"] = build_quant_config(cfg.TEACHER_QUANT_BITS)
        teacher_kwargs["device_map"] = "auto"
    else:
        student_kwargs["device_map"] = "cpu"
        student_kwargs["torch_dtype"] = torch.float32
        teacher_kwargs["device_map"] = "cpu"
        teacher_kwargs["torch_dtype"] = torch.float32

    student_model = AutoModelForCausalLM.from_pretrained(cfg.MODEL_NAME, **student_kwargs)
    if cfg.DEVICE == "cuda":
        student_model = prepare_model_for_kbit_training(student_model)
    if cfg.USE_GRADIENT_CHECKPOINT:
        student_model.gradient_checkpointing_enable()
        student_model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=cfg.LORA_R,
        lora_alpha=cfg.LORA_ALPHA,
        lora_dropout=cfg.LORA_DROPOUT,
        target_modules=cfg.LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    student_model = get_peft_model(student_model, lora_cfg)

    teacher_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    teacher_model = AutoModelForCausalLM.from_pretrained(
        cfg.TEACHER_MODEL_NAME,
        **teacher_kwargs
    )
    teacher_model.eval()
    for parameter in teacher_model.parameters():
        parameter.requires_grad_(False)

    return student_model, teacher_model, tokenizer


def train_phase3():
    from torch.utils.data import DataLoader

    train_data, val_data = get_or_create_train_val_split()

    # ── 合并 Level-1 + Level-2 扰动语料（如已生成则载入）────────────────
    level2_path = cfg.DATA_DIR / "tcm_level2_perturbed.json"
    if level2_path.exists():
        with open(level2_path, "r", encoding="utf-8") as file:
            level2_raw = json.load(file)
        level2_data = [_convert_inputv2_sample(s) for s in level2_raw]
        # Stage1 仅用 Level-1（无扰动），从 train_data 里筛出
        level1_train = [item for item in train_data if "Level-2" not in item.get("数据来源", "")]
        mixed_train = train_data + [item for item in level2_data
                                    if item.get("数据来源", "").startswith("Level-2")]
        write_log(f"载入 Level-2 扰动语料 {len(level2_data)} 条，混合训练集 {len(mixed_train)} 条")
    else:
        level1_train = train_data
        mixed_train = train_data
        write_log("未找到 Level-2 语料，Stage1/2 均使用原始训练集（inputv2 模式）")

    level1_samples = [build_training_sample(item) for item in level1_train]
    mixed_samples  = [build_training_sample(item) for item in mixed_train]
    val_samples    = [build_training_sample(item) for item in val_data]

    student_model, teacher_model, tokenizer = load_teacher_student_models()

    def make_loader(samples, shuffle=True):
        return DataLoader(
            TCMSFTDataset(samples, tokenizer),
            batch_size=cfg.TRAIN_BATCH_SIZE,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=False,
        )

    val_loader = make_loader(val_samples, shuffle=False)
    cfg.SFT_BASELINE_PATH.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    # ══════════════════════════════════════════════════════════════
    # Stage 1: 语义锚定（专利 S4）
    # 冻结 Layer 0-22，仅开放 Layer 23-31 + LM Head
    # 数据：Level-1 金标准；温度固定 T=4.0；LR=1e-4；Epoch×3
    # ══════════════════════════════════════════════════════════════
    write_log("=== Stage 1: 语义锚定训练 (冻结 Layer 0-22) ===")
    freeze_layers(student_model, cfg.STAGE1_FROZEN_LAYERS)
    # LoRA 层不受 freeze 影响，PEFT 管理自身可训参数

    stage1_loader = make_loader(level1_samples)
    stage1_total_steps = len(stage1_loader) * cfg.STAGE1_EPOCHS
    optimizer_s1 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, student_model.parameters()),
        lr=cfg.STAGE1_LR,
        weight_decay=0.01,
    )
    trainer = PatentTrainer(student_model, teacher_model, tokenizer, optimizer_s1, total_steps=stage1_total_steps)

    for epoch in range(1, cfg.STAGE1_EPOCHS + 1):
        train_loss = trainer.train_epoch(
            stage1_loader, epoch,
            use_annealing=False,
            fixed_temperature=cfg.STAGE1_TEMPERATURE,
            kl_weight=cfg.STAGE1_LOSS_KL_W,
        )
        val_loss = trainer.evaluate_epoch(val_loader)
        write_log(f"[Stage1] epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        stage1_epoch_dir = cfg.SFT_BASELINE_PATH / f"stage1-epoch-{epoch}"
        stage1_epoch_dir.mkdir(parents=True, exist_ok=True)
        student_model.save_pretrained(stage1_epoch_dir)
        tokenizer.save_pretrained(stage1_epoch_dir)

    write_log("Stage 1 完成")

    # ══════════════════════════════════════════════════════════════
    # Stage 2: 推理能力拓展（专利 S4）
    # 解冻 Layer 12-31，保持 Layer 0-11 冻结
    # 数据：Level-1 + Level-2 混合；余弦温度退火 T: 4.0→1.0；LR=5e-5；Epoch×5
    # ══════════════════════════════════════════════════════════════
    write_log("=== Stage 2: 推理能力拓展训练 (Layer 12-31) ===")
    unfreeze_layers(student_model, cfg.STAGE2_OPEN_LAYERS)
    freeze_layers(student_model, cfg.STAGE2_FROZEN_LAYERS)   # 确保 0-11 仍冻结

    stage2_loader = make_loader(mixed_samples)
    stage2_total_steps = len(stage2_loader) * cfg.STAGE2_EPOCHS
    optimizer_s2 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, student_model.parameters()),
        lr=cfg.STAGE2_LR,
        weight_decay=0.01,
    )
    # 余弦学习率衰减（与温度退火协同）
    scheduler_s2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_s2, T_max=stage2_total_steps)
    trainer.optimizer = optimizer_s2
    trainer.total_steps = stage2_total_steps
    trainer.current_step = 0   # 重置步数用于退火计算

    for epoch in range(1, cfg.STAGE2_EPOCHS + 1):
        train_loss = trainer.train_epoch(
            stage2_loader, epoch,
            use_annealing=True,                     # 开启余弦温度退火
            kl_weight=cfg.STAGE2_LOSS_KL_W,
        )
        scheduler_s2.step()
        val_loss = trainer.evaluate_epoch(val_loader)
        write_log(f"[Stage2] epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        stage2_epoch_dir = cfg.SFT_BASELINE_PATH / f"stage2-epoch-{epoch}"
        stage2_epoch_dir.mkdir(parents=True, exist_ok=True)
        student_model.save_pretrained(stage2_epoch_dir)
        tokenizer.save_pretrained(stage2_epoch_dir)
        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            cfg.BEST_MODEL_PATH.mkdir(parents=True, exist_ok=True)
            student_model.save_pretrained(cfg.BEST_MODEL_PATH)
            tokenizer.save_pretrained(cfg.BEST_MODEL_PATH)
            write_log(f"[Stage2] 保存最优检查点至 {cfg.BEST_MODEL_PATH}")

    write_log("Stage 2 完成")

    # Stage 3 is not used in the manuscript experiments; save the Stage-2 final LoRA adapter
    student_model.save_pretrained(cfg.SFT_BASELINE_PATH)
    tokenizer.save_pretrained(cfg.SFT_BASELINE_PATH)
    write_log(f"Phase3 (Stage1+2) 权重已存盘至 {cfg.SFT_BASELINE_PATH}")

    del student_model, teacher_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 3: SFT Training")
    parser.add_argument("--group", type=str, default="tcm_mini_final",
                        help="Training group name")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to group-specific config.py")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Max training steps per epoch (for smoke test)")
    args = parser.parse_args()

    # Load group config if specified
    if args.config:
        import importlib.util
        cfg_path = Path(args.config)
        if cfg_path.exists():
            spec = importlib.util.spec_from_file_location("group_cfg", str(cfg_path))
            group_cfg = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(group_cfg)
            # Override cfg attributes with group config
            _OVERRIDE_ATTRS = [
                "MODEL_DIR", "LOG_DIR", "CHECKPOINT_DIR", "BEST_MODEL_PATH",
                "SFT_BASELINE_PATH", "PATENT_MODEL_PATH", "CHECKPOINT_PREFIX",
                "OUTPUT_ROOT", "GROUP_NAME",
                "USE_RULE_LOSS", "USE_KD", "USE_DPO",
                "LOSS_LAMBDA_SFT", "LOSS_LAMBDA_KL", "LOSS_LAMBDA_RULE",
                "LOSS_LAMBDA_SMOOTH",
                "INPUT_VERSION", "DATA_DIR", "BASE_DIR",
                "TRAIN_FILE", "DEV_FILE", "TEST_FILE",
                "RULE_DICT_FILE", "SYNDROME_CONSTRAINTS_FILE",
                "LORA_R", "LORA_ALPHA", "LORA_DROPOUT",
                "LEARNING_RATE", "TRAIN_BATCH_SIZE", "GRAD_ACCUM_STEPS",
                "MAX_SEQ_LENGTH",
                "STAGE1_LR", "STAGE1_EPOCHS", "STAGE1_LOSS_KL_W", "STAGE1_TEMPERATURE",
                "STAGE2_LR", "STAGE2_EPOCHS", "STAGE2_LOSS_KL_W",
                "STAGE2_LOSS_RULE_W", "STAGE2_LOSS_SMOOTH_W",
                "STAGE3_LORA_R", "STAGE3_LORA_ALPHA",
                "DISTILL_TEMP_MAX", "DISTILL_TEMP_MIN", "DISTILL_TEMPERATURE",
                "DPO_BETA", "DPO_LEARNING_RATE", "DPO_NUM_EPOCHS",
            ]
            for attr in _OVERRIDE_ATTRS:
                if hasattr(group_cfg, attr):
                    setattr(cfg, attr, getattr(group_cfg, attr))

            # Re-create output directories after path overrides
            for _d in (cfg.MODEL_DIR, cfg.LOG_DIR, cfg.CHECKPOINT_DIR):
                _d.mkdir(parents=True, exist_ok=True)

            # 重新计算派生路径（否则仍指向旧 MODEL_DIR）
            cfg.SFT_BASELINE_PATH = cfg.MODEL_DIR / "round-0"
            cfg.BEST_MODEL_PATH = cfg.MODEL_DIR / "pilot_sft_clean_v3_lora_adapter"
            cfg.PATENT_MODEL_PATH = cfg.BEST_MODEL_PATH
            cfg.CHECKPOINT_PREFIX = cfg.CHECKPOINT_DIR / "round"

            write_log(f"Loaded group config: {args.group} from {args.config}")
            write_log(f"MODEL_DIR={cfg.MODEL_DIR}")
            write_log(f"SFT_BASELINE_PATH={cfg.SFT_BASELINE_PATH}")
            write_log(f"BEST_MODEL_PATH={cfg.BEST_MODEL_PATH}")

            # Re-init rules so they load from the new DATA_DIR paths
            import json as _json
            def _load_json(path):
                if path.exists():
                    with open(path, "r", encoding="utf-8") as f:
                        return _json.load(f)
                return {}
            cfg.RULE_DICT = {}
            for sx, info in _load_json(cfg.RULE_DICT_FILE).items():
                cfg.RULE_DICT[sx] = info.get("formulas", info.get("方剂", []))
            cfg.SYNDROME_PRESCRIPTION_CONSTRAINTS = _load_json(cfg.SYNDROME_CONSTRAINTS_FILE)
            write_log(f"Reloaded RULE_DICT ({len(cfg.RULE_DICT)} keys) and SYNDROME_PRESCRIPTION_CONSTRAINTS from new DATA_DIR")
        else:
            write_log(f"[WARN] Config not found: {args.config}, using default config")

    write_log(f"Group: {getattr(cfg, 'GROUP_NAME', args.group)}")
    write_log(f"USE_RULE_LOSS={getattr(cfg, 'USE_RULE_LOSS', True)}, "
              f"USE_KD={getattr(cfg, 'USE_KD', True)}, "
              f"USE_DPO={getattr(cfg, 'USE_DPO', True)}")

    # Smoke test: 限制每 epoch 最多 N 步
    if args.max_steps:
        cfg.SMOKE_MAX_STEPS = args.max_steps
        write_log(f"Smoke mode: max_steps={args.max_steps}")

    start_time = time.time()
    train_phase3()
    write_log(f"phase3 finished in {time.time() - start_time:.1f}s")
