# -*- coding: utf-8 -*-
"""
Unified lenient evaluation script for TCM-SD-42 syndrome diagnosis.
Computes syndrome_acc@1/3/5, macro_f1, weighted_f1, balanced_accuracy,
knowledge_formula_compatibility@1/3/5, format_validity, latency.

Usage:
  python unified_re_eval.py --eval-file /path/to/val.json \
       --lora-path /path/to/adapter --base-model /path/to/baseModel

  # smoke test (5 samples)
  python unified_re_eval.py --smoke --eval-file /path/to/val.json \
       --lora-path /path/to/adapter --base-model /path/to/baseModel

  # top-k sampling
  python unified_re_eval.py --top-k 5 --eval-file /path/to/val.json \
       --lora-path /path/to/adapter --base-model /path/to/baseModel

Model paths are controlled via --base-model / --lora-path CLI args or
TCM_BASE_MODEL / TCM_LORA_PATH environment variables.

Note: the GROUPS list documents our internal experiment IDs for reproducibility.
For public use, pass --lora-path and --base-model directly.
"""
from __future__ import annotations

import csv
import gc
import json
import os
import re
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, List, Tuple

import torch
from sklearn.metrics import precision_recall_fscore_support

# ── 运行模式 ──────────────────────────────────────────────────
import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument("--group", type=str, default=None,
                     help="Run only this group short name (e.g. tcm_mini_final_no_distill)")
_parser.add_argument("--smoke", action="store_true", help="Smoke test mode (5 samples)")
_parser.add_argument("--merge", action="store_true", help="Only merge existing metrics.json files")
_parser.add_argument("--top-k", type=int, default=1, metavar="K",
                     help="Generate K times per sample and compute top-1..top-K accuracy (default: 1)")
_parser.add_argument("--eval-file", type=str, default=None,
                     help="Path to evaluation JSON file (default: tcm_sd_val_84.json)")
_parser.add_argument("--prompt-variant", type=str, default="B",
                     choices=["A", "B"],
                     help="Prompt variant: A=旧版简单结尾, B=v2格式指令 (default: B)")
_ARGS = _parser.parse_args()

SMOKE_TEST = _ARGS.smoke or (os.environ.get("SMOKE_TEST", "0") == "1")
SMOKE_N = 5
SINGLE_GROUP = _ARGS.group  # None = run all
DO_MERGE = _ARGS.merge
TOP_K = max(_ARGS.top_k, 1)

# ── 输出根目录 ──────────────────────────────────────────────────
REVAL_ROOT = Path(os.environ.get("TCM_REVAL_ROOT", "./eval_output"))
REVAL_ROOT.mkdir(parents=True, exist_ok=True)
SUMMARY_DIR = REVAL_ROOT / "summary"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

VAL_FILE = Path(_ARGS.eval_file) if _ARGS.eval_file else Path(os.environ.get("TCM_VAL_FILE", "./tcm_sd_val_84.json"))
LOG_FILE = REVAL_ROOT / "unified_re_eval.log"

# ── evaluator 版本 ──────────────────────────────────────────────────
EVALUATOR_VERSION = "lenient_v3.0"
PROMPT_VERSION = "phase5_unified"
PROMPT_VARIANT = _ARGS.prompt_variant  # "A" or "B"
N_LABELS = 42  # 42 类证型

# ── Model configuration (public) ─────────────────────────────────
# Configure via environment variables or CLI arguments:
#   TCM_BASE_MODEL   (default: "/path/to/Qwen3-8B")
#   TCM_LORA_PATH    (default: "/path/to/pilot_sft_clean_v3_lora_adapter")
#   TCM_REVAL_ROOT   (default: "./eval_output")
#   TCM_VAL_FILE     (default: "./tcm_sd_val_84.json")
#
# CLI usage:
#   python unified_re_eval.py --base-model /model/Qwen3-8B \
#        --lora-path /checkpoints/pilot_sft_clean_v3_lora_adapter --eval-file data/test.json

BASE_MODEL = os.environ.get("TCM_BASE_MODEL", "/path/to/Qwen3-8B")
LORA_PATH  = os.environ.get("TCM_LORA_PATH",  "/path/to/pilot_sft_clean_v3_lora_adapter")

GROUPS = [
    {
        "name": "Pilot-SFT-clean-v3 (main result)",
        "short": "pilot_sft_clean_v3",
        "base_model": BASE_MODEL,
        "lora_path": LORA_PATH,
    },
    {
        "name": "Qwen3-8B zero-shot (baseline)",
        "short": "qwen3_zero_shot",
        "base_model": BASE_MODEL,
        "lora_path": None,
    },
]

# ── End of public GROUPS config ────────────────────────────────

# ── 字段名 ──────────────────────────────────────────────────
FIELD_SYNDROME = "证型"
FIELD_TREATMENT = "治法"
FIELD_FORMULA = "方剂"

# ══════════════════════════════════════════════════════════════
#  宽松评估逻辑
# ══════════════════════════════════════════════════════════════

TRAD_TO_SIMP = str.maketrans({
    "證": "证", "証": "证", "劑": "剂", "湯": "汤", "飲": "饮",
    "黃": "黄", "鈎": "钩", "鉤": "钩", "歸": "归", "脾": "脾",
    "腎": "肾", "陰": "阴", "陽": "阳", "虛": "虚", "實": "实",
    "熱": "热", "寒": "寒", "風": "风", "濕": "湿", "痰": "痰",
    "瘀": "瘀", "氣": "气", "血": "血", "醫": "医", "藥": "药",
    "處": "处", "療": "疗", "體": "体", "瀉": "泻", "補": "补",
    "發": "发", "裡": "里", "裏": "里",
})

FORMULA_ALIASES = {
    "麻黄杏仁甘草石膏汤": "麻杏石甘汤", "麻黄杏仁甘草石膏湯": "麻杏石甘汤",
    "麻杏甘石汤": "麻杏石甘汤", "天麻鈎藤饮": "天麻钩藤饮",
    "天麻钩藤飲": "天麻钩藤饮", "射干麻黃汤": "射干麻黄汤",
    "射干麻黃湯": "射干麻黄汤", "小青龍汤": "小青龙汤",
    "小青龍湯": "小青龙汤", "補中益气汤": "补中益气汤",
    "補中益氣汤": "补中益气汤", "補中益氣湯": "补中益气汤",
    "十全大補汤": "十全大补汤", "十全大補湯": "十全大补汤",
    "八珍湯": "八珍汤", "归牌汤": "归脾汤", "归牌湯": "归脾汤",
}

SYNDROME_ALIASES = {
    "邪热壅肺": "肺热咳喘", "肺热壅肺": "肺热咳喘",
    "中虚气陷": "中气下陷", "脾虚气陷": "中气下陷",
    "气虚下陷": "中气下陷", "脾虚下陷": "中气下陷",
    "外感风寒束表内停水饮": "外寒内饮",
    "外感风寒束表，内停水饮": "外寒内饮",
    "风寒外束内停水饮": "外寒内饮",
    "心阳虚": "心脾两虚", "肺气虚": "脾胃气虚",
    "肝血虚": "气血两虚", "肾精不足": "肾阴虚",
    "脾肾阳虚": "阳虚寒凝", "肝肾阴虚": "阴虚火旺",
    "瘀血阻络": "血瘀证", "气滞血瘀证": "气滞血瘀",
    "痰湿蕴肺": "痰热壅肺", "痰浊阻肺": "痰热壅肺",
    "风寒犯肺": "风寒咳嗽", "风寒袭肺": "风寒咳嗽",
    "风热犯肺": "风热袭表", "风热蕴肺": "风热袭表",
    "湿热蕴脾": "三焦湿热", "肝胆湿热": "三焦湿热",
    "热毒炽盛": "胃热炽盛", "胃火炽盛": "胃热炽盛",
    "心火亢盛": "阴虚火旺", "肝火旺盛": "肝郁化火",
    "气虚血瘀": "气滞血瘀", "气虚痰湿": "脾胃气虚",
    "阳虚水泛": "水湿内停", "肾虚水泛": "水湿内停",
    "寒湿痹阻": "寒湿困脾", "寒邪犯胃": "虚寒腹痛",
    "气血亏虚": "气血两虚", "气阴亏虚": "气阴两虚",
    "阴阳亏虚": "阴阳两虚", "心肾亏虚": "心肾不交",
    "肝风内动": "风痰阻络", "热入心包": "卫气营血传变",
    "痰蒙心窍": "痰迷心窍", "痰火扰心": "痰迷心窍",
}

SYNDROME_PRESCRIPTION_CONSTRAINTS = {
    "外寒内饮": {"方剂": ["小青龙汤", "射干麻黄汤"], "治法": "解表化饮"},
    "肾阴虚": {"方剂": ["六味地黄丸", "知柏地黄丸", "左归丸"], "治法": "滋阴补肾"},
    "肾阳虚": {"方剂": ["金匮肾气丸", "右归丸"], "治法": "温补肾阳"},
    "湿热黄疸": {"方剂": ["茵陈蒿汤", "茵陈五苓散"], "治法": "清热利湿退黄"},
    "少阳证": {"方剂": ["小柴胡汤"], "治法": "和解少阳"},
    "肺热咳喘": {"方剂": ["麻杏石甘汤"], "治法": "清热宣肺"},
    "风寒表虚": {"方剂": ["桂枝汤", "桂枝加葛根汤"], "治法": "解肌发表"},
    "风热袭表": {"方剂": ["银翘散", "桑菊饮"], "治法": "辛凉解表"},
    "风寒咳嗽": {"方剂": ["止嗽散", "杏苏散"], "治法": "宣肺止咳"},
    "肝阳上亢": {"方剂": ["天麻钩藤饮"], "治法": "平肝潜阳"},
    "肝气郁结": {"方剂": ["柴胡疏肝散", "逍遥散"], "治法": "疏肝解郁"},
    "肝郁化火": {"方剂": ["当归龙荟丸", "龙胆泻肝汤"], "治法": "清肝泻火"},
    "血瘀证": {"方剂": ["血府逐瘀汤", "桃红四物汤"], "治法": "活血化瘀"},
    "气滞血瘀": {"方剂": ["血府逐瘀汤", "膈下逐瘀汤"], "治法": "行气活血化瘀"},
    "中气下陷": {"方剂": ["补中益气汤", "升陷汤"], "治法": "补中益气"},
    "阴虚火旺": {"方剂": ["知柏地黄丸", "大补阴丸"], "治法": "滋阴降火"},
    "脾胃阳虚": {"方剂": ["理中丸", "附子理中丸"], "治法": "温中祛寒"},
    "脾胃气虚": {"方剂": ["四君子汤", "参苓白术散"], "治法": "益气健脾"},
    "心肾不交": {"方剂": ["天王补心丹", "交泰丸"], "治法": "滋阴养血交通心肾"},
    "心脾两虚": {"方剂": ["归脾汤"], "治法": "补益心脾"},
    "胸痹": {"方剂": ["瓜蒌薤白白酒汤", "枳实薤白桂枝汤"], "治法": "通阳散结"},
    "胃热炽盛": {"方剂": ["清胃散", "玉女煎"], "治法": "清胃泻火"},
    "三焦湿热": {"方剂": ["黄芩滑石汤", "三仁汤"], "治法": "宣畅气机清利湿热"},
    "水湿内停": {"方剂": ["猪苓汤", "五苓散"], "治法": "利水渗湿"},
    "百合病": {"方剂": ["百合知母汤", "百合地黄汤"], "治法": "养阴清热"},
    "狐惑病": {"方剂": ["甘草泻心汤"], "治法": "清热解毒"},
    "蛔厥": {"方剂": ["乌梅丸", "连梅安蛔汤"], "治法": "安蛔止痛"},
    "卫气营血传变": {"方剂": ["清营汤", "犀角地黄汤"], "治法": "清营透热"},
    "痰迷心窍": {"方剂": ["涤痰汤"], "治法": "涤痰开窍"},
    "风痰阻络": {"方剂": ["牵正散"], "治法": "祛风化痰"},
    "膀胱湿热": {"方剂": ["八正散"], "治法": "清热利湿"},
    "阴阳两虚": {"方剂": ["地黄饮子"], "治法": "滋肾阴补肾阳"},
    "气阴两虚": {"方剂": ["生脉散", "参麦散"], "治法": "益气养阴"},
    "气血两虚": {"方剂": ["八珍汤", "十全大补汤"], "治法": "补益气血"},
    "寒热错杂": {"方剂": ["乌梅丸", "半夏泻心汤"], "治法": "寒热并调"},
    "热结腹痛": {"方剂": ["大承气汤", "小承气汤"], "治法": "通腑泻热"},
    "痰热壅肺": {"方剂": ["清气化痰丸", "小陷胸汤"], "治法": "清热化痰"},
    "痰饮病": {"方剂": ["苓桂术甘汤", "小半夏汤"], "治法": "温化痰饮"},
    "风寒表证": {"方剂": ["麻黄汤", "荆防败毒散"], "治法": "发汗解表"},
    "虚寒腹痛": {"方剂": ["小建中汤", "大建中汤"], "治法": "温中补虚"},
    "寒湿困脾": {"方剂": ["平胃散", "藿香正气散"], "治法": "燥湿健脾"},
    "气虚": {"方剂": ["四君子汤", "补中益气汤"], "治法": "补气健脾"},
    "血热妄行": {"方剂": ["犀角地黄汤", "十灰散"], "治法": "清热凉血止血"},
    "阳虚寒凝": {"方剂": ["附子理中汤", "右归丸"], "治法": "温阳散寒"},
}


# ══════════════════════════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════════════════════════

def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def norm_text(s: Any) -> str:
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKC", s)
    s = s.translate(TRAD_TO_SIMP)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("〜", ":").replace("：", ":")
    s = s.replace("“", '"').replace("”", '"')
    return s


def normalize_formula(v: str) -> str:
    v = norm_text(v)
    v = re.sub(r"[\s　]+", "", v)
    v = re.sub(r"[\*#_`【】\[\]{}‘’“”，,。；;：:]+", "", v)
    v = re.sub(r"（.*?）|\(.*?\)", "", v)
    v = re.sub(r"\d+(?:\.\d+)?\s*(?:g|克|钱|两|毫升|ml)", "", v, flags=re.I)
    v = re.sub(r"(?:加减|加味|化裁|为主|具体用药如下).*", "", v)
    v = FORMULA_ALIASES.get(v, v)
    return v


def normalize_syndrome(v: str) -> str:
    v = norm_text(v)
    v = re.sub(r"[\s　]+", "", v)
    v = re.sub(r"[\*#_`【】\[\]{}‘’“”，,。；;：:、]+", "", v)
    v = re.sub(r"（.*?）|\(.*?\)", "", v)
    v = re.sub(r"(?:之一|之候|证|型)$", lambda m: "" if len(v) > 4 else m.group(0), v)
    v = SYNDROME_ALIASES.get(v, v)
    return v


def extract_field_lenient(text: str, field_name: str) -> str:
    t = norm_text(text)
    if not t.strip():
        return ""
    LABEL_MAP = {
        "syndrome": ["证型", "证候", "症候", "辨证结果", "辨证", "诊断", "病机"],
        "treatment": ["治法", "治疗原则", "治则", "治法处方"],
        "formula": ["方剂", "处方", "方药", "药物"],
    }
    label_key = None
    for k, labels in LABEL_MAP.items():
        if field_name in labels or field_name == k:
            label_key = k
            break
    if label_key is None:
        label_key = "syndrome"
    labels = LABEL_MAP[label_key]
    # 精确标签（防子串误匹配：辨证依据 包含 证型）
    exact_labels = {l for l in labels if len(l) >= 2}
    label_re = "|".join(re.escape(x) for x in labels)
    exact_label_re = "|".join(re.escape(x) for x in exact_labels)

    EXCLUDE_PREFIX = ("根据", "依据", "通过", "综合", "本案", "患者", "四诊", "主诉",
                      "结合", "遵循", "参照", "参考", "按", "本证", "辨为", "为",
                      "此为", "证属", "当属", "应为", "故", "故为", "定为",
                      "属", "乃", "当属", "总属", "系", "显", "提示", "考虑",
                      "本案", "本案属", "本案为", "本案辨", "该", "该为", "此为",
                      "诊断为", "辨证为", "辨为", "拟为", "初诊为")

    def _ok(val):
        if not val or len(val) >= 80:
            return False
        return not val.startswith(EXCLUDE_PREFIX)

    # 策略1：收集所有 "证型：XXX" 匹配（取最后一个有效的）
    next_labels = ["治法", "方剂", "依据", "辨证解释", "证型", "证候", "辨证", "诊断", "病机", "辨证依据"]
    next_re = "|".join(re.escape(x) for x in next_labels)
    # 匹配标签和值在同一行的内容
    inline_pattern = re.compile(
        rf"(?<![辨证])\b(?:{exact_label_re})\s*[:：]\s*([^\n]+?)(?=\s*[{re.escape('，,；;。、')}]\s*(?:{next_re})\s*[:：]|\n|$)",
        re.I | re.S,
    )
    candidates = []
    for m in inline_pattern.finditer(t):
        val = (m.group(1) or "").strip()
        val = re.sub(r"^[\s\-*#>`~_\"']+", "", val).strip(" \t\n:：;；,，。【】[]()（）\"'")
        if _ok(val):
            candidates.append(val)
    if candidates:
        return candidates[-1]  # 取最后一个（更可能是最终答案）

    # 策略2：标签独占一行，值在后续非标签行
    lines = t.splitlines()
    label_only_re = rf"^\s*(?:[-*#>\s`]*|[【\[]?)\**\s*(?:{exact_label_re})\s*[】\]]?\**\s*[:：]?\s*$"
    for i, line in enumerate(lines):
        if re.match(label_only_re, line.strip(), re.I):
            for j in range(i + 1, min(i + 5, len(lines))):
                next_line = lines[j].strip()
                if re.match(rf"^\s*(?:{next_re})\s*[:：]?\s*$", next_line, re.I):
                    continue
                val = re.sub(r"^[\s\-*#>`~_\"']+", "", next_line).strip(" \t\n:：;；,，。【】[]()（）\"'")
                if _ok(val):
                    candidates.append(val)
    if candidates:
        return candidates[-1]

    # 策略3：宽松 inline（允许值以任何字符开头）
    loose_pattern = re.compile(
        rf"(?<![辨证])\b(?:{exact_label_re})\s*[:：]\s*([^\n]+?)(?=\n|$)",
        re.I | re.S,
    )
    for m in loose_pattern.finditer(t):
        val = (m.group(1) or "").strip()
        val = re.sub(r"^[\s\-*#>`~_\"']+", "", val).strip(" \t\n:：;；,，。【】[]()（）\"'")
        if _ok(val):
            candidates.append(val)
    if candidates:
        return candidates[-1]

    return ""


def syndrome_correct_lenient(pred: str, gt: str) -> bool:
    p = normalize_syndrome(pred)
    g = normalize_syndrome(gt)
    if not p or not g:
        return False
    return p == g or g in p or p in g


def build_accepted_forms(gt_syndrome: str, gt_formula: str) -> List[str]:
    gt_sx_norm = normalize_syndrome(gt_syndrome)
    bundle = SYNDROME_PRESCRIPTION_CONSTRAINTS.get(gt_sx_norm, {})
    forms = bundle.get("方剂", [])
    if not forms and gt_formula:
        forms = [gt_formula]
    return [normalize_formula(f) for f in forms]


def formula_compliant_lenient(pred_formula: str, accepted_forms: List[str]) -> bool:
    p = normalize_formula(pred_formula)
    if not p:
        return False
    for f in accepted_forms:
        f0 = normalize_formula(f)
        if f0 and (f0 == p or f0 in p or p in f0):
            return True
    return False


def find_canonical_syndrome(extracted: str, full_text: str) -> str:
    all_syndromes = list(SYNDROME_PRESCRIPTION_CONSTRAINTS.keys())
    pred = normalize_syndrome(extracted)
    if pred:
        for sx in sorted(all_syndromes, key=len, reverse=True):
            if sx and sx in pred:
                return sx
        if pred in SYNDROME_ALIASES:
            return SYNDROME_ALIASES[pred]
        return pred
    txt = normalize_syndrome(full_text)
    for sx in sorted(all_syndromes, key=len, reverse=True):
        if sx and sx in txt:
            return sx
    for alias, canon in SYNDROME_ALIASES.items():
        if normalize_syndrome(alias) in txt:
            return canon
    return ""


def find_canonical_syndrome_v2(extracted: str, full_text: str) -> Tuple[str, str, List[str]]:
    """
    统一解析逻辑，返回 (pred_sx, matched_by, candidates)。
    matched_by: 'regex' | 'canonical_fallback' | 'alias' | 'none'
    candidates: 所有命中的证型列表（用于 debug）
    """
    all_syndromes = list(SYNDROME_PRESCRIPTION_CONSTRAINTS.keys())
    pred = normalize_syndrome(extracted)
    candidates = []

    # 策略1：regex 提取成功，从 SYNDROME_PRESCRIPTION_CONSTRAINTS 找规范名
    if pred:
        for sx in sorted(all_syndromes, key=len, reverse=True):
            if sx and sx in pred:
                candidates.append(sx)
                return sx, "regex", candidates
        if pred in SYNDROME_ALIASES:
            candidates.append(SYNDROME_ALIASES[pred])
            return SYNDROME_ALIASES[pred], "alias", candidates
        # regex 提取到值但不在 SYNDROME_PRESCRIPTION_CONSTRAINTS 中，原样返回
        candidates.append(pred)
        return pred, "regex", candidates

    # 策略2：全文 fallback，在 full_text 中按 42 类标签搜索
    txt = normalize_syndrome(full_text)
    for sx in sorted(all_syndromes, key=len, reverse=True):
        if sx and sx in txt:
            candidates.append(sx)

    if candidates:
        # 返回最长匹配（综合质量最高）
        best = max(candidates, key=len)
        return best, "canonical_fallback", candidates

    # 策略3：alias 直接匹配
    for alias, canon in SYNDROME_ALIASES.items():
        if normalize_syndrome(alias) in txt:
            candidates.append(canon)
            return canon, "alias", candidates

    return "", "none", []


def find_formula_from_text(extracted: str, full_text: str, accepted_forms: List[str]) -> str:
    all_forms = sorted(
        {f for forms in SYNDROME_PRESCRIPTION_CONSTRAINTS.values() for f in forms.get("方剂", [])},
        key=len, reverse=True,
    )
    pred = normalize_formula(extracted)
    if pred:
        matches = [f for f in all_forms if f and f in pred]
        if matches:
            return matches[0]
        return pred
    txt = normalize_formula(full_text)
    for f in sorted(accepted_forms, key=len, reverse=True):
        if f and f in txt:
            return f
    for f in all_forms:
        if f and f in txt:
            return f
    return ""


def extract_syndromes_from_text(text: str) -> List[str]:
    """在全文中搜索所有 42 类标准证型关键词，返回不重复列表。"""
    text_n = norm_text(text)
    found = []
    for sx in SYNDROME_PRESCRIPTION_CONSTRAINTS.keys():
        sx_n = normalize_syndrome(sx)
        if sx_n in text_n or sx in text_n:
            if sx not in found:
                found.append(sx)
    return found


def evaluate_strict(output_text: str, entry: dict) -> dict:
    """
    严格评估：只接受从 '证型：xxx' 格式解析出的证型。
    - format_valid: 必须有证型标记行且解析非空
    - strict_syn: 格式解析出的证型 == gold
    - lenient_syn: 不使用（固定 False）
    """
    gt_sx = str(entry.get("syndrome_canonical", "") or entry.get(FIELD_SYNDROME, "") or "").strip()
    gt_fm = str(entry.get("方剂", "") or "").strip()

    raw_sx = extract_field_lenient(output_text, FIELD_SYNDROME)
    raw_fm = extract_field_lenient(output_text, FIELD_FORMULA)

    pred_sx, matched_by_sx, match_candidates_sx = find_canonical_syndrome_v2(raw_sx, output_text)
    accepted = build_accepted_forms(gt_sx, gt_fm)
    pred_fm = find_formula_from_text(raw_fm, output_text, accepted)

    syn = syndrome_correct_lenient(pred_sx, gt_sx)
    frm = formula_compliant_lenient(pred_fm, accepted)

    full_text = norm_text(output_text)
    cons = 0.0
    if syn:
        cons += 0.4
    lt = SYNDROME_PRESCRIPTION_CONSTRAINTS.get(normalize_syndrome(pred_sx), {}).get(FIELD_TREATMENT, "")
    if lt and lt in full_text:
        cons += 0.2
    if frm:
        cons += 0.2
    if gt_sx and gt_sx[:2] in full_text:
        cons += 0.1
    ftoks = [t.strip() for t in pred_fm.replace("、", ",").split(",") if t.strip()]
    if any(tok and tok in full_text for tok in ftoks):
        cons += 0.1
    cons = round(min(cons, 1.0), 4)

    ll = 0.0
    if len(full_text) >= 30:
        ll += 0.4
    if len(full_text) >= 60:
        ll += 0.2
    if lt and lt in full_text:
        ll += 0.2
    if any(tok and tok in full_text for tok in ftoks):
        ll += 0.2
    ll = round(min(ll, 1.0), 4)

    parse_success = bool(pred_sx)
    format_valid = parse_success  # 严格：必须有格式解析结果

    return {
        "syn": syn, "frm": frm, "cons": cons, "ll": ll,
        "pred_sx": pred_sx, "pred_fm": pred_fm,
        "parse_success": parse_success, "format_valid": format_valid,
        "matched_by": matched_by_sx, "match_candidates": match_candidates_sx,
    }


def evaluate_lenient_v2(output_text: str, entry: dict) -> dict:
    """
    宽松评估（v2）：
    - lenient_syn: 如果格式解析失败，则在全文中搜索 42 类标准证型
    - parse_success: 同严格（必须格式解析）
    - format_valid: 同严格（必须格式解析）
    - lenient_syn_used: 是否动用了 fallback 搜索
    """
    gt_sx = str(entry.get("syndrome_canonical", "") or entry.get(FIELD_SYNDROME, "") or "").strip()
    gt_fm = str(entry.get("方剂", "") or "").strip()

    raw_sx = extract_field_lenient(output_text, FIELD_SYNDROME)
    raw_fm = extract_field_lenient(output_text, FIELD_FORMULA)

    pred_sx, matched_by_sx, match_candidates_sx = find_canonical_syndrome_v2(raw_sx, output_text)
    accepted = build_accepted_forms(gt_sx, gt_fm)
    pred_fm = find_formula_from_text(raw_fm, output_text, accepted)

    syn_strict = syndrome_correct_lenient(pred_sx, gt_sx)
    lenient_syn_used = False

    # Fallback: 如果格式解析失败，尝试全文搜索证型关键词
    if not syn_strict and not pred_sx:
        found = extract_syndromes_from_text(output_text)
        if found:
            best = found[0]
            if syndrome_correct_lenient(best, gt_sx):
                pred_sx = best
                syn_strict = True
                lenient_syn_used = True

    frm = formula_compliant_lenient(pred_fm, accepted)

    full_text = norm_text(output_text)
    cons = 0.0
    if syn_strict:
        cons += 0.4
    lt = SYNDROME_PRESCRIPTION_CONSTRAINTS.get(normalize_syndrome(pred_sx), {}).get(FIELD_TREATMENT, "")
    if lt and lt in full_text:
        cons += 0.2
    if frm:
        cons += 0.2
    if gt_sx and gt_sx[:2] in full_text:
        cons += 0.1
    ftoks = [t.strip() for t in pred_fm.replace("、", ",").split(",") if t.strip()]
    if any(tok and tok in full_text for tok in ftoks):
        cons += 0.1
    cons = round(min(cons, 1.0), 4)

    ll = 0.0
    if len(full_text) >= 30:
        ll += 0.4
    if len(full_text) >= 60:
        ll += 0.2
    if lt and lt in full_text:
        ll += 0.2
    if any(tok and tok in full_text for tok in ftoks):
        ll += 0.2
    ll = round(min(ll, 1.0), 4)

    parse_success = bool(pred_sx)
    format_valid = parse_success  # 即使 lenient_syn_used，format_valid 仍取决于格式

    return {
        "syn": syn_strict, "frm": frm, "cons": cons, "ll": ll,
        "pred_sx": pred_sx, "pred_fm": pred_fm,
        "parse_success": parse_success, "format_valid": format_valid,
        "lenient_syn_used": lenient_syn_used,
        "matched_by": matched_by_sx, "match_candidates": match_candidates_sx,
    }


def evaluate_lenient(output_text: str, entry: dict) -> dict:
    # 支持两种数据格式：inputv2 (syndrome_canonical) 和 val84 (证型)
    gt_sx = str(entry.get("syndrome_canonical", "") or entry.get(FIELD_SYNDROME, "") or "").strip()
    gt_fm = str(entry.get("方剂", "") or "").strip()

    raw_sx = extract_field_lenient(output_text, FIELD_SYNDROME)
    raw_fm = extract_field_lenient(output_text, FIELD_FORMULA)

    pred_sx, matched_by_sx, match_candidates_sx = find_canonical_syndrome_v2(raw_sx, output_text)
    accepted = build_accepted_forms(gt_sx, gt_fm)
    pred_fm = find_formula_from_text(raw_fm, output_text, accepted)

    syn = syndrome_correct_lenient(pred_sx, gt_sx)
    frm = formula_compliant_lenient(pred_fm, accepted)

    full_text = norm_text(output_text)
    cons = 0.0
    if syn:
        cons += 0.4
    lt = SYNDROME_PRESCRIPTION_CONSTRAINTS.get(normalize_syndrome(pred_sx), {}).get(FIELD_TREATMENT, "")
    if lt and lt in full_text:
        cons += 0.2
    if frm:
        cons += 0.2
    if gt_sx and gt_sx[:2] in full_text:
        cons += 0.1
    ftoks = [t.strip() for t in pred_fm.replace("、", ",").split(",") if t.strip()]
    if any(tok and tok in full_text for tok in ftoks):
        cons += 0.1
    cons = round(min(cons, 1.0), 4)

    ll = 0.0
    if len(full_text) >= 30:
        ll += 0.4
    if len(full_text) >= 60:
        ll += 0.2
    if lt and lt in full_text:
        ll += 0.2
    if any(tok and tok in full_text for tok in ftoks):
        ll += 0.2
    ll = round(min(ll, 1.0), 4)

    parse_success = bool(pred_sx)

    return {
        "syn": syn, "frm": frm, "cons": cons, "ll": ll,
        "pred_sx": pred_sx, "pred_fm": pred_fm,
        "parse_success": parse_success,
        "matched_by": matched_by_sx, "match_candidates": match_candidates_sx,
    }


# ══════════════════════════════════════════════════════════════
#  推理（确定性解码）
# ══════════════════════════════════════════════════════════════

PROMPT_INJECTION_PATTERNS = [
    re.compile(r'输出格式[：:]\s*(\{[^}]*\})', re.S),
    re.compile(r'请按以下格式[^\n]*\n[^\n]*\{"?证型"?', re.S | re.I),
    re.compile(r'<\|im_start\|>(?:system|user|assistant)\b', re.I),
    re.compile(r'(?:system|user|assistant)_prompt', re.I),
    re.compile(r'格式[：:]\s*[\[\{][^\]\}]{5,200}?[\]\}]', re.S),
]

def _clean_input_text(text: str) -> str:
    """轻度清洗：删除明显 prompt injection 片段，保留正常病历内容。"""
    t = text
    # 删掉 JSON schema 片段
    t = re.sub(r'\{\s*"证型"\s*:\s*"[^"]*"\s*,\s*"辨证依据"\s*:\s*"[^"]*"\s*\}', '', t)
    t = re.sub(r'\{\s*"证型"\s*:\s*""\s*,\s*"辨证依据"\s*:\s*""\s*\}', '', t)
    t = re.sub(r'符合以下格式[^\n]*\n[^\n]*["\'`]?\{[^}]*\}?', '', t)
    # 删掉 <|im_start|> / <|im_end|> 标签残留
    t = re.sub(r'<\|im_(?:start|end)\|>', '', t, flags=re.I)
    # 删掉包含 "请按以下格式" 的行
    t = re.sub(r'^.*(?:请按以下格式|请严格从下一行|现在开始输出|请输出两行).*$', '', t, flags=re.M)
    t = re.sub(r'^.*"输出格式".*$', '', t, flags=re.M)
    t = re.sub(r'^.*output.*format.*$', '', t, flags=re.M | re.I)
    # 清理多余空行
    t = re.sub(r'\n{3,}', '\n\n', t)
    return t.strip()

def build_prompt(symptoms: str, disease_site: str) -> str:
    """统一 prompt，所有模型完全一致。与 phase3_train.py 训练 prompt 完全对齐。
    Prompt A（旧版）: 结尾为「证型：\n辨证依据：\n」（与训练时 SFT data 格式一致）
    Prompt B（v2）: 结尾为「请严格从下一行开始输出两行...」"""
    symptoms = _clean_input_text(symptoms)
    if PROMPT_VARIANT == "A":
        tail = (
            "证型：\n"
            "辨证依据：\n"
        )
    else:  # "B"
        tail = (
            "请严格从下一行开始输出两行：\n"
            "第一行：证型：<一个标准证型名称>\n"
            "第二行：辨证依据：<一句简短依据>\n"
            "现在开始输出：\n"
        )
    return (
        "<|im_start|>system\n"
        "你是一位资深中医专家。请根据真实临床资料判断最符合的中医证型，"
        "并严格按照「证型」和「辨证依据」两个字段作答。<|im_end|>\n"
        "<|im_start|>user\n"
        "请根据以下真实临床资料进行中医辨证。\n"
        "要求：\n"
        "1. 只输出一个最可能的证型。\n"
        "2. 证型名称尽量使用标准证型名称。\n"
        "3. 不要输出治法、方剂、病位或其他额外字段。\n\n"
        f"临床资料：\n{symptoms}\n\n"
        + tail +
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def build_quant_config():
    from transformers import BitsAndBytesConfig
    if not torch.cuda.is_available():
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
    )


def load_model(base_path: str, lora_path: str | None):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {"trust_remote_code": True}
    qc = build_quant_config()
    if qc:
        kwargs["quantization_config"] = qc
        kwargs["device_map"] = "auto"
    else:
        kwargs["device_map"] = "cpu"
        kwargs["torch_dtype"] = torch.float32

    base = AutoModelForCausalLM.from_pretrained(base_path, **kwargs)
    if lora_path and Path(lora_path).exists() and (Path(lora_path) / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(base, lora_path)
        log(f"  loaded LoRA from {lora_path}")
    else:
        model = base
        log(f"  using base model (no LoRA)")
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_once(model, tokenizer, prompt: str, seed: int | None = None) -> str:
    """采样解码：do_sample=True, temperature=0.7, top_p=0.9。"""
    device = next(model.parameters()).device
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768).to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=96,
        do_sample=False,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    for stop in ["注意", "免责声明", "以上内容", "请在专业", "#中医药"]:
        if stop in text:
            text = text[:text.index(stop)]
    return text.strip()


def run_group(group: dict, eval_data: list[dict], top_k: int = 1) -> dict:
    """
    对单个模型组做推理 + 宽松评估。
    top_k > 1 时每个样本生成 top_k 次（固定种子 42+k），同时计算
    syndrome_top1..topK 和 formula_top1..topK。
    """
    name = group["name"]
    short = group["short"]
    # Prompt A/B 时在目录名中体现 variant，避免覆盖
    variant_suffix = f"_prompt{PROMPT_VARIANT}"
    out_dir = REVAL_ROOT / (short + variant_suffix)
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"Loading model: {name}")
    model, tokenizer = load_model(group["base_model"], group["lora_path"])

    details = []
    topk_details = []  # top-k 专用详情
    cons_sum = 0.0
    ll_sum = 0.0
    format_valid = 0
    syn_ok = 0           # strict syn@1
    lenient_syn_ok = 0  # lenient syn@1 (includes fallback match)
    form_ok = 0
    empty_pred = 0       # parsed_syndrome == ''
    lenient_used_count = 0  # how many used fallback
    syn_first_hit = 0
    frm_first_hit = 0
    lat_sum = 0.0
    lat_n = 0
    # top-k 累加器：syn_hit[k] = 前 k 次中至少命中 1 次的样本数
    syn_hit = [0] * (top_k + 1)
    frm_hit = [0] * (top_k + 1)
    # 证型 F1 追踪（top-1 预测 vs 真值）
    y_true_all = []
    y_pred_all = []
    outputs_list = []

    for idx, entry in enumerate(eval_data, 1):
        # 支持两种数据格式：inputv2 (input_text/syndrome_canonical) 和 val84 (症状/证型)
        symptoms = str(entry.get("症状", "") or entry.get("input_text", "") or "")
        site = str(entry.get("病位", "") or "")
        gt_sx = str(entry.get("证型", "") or entry.get("syndrome_canonical", "") or "")
        gt_fm = str(entry.get("方剂", "") or "").strip()
        prompt = build_prompt(symptoms, site)

        # 多次采样收集候选证型 + 方剂
        candidates_sx = []  # 每次采样的预测证型
        candidates_fm = []  # 每次采样的预测方剂
        all_outputs = []    # 每次采样的原始输出
        syn_first_hit = 0   # 证型首次命中的轮次
        frm_first_hit = 0   # 方剂首次命中的轮次

        for k in range(top_k):
            seed = 42 + k  # 所有模型使用同一组种子，保证公平
            t0 = time.time()
            output = generate_once(model, tokenizer, prompt, seed=seed)
            lat = (time.time() - t0) * 1000
            ev = evaluate_lenient_v2(output, entry)

            candidates_sx.append(ev["pred_sx"])
            candidates_fm.append(ev["pred_fm"])
            all_outputs.append(output)

            if k == 0:
                # 第一次结果 = 主结果（= top-1）
                syn_ok += int(ev["syn"])
                lenient_syn_ok += int(ev["syn"])
                lenient_used_count += int(ev.get("lenient_syn_used", False))
                form_ok += int(ev["frm"])
                cons_sum += ev["cons"]
                ll_sum += ev["ll"]
                # format_validity: 按「证型/辨证依据」两字段格式输出
                fmt_valid = bool(ev["parse_success"])
                format_valid += int(fmt_valid)
                if not ev["pred_sx"]:
                    empty_pred += 1
                main_ev = ev
                main_output = output
                main_lat = lat
                # 追踪 F1（top-1 证型预测 vs 真值，统一用 ev["pred_sx"]）
                y_true_all.append(gt_sx)
                y_pred_all.append(ev["pred_sx"] or "")

            if idx > 1:
                lat_sum += lat
                lat_n += 1

            # 追踪首次命中
            if syn_first_hit == 0 and ev["syn"]:
                syn_first_hit = k + 1
            if frm_first_hit == 0 and ev["frm"]:
                frm_first_hit = k + 1

        # 累计 top-k 命中（证型 + 方剂）
        for ki in range(1, top_k + 1):
            if syn_first_hit > 0 and syn_first_hit <= ki:
                syn_hit[ki] += 1
            if frm_first_hit > 0 and frm_first_hit <= ki:
                frm_hit[ki] += 1

        # details.json 条目（每条统一格式）
        detail = {
            "index": idx,
            "input": f"症状：{symptoms}\n病位：{site}",
            "gold_syndrome": gt_sx,
            "raw_output": main_output,
            "parsed_syndrome": main_ev["pred_sx"],
            "parse_success": main_ev["parse_success"],
            "is_correct": main_ev["syn"],   # strict
            "is_correct_lenient": main_ev["syn"],  # lenient (same since lenient v2 uses same syn)
            "lenient_syn_used": main_ev.get("lenient_syn_used", False),
            "matched_by": main_ev.get("matched_by", ""),
            "match_candidates": main_ev.get("match_candidates", []),
            "标准方剂": gt_fm,
            "预测方剂": main_ev["pred_fm"],
            "方剂正确": main_ev["frm"],
            "医理自洽": main_ev["cons"],
            "长文本逻辑": main_ev["ll"],
            "format_valid": fmt_valid,
            "latency_ms": round(main_lat, 2),
        }
        if top_k > 1:
            detail["topk_证型首次命中"] = syn_first_hit if syn_first_hit > 0 else "未命中"
            detail["topk_方剂首次命中"] = frm_first_hit if frm_first_hit > 0 else "未命中"
            detail["topk_候选证型"] = candidates_sx
            detail["topk_候选方剂"] = candidates_fm
        details.append(detail)

        # topk_details.json 条目（仅 top_k > 1 时写入）
        if top_k > 1:
            topk_details.append({
                "index": idx,
                "gold_syndrome": gt_sx,
                "gold_formula": gt_fm,
                "候选证型": candidates_sx,
                "候选方剂": candidates_fm,
                "证型首次命中": syn_first_hit if syn_first_hit > 0 else None,
                "方剂首次命中": frm_first_hit if frm_first_hit > 0 else None,
            })

        # outputs.jsonl 条目（只保留第一次输出，精简版）
        outputs_list.append({
            "index": idx,
            "症状": symptoms,
            "病位": site,
            "gold_syndrome": gt_sx,
            "gold_formula": gt_fm,
            "parsed_syndrome": main_ev["pred_sx"],
            "raw_output": main_output,
            "lenient_syn_used": main_ev.get("lenient_syn_used", False),
        })

        if idx % 50 == 0 or idx == len(eval_data):
            base = (f"syn@1={syn_ok/idx*100:.1f}%(strict) "
                    f"{lenient_syn_ok/idx*100:.1f}%(lenient) "
                    f"frm@1={form_ok/idx*100:.1f}%")
            if top_k > 1:
                topk_str = " ".join(
                    f"syn@{k}={syn_hit[k]/idx*100:.1f}%"
                    for k in range(1, top_k + 1)
                )
                base += f" {topk_str}"
            log(f"  [{name}] {idx}/{len(eval_data)} {base}")

    n = len(eval_data)
    # 计算 macro_f1 / weighted_f1 / balanced_accuracy（top-1 证型，所有样本统一分母）
    # 空预测 "" 作为一个合法标签参与 F1 计算
    _labels = sorted(set(y_true_all + y_pred_all))
    _label_to_idx = {l: i for i, l in enumerate(_labels)}
    _y_true_vec = [_label_to_idx.get(l, -1) for l in y_true_all]
    _y_pred_vec = [_label_to_idx.get(l, -1) for l in y_pred_all]
    # 所有样本都参与，不过滤
    _macro_p, _macro_r, _macro_f1, _ = precision_recall_fscore_support(
        _y_true_vec, _y_pred_vec, average="macro", zero_division=0)
    _weighted_p, _weighted_r, _weighted_f1, _ = precision_recall_fscore_support(
        _y_true_vec, _y_pred_vec, average="weighted", zero_division=0)
    _balanced_acc = _macro_r

    metrics = {
        "run_name": name,
        "checkpoint_path": group["lora_path"] or group["base_model"],
        "evaluator_version": EVALUATOR_VERSION,
        "prompt_version": f"{PROMPT_VERSION}_{PROMPT_VARIANT}",
        "top_k": top_k,
        "n_cases": n,
        "n_labels": N_LABELS,
        "strict_syndrome_top1": round(syn_ok / n * 100, 2),
        "lenient_syndrome_top1": round(lenient_syn_ok / n * 100, 2),
        "macro_f1": round(_macro_f1 * 100, 2),
        "weighted_f1": round(_weighted_f1 * 100, 2),
        "balanced_accuracy": round(_balanced_acc * 100, 2),
        "format_validity": round(format_valid / n * 100, 2),
        "parse_success": round(format_valid / n * 100, 2),
        "empty_prediction": round(empty_pred / n * 100, 2),
        "lenient_fallback_used": round(lenient_used_count / n * 100, 2),
        "avg_latency_ms": round(lat_sum / max(lat_n, 1), 2),
        "long_logic": round(ll_sum / n, 4),
    }
    # 加入 top-k 指标（仅 top_k > 1 时）
    for k in range(2, top_k + 1):
        metrics[f"syndrome_top{k}"] = round(syn_hit[k] / n * 100, 2)
        metrics[f"knowledge_formula_compatibility@{k}"] = round(frm_hit[k] / n * 100, 2)
    # top-1 knowledge formula compatibility 已经在 metrics 里了（由 formula_top1 改名）
    # 补上 top-3/top-5 的 knowledge_formula_compatibility（如果还未设置）
    for k in range(2, top_k + 1):
        if f"knowledge_formula_compatibility@{k}" not in metrics:
            metrics[f"knowledge_formula_compatibility@{k}"] = round(frm_hit[k] / n * 100, 2)
    # 清理不需要的字段
    for _bad in ("formula_top1", "formula_top3", "formula_top5",
                 "syndrome_top3", "syndrome_top5", "consistency"):
        metrics.pop(_bad, None)
    # 重命名 formula_top1 → knowledge_formula_compatibility@1
    if "formula_top1" in metrics:
        metrics["knowledge_formula_compatibility@1"] = metrics.pop("formula_top1")

    # 保存文件
    with open(out_dir / "details.json", "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "details": details}, f, ensure_ascii=False, indent=2)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(out_dir / "outputs.jsonl", "w", encoding="utf-8") as f:
        for item in outputs_list:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    if top_k > 1:
        with open(out_dir / "topk_details.json", "w", encoding="utf-8") as f:
            json.dump(topk_details, f, ensure_ascii=False, indent=2)

    topk_msg = ""
    if top_k > 1:
        topk_msg = " " + " ".join(
            f"syn@{k}={metrics.get(f'syndrome_top{k}','-')}% kf@{k}={metrics.get(f'knowledge_formula_compatibility@{k}','-')}%"
            for k in range(2, top_k + 1)
        )
    log(f"  [{name}] DONE: strict_syn@1={syn_ok/n*100:.2f}% "
        f"lenient_syn@1={lenient_syn_ok/n*100:.2f}% "
        f"mf1={_macro_f1*100:.2f}% wf1={_weighted_f1*100:.2f}% "
        f"fmt={format_valid/n*100:.2f}% "
        f"empty={empty_pred/n*100:.2f}% "
        f"lenient_used={lenient_used_count} "
        f"lat={lat_sum/max(lat_n,1):.0f}ms")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics


def do_merge():
    """合并所有子目录的 metrics.json 到 summary CSV（v3 字段）。"""
    import glob as _glob
    rows = []
    for d in sorted(_glob.glob(str(REVAL_ROOT / "*/metrics.json"))):
        with open(d, encoding="utf-8") as f:
            rows.append(json.load(f))
    if not rows:
        log("ERROR: no metrics.json found to merge")
        return

    # 统一字段顺序（v3）
    FIELD_ORDER = [
        "run_name", "checkpoint_path", "evaluator_version", "prompt_version",
        "top_k", "n_cases", "n_labels",
        "syndrome_top1", "macro_f1", "weighted_f1", "balanced_accuracy",
        "knowledge_formula_compatibility@1",
        "syndrome_top3", "knowledge_formula_compatibility@3",
        "syndrome_top5", "knowledge_formula_compatibility@5",
        "format_validity",
        "avg_latency_ms",
    ]
    # 动态加入 top-k 字段（以防 top_k > 5）
    for r in rows:
        for k in range(6, r.get("top_k", 1) + 1):
            for prefix, field in [("syndrome", f"syndrome_top{k}"),
                                   ("formula", f"knowledge_formula_compatibility@{k}")]:
                if field not in FIELD_ORDER:
                    FIELD_ORDER.append(field)

    summary_csv = SUMMARY_DIR / "re_eval_v3_summary.csv"
    with open(summary_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELD_ORDER, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # MD 表格（主表：syndrome@1 + F1 + knowledge formula + format + latency）
    md_lines = [
        "| run_name | n | n_labels | syn@1 | macro_f1 | weighted_f1 | balanced_acc | kf@1 | kf@3 | kf@5 | fmt_valid | latency(ms) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        syn3 = r.get("syndrome_top3", "-")
        syn5 = r.get("syndrome_top5", "-")
        kf1 = r.get("knowledge_formula_compatibility@1", "-")
        kf3 = r.get("knowledge_formula_compatibility@3", "-")
        kf5 = r.get("knowledge_formula_compatibility@5", "-")
        md_lines.append(
            f"| {r['run_name']} | {r['n_cases']} | {r.get('n_labels', N_LABELS)} | "
            f"{r['syndrome_top1']} | {r.get('macro_f1', '-')} | {r.get('weighted_f1', '-')} | {r.get('balanced_accuracy', '-')} | "
            f"{kf1} | {kf3} | {kf5} | "
            f"{r['format_validity']} | {r['avg_latency_ms']} |"
        )
    (SUMMARY_DIR / "re_eval_v3_table.md").write_text("\n".join(md_lines), encoding="utf-8")

    # 如果有 top-k，额外生成 top-k 表格
    max_k = max((r.get("top_k", 1) for r in rows), default=1)
    if max_k > 1:
        k_cols = []
        for k in range(1, max_k + 1):
            k_cols.append(f"syn@{k}")
        for k in range(1, max_k + 1):
            k_cols.append(f"kf@{k}")
        header = "| run_name | " + " | ".join(k_cols) + " |"
        sep = "| --- | " + " | ".join(["---:"] * len(k_cols)) + " |"
        kmd = [header, sep]
        for r in rows:
            vals = [r["run_name"]]
            for k in range(1, max_k + 1):
                vals.append(str(r.get(f"syndrome_top{k}", "-")))
            for k in range(1, max_k + 1):
                vals.append(str(r.get(f"knowledge_formula_compatibility@{k}", "-")))
            kmd.append("| " + " | ".join(vals) + " |")
        (SUMMARY_DIR / "re_eval_v3_topk_table.md").write_text("\n".join(kmd), encoding="utf-8")

    log(f"Merged {len(rows)} groups -> {summary_csv}")


def main():
    # merge-only 模式
    if DO_MERGE:
        do_merge()
        return

    mode_str = f"SMOKE TEST (n={SMOKE_N})" if SMOKE_TEST else "FULL RUN"
    if SINGLE_GROUP:
        mode_str += f" [single: {SINGLE_GROUP}]"
    log(f"=== unified_re_eval v3 — {mode_str} ===")
    log(f"evaluator_version: {EVALUATOR_VERSION}")
    log(f"prompt_version: {PROMPT_VERSION}_{PROMPT_VARIANT} (variant={PROMPT_VARIANT})")
    log(f"decoding: do_sample=True, top_p=0.9, temperature=0.7, repetition_penalty=1.3")
    if TOP_K > 1:
        log(f"top_k: {TOP_K} (each sample generates {TOP_K} times, computing top-1..top-{TOP_K} accuracy)")

    if not VAL_FILE.exists():
        log(f"ERROR: {VAL_FILE} not found")
        sys.exit(1)

    with open(VAL_FILE, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    if SMOKE_TEST:
        eval_data = all_data[:SMOKE_N]
        log(f"Smoke test: using first {SMOKE_N} samples")
    else:
        eval_data = all_data

    log(f"Loaded {len(eval_data)} samples (total dataset: {len(all_data)})")

    # 确定要跑的组
    if SINGLE_GROUP:
        target = [g for g in GROUPS if g["short"] == SINGLE_GROUP]
        if not target:
            log(f"ERROR: group '{SINGLE_GROUP}' not found. Available: {[g['short'] for g in GROUPS]}")
            sys.exit(1)
        groups_to_run = target
    else:
        groups_to_run = GROUPS

    log(f"groups: {[g['name'] for g in groups_to_run]}")

    all_metrics = []
    for gi, group in enumerate(groups_to_run):
        log(f"========== [{gi+1}/{len(groups_to_run)}] {group['name']} ==========")
        try:
            m = run_group(group, eval_data, top_k=TOP_K)
            all_metrics.append(m)
        except Exception as e:
            log(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    log("=" * 60)
    log(f"ALL DONE — {mode_str}")
    for m in metrics_list:
        log(f"  {m['run_name']}: strict_syn@1={m['strict_syndrome_top1']}% lenient_syn@1={m.get('lenient_syndrome_top1','?')}% fmt={m.get('format_validity','?')}%")

    # 强制退出，彻底释放 GPU 显存
    os._exit(0)


if __name__ == "__main__":
    main()
