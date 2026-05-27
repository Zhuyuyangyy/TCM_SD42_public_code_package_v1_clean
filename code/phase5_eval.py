# -*- coding: utf-8 -*-
"""
Phase 5: evaluation with SafeTCMGenerator post-check.
"""

from __future__ import annotations

import csv
import gc
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg


LOG_FILE = Path(os.environ.get("TCM_REVAL_ROOT", "./eval_output")) / "phase5_eval.log"

FIELD_SYNDROME = "证型"
FIELD_TREATMENT = "治法"
FIELD_FORMULA = "方剂"
FIELD_BASIS = "依据"
FIELD_EXPLANATION = "辨证解释"
FIELD_TONGUE = "舌象"
FIELD_PULSE = "脉象"

ENTRY_KEY_ALIASES = {
    "症状": ["症状", "鐥囩姸"],
    "病位": ["病位", "鐥呬綅"],
    FIELD_EXPLANATION: [FIELD_EXPLANATION, "杈ㄨ瘉瑙ｉ噴"],
}


def write_log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] [Phase5] {message}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as file:
        file.write(line + "\n")


def normalize_formula(value) -> str:
    if isinstance(value, list):
        return "、".join(str(item).strip() for item in value if item)
    if value is None:
        return ""
    return str(value).strip()


def get_formula_list(value) -> list[str]:
    text = normalize_formula(value)
    if not text:
        return []
    if "、" in text:
        return [item.strip() for item in text.split("、") if item.strip()]
    if "," in text:
        return [item.strip() for item in text.split(",") if item.strip()]
    return [text]


def get_entry_value(entry: dict, field_name: str, default=""):
    for key in ENTRY_KEY_ALIASES.get(field_name, [field_name]):
        if key in entry and entry[key] is not None:
            return entry[key]
    return default


def extract_field(text: str, field_name: str) -> str:
    keywords = [
        FIELD_SYNDROME,
        FIELD_TREATMENT,
        FIELD_FORMULA,
        FIELD_BASIS,
        FIELD_EXPLANATION,
        FIELD_TONGUE,
        FIELD_PULSE,
    ]
    other_keywords = [k for k in keywords if k != field_name]
    lookahead = "|".join(other_keywords)

    # 允许字段值跨行，直到下一个字段名开始。
    pattern = rf"{re.escape(field_name)}\s*[:：]?\s*(.*?)(?=\n(?:{lookahead})\s*[:：]?|$)"
    match = re.search(pattern, text, re.DOTALL | re.MULTILINE)

    if match:
        return match.group(1).strip().replace("<|im_end|>", "")
    return ""


def build_prompt(symptoms: str, disease_site: str) -> str:
    return (
        "请用中文回答。\n"
        f"症状：{symptoms}\n"
        f"病位：{disease_site}\n"
        "请给出辨证论治，用中文按以下格式输出：\n"
        "证型：\n治法：\n方剂：\n依据：\n辨证解释：\n"
    )


def build_quant_config():
    from transformers import BitsAndBytesConfig

    if cfg.DEVICE != "cuda":
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def load_model(lora_path: Path | None, base_model_path: Path | None = None):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(base_model_path or cfg.MODEL_NAME), trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"trust_remote_code": True}
    quant_config = build_quant_config()
    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = "cpu"
        model_kwargs["torch_dtype"] = torch.float32

    base_model = AutoModelForCausalLM.from_pretrained(str(base_model_path or cfg.MODEL_NAME), **model_kwargs)
    if lora_path and Path(lora_path).exists() and (Path(lora_path) / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(base_model, str(lora_path))
        write_log(f"loaded model from {lora_path}")
    else:
        model = base_model
        write_log("fallback to base model for eval")
    model.eval()
    return model, tokenizer


def violates_rule(text: str) -> bool:
    syndrome = extract_field(text, FIELD_SYNDROME)
    formula_text = extract_field(text, FIELD_FORMULA)
    taboo_list = cfg.RULE_DICT.get(syndrome, [])
    return any(taboo in formula_text for taboo in taboo_list)


class SafeTCMGenerator:
    def __init__(self, model, tokenizer, max_retry: int = 2):
        self.model = model
        self.tokenizer = tokenizer
        self.max_retry = max_retry

    @torch.no_grad()
    def generate_once(self, prompt: str) -> str:
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=cfg.MAX_SEQ_LENGTH,
        ).to(device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=320,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            repetition_penalty=1.3,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        output = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        for stop_token in ["注意", "免责声明", "以上内容", "请在专业", "#中医药"]:
            if stop_token in output:
                output = output[:output.index(stop_token)].strip()
        return output

    def generate(self, prompt: str) -> str:
        last_output = ""
        for i in range(self.max_retry + 1):
            if i > 0:
                print(f"检测到禁忌或格式错误，正在进行第 {i} 次重试...")
            last_output = self.generate_once(prompt)
            if not violates_rule(last_output):
                return last_output
        return last_output + cfg.DISCLAIMER


def syndrome_match(predicted: str, expected: str) -> bool:
    return bool(predicted) and (predicted == expected or predicted in expected or expected in predicted)


def formula_match(predicted_formula: str, syndrome: str, expected_syndrome: str = "") -> bool:
    if not predicted_formula:
        return False
    check_syndrome = expected_syndrome if expected_syndrome else syndrome
    legal_bundle = cfg.SYNDROME_PRESCRIPTION_CONSTRAINTS.get(check_syndrome, {})
    legal_formulas = legal_bundle.get(FIELD_FORMULA, [])
    if not legal_formulas:
        # 标准证型不在字典时，退而使用预测证型。
        legal_bundle = cfg.SYNDROME_PRESCRIPTION_CONSTRAINTS.get(syndrome, {})
        legal_formulas = legal_bundle.get(FIELD_FORMULA, [])
    if not legal_formulas:
        return True
    return any(
        legal_formula in predicted_formula or predicted_formula in legal_formula
        for legal_formula in legal_formulas
    )


def anti_hallucination_flags(text: str) -> dict:
    hits = [term for term in cfg.HALLUCINATION_TERMS if term in text]
    return {"hits": hits, "is_clean": len(hits) == 0}


def internal_consistency_score(output_text: str, entry: dict) -> float:
    syndrome = extract_field(output_text, FIELD_SYNDROME)
    treatment = extract_field(output_text, FIELD_TREATMENT)
    formula = extract_field(output_text, FIELD_FORMULA)
    explanation = extract_field(output_text, FIELD_EXPLANATION)

    score = 0.0
    expected_syndrome = get_entry_value(entry, FIELD_SYNDROME, "")
    expected_treatment = get_entry_value(entry, FIELD_TREATMENT, "")
    expected_site = get_entry_value(entry, "病位", "")
    if syndrome_match(syndrome, expected_syndrome):
        score += 0.4
    legal_treatment = cfg.SYNDROME_PRESCRIPTION_CONSTRAINTS.get(syndrome, {}).get(FIELD_TREATMENT, "")
    if legal_treatment and legal_treatment in treatment:
        score += 0.2
    if formula_match(formula, syndrome):
        score += 0.2
    if any(anchor and anchor[:2] in explanation for anchor in [expected_syndrome, expected_treatment, expected_site]):
        score += 0.1
    if any(token and token in explanation for token in get_formula_list(formula)):
        score += 0.1
    return round(min(score, 1.0), 4)


def long_text_logic_score(output_text: str) -> float:
    treatment = extract_field(output_text, FIELD_TREATMENT)
    formula = extract_field(output_text, FIELD_FORMULA)
    explanation = extract_field(output_text, FIELD_EXPLANATION)
    score = 0.0
    if len(explanation) >= 30:
        score += 0.4
    if len(explanation) >= 60:
        score += 0.2
    if treatment and treatment in explanation:
        score += 0.2
    if any(token and token in explanation for token in get_formula_list(formula)):
        score += 0.2
    return round(min(score, 1.0), 4)


def evaluate_model_group(group_name: str, model_path: Path, eval_data: list[dict], base_model_path: Path | None = None) -> dict:
    model, tokenizer = load_model(model_path, base_model_path)
    generator = SafeTCMGenerator(model, tokenizer, max_retry=cfg.MAX_RETRIES)

    syndrome_correct = 0
    formula_correct = 0
    clean_count = 0
    total_latency = 0.0
    total_consistency = 0.0
    total_long_logic = 0.0
    latency_count = 0
    details = []

    for index, entry in enumerate(eval_data, start=1):
        write_log(f"[{group_name}] 进度 {index}/{len(eval_data)}")
        prompt = build_prompt(get_entry_value(entry, "症状", ""), get_entry_value(entry, "病位", ""))
        start_time = time.time()
        output_text = generator.generate(prompt)
        latency_ms = (time.time() - start_time) * 1000

        expected_syndrome = get_entry_value(entry, FIELD_SYNDROME, "")
        syndrome = extract_field(output_text, FIELD_SYNDROME)
        formula = extract_field(output_text, FIELD_FORMULA)
        syndrome_ok = syndrome_match(syndrome, expected_syndrome)
        formula_ok = formula_match(formula, syndrome, expected_syndrome)
        anti_hallucination = anti_hallucination_flags(output_text)
        consistency = internal_consistency_score(output_text, entry)
        long_logic = long_text_logic_score(output_text)

        syndrome_correct += int(syndrome_ok)
        formula_correct += int(formula_ok)
        clean_count += int(anti_hallucination["is_clean"])
        if index > 1:  # 跳过第一个样本（含模型加载时间）
            total_latency += latency_ms
            latency_count += 1
        total_consistency += consistency
        total_long_logic += long_logic

        details.append(
            {
                "index": index,
                "症状": get_entry_value(entry, "症状", ""),
                "标准证型": expected_syndrome,
                "标准方剂": normalize_formula(get_entry_value(entry, FIELD_FORMULA, "")),
                "预测证型": syndrome,
                "预测方剂": formula,
                "证型准确": syndrome_ok,
                "方剂合规": formula_ok,
                "抗幻觉通过": anti_hallucination["is_clean"],
                "hallucination_hits": anti_hallucination["hits"],
                "医理自洽评分": consistency,
                "long_text_logic": long_logic,
                "输出": output_text,
            }
        )

    total = max(len(eval_data), 1)
    metrics = {
        "group": group_name,
        "accuracy_syndrome": round(syndrome_correct / total * 100, 2),
        "accuracy_formula": round(formula_correct / total * 100, 2),
        "internal_consistency_score": round(total_consistency / total, 4),
        "anti_hallucination_rate": round(clean_count / total * 100, 2),
        "long_text_logic_score": round(total_long_logic / total, 4),
        "avg_latency_ms": round(total_latency / max(latency_count, 1), 2),
        "details": details,
    }

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def save_results(results: list[dict]) -> None:
    fieldnames = [
        "group",
        "accuracy_syndrome",
        "accuracy_formula",
        "internal_consistency_score",
        "anti_hallucination_rate",
        "long_text_logic_score",
        "avg_latency_ms",
    ]
    with open(cfg.EXPERIMENT_CSV, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            writer.writerow({field: item[field] for field in fieldnames})

    lines = [
        "| Group | Syndrome Acc | Formula Acc | Consistency | Anti-Hallucination | Long-Logic | Avg Latency(ms) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in results:
        lines.append(
            f"| {item['group']} | {item['accuracy_syndrome']} | {item['accuracy_formula']} | "
            f"{item['internal_consistency_score']} | {item['anti_hallucination_rate']} | "
            f"{item['long_text_logic_score']} | {item['avg_latency_ms']} |"
        )
    cfg.EXPERIMENT_MD.write_text("\n".join(lines), encoding="utf-8")

    with open(cfg.BASE_DIR / "phase5_eval_details.json", "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)


def main():
    if not cfg.VAL_UNSEEN_FILE.exists():
        raise FileNotFoundError(f"validation file not found: {cfg.VAL_UNSEEN_FILE}")
    with open(cfg.VAL_UNSEEN_FILE, "r", encoding="utf-8") as file:
        eval_data = json.load(file)
    groups = [
        ("SFT Baseline", cfg.SFT_BASELINE_PATH, None),
        ("Patent Method", cfg.PATENT_MODEL_PATH, None),
        ("Qwen2.5-14B-Instruct", Path(os.environ.get("QWEN25_14B_PATH", "/path/to/Qwen2.5-14B-Instruct")), Path(os.environ.get("QWEN25_14B_PATH", "/path/to/Qwen2.5-14B-Instruct"))),
    ]
    results = []
    for name, model_path, base_path in groups:
        try:
            r = evaluate_model_group(name, model_path, eval_data, base_path)
            results.append(r)
            save_results(results)
            write_log(f"[OK] {name} done, results saved ({len(results)}/{len(groups)})")
        except Exception as e:
            write_log(f"[FAIL] {name} crashed: {e}")
            save_results(results)
    write_log(f"saved eval outputs -> {cfg.EXPERIMENT_CSV} / {cfg.EXPERIMENT_MD}")

if __name__ == "__main__":
    main()
