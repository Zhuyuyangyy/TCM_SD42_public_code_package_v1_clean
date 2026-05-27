# TCM-SD-42 Public Reproducibility Package

**Version:** v1.1 git-ready  
**Manuscript:** *Automated TCM Syndrome Differentiation via Supervised Fine-Tuning of a Large Language Model: A Stratified Pilot Evaluation with Within-Corpus Held-Out Confirmation on the TCM-SD-42 Dataset*

## Overview

This repository provides a public, non-sensitive reproducibility package for the TCM-SD-42 LoRA-SFT syndrome classification pilot study. It supports the manuscript's label-level evaluation protocol for 42-class Traditional Chinese Medicine syndrome differentiation using Qwen3-8B with LoRA supervised fine-tuning.

This public package intentionally does **not** include:

- Raw clinical case records or full train/dev/test JSON files
- Trained LoRA adapter weights or base model weights
- Per-case model outputs, `details.json`, or `outputs.jsonl`
- Historical TCM-Mini, KD, DPO, Rule Loss, cMedQA2, or other unrelated experiment lines

## Package Contents

```
code/
  phase3_train.py      # LoRA supervised fine-tuning script retained for reproducibility reference
  phase5_eval.py       # Evaluation utilities
  unified_re_eval.py   # Evaluation pipeline and label extraction utilities

configs/
  config.py            # Public global config with environment-variable paths
  config_pilot.py      # Pilot dataset config
  config_pilot_sft_clean_v3_public.yaml

train_configs/
  config_pilot_sft_clean_v3.py  # Clean-v3 public config; KD/DPO/Rule Loss disabled

label_space/
  syndrome_label_list.json       # 42-class syndrome label list
  label_mapping_public.json      # Label ID ↔ syndrome name mapping

prompts/
  zero_shot_prompt_template.md
  few_shot_prompt_template.md

docs/
  reproducibility_details.md
  data_availability_statement.md
  model_card_summary.md
  known_limitations.md

aggregated_results/
  main_results_table.csv
  pilot_420_metrics.json
  full_4452_metrics.json
  per_class_metrics_exact42_pilot420.csv
  top_confusion_pairs_exact42_pilot420.csv
  confusion_matrix_exact42_pilot420.csv

requirements.txt
MANIFEST.md
SHA256SUMS.txt
public_release_checklist.md
README.md
```

## Quick Start

### 1. Environment

```bash
pip install -r requirements.txt
```

Python >= 3.10 is recommended. GPU is required for full model training/inference; aggregated results can be inspected without a GPU.

### 2. Configure paths

Set environment variables or edit config files locally:

```bash
export TCM_DATA_DIR=/path/to/your/TCM-SD-42-data
export TCM_OUTPUT_ROOT=/path/to/output
export TCM_MODEL_NAME=/path/to/Qwen3-8B
export TCM_LORA_PATH=/path/to/pilot_sft_clean_v3_lora_adapter
```

Raw data and adapter weights are not included in this public repository.

### 3. Train / evaluate

The public scripts are provided for reproducibility reference. A typical evaluation command is:

```bash
python code/unified_re_eval.py \
    --eval-file /path/to/test.json \
    --lora-path /path/to/pilot_sft_clean_v3_lora_adapter \
    --base-model /path/to/Qwen3-8B
```

The manuscript-reported main metrics are available as aggregated files under `aggregated_results/`.

## Expected Results

| Dataset   | Model             | syn@1  | macro-F1 |
|-----------|-------------------|-------:|--------:|
| Pilot 420 | Pilot-SFT-clean-v3 | 69.52% |  50.37% |
| Pilot 420 | Qwen3-8B zero-shot |  5.71% |   0.76% |
| Full 4452 | Pilot-SFT-clean-v3 | 60.60% |  41.74% |

**Runtime note:** The manuscript-level pilot result is tied to bitsandbytes 4-bit NF4 quantization with bfloat16 compute. A pure float16 rerun yielded a lower sensitivity result and is not used as the main reported metric. See `docs/reproducibility_details.md`.

## Evaluation Protocol

- **Formal label protocol:** deterministic closed-set 42-label parsing. Predictions are matched only against the predefined 42 syndrome labels. Out-of-label or synonym-only outputs are not counted as valid formal labels.
- **Main paper metrics:** Table 2 reports the original manuscript evaluation values. The NF4-bfloat16 reproducibility audit confirms the pilot syn@1 under the manuscript-style closed-set parsing protocol.
- **Per-class audit files:** `per_class_metrics_exact42_pilot420.csv`, `top_confusion_pairs_exact42_pilot420.csv`, and `confusion_matrix_exact42_pilot420.csv` are aggregate, non-sensitive files derived from the NF4-bfloat16 reproducibility rerun.
- **Training target:** the manuscript's formal task is label-level 42-class syndrome classification. Public scripts may retain internal prompt-format utilities, but the reported formal evaluation is based on single closed-set syndrome-label extraction.

## Data Format

The private data use input-style JSON records containing clinical presentation text and a canonical syndrome label. The raw records are not included here. A simplified conceptual schema is:

```json
{
  "input_text": "疾病名称：...\n主诉：...\n四诊信息：...",
  "syndrome_canonical": "气虚血瘀"
}
```

The `label_space/syndrome_label_list.json` file defines the 42-class taxonomy used for formal evaluation.

## Data and Model Availability

See `docs/data_availability_statement.md`. Raw clinical records are subject to institutional data governance and privacy restrictions. The trained LoRA adapter weights are not included in this public package and may be made available only subject to journal policy and institutional approval.

## Citation

To be updated with the final DOI, arXiv, or journal link once available.
