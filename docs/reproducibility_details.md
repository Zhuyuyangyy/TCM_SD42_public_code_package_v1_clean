# Reproducibility Details

## Scope

This public repository supports the non-sensitive reproduction of the TCM-SD-42 LoRA-SFT manuscript results. Raw clinical records, LoRA adapter weights, and per-case outputs are not included.

## Data

- Pilot evaluation: `mini_42_pilot/test.json` in the private evidence package, N = 420.
- Full within-corpus held-out evaluation: private full test split, N = 4,452.
- Public release: only label lists, aggregate metrics, and non-sensitive figure/table data.

## Model

- Base model: Qwen3-8B instruction-tuned checkpoint.
- Fine-tuning: LoRA supervised fine-tuning.
- LoRA rank: 16.
- LoRA alpha: 32.
- LoRA dropout: 0.1.
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.

## Main Inference Runtime

The manuscript-level pilot result is tied to the following inference runtime configuration:

```python
from transformers import BitsAndBytesConfig
import torch

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
```

Generation configuration:

```text
do_sample = False
temperature = 0
max_new_tokens = 96
repetition_penalty = 1.1
stop sequences = ["注意", "免责声明", "以上内容", "请在专业", "#中医药"]
```

A pure float16 rerun produced a lower pilot accuracy of approximately 64.5% and is treated as a runtime sensitivity analysis, not as the main manuscript result. Restoring the original NF4-bfloat16 runtime reproduced the manuscript-level pilot result under the closed-set 42-label parsing protocol.

## Formal Label Parsing

The formal reported label-level evaluation uses a deterministic closed-set 42-label parser:

1. Search the model output for occurrences of the 42 predefined syndrome labels.
2. If exactly one label is found, record that label.
3. If no label is found, record an empty prediction.
4. If multiple distinct labels are found, use the first-occurring label.
5. Synonym-only or out-of-label predictions are not counted as valid formal labels.

## Public Aggregate Files

- `aggregated_results/main_results_table.csv`: manuscript main table values.
- `aggregated_results/pilot_420_metrics.json`: pilot split aggregate metrics.
- `aggregated_results/full_4452_metrics.json`: full held-out aggregate metrics.
- `aggregated_results/per_class_metrics_exact42_pilot420.csv`: aggregate per-class F1 audit data.
- `aggregated_results/top_confusion_pairs_exact42_pilot420.csv`: aggregate top confusion pairs.
- `aggregated_results/confusion_matrix_exact42_pilot420.csv`: aggregate confusion matrix.
