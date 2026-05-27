# Model Card Summary

## Model

Pilot-SFT-clean-v3 is a LoRA-supervised-fine-tuned Qwen3-8B model for 42-class Traditional Chinese Medicine syndrome differentiation.

## Intended Use

Offline retrospective research evaluation of closed-set TCM syndrome classification under the TCM-SD-42 pilot protocol.

## Out-of-Scope Use

This model is not intended for clinical deployment, autonomous diagnosis, treatment recommendation, or patient-facing decision-making.

## Important Runtime Note

The manuscript-level pilot result is tied to bitsandbytes 4-bit NF4 quantization with bfloat16 compute. A pure float16 runtime produced a lower sensitivity result and is not the main reported metric.
