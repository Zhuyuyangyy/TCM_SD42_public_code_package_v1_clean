# Public Release Checklist — TCM-SD-42 Public Reproducibility Package v1.1

## Scope of this release

**Included:**
- LoRA-SFT training/evaluation reference scripts
- Public configuration files with environment-variable paths
- 42-class label space files
- Prompt templates
- Documentation: reproducibility details, data availability, model card summary, known limitations
- Aggregated, non-sensitive metrics and figure/error-analysis data
- `requirements.txt`, `MANIFEST.md`, `SHA256SUMS.txt`, `README.md`

**Explicitly excluded:**
- Raw clinical records (`train.json`, `dev.json`, `test.json` from TCM-SD-42)
- Per-case model outputs (`outputs.jsonl`, `details.json`, `details_exact42.jsonl`)
- Trained LoRA adapter weights (`adapter_model.safetensors`) and model binaries
- Manuscript draft files
- Historical TCM-Mini, KD, DPO, Rule Loss, cMedQA2, or other unrelated experiment lines

## File-level checks

- [x] No raw clinical record files are included.
- [x] No adapter weights or model binaries are included.
- [x] No per-case outputs/details are included.
- [x] No manuscript drafts are included.
- [x] No `__pycache__` or `.pyc` files are included.
- [x] `.gitignore` excludes common model/data/output artifacts.
- [x] Aggregated result files only contain non-sensitive metrics/tables.

## Content-level checks

- [x] README describes the package as a TCM-SD-42 LoRA-SFT reproducibility package, not as a small-model distillation package.
- [x] README no longer references nonexistent `figure3_per_class_f1_data.csv`.
- [x] README describes formal evaluation as deterministic closed-set 42-label parsing.
- [x] `docs/reproducibility_details.md` documents bitsandbytes 4-bit NF4 + bfloat16 compute as the main inference runtime.
- [x] Public configs make KD/DPO/Rule Loss disabled or mark legacy constants as not used in manuscript experiments.
- [x] `TCM-Mini-best` default paths were replaced with `pilot_sft_clean_v3_lora_adapter`.
- [x] `SHA256SUMS.txt` excludes itself and can be verified with `sha256sum -c SHA256SUMS.txt`.

## Recommended verification commands

```bash
# Count files
find . -type f | wc -l

# Check forbidden file types / paths
find . -type f \( -name '*.safetensors' -o -name '*.bin' -o -name '*.pt' -o -name '*.pth' -o -name 'outputs.jsonl' -o -name 'details.json' -o -name 'details_exact42.jsonl' \)

# Check for Python cache files
find . -type f \( -name '*.pyc' -o -path '*/__pycache__/*' \)

# Verify checksums
sha256sum -c SHA256SUMS.txt

# Verify Python syntax
for f in $(find . -name '*.py'); do python3 -m py_compile "$f" || exit 1; done
```

## Final sign-off

| Item | Status |
|------|--------|
| No raw clinical data | ✅ |
| No adapter weights | ✅ |
| No per-case outputs | ✅ |
| No old experiment packages | ✅ |
| No manuscript drafts | ✅ |
| Repository wording aligned with LoRA-SFT manuscript | ✅ |
| Requirements present | ✅ |
| Documentation present | ✅ |
