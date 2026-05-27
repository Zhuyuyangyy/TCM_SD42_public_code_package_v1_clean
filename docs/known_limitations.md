# Known Limitations

- Evaluation is retrospective and within-corpus, not externally curated or prospective.
- Raw clinical records are not public, limiting full third-party reproduction without data access approval.
- No practitioner benchmark or multi-institutional validation is included.
- Runtime configuration affects pilot accuracy; the main result is tied to the documented NF4-bfloat16 inference setup.
- Few-shot prompting was excluded from the formal comparison because the constructed multi-example prompt exceeded the sequence-length constraint and caused format collapse.
