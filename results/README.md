# Core paper result records

| Directory | Contents |
|---|---|
| `static_cases/` | Native prediction errors and same-feedback Adam controls in the DeepSeek-selected parameter pool, feedback-prefix metrics, native updates and protocols. |
| `additional_functional/` | Functional MLP ranking before and after feedback editing, with three model-training seeds. |
| `additional_sequential/` | Three-seed discovery trajectories for the GPT-6 Astra coding-workflow program, the official DeepSeek program and frozen initial MLP ranking. |
| `rna_calibration/` | MSE calibration of native scGPT and scFoundation+GEARS models: Original, same-feedback Adam and the reference editor, with three training seeds. |

Native prediction cases use execution seed17. Functional and sequential experiments first average the five gene folds within each training seed, then report mean and sample SD across three training-seed means. Protocol files provide task partitions, optimizer settings and original source identities. Original data method IDs are retained to bind each value to its source run.

RNA calibration uses the H1 CRISPRi dataset with 40 base-adaptation, 16 feedback and 20 query targets per fixed partition. Adam and WeightLoop start from the same base model and share the 16 feedback targets, two-fold feedback validation and complete v3 native parameter allowlist. Each training seed averages three fixed target partitions; the summary reports the mean and sample SD across the three seed means. The MSE export contains all three compared methods, 54 split-level records and 18 seed means. `rna_calibration/protocol.json` records the original configuration and source hashes.

Run `python scripts/reproduce_tables.py` to verify the native error reductions, feedback identity parity, three-seed summaries and result-file hashes. Scientific fitting uses the source datasets, pretrained checkpoints and native task adapters.
