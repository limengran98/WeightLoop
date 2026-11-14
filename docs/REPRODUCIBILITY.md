# Reproduction guide

## CPU example and table replay

```bash
python -m pip install -e '.[scientific]'
python examples/feedback_demo.py --output outputs/demo
python -m unittest discover -s tests -v
python scripts/reproduce_tables.py
```

The NumPy example uses 75 synthetic feedback observations and 150 query examples. It demonstrates feedback selection, native parameter saving and prediction replay. Eighteen tests cover numerical behavior, input validation, feedback isolation and persistence. The `scientific` extra installs SciPy for the static fitting module.

## Constructors and shared functional setup

[GPT-6 Astra](https://developers.openai.com/api/docs/models/gpt-6-astra) constructs the editor in an interactive coding workflow. [DeepSeek](https://api-docs.deepseek.com/quick_start/pricing/) independently generates the official functional editor through its API. Both editors use a functional MLP with a 128-unit final hidden layer. Inputs combine GO, scGPT and MAP-KG features; updates act on output weights and final hidden biases.

Functional ranking starts from an MLP trained with 25 initial labels and supplies 75 further feedback assays to each editor. The initial MLP remains the starting predictor. Sequential discovery acquires five batches of 20 targets, and each policy uses its own returned outcomes. Functional and sequential summaries average five gene folds within each of exactly three training seeds, then report mean and sample SD across the three seed means.

## Native prediction tasks

The native tasks use execution seed17. Fixed editing and Adam share original parameters, feedback identities and target arrays, DeepSeek-selected native parameter pools, folds, twelve objectives and acceptance checks. The optimizer configuration is specified in `programs/STATIC_REFERENCE.json` and the fitting arguments in `programs/static_reference.py`. Fixed editing uses the analytic squared-error solve or L-BFGS for smooth-MAE. Native forward validation and float32 commitment follow fitting.

| Task | Predictor | Metric | Feedback prefixes | Final evaluation | Native pool / final changed |
|---|---|---|---|---|---|
| Adamson CRISPRi | GEARS | RNA MSE | 8,12,16 whole genes | 9 whole target genes | 1024 / 1024 |
| INSTANCE | PhaseNet | P-arrival MAE, seconds | 24,40,56 whole earthquakes | 64 whole earthquakes | 27 / 26 |
| ERA5 Z500 | ClimaX | Area-weighted geopotential RMSE, m² s⁻² | 20,41,61 weather states | 59 states in six whole months | 192 / 4 |
| Radial dam-break | FNO2d | Full-field relative L2 | 10,18,26 initial-state groups | 32 groups | 129 / 129 |

DeepSeek supplies the pool selections for these four tasks, and the fixed numerical program estimates their corrections. Source units are whole target genes, earthquakes, calendar months and initial-state groups. The shallow-water split includes training-equivalent initial states. Detailed case protocols, source hashes and model provenance are provided in [`case_protocols.json`](../results/static_cases/case_protocols.json); [`feedback_parity.json`](../results/static_cases/feedback_parity.json) binds feedback identities and target arrays.

Sources: [Adamson CRISPRi](https://doi.org/10.1016/j.cell.2016.11.048), [GEARS](https://doi.org/10.1038/s41587-023-01905-6), [INSTANCE](https://doi.org/10.5194/essd-13-5509-2021), [PhaseNet](https://doi.org/10.1093/gji/ggy423), [ClimaX](https://proceedings.mlr.press/v202/nguyen23a.html), [WeatherBench2](https://doi.org/10.1029/2023MS004019), [PDEBench](https://arxiv.org/abs/2210.07182), [FNO](https://arxiv.org/abs/2010.08895).

Functional assay sources: [DepMap 24Q4](https://doi.org/10.25452/figshare.plus.27993248.v1), [HL60 assays](https://doi.org/10.1038/s41467-023-41452-x), [autophagy](https://doi.org/10.1371/journal.pbio.2007044), [LDL uptake](https://doi.org/10.1371/journal.pgen.1009285).

## RNA output calibration

The RNA study uses scGPT and scFoundation+GEARS on H1 CRISPRi perturbations from VCC2025, predicting 1024 RNA output coordinates. Base is the model adapted on 40 target perturbations. Adam and WeightLoop-A start from the same Base checkpoint and share 16 feedback targets, two feedback-validation folds, the complete native writable allowlist and 20 query targets per partition. WeightLoop-A is constructed through the GPT-6 Astra interactive coding workflow. Both update procedures select on feedback folds and refit on all available feedback before query evaluation.

The reported metric is overall query MSE. Each partition averages the target-level errors over 20 query perturbations. Three fixed partitions are averaged within each model-training seed, followed by the mean and sample SD across exactly three training-seed means. These RNA statistics use their own protocol and are separate from the execution-seed17 native case studies.

[`protocol.json`](../results/rna_calibration/protocol.json) describes the data, allowlist and selection configuration. [`split_metrics.tsv`](../results/rna_calibration/split_metrics.tsv) contains 54 split records; [`seed_metrics.tsv`](../results/rna_calibration/seed_metrics.tsv) contains 18 seed-level means; [`summary.tsv`](../results/rna_calibration/summary.tsv) contains the six model–procedure summaries. Source arm names `original`, `block_adapt` and `weightloop` correspond to Base, Adam and WeightLoop-A, respectively. `scripts/reproduce_tables.py` checks these aggregates and the relative MSE reductions.

## Scientific fitting and manuscript versions

Scientific fitting uses source datasets, fixed partitions, preprocessing assets, pretrained backbones and native task adapters under their original terms. The release provides numerical editor code, a CPU example, source tables and scripts for summary replay.

The [author/arXiv PDF](../assets/WeightLoop.pdf) and [anonymous submission PDF](../assets/WeightLoop_submission.pdf) have the same scientific content. The anonymous submission has four body pages with ethics and references following; the author version adds author information. Editable manuscript and figure sources are maintained privately. See [MAINTAINING.md](MAINTAINING.md) for PDF synchronization.

Code: <https://github.com/limengran98/WeightLoop>. Anonymous code: <https://anonymous.4open.science/r/WeightLoop-C75A/>. The `submission-anonymous` branch contains code, configuration, tests and usage instructions; it omits papers, figures and result records.
