# Program construction and fixed execution

[GPT-6 Astra](https://developers.openai.com/api/docs/models/gpt-6-astra) constructs the editor through an interactive coding workflow that inspects the scientific objective, model computation, writable tensors and feedback schema. [DeepSeek](https://api-docs.deepseek.com/quick_start/pricing/) independently generates model-specific proposal code through its official API. The host fits numerical corrections from returned observations, validates proposals and commits accepted native parameters. Each editor stays fixed across feedback updates.

`programs/official_functional.py` and its requests provide the independently generated DeepSeek editor used in the functional and sequential studies. The supplied-rule example in `programs/hepg2/` implements a specified numerical recipe. The construction workflows are associated with their recorded programs and source bindings.

For GEARS, PhaseNet, ClimaX and FNO2d, DeepSeek selects the editable parameter pools. `programs/static_reference.py` supplies the fixed numerical program and Adam control; `anchors(False) + fixed_extra(False)` defines twelve candidate settings. Native adapters provide response arrays, original coordinates, parameter scales and reference inputs.

The HepG2 illustration shows how an accepted MLP update is saved and reloaded through the original inference interface. Functional and sequential results use three model-training seeds. The four native case studies use execution seed17 with matched feedback and writable parameters.

RNA output calibration uses WeightLoop-A from the GPT-6 Astra interactive coding workflow. Adam and WeightLoop-A share the same Base checkpoint adapted on 40 targets, 16 feedback targets, two feedback-validation folds, native parameter allowlist and 20 query targets per partition. Overall MSE is summarized over three partitions within each of three model-training seeds; the [RNA protocol](../results/rna_calibration/protocol.json) binds this construction to its separate evaluation records.
