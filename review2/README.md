# Review-2: M1 → M6

Separate implementation of the supplied Technical Handoff. Module identifiers are
functional, pending manual thesis-title review. No M7, M8 or M9 implementation.

For a ready-to-run GPU notebook and upload archive, see [Kaggle instructions](docs/KAGGLE.md).

Run commands from the repository root using the existing Python environment:

```powershell
.\fypvenv\Scripts\python.exe -m pytest review2/tests -q --junitxml=review2/outputs/tests.xml
.\fypvenv\Scripts\python.exe -m review2.pipeline --config review2/configs/smoke.json --output review2/outputs/smoke_new --evaluate-test
.\fypvenv\Scripts\python.exe -m review2.pipeline --config review2/configs/feasibility.json --output review2/outputs/pretrained_new --evaluate-test
```

Each module also has a runner:

```powershell
.\fypvenv\Scripts\python.exe -m review2.m1.run --config review2/configs/smoke.json --output review2/outputs/example
.\fypvenv\Scripts\python.exe -m review2.m2.run --config review2/configs/smoke.json --output review2/outputs/example
# Likewise review2.m3.run through review2.m6.run, in order.
```

Use `--stages m3 m4 m5 m6` to continue from compatible prior artifacts. Training
resumes `last.pt` with optimizer and RNG state, then exports `best.pt`. A changed
configuration requires a new output directory. Checkpoints are trusted local
pipeline outputs; do not load arbitrary pickle checkpoints.

After development completes, `--test-only` evaluates or resumes the frozen test
path without retraining. A run with frozen test selection rejects further development
stage execution. Use a new output directory for any changed research configuration.

`smoke.json` trains tiny random DeBERTa/DistilBERT architectures on a bounded external
PIIBench sample. It verifies execution and is **not a pretrained benchmark result**.
`feasibility.json` uses pretrained models on the same small official-file prefixes.
`full.json` removes the dataset cap and increases training/FGSM experiments. Full
training is expensive and is not run automatically. The current runner materializes
records, tokenized splits and MC logits in memory; a full-release MC sweep requires
substantial RAM/storage and may need smaller explicitly configured acquisition
limits on ordinary hardware. Prefix experiments are not representative estimates.

All three configs use explicitly marked controlled policy profiles. Replace
`m6.observations` with measured validation observations and set `allow_controlled=false`
when M7/M8 measurements exist. M6 returns executable parameter **contracts**, not
an implemented privacy mechanism. The central RAG privacy/utility hypothesis cannot
be validated within this M1–M6-only scope.

Outputs are written under the selected run directory:

- `m1/`: four processed splits, label map, integrity/statistics/source/split manifests,
  optional application corpus.
- `m2/`: model/loss comparisons, coverage reports, regex baseline, best/last checkpoints,
  exported models/tokenizers, metrics and validation predictions.
- `m3/`: validation attack records, clean/FGSM robustness, hyperparameter candidates,
  checkpoints, predictions and selected robust model.
- `m4/`: fitted temperature, calibration/MC logits, all reliability variants,
  metrics, pass comparisons, reliability/risk–coverage SVG and selected interface.
- `m5/`: three risk outputs, evidence/mass traces, stability/sensitivity/conflict results.
- `m6/`: policies/profiles, decisions, candidate checks, fixed/adaptive controlled
  comparison, sensitivity and downstream risk-method comparison.
- `final_test/`: frozen selection, final predictions, detector/robustness/calibration
  metrics and risk/policy outputs, only with `--evaluate-test`.
- Run-level configuration, environment versions, code hash, logs, status/failures.

Every completed stage contains a file-hash manifest and upstream artifact hash.
No raw sensitive text is written to decision logs; processed source/attack datasets
retain text because they are explicit research artifacts.

See [migration map](docs/MIGRATION.md), [implemented algorithms](docs/ALGORITHMS.md),
the [configuration/contracts](docs/CONFIGURATION_AND_CONTRACTS.md), and
[actual execution report](docs/RESULTS.md). The source mapping and policy
assumptions are editable configuration, not authoritative privacy annotations.

To reuse verified legacy application corpora (without PII supervision), run
`python -m review2.scripts.export_application_corpora`. To regenerate the saved
result tables and tree after a run, use `python -m review2.scripts.report`.
