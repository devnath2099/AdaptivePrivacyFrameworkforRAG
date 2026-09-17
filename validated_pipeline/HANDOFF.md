# Revised M1-M5 handoff

## Files and baseline preservation

All new code, configuration, tests, manifests, caches and outputs live in `validated_pipeline/`. `FILE_INVENTORY.txt` lists every deliverable source/config/test/document file relative to this directory. Runtime data/checkpoints are excluded from that source inventory and from Git; the expected runtime tree is in README.md.

Intentionally unchanged: root `src/`, `updated_pipeline/`, `scripts/`, `configs/`, `tests/`, `README.md`, `requirements.txt`, `data/`, `outputs/`, `outputs01/`, `output1/`, UI and notebook. No old checkpoint was copied, no commit was made and nothing was pushed. Initial Git status was clean; the subsequent status showed only `?? validated_pipeline/`.

## Implemented

Pinned natural source adapters, exact group-safe 50k splits, query-only direct evidence, abstaining semantic LFs, train-only Snorkel fitting and majority diagnostics, dynamic masked three-head losses, frozen encoder/default and optional last-N unfreezing, M3/M4 resumable checkpoints, raw-embedding FGSM, calibration-only positive temperatures and MC uncertainty, deterministic disjoint controlled benchmark definitions, optional training-only verified augmentation, isolated auxiliary evidence evaluation, one-command launcher, and explicit future validation/controlled evaluation launchers.

Artifact contracts bind source/configuration/code/splits/checkpoint lineage. Test data is written but not evaluated by the default flow. M6 fields appear only as declared controlled-scenario contracts. There is no RAG enforcement implementation.

## M2 v2 correction

The original real M2 run showed that `personal_finance` matched ambiguous words (`bank`, `balance`, `financial`) inside medical queries. Its medium LF was a phrase-only rule that fired zero times. M2 v2 narrows personal-financial evidence to an owned financial artifact or account action, separates direct personal medical evidence from a limited health/financial topic, and adds a narrow medium LF for explicit limited personal health/planning topics only. It does not classify generic medical/financial vocabulary as high or medium.

The original Snorkel model suppressed personal-financial argmax support despite direct majority support. M2 v2 uses a transparent, train-only fallback only for a categorical class that has direct majority support but zero Snorkel argmax support: a record must have one unanimous direct LF class vote, then its output is recorded as `unanimous_direct_lf_fallback`. No support is created when no LF fires. Threat labels do not use this fallback.

Every M2 diagnostic now records per-LF coverage/firing counts, all-abstain/overlap/conflict rates, per-class LF firing/record support, majority/final support and disagreement, Snorkel-suppression flags, and aggregation provenance. `scripts/summarize_m2.py` prints a compact version.

## Actual verification

Command, from the repository root using its existing Python environment:

```powershell
.\fypvenv\Scripts\python.exe -m pytest validated_pipeline/tests -q -o cache_dir=validated_pipeline/.pytest_cache --basetemp=validated_pipeline/outputs/test_tmp_verified
```

Actual final output:

```text
...............................                                          [100%]
31 passed, 8 warnings in 10.39s
```

Warnings are `torch.jit.script` deprecation warnings from PyTorch during the small actual-DeBERTa interface test. Full output is saved in `outputs/pytest_result.txt`. An earlier run caught a YAML quoting error; it was fixed before the final run. Exact dependency versions are in TEST_ENVIRONMENT.txt; requirements.txt references that file.

The suite includes query/context independence, TAT-QA query-only loading, fallback prohibition, deterministic dedup/group split and deficits, train-only synthetic augmentation, default and alternate task shapes, explicit freezing/unfreezing, taxonomy rejection, all-head FGSM with padding and evaluation gradients, both temperature objectives and calibration-only fitting, population ECE, uncertainty, benchmark separation, majority/Snorkel abstention handling, checkpoint resume, exact interrupted CPU replay, stage artifact corruption rejection, and the optional span contract.

```powershell
.\fypvenv\Scripts\python.exe validated_pipeline/scripts/check_sources.py
```

Actual output:

```text
{'train': 35000, 'calibration': 5000, 'validation': 5000, 'test': 5000}
actual source load and group-safe split passed; no training or test evaluation
```

All three pinned public sources loaded successfully. `outputs/source_check/result.json` records revisions, source-cache hashes and split ID. Separately counted official TAT-QA files: train 13,215, dev 1,668, test 1,669; combined 16,552, normalized unique 15,764. There was no deficit or substitution.

```powershell
.\fypvenv\Scripts\python.exe validated_pipeline/scripts/check_token_lengths.py --run outputs/overnight_50k
```

Actual 35,000 natural-training-query tokenizer diagnostics: p50=19, p95=149, p99=245; 7.694285714% beyond 128, 0.848571429% beyond 256. Retained requested max_length=128; 92.3% fit without truncation. The analytical padded token/attention cost ratios for 256 versus 128 are 2x/4x, not measured end-to-end costs. See `outputs/overnight_50k/pretraining_token_diagnostics.json`.

The offline integration smoke trained the three heads, continued with FGSM, fitted temperatures and wrote M5 outputs using tiny contextual fixtures. Another test exercised the real DeBERTa API with random miniature weights. These are engineering smoke checks, not pretrained research runs or reported thesis metrics.

The real partial pipeline also executed successfully:

```powershell
.\fypvenv\Scripts\python.exe validated_pipeline/scripts/run_validated_pipeline.py --profile overnight_50k --until m2
```

It built the exact 50k corpus, extracted query-only spaCy/regex evidence for 35k train + 5k calibration + 5k validation, fitted five train-only Snorkel LabelModels (two categorical and three independent binary models), and wrote labels/diagnostics and stage receipts. Final status is `partial_requested_stop` at `m1_m2`, not full experiment completion. Its code hash was checked against the delivered implementation and matched. Test was not labeled or evaluated.

Repeating the same partial-run command validated stage identities and artifact hashes and reused the completed stages successfully (exit code 0, 1.22 seconds), without re-extracting evidence or refitting LabelModels.

**Observed training-label limitation:** sensitivity all-abstain=48.802857%, intent=49.494286%, and individual threat all-abstain about 66.2%. Among observed rows, Snorkel argmax distributions were sensitivity `[11812, 0, 6107]` and intent `[11812, 184, 5673, 0, 8]` in configured class order. Thus medium sensitivity and personal-financial intent have no natural-training Snorkel argmax support in this run. This is a weak-supervision diagnostic, not accuracy. It is a material limitation for subsequent model interpretation; controlled rule/model evaluation and optional training-only augmentation are available, but their benefit has not been established. No validation/test-driven rule adjustment was made to conceal this result.

The majority-vote intent distribution was `[11812, 209, 5606, 42, 8]`; its 42 personal-financial argmax rows disappear under Snorkel argmax. Both methods had zero medium-sensitivity argmax rows. These observations establish scarcity/disagreement, not which method is correct.

## Kaggle command and expected outputs

From a writable repository checkout with Internet and a GPU:

```bash
cd validated_pipeline
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python scripts/run_validated_pipeline.py --profile overnight_50k
```

The full expected output tree is documented in README.md. M3/M4 checkpoints and M5 predictions only exist for a full run after those stages execute; they are not implied by the source/M2 checks. `PROJECT_ROOT`, if set, must resolve to the revised directory. Use a new `--output outputs/<experiment>` after changing source/configuration/code; mismatched stages are intentionally not resumed.

## Values requiring later experimental validation

* Scale: 50k and source caps 20k/15k/15k; optional 25k caps 10k/7.5k/7.5k. Neither scale is claimed optimal. Source candidate budgets, grouping proxies and resulting language/domain distributions need research review.
* Split seed 42; 70/10/10/10 allocation; exact-group selection. Cross-seed variability is unmeasured.
* Entire LF ontology/cue vocabulary, correlated LFs, abstention handling, Snorkel 500 epochs, and selection of Snorkel versus majority targets. Controlled compliance is not external natural-user validation.
* Pinned DeBERTa-base, frozen encoder (0 unfrozen layers), dropout 0.1, maximum length 128, batch size 16, learning rate 1e-5, weight decay 0.01, task weights 1/1/1, M3 5 epochs. The measured truncation tradeoff remains an ablation question.
* FGSM epsilon=0.0005, lambda_adv=0.5, M4 3 epochs, adversarial validation batch size 4, and combined clean/FGSM weak-validation selection. Optional candidates: epsilon 0.0001/0.0005/0.001 and lambda 0.25/0.5/1.0. No sweep was run.
* MC passes=20, temperature iterations=100, ECE bins=15, per-head scalar/shared threat temperature assumption, uniform composite-entropy weighting. Optional 10/20/30 ranking/runtime diagnostic is not run. No uncertainty threshold exists.
* Optional augmentation is off; requested count 1,500, cap 1,500, ratio limit 5% of natural train, five scarce scenarios, placeholder lexicons. Its benefit and possible template shortcut effects are unmeasured.
* Optional auxiliary cap=1,000 English records and its limited supported structured types/exact-span scoring. It is not a full general-PII or medical detector evaluation.
* Controlled scenario definitions and future risk/policy contracts are operational research choices, not validated enforcement policies. Future M6 must validate any decision thresholds/rules separately.

## Experiments not run

No full pretrained 50k M3/M4 training, natural M5 calibration experiment, natural untouched-test evaluation, controlled held-out model evaluation, 25k/50k ablation, FGSM grid, MC sweep, or optional AI4Privacy dataset evaluation has been run. This local environment reports CUDA unavailable. There is no fabricated checkpoint or metric for those experiments.
