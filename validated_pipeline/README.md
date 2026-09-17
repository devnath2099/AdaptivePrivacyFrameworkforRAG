# Validated query-level privacy intelligence (M1-M5)

This independent revision implements the Review-2 pipeline for **Deep Learning-Based Adaptive Privacy Framework for Cloud-Based Retrieval-Augmented Generation**. The research targets are sensitivity, privacy-relevant intent and independent threat content. M1 supplies auditable query evidence. This is multi-domain privacy intelligence, not a medical PII detector. RAG enforcement and M6 policy selection are not implemented.

Read `INSPECTION.md` for the baseline audit and `HANDOFF.md` for executed checks and remaining experiments. Everything this revision writes stays under this directory. No baseline code, configs, scripts or outputs are imported or modified.

## Kaggle execution

From the repository checkout, with Internet enabled and a GPU selected:

```bash
cd validated_pipeline
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python scripts/run_validated_pipeline.py --profile overnight_50k
```

`PROJECT_ROOT`, if set, must name this `validated_pipeline` directory. Otherwise it is derived from the source file. All configured paths are relative to that root. To use a writable Kaggle checkout, place the repository in the notebook working directory before running these commands. No Windows or Colab paths are embedded in configuration.

The default command builds the exact corpus, extracts M1 evidence, fits M2 on natural training records only, prepares optional training augmentation, trains M3 for 5 epochs, trains M4 for 3 epochs from selected M3, then fits temperatures on calibration and reports M5 on validation from selected M4. It never evaluates the natural test split or the controlled held-out benchmark. It performs no hyperparameter search.

Use `--until data` or `--until m2` for explicit partial runs. These write `partial_requested_stop`, never a misleading complete status. An unavailable dataset, insufficient unique queries, infeasible whole-group quotas, missing spaCy model or incompatible resume is a hard error. No synthetic or model fallback occurs.

## Source and split contract

`configs/sources.v1.yaml` pins exact revisions and field mappings:

| Source | Natural cap | Query | Separate context | Grouping |
|---|---:|---|---|---|
| HealthcareMagic | 20,000 | input | output | conversation/source row |
| Natural Questions | 15,000 | query | answer | equal answer hash proxy |
| TAT-QA | 15,000 | questions[].question | table and paragraphs | table UID and equal context |

The TAT-QA official train/dev/test source files are pooled as **unlabeled query text**. These are not the revised pipeline's evaluation splits. QA answer labels and `test_gold` are unused. A direct source check found 16,552 questions and 15,764 normalized unique queries across those three files. Source licenses/limitations are recorded in YAML; raw datasets are not included in Git.

Upstream references: [HealthcareMagic](https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k), [NQ pairs](https://huggingface.co/datasets/sentence-transformers/natural-questions), [official TAT-QA](https://github.com/NExTplusplus/TAT-QA).

HealthcareMagic and NQ stream fixed candidate pools of twice their 50k-profile caps; TAT-QA loads its three source question files. Final corpus size is exactly the configured cap, not the entire candidate pool. Source caches retain content hashes. Deficits are reported without substituting records.

Normalization is NFKC plus whitespace collapsing, with casefold for duplicate comparison. Duplicate query aliases are unioned with source groups before selecting a canonical query. This conservatively links groups even when one duplicate record is removed. An integer allocation over group-size signatures selects whole groups for exact per-source 70/10/10/10 quotas. The remainder of candidate groups is excluded. NQ's derived pair dataset does not expose original Wikipedia IDs: equal-answer grouping is explicitly a proxy, not a guarantee that all questions from one Wikipedia page can be linked.

The default exact counts are 35,000 / 5,000 / 5,000 / 5,000. Natural test is saved as raw query/context records and sealed: no evidence labeling, diagnostics, LabelModel fitting, model selection or calibration uses it in the default run.

## Weak supervision

Only `query_normalized_text` enters M1/M2/M3. Direct regex spans and spaCy entities remain in evidence; they are not learned entity targets. LFs use request semantics, personal references and direct evidence, never source/domain. Generic medicine/finance is not automatically high. Unknown queries abstain; explicit general-information constructions can vote low and benign. Threat positives require request semantics. Neither named entities alone nor missing threat cues produce threat labels.

Every labeled record preserves LF names, votes including -1 abstentions, soft targets and observation masks. A true Snorkel LabelModel is fitted for each categorical target and each independent threat binary target on **train only**. Majority vote produces normalized vote fractions as a comparison. Per-source/domain diagnostics report both posterior summaries and coverage/abstention/overlap/conflict. All-abstain posteriors are stored uniformly for audit, but their masks exclude them from training, calibration and agreement metrics. Missing coverage never becomes a fabricated label.

The LFs are weak heuristic supervision and can be correlated or wrong. Snorkel fitting does not establish superiority over majority vote. Baseline aggregate artifacts have different input scopes, sources and taxonomies; the comparison artifact explicitly marks them not comparable instead of presenting a false paired improvement.

## Model, robustness and calibration

The configuration drives head construction, targets, masks, losses, checkpoint checks and metrics. Defaults: 3-class sensitivity, 5-class privacy intent, and 3 independent sigmoid threat outputs. Categorical soft-label cross entropy and masked BCE-with-logits use explicit YAML weights. No GradNorm claim is made.

The pinned DeBERTa encoder is frozen by default; head dropout remains active. Optional last-N transformer layers can be unfrozen explicitly. Train-query token diagnostics are written before model training: p50/p95/p99, percentages beyond 128/256, and analytical 2x token / 4x attention cost estimates. These estimates are not measured runtime or memory. Review these diagnostics before committing to 128 for a research experiment; no second 256 experiment is launched automatically.

M4 perturbs raw word embeddings, not already position-normalized encoder embeddings. It masks padding and detaches the attack tensor. Attack construction enables gradients even during evaluation. The training objective is clean loss + lambda_adv * adversarial loss. Best M3 uses clean validation loss; best M4 uses clean validation loss + lambda_adv * FGSM validation loss. Both are weak-target objectives, not gold task accuracy. Robustness statements are restricted to this tested embedding-space FGSM threat model and epsilon.

M5 enables standard and DeBERTa StableDropout modules, gathers 20 fixed MC logit samples, and fits a positive softplus temperature per categorical head plus one shared positive scalar for the three independent threat logits. It fits categorical MC mixture NLL and binary MC mixture BCE-with-logits on observed calibration targets. Validation NLL and population-bin ECE before/after use those same probability definitions. ECE reports expected agreement with soft weak labels; it does not establish natural-user correctness calibration. No improvement is assumed or forced.

Each validation record gets task-wise mean probabilities, confidence, entropy and per-output predictive variance. Composite uncertainty is the unweighted mean of task normalized entropies: categorical entropy/log(K), mean binary entropy/log(2). No uncertainty threshold is chosen.

## Resume and artifacts

Stage receipts hash artifacts and bind configuration, source manifest, code, split and checkpoint lineage. A mismatch fails; use a new `--output outputs/<name>` directory after an intentional configuration/code change. M3/M4 atomically save model/optimizer, epoch/batch offset, RNG states, selected loss/history and taxonomy. An interrupted batch interval is replayed from the last durable checkpoint. Tests check exact CPU replay after a mid-epoch interruption. Bitwise equivalence across different hardware/library versions is not promised; the manifest records installed versions.

Expected full-run tree (some entries are only created after their stage actually executes):

```text
outputs/overnight_50k/
  run_identity.json, manifest.json, status.json
  split_manifest.json, data.done.json
  corpus/{train,calibration,validation,test}.jsonl
  corpus/*.manifest.json
  labels/{train,calibration,validation}.jsonl
  labels/*_diagnostics.json, labels/*.manifest.json
  labels/label_models.pt, labels/baseline_comparison.json
  labels/augmented_train.jsonl
  m1_m2.done.json, token_diagnostics.json
  m3/{best.pt,last.pt,history.json}, m3.done.json
  m4/{best.pt,last.pt,history.json}, m4.done.json
  m5/calibration_metrics.json
  m5/validation_uncertainty.jsonl
  m5/validation_uncertainty.manifest.json, m5.done.json
```

## Optional experiments

```bash
# Scale ablations: validation experiments, not a claim of optimal corpus size
python scripts/run_validated_pipeline.py --profile scale_25k
python scripts/run_validated_pipeline.py --profile scale_50k

# Nine separately saved M4 runs, all from the same revised selected M3
python scripts/run_validation_sweep.py --run outputs/overnight_50k

# Optional MC 10/20/30 runtime/ranking diagnostic on at most 128 calibration records
python scripts/diagnose_mc.py --run outputs/overnight_50k --count 128

# Isolated general synthetic structured-PII span benchmark, never merged into M2-M5
python scripts/run_validated_pipeline.py --profile m1_auxiliary_pii_benchmark

# Explicit controlled held-out evaluation of a selected revised M4
python scripts/evaluate_controlled.py --run outputs/overnight_50k
```

Training augmentation is off by default. Enable it in a separate config/run if needed: at most 1,500 verified examples and at most 5% of natural train size. On the 25k profile, 1,500 exceeds that ratio: choose at most 875 explicitly. The generator cycles five scarce scenarios using placeholders, with one-hot verified targets marked `synthetic_verified`. It never uses Snorkel to pretend these labels were inferred. Template families, IDs and lexical choices are disjoint from held-out evaluation. The full controlled ontology also includes general, limited-topic and personal-data scenarios. Finite generic templates are deduplicated, so the benchmark generator records actual output count, which can be smaller than requested.

Controlled labels are construction-based operational definitions, not student-created gold. Future risk/policy fields are declared scenario contracts only. Results must be described as **controlled ontology-compliance evaluation**, not natural-user generalization or gold clinical evaluation.

The optional AI4Privacy profile reports **synthetic general-PII evidence evaluation** over configured structured types and exact provided value spans. Unsupported gold span types are counted, not silently included in recall. It is neither a medical benchmark nor part of the default overnight run.

## Tests

From the repository root using an environment with the requirements installed:

```bash
python -m pytest validated_pipeline/tests -q -o cache_dir=validated_pipeline/.pytest_cache --basetemp=validated_pipeline/outputs/test_tmp
python validated_pipeline/scripts/check_sources.py
python validated_pipeline/scripts/check_token_lengths.py --run outputs/overnight_50k
python validated_pipeline/scripts/summarize_m2.py --run outputs/overnight_50k_m2_v2
```

The unit/integration suite uses explicitly named offline fixtures and a tiny random DeBERTa config. Passing these tests does not mean the pretrained 50k training experiment has run.
