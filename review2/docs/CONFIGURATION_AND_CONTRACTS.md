# Configuration assumptions and artifact contracts

No number below is claimed to be scientifically optimal. Validation comparison is
the mechanism for challenging candidates; the final test is excluded from that loop.

| Parameter | Origin and interpretation |
|---|---|
| Seed 42 | Retained reproducibility convention; no claim of superiority |
| PIIBench commit | Actual inspected release; source label file retained in manifest |
| 60 records per official file | CPU feasibility budget, not representative sampling |
| Calibration carve-out 0.2 | Explicit engineering split choice within official train |
| Lengths 128/256/512 | Handoff candidates, measured on training data |
| Maximum span truncation 0.01 | Configurable coverage tolerance; residual truncation reported |
| CE and inverse-square-root weighted CE | Required loss comparison; weights fit only on train counts |
| Learning rate 2e-5, weight decay 0.01 | Initial fine-tuning candidates; not tuned by this tiny run |
| Tiny smoke learning rate 0.001 | Test-runtime setting for randomly initialized small networks |
| Epsilon 0.0001/0.0005/0.001, lambda 0.25/0.5/1 | Configurable candidates adapted from existing FGSM infrastructure |
| Matched lambda=0 continuation | Extra-training control; not a replacement for FGSM |
| MC 5/10/20/30/50 | Handoff candidates; nested validation passes measure cost/stability |
| 15 calibration bins | Configurable histogram resolution |
| MC maximum latency ratio 12 | Hardware-budget constraint, not a learned scientific constant |
| Unknown risk tier High | Explicit conservative test assumption; no type-specific authority |
| DS conflict limit 0.999999 | Numerical near-total-conflict safeguard returning ignorance |
| Uniform dependent-entity pooling | Avoids treating correlated model predictions as independent |
| M6 uncertainty strength 0.5 | Controlled design assumption; swept at 0/0.5/1 |
| Utility minimum 0.6 | Controlled requirement; swept at 0.5/0.6/0.8 |
| Policy epsilon 8/2/0.5 | Parameter-contract candidates only; no LDP enforcement claimed |
| Controlled P/U/L observations | Explicit optimizer fixtures, unrelated to measured privacy guarantees |

## M1 → M2

`train.jsonl`, `validation.jsonl`, `calibration.jsonl`, `test.jsonl` contain:

```text
record_id, source_id, source_record_id, text,
source_annotations,
canonical_spans: [{start, end, type, optional value}],
split, group_id, component_id, metadata
```

Offsets are half-open Unicode character positions. Source text is not cleaned after
annotation alignment. BIO labels are generated dynamically from `labels.json`.
Version/large provenance stays in manifests instead of every row.

Application corpus records are explicitly `pii_gold=false` and cannot feed M2 as
authoritative PII supervision. Legacy validated QA caches are hash-verified before reuse.

## M2 → M3 → M4

Best/last training checkpoints contain a compatibility contract, model state,
optimizer state, epoch and batch offset, best validation score, patience counter,
history, elapsed time and Python/NumPy/PyTorch/CUDA RNG states.
Exported `model/` directories contain model configuration, tokenizer and weights.
Each selection file identifies the checkpoint and maximum sequence length.

Original M2 state initializes every FGSM/control candidate. M4 does not update any
model parameter. Temperature fitting uses only detached calibration logits.

## M4 → M5

```text
record_id, source_id, split,
entities: [{start, end, type, confidence, calibrated_confidence,
            uncertainty, token_indices}],
confidence, uncertainty, method, model_hash, uncertainty_definition
```

`method` distinguishes raw, temperature and temperature+MC. The confidence field
contains the probability under that selected method; raw means T=1 and is explicitly
an uncalibrated baseline. Token uncertainty is normalized predictive entropy. Max
entropy and minimum span confidence are configurable implementation conventions
documented in the algorithm, not assertions of calibrated entity correctness.

## M5 → M6

```text
record_id, split, method, score, tier, ignorance, uncertainty,
entity_types, entity_count, type_diversity,
probabilities, belief, plausibility, masses,
conflicts, conflict_fallback, evidence_trace, assumption_basis
```

Non-DS methods set unavailable DS fields to null. The normalized score is a design
risk characteristic, not a probability that an actual privacy breach will occur.
Confidence/uncertainty never become separate independent DS sources.

## M6 → future enforcement

```text
record_id, split, action, policy or null,
required_protection, profile, feasible, fallback, reason,
candidates: [{policy_id, feasible, checks, privacy_margin,
              utility_margin, latency_ms}],
enforcement_executed: false
```

Policy schemas carry epsilon, coverage, protected categories, word/number/phrase
mechanism identifiers and safe fallback. They are interfaces for future M7. No
privacy budget is spent and no data transformation is performed by this project scope.

Measured profile rows require known policy IDs, validation split, experiment ID,
provenance, privacy/utility in [0,1] and nonnegative latency. Definitions and scale
of the measured privacy/utility characteristics must be agreed when M7/M8 exist.
The current minimum/maximum profiler is conservative over observed cases only and
provides no statistical out-of-sample bound.

## Reproduction and limitations

The stage manifest records configuration/hash, code revision and source-content
hash, random seed, artifact checksums and parent hash. Environment versions are
saved at run level. Timing comparisons are hardware/load-dependent; use an idle
dedicated machine for publishable latency measurements. This session's smoke and
CPU feasibility workloads overlapped, so their timing is diagnostic, not publication-grade.

The full configuration requires substantially more memory/storage than the bounded
example. The code materializes source records and model/MC arrays for transparency.
Large-scale streaming/memory optimization is not represented as already implemented.
This limitation affects feasible experiment size, not the underlying six algorithms.
