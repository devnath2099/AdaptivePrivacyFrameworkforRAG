# Implemented algorithms and code correspondence

These describe the actual code flow. Functional module identifiers remain M1–M6.

## M1

**Input:** pinned external JSONL records with BIO labels or typed character spans;
optional application/control records; split seed and source configuration.
**Output:** canonical records, four split files, label map, source/split manifests,
annotation integrity, entity/source statistics and optional RAG corpus.

1. Load a checksum-verified local source or acquire a pinned remote release.
2. Preserve original text and source annotations. Locate token offsets; optionally
   handle explicit WordPiece continuations and repair orphan I labels.
3. Convert BIO runs to half-open typed character spans. Reject invalid or overlapping
   spans; audit configured quarantine instead of silently discarding errors.
4. Union rows connected by normalized text or source group. Quarantine components
   crossing official splits. Reject contradictory annotations for identical text.
5. Deduplicate exact text. Assign whole components, preserving official validation
   and test, and carving calibration from official train.
6. Assert record/text/group/component isolation, count source/entity distributions,
   write records, reports and file checksums. Return the M1 manifest hash.

**Code:** loaders.py → normalization.py → validation.py → splitting.py → statistics.py → run.py.
**Validation:** round-trip spans, transitive duplicate groups, conflicting annotation
rejection, official conflict quarantine, reproducibility, and leakage tests.

## M2

**Input:** M1 train/validation, externally defined type inventory, model/loss/length candidates.
**Output:** clean token classifiers, best/last checkpoints, validation predictions,
coverage analysis, strict span metrics and model/loss comparisons.

For token j, z_j is the label logit vector, y_j its BIO target, w_y its training-only
class weight, and V the unmasked positions. CE uses w=1; weighted CE uses normalized
inverse-square-root training frequency. The weighted objective is
`sum(j in V, w[y_j] * -log softmax(z_j)[y_j]) / sum(j in V, w[y_j])`.

1. For each DeBERTa/lighter-transformer and CE/weighted-CE candidate, reset the seed.
2. Measure train truncation and affected-gold-span rates at candidate lengths.
   Choose the shortest length under the configured span-truncation tolerance,
   or the largest candidate and report its remaining truncation.
3. Tokenize with original character offsets. Mask special/padding positions and
   boundary-straddling tokens; assign BIO labels to representable entity tokens.
4. Load an existing compatible last checkpoint, including optimizer, epoch/batch,
   best score, early-stopping counter and all RNG states, or initialize.
5. For each epoch, produce a deterministic train permutation. For each batch:
   forward → masked CE → backward → gradient clipping → AdamW update → periodic save.
6. Reconstruct validation typed spans. Save best checkpoint on improved strict F1;
   stop after configured patience. Export best weights/tokenizer.
7. Compare strict precision/recall/F1, macro/per-type/per-source results, rare-type
   recall, untyped PII span recall, non-PII-record recall, inference and training cost.
8. Keep the best validation DeBERTa loss variant as the primary handoff candidate,
   while reporting the best overall architecture for manual review.

**Code:** coverage.py, alignment.py, loss.py, training.py, inference.py, metrics.py, run.py.
**Failure criterion:** inadequate entity recall, unrepresentable/truncated annotations,
or no accuracy benefit sufficient to justify DeBERTa cost. Token accuracy is not the
model-selection metric. Tiny random models are test infrastructure only.

## M3

**Input:** selected M2 checkpoint and train/validation records; epsilon and lambda grid.
**Output:** FGSM checkpoints and clean/textual robustness comparisons.

For raw input embedding H, define `delta = epsilon * sign(d L / d H) * mask`.
The mask excludes padding and positions without a supervised token label.
The update objective is `L_clean + lambda_adv * L(H + delta)`.

1. Build deterministic obfuscations inside externally annotated validation spans,
   updating all subsequent character offsets and retaining parent/split identifiers.
2. Evaluate the clean checkpoint on clean and each applicable attack set.
3. For each epsilon/lambda pair, restart from the same M2 checkpoint and seed.
   Also continue standard training with lambda=0 for the same number of epochs,
   using the same validation selection criterion as a matched control.
4. For each train batch: clean loss; a detached raw-embedding forward to obtain
   embedding gradient; bounded FGSM perturbation; adversarial forward; combined
   objective backward; optimizer update.
5. Select epochs and configurations using the mean of clean and pooled textual-
   attack validation strict F1. Export the best robust checkpoint.
6. Report attack-specific strict F1 and attack success only among changed entities
   that were correctly detected before the attack. Undefined denominators are null.

**Code:** attacks.py, fgsm.py, evaluation.py, run.py; M2 training loop is reused.
**Failure criterion:** worse clean detection without demonstrated robustness benefit.
Homoglyphs, separators and obfuscations simulate the same underlying entity; they
are not a human-certified semantic attack benchmark. Case changes exclude credentials.

## M4

**Input:** fixed M3 detector, calibration split, validation split and MC pass candidates.
**Output:** positive temperature, selected generic reliability records, comparisons,
calibration/MC logits, reliability and risk–coverage plot.

1. Freeze the inference path and collect detached calibration logits.
2. Parameterize temperature as `softplus(a) + 1e-6`; use LBFGS to minimize calibration
   token NLL. Reject nonfinite or worse objectives. Never optimize detector parameters.
3. On validation only, compare raw softmax, softmax(z/T), and MC mean softmax(z_t/T).
4. For MC, keep the model in evaluation mode and enable dropout modules only;
   restore every module's original mode afterward. Use nested prefixes of a fixed
   maximum-pass sample set for 5/10/20/30/50 comparisons.
5. Compute normalized predictive entropy, mutual information, variance and variation
   ratio; compare ECE, NLL, Brier, error AUROC/AP, AURC, stability and latency.
6. Select minimum validation AURC subject to the configured relative-latency limit.
   Report all candidates; this does not establish that MC dropout is beneficial.
7. Reconstruct spans from the selected probabilities. Entity confidence is the
   minimum token confidence; entity/input uncertainty is maximum normalized entropy.
   Pass this generic reliability contract to M5.

**Code:** calibration.py, uncertainty.py, metrics.py, run.py.
**Unit:** calibration metrics are token-level, with a separate gold-PII-only report;
entity confidence aggregation is an explicit convention, not entity-level calibration.
**Failure criterion:** no useful error discrimination or excessive latency.

## M5

**Input:** generic M4 entity/confidence/uncertainty records and traceable mapping config.
**Output:** interchangeable risk records, mass/conflict traces and method comparisons.

Let Theta={Low, Medium, High, Critical}. Focal subsets use four-bit masks.
For a mapped entity tier r and confidence c, assign `m({r})=c`, `m(Theta)=1-c`.
The tier mapping is a configurable assumption, never benchmark gold.

1. Group all outputs of the same detector as dependent evidence. Pool their masses
   using a convex mixture (uniform unless explicit positive weights are configured).
2. Require documented independent group names before applying DS across groups.
3. For independent masses m1,m2 compute `K=sum(A intersect B empty) m1(A)m2(B)`.
   For nonempty C, `m(C)=sum(A intersect B=C) m1(A)m2(B)/(1-K)`.
   At the configured near-total-conflict bound return full ignorance and flag fallback.
4. Calculate belief, plausibility and pignistic probabilities. DS scalar score is
   expected normalized ordinal tier under the pignistic distribution.
5. In parallel, compute max and weighted averages of confidence-discounted tier scores.
   Return the same outer risk interface for all methods.
6. Compare sensitivity to confidence shifts, repeated evidence, conflict and runtime;
   M6 additionally compares each aggregator's downstream decisions.

**Code:** evidence.py, dempster_shafer.py, aggregation.py, experiments.py, run.py.
**Restriction:** confidence and entropy are never two independent DS sources. With
one neural source, DS is a mass representation of pooled evidence, not a claim of
independent multi-source fusion. No risk ground truth exists in this repository.
**Failure criterion:** no advantage over transparent baselines in later validation.

## M6

**Input:** M5 risk, M4 uncertainty, policy parameter repository and validation profiles.
**Output:** selected policy parameters or explicit block fallback, constraint margins,
candidate audit, fixed/adaptive comparisons and sensitivity results.

For risk score r, uncertainty u=max(entropy,ignorance), and configured strength a,
set `P_required = r + (1-r)*a*u`. This conservative interpolation is a replaceable
design assumption. Profile each candidate from supplied validation observations:
minimum observed protection P, minimum utility U and maximum latency L.

1. Validate explicit policies or generate the Cartesian candidate grid. Each includes
   epsilon, category coverage, WPM/NPM/PPM routing and fallback parameters.
2. Validate measurement provenance and split. Controlled profiles require explicit
   permission in configuration; no epsilon-to-guarantee mapping is invented.
3. For every candidate, check P>=P_required, U>=U_min, latency<=limit, sufficient
   entity/category coverage and profile availability. Save every check and margin.
4. Among feasible policies choose minimum profiled latency, breaking ties by id.
5. If no policy is feasible, return block with violated constraints; do not falsely
   claim successful protection or utility. No enforcement executes in this module.
6. Compare no-privacy, each actually fixed policy, and adaptive selection. Fixed
   policies retain their identity and log violations rather than silently adapting.
7. Sweep uncertainty strength and utility minimum; save selected ids/fallback counts.

**Code:** policies.py, profiling.py, constraints.py, selection.py, experiments.py, run.py.
**Failure criterion:** infeasible/unstable policy selection or no benefit under future
measured privacy–utility–latency evaluation. Controlled tests validate the algorithm,
not the central thesis hypothesis. M7–M9 remain unimplemented.

## Split discipline and sources

Train fits model weights and class frequencies and supplies length-coverage analysis.
Validation selects models, FGSM parameters, reliability method and develops risk/policy settings. Calibration
fits temperature only. `final_evaluation.py` saves a frozen configuration and artifact
chain before reading final test rows. It never selects parameters from test metrics.

Method references: [PIIBench source repository](https://github.com/pritesh-2711/pii-bench),
[DeBERTa](https://arxiv.org/abs/2006.03654),
[FGSM](https://arxiv.org/abs/1412.6572),
[temperature scaling](https://proceedings.mlr.press/v70/guo17a),
[MC dropout](https://proceedings.mlr.press/v48/gal16.html).
DS and constrained optimization equations follow the supplied handoff; category
mappings, utility targets, uncertainty interpolation and controlled policy numbers
are configuration assumptions, not claims drawn from these papers.
