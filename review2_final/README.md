# Fresh PIILO implementation: M1–M4

This directory is an independent implementation of the fresh thesis handoff. It
does not import or reuse the previous project's code, taxonomy, checkpoints,
calibration parameters, risk constants or outputs. Only the existing Python
environment and the public `microsoft/deberta-base` pretrained backbone are reused.

The boundary is PII in the **user query before RAG exposure**. Student essays are
the detector's supervision domain; they are not evidence of performance on real
RAG queries. Query-domain transfer remains a limitation to evaluate separately.
There is no retrieved-context sanitization pipeline.

**M5 is unresolved and unimplemented. M6–M9 are intentionally unimplemented.**
There are no placeholder risk labels, DS masses, severity constants, or policies.
Nemotron and all other auxiliary datasets are excluded from the baseline.

## Acquired release and audit

Publisher: [The Learning Agency](https://the-learning-agency.com/guides-resources/datasets/).
Release: [lburleigh/piilo-dataset, version 2](https://www.kaggle.com/datasets/lburleigh/piilo-dataset/versions/2),
updated 2024-10-03. License: CC BY 4.0. Attribution: PIILO © 2024 The Learning
Agency Lab; the Kaggle release credits Vanderbilt University and The Learning
Agency Lab and cites Holmes et al. (2023) and the 2024 Kaggle competition.
The metadata snapshots are saved alongside this README. The publisher explicitly
uses PIILO as the dataset name; the earlier conversation's naming caution is
superseded by this primary-source verification.

The release description discusses approximately 22,000 essays, but **the acquired
labelled file contains 6,807 documents**, not 22,000 accessible labelled records.
Only `train.json` is provided; no suitable official labelled validation/test files
were acquired. No hidden competition test data is used.

Verified native columns: `document`, `full_text`, `tokens`,
`trailing_whitespace`, `labels`. The raw SHA-256 is
`8276cd44f3b2eb357dfb405b3c5d8e9f821388e984cbf66e92e7df03f1b13117`.

| Category | Entities | Documents containing category |
|---|---:|---:|
| NAME_STUDENT | 1,365 | 891 |
| URL_PERSONAL | 110 | 72 |
| ID_NUM | 78 | 33 |
| EMAIL | 39 | 24 |
| USERNAME | 6 | 5 |
| PHONE_NUM | 6 | 4 |
| STREET_ADDRESS | 2 | 2 |

There are 4,992,533 native tokens, 945 PII-positive documents and 5,862 negatives.
BIO validation and exact text reconstruction pass without annotation repairs.
See `data/prepared/audit.json` for computed counts and duplicate checks.

## Splits and limitations

After inspection, 70/10/10/10 was selected to retain a substantial training pool
while separating calibration, model selection and final evaluation. This is a
document-level, rarest-label-first approximate multilabel allocation, not token
or sentence splitting. Positive/negative presence participates in stratification.
Exact duplicates are reduced to one exemplar; normalized-text groups cannot cross
splits. There is no author grouping field, so author-level independence and
semantic near-duplicate isolation are not established.

| Split | Documents | PII-positive |
|---|---:|---:|
| Train | 4,764 | 660 |
| Calibration | 681 | 96 |
| Validation | 683 | 96 |
| Test | 679 | 93 |

Only two documents contain addresses. One is allocated to training and one to
test. Calibration and validation address recall is **undefined**, reported as
null, not zero or a fabricated estimate. Phone and username supports are also
too small for strong generalization claims. Split support is reported in full.
The allocation prioritizes train/test coverage for the rarest categories; it
cannot make four-way stratification statistically adequate where data is absent.

The raw file and canonical records preserve text, tokens, native labels, native
whitespace flags, character spans, document IDs and source hashes. Source text is
not normalized for model input. Normalization is used only for duplicate grouping.
Model BIO labels are derived from observed entity types, with both B and I tags
for each type. Added valid-but-unobserved continuation labels are disclosed in
the audit; they are not additional PII categories.

## Model and experiment design

M2 uses `microsoft/deberta-base` token classification with unweighted cross-entropy.
The public backbone/tokenizer revision is pinned to
`0d1b43ccf21b5acd9f4e5f7b077fa698f05cf195`; new run manifests record source-code
hashes and the environment. GPU/CPU numerical results need not be bitwise identical.
Only the first subword of each native token contributes a label/loss. Documents
are divided into complete-word windows, retaining every token; window predictions
are reassembled before strict typed entity evaluation. There is no silent tail
document-tail truncation or overlap that double-counts tokens. A single native
token can itself exceed the subword budget (observed in validation document 8447,
token 336). Such tokens retain the first and last subwords up to the window budget;
their middle subwords are omitted **only from the model input**. Native text,
annotations, offsets and one supervised/scored position per token are preserved.
Coverage reports identify each affected token and its discarded subword count.
This bounded representation loses internal context and is a disclosed limitation,
not a silent data repair. Context across window boundaries
is limited and explicitly part of this simple baseline. Native tokens disappearing
under tokenization are represented by UNK and counted in the coverage report.

The headline metrics are exact typed entity precision, recall and F1, with per-class
support and macro F1 over supported classes. An invalid predicted I tag starts a
new entity, and invalid BIO document counts are disclosed. Gold BIO must be valid.
Negative-document false detections and positive documents with no detections are
also reported. `subsets.json` fixes nested approximate 25/50/75/100% training
subsets; validation/calibration/test membership never changes. Full per-class
recall is the rare-class report; no post-hoc category cutoff hides small classes.

M3 compares the selected original M2 detector, clean continued training (controls
additional optimization steps), and FGSM-trained candidates. Each candidate is
evaluated clean and under the same epsilon grid. FGSM is an untargeted white-box
embedding-space attack using evaluation gold labels for attack construction only.
It excludes padding/special tokens. This is **not a textual attack, privacy attack,
or certified robustness guarantee**. The training loss averages clean and attacked
cross-entropy; epsilon and training budgets are experiment parameters, not privacy
severity constants. Both clean and attacked results are retained, even if FGSM
hurts. A smaller M3 subset is optional, reproducible and reported with coverage.

M4 fits a scalar temperature on the calibration split only. For MC dropout it
scales the logarithm of the averaged predictive distribution, with a separate
temperature for each pass count. It does not confuse mean logits with mean
probabilities. MC dropout activates dropout layers while leaving the model in
evaluation mode otherwise. It reports predictive entropy, expected entropy and
mutual information from the **uncalibrated MC distribution**; these are detector
uncertainty measures, not gold privacy risk or proven DS ignorance.

Validation compares 5/10/20/30/50 passes using ECE, NLL, multiclass Brier, AURC,
error AUROC/AP, entity metrics and latency. Calibration metrics are native-token
metrics, not calibrated entity probabilities. PII-gold-token results are reported
separately to expose O dominance. Risk–coverage refers to **classification error**.
Latency excludes metric computation/temperature fitting; model/tokenizer loading
and preprocessing are not included. MC pass budgets use nested stochastic draws.
Token probabilities and uncertainty arrays are saved with document-index metadata.

Selection rules are explicit design choices in config/manifests, not empirical
claims: M2 chooses validation micro F1 (ties: smaller fraction); M3 chooses mean
clean/attacked validation micro F1 (ties: original detector); M4 chooses the fewest
passes within 0.01 absolute token NLL of the best MC candidate. Macro/per-class
metrics remain mandatory context; token NLL can favor the majority O class.
The deterministic M4 baseline is retained, so MC superiority is never assumed.

## Commands

The GPU notebook calls M2/M3/M4 directly in Python cells, with normal notebook
tracebacks. `tqdm.auto` displays document tokenization, training batches, evaluation
batches and MC-pass progress. Training bars show current and running mean loss;
intermediate lines at the first, every 50th and final batch report elapsed time
and estimated time remaining. Stage/fraction/epoch and checkpoint-save messages
identify the active work. CLI calls expose the same progress. After updating code
in an already-imported notebook, restart the kernel or reload modules before running.

Run from the parent repository directory. This package has no dependency on its
other source directories. Use Python 3.10+ and install `review2_final/requirements.txt`
in a suitable environment. A CUDA runtime is strongly recommended for experiments.
The local verified environment is recorded in run artifacts; no model results
from other project directories are loaded.

```powershell
# Reacquire the exact public release if raw data is absent.
python -m review2_final.acquire
python -m review2_final audit

# Already completed locally. Fresh output directory required if repeated.
python -m review2_final prepare --proportions 0.7 0.1 0.1 0.1

python -m pytest review2_final/tests -q
python -m review2_final smoke --output review2_final/runs/smoke_new

# Research runs: no hidden subset caps or synthetic substitutions.
python -m review2_final baseline --output review2_final/runs/baseline
python -m review2_final robustness --baseline review2_final/runs/baseline --output review2_final/runs/fgsm
python -m review2_final uncertainty --robustness review2_final/runs/fgsm --output review2_final/runs/uncertainty

# Only after all methods, hyperparameters and selections are final:
python -m review2_final final-test --baseline review2_final/runs/baseline --robustness review2_final/runs/fgsm --uncertainty review2_final/runs/uncertainty --output review2_final/runs/final_test
```

The local interpreter is `..\fypvenv\Scripts\python.exe` when inside this directory,
or `.\fypvenv\Scripts\python.exe` from the repository root.

Output directories cannot be silently overwritten. Split hashes are checked on
load. `final-test` freezes the selections and creates `TEST_OPENED.json` before
reading test records. Subsequent development commands reject that prepared dataset.
This workflow guard is not an access-control system: do not bypass it or tune from
test results. Smoke runs cannot open the research test. During M1, test annotations
are necessarily used for split construction/structural checks; after creation they
are not consumed by development experiments.

If final evaluation is interrupted, rerun the same `final-test` command with
`--resume`. It verifies frozen run/selection/comparison hashes and checkpoint
contents, and skips completed candidate evaluations. This does not reopen
development. Training itself saves the best checkpoint at each improved epoch;
there is no mid-epoch optimizer-state resume in this initial baseline.

Run outputs distinguish **smoke execution checks** from research experiments.
For the notebook's subprocess helper, stdout and stderr are streamed together and each command's complete
output in `logs/`. If a command fails, its exception includes the final 60 output
lines. A bare `CalledProcessError` is not enough to diagnose a training failure.
An existing output directory triggers `FileExistsError`, including after a failed
attempt. Preserve its log/artifacts and use a new output path for a fresh run;
point subsequent M3/M4 commands at the successful path. Do not delete prepared
splits or assume a CUDA memory error without the underlying traceback.
Random tiny models in unit tests validate mechanics only. No fabricated data is
used to report thesis results. Full learning-curve/FGSM/MC experiments require
actual training; successful unit or smoke tests do not establish accuracy or
robustness. GPU training is provided through `run_gpu.ipynb`.
