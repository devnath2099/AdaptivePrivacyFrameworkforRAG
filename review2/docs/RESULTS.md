# Actual execution report

Results below are read from saved artifacts, not expected outcomes.

Tests: 45 executed; 0 failures, 0 errors, 0 skipped.
Artifacts: `outputs/tests.xml`, `outputs/pytest_result.txt`.

## verified_smoke

Completed stages: m1, m2, m3, m4, m5, m6. Final test evaluated: True.

Run kind: `external_data_tiny_random_model_smoke`. Dataset cap: 60 rows per official split.

| Split | Records | Gold entities |
|---|---:|---:|
| train | 41 | 99 |
| validation | 50 | 188 |
| calibration | 9 | 12 |
| test | 54 | 127 |

Accepted 154; quarantined 26; BIO repairs 10.
Release/canonical inventory: 82 entity types.

| M2 variant | Validation strict F1 | Recall | Inference ms/record | Training seconds |
|---|---:|---:|---:|---:|
| deberta_ce | 0.000000 | 0.000000 | 5.17 | 2.25 |
| deberta_weighted | 0.000708 | 0.005319 | 8.05 | 1.54 |
| distilbert_ce | 0.000000 | 0.000000 | 4.25 | 1.23 |
| distilbert_weighted | 0.000611 | 0.005319 | 5.12 | 1.43 |

| M3 variant | Clean validation F1 | Mean attack F1 (applicable families) |
|---|---:|---:|
| standard_continuation | 0.000000 | 0.000000 |
| fgsm_e0.0001_l0.5 | 0.000000 | 0.000000 |
| fgsm_e0.001_l0.5 | 0.000000 | 0.000000 |

Matched continuation control separates the extra training epoch from FGSM. Detailed paired attack metrics, eligible attack-success denominators and overhead are in `m3/`.

M4 selected `raw` on validation. Fitted T=0.115791; calibration NLL 4.140464 → 0.863367.

| Reliability method | ECE | NLL | Brier | AURC | Seconds |
|---|---:|---:|---:|---:|---:|
| raw | 0.726507 | 4.268755 | 0.976748 | 0.234089 | 0.16 |
| temperature | 0.167811 | 1.651478 | 0.475231 | 0.239769 | 0.16 |
| temperature_mc5 | 0.158612 | 1.623287 | 0.474556 | 0.240653 | 1.37 |
| temperature_mc10 | 0.156851 | 1.611735 | 0.472678 | 0.240617 | 3.07 |
| temperature_mc20 | 0.155739 | 1.605807 | 0.472260 | 0.239081 | 5.96 |
| temperature_mc30 | 0.155965 | 1.605063 | 0.472190 | 0.239258 | 8.63 |
| temperature_mc50 | 0.156452 | 1.604211 | 0.472257 | 0.239833 | 14.11 |

These are token metrics including O; PII-only metrics are also saved. Plot: `m4/reliability.svg`.

| Risk method | Mean validation risk score | Duplicate evidence max change |
|---|---:|---:|
| max | 0.002655 | 0 |
| weighted | 0.002440 | 8.67362e-19 |
| ds | 0.497184 | 2.22045e-16 |

Risk scores use a disclosed category-ordering assumption. No risk accuracy/superiority claim is possible without independent outcomes.

| Controlled policy mode | Constraint violations | Block fallbacks |
|---|---:|---:|
| no_privacy | 3 | 0 |
| weak | 2 | 0 |
| medium | 1 | 0 |
| strong | 0 | 0 |
| adaptive | 0 | 0 |

Detector-derived validation decisions: 50; block fallbacks: 0.
Controlled profile numbers are fixtures. Lower simulated cost/violations does not establish real privacy or RAG utility gains.

| Frozen final-test M2 variant | Strict F1 | Recall |
|---|---:|---:|
| deberta_ce | 0.002045 | 0.007874 |
| deberta_weighted | 0.001799 | 0.015748 |
| distilbert_ce | 0.003185 | 0.015748 |
| distilbert_weighted | 0.000789 | 0.007874 |

Final test results were not used for tuning. Near-zero scores on these small runs are failures to establish useful detection, not successful thesis results.

## pretrained_cpu

Run has no completion marker. Inspect run.log and failure.json before claiming completion.

Run kind: `pretrained_cpu_feasibility`. Dataset cap: 60 rows per official split.

| Split | Records | Gold entities |
|---|---:|---:|
| train | 41 | 99 |
| validation | 50 | 188 |
| calibration | 9 | 12 |
| test | 54 | 127 |

Accepted 154; quarantined 26; BIO repairs 10.
Release/canonical inventory: 82 entity types.

| M2 variant | Validation strict F1 | Recall | Inference ms/record | Training seconds |
|---|---:|---:|---:|---:|
| deberta_ce | 0.000000 | 0.000000 | 450.44 | 183.00 |
| deberta_weighted | 0.000000 | 0.000000 | 539.28 | 161.13 |
| distilbert_ce | 0.000000 | 0.000000 | 128.91 | 65.83 |
| distilbert_weighted | 0.000000 | 0.000000 | 177.52 | 60.67 |

## Failures and limits

- The first strict M1 pass quarantined WordPiece and orphan-I records; explicit audited normalization was added, and new runs were created.
- The initial smoke attempt failed at M4 because Matplotlib was absent. The SVG writer removes that dependency; the failure remains in `outputs/smoke/failure.json`.
- Python could not launch inside the sandbox; the existing environment worked with approved execution outside it. CPU-only PyTorch was used.
- The complete million-record benchmark and large hyperparameter sweeps were not run. Full configuration is supplied, with memory/storage requirements documented in README.
- Full inference currently truncates to the selected maximum length rather than implementing a new sliding-window methodology. Coverage loss is measured and reported.
- M5 risk mapping and M6 controlled profiles need manual methodological review and future measured validation; they are explicit replaceable assumptions.
- No M7–M9 code, mechanism enforcement, local RAG, or real end-to-end privacy/utility result is claimed.

## Artifacts and tree

All run artifacts are below `review2/outputs/`. `outputs/application_corpora/` contains 200 reused medical/public application records with no PII supervision claim.
See `TREE.txt` for the source tree and artifact layout. Checkpoint files are retained; outputs are git-ignored to avoid committing datasets and multi-GB weights.
