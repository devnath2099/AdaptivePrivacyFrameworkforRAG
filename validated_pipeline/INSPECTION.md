# Inspection before implementation

Initial git status was clean (Git required a command-local safe.directory override due to sandbox ownership). No baseline files were edited. The three requested project_sources Markdown files and AGENTS.md were not present in this checkout or the inspected parent locations.

The README describes the older M1/M2 prototype; the checkout also contains M3-M5, and updated_pipeline adds temperature scaling and an M6 prototype. Inventory: 134 Python files across both source, script and test trees, with 75 distinct content hashes. Most updated M1-M4 files duplicate the baseline.

Concrete mismatches:

* Four loaders: HealthcareMagic input/output, FiQA corpus text/title, HotpotQA question/context, Natural Questions query/answer. Load failures lead to synthetic generation. Cached rows do not preserve upstream conversation/document IDs. No TAT-QA adapter.
* cleaner.py concatenates query and context into normalized_text; evidence and dedup use that combined field. M3's dataset default is also normalized_text, while some launchers override text fields. There is no consistent query-only contract.
* Five heads: sensitivity 3, intent 5, disclosure_scope 2, entity_tags 4, threat_content 3. LFs include medical-term high sensitivity, default-low, domain-based disclosure, person/location re-identification, and absence-based negatives. Entity tags have already changed from the README's categorical description to multilabel.
* M1 defaults to 80/10/10 with no separate calibration split and record-level rather than document-group allocation. Label generation can operate over the unified corpus.
* M3 does not freeze its encoder by default. Checkpoints are plain state dictionaries without taxonomy, source, split or optimizer lineage. Trainer task weights/configuration are not consistently passed to the loss.
* M4 uses encoder.embeddings followed by inputs_embeds, risking embedding transforms twice. Revised code must perturb raw word embeddings. Existing M4 has useful gradient-enabled validation and memory cleanup, but static task contracts and no interruption-safe optimizer resume.
* updated temperature_scaling.py converts losses to Python floats before backward, reuses sensitivity logits during categorical fitting, and calculates per-record rather than population-bin categorical ECE. The revised implementation requires independent tests.
* updated run_pipeline.py launches M5 and M6, not the requested M1-M5 experiment. Reference-set generation requests subjective annotation and uses model predictions to prioritize examples.

Existing artifact metadata (read-only, not rerun or independently validated): outputs/m1_statistics.json reports 277352 records and checks_passed=false for count reconciliation; M3 smoke config records 10000 train/2000 validation, 5 epochs; M4 config records 50 train/10 validation, 1 epoch, epsilon=0.001, lambda=1.0; M5 records 20 passes from that M4 path. Baseline results are not revised-pipeline evidence. Full checkpoints and large outputs are not copied.

Revision strategy: isolated modules, one active-task configuration, query-only evidence and labels, train-only weak-label estimation, masked all-abstain targets, exact source quotas, group isolation, versioned controlled scenarios, raw-word-embedding FGSM, calibration-only temperature fitting, and content-checked resumability. Source feasibility is checked before training; deficits are fatal.
