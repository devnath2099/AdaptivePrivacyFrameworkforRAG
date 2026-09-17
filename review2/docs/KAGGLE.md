# Run the complete M1–M6 pipeline on Kaggle

Use the provided `kaggle/Review2_M1_M6_Kaggle.ipynb` notebook and
`kaggle/review2_kaggle_bundle.zip` archive.

1. Create a private Kaggle Dataset and upload the ZIP. The ZIP contains source,
   configs, tests and documentation, not your Windows virtual environment or outputs.
2. Create a Kaggle Notebook and import the `.ipynb` file.
3. Attach the uploaded dataset using **Add Input**.
4. Enable a **GPU accelerator** and **Internet** in notebook/session settings.
5. Run the cells in order. The notebook copies source into `/kaggle/working`,
   installs dependencies without replacing PyTorch, checks CUDA, runs tests,
   constructs a GPU configuration, and runs M1 → M2 → M3 → M4 → M5 → M6 → frozen test.
6. Save a notebook version with outputs retained. Download the saved artifacts.

The notebook automatically handles either an extracted `review2/` directory or
an attached ZIP. It uses one GPU, including when Kaggle allocates two GPUs.

## Dataset size and 50k profile

The published Pritesh-2711/pii-bench release has **999,940 records**:
799,948 train, 99,990 validation, and 100,002 test. Original JSONL files total
1,391,776,736 bytes (about 1.39 GB). Counts were verified using the
[Hugging Face size API](https://datasets-server.huggingface.co/size?dataset=Pritesh-2711/pii-bench).
These are source counts, before integrity filtering and duplicate quarantine.

The earlier 200-per-split notebook was only a smoke/feasibility configuration:
600 source rows, not a substantive benchmark run. Merely changing that old
setting to 50,000 would request 150,000 source rows.

The notebook now loads `configs/kaggle_50k.json` and uses:

```python
RUN_NAME = 'kaggle_review2_50k'
PROFILE = 'kaggle_50k.json'
```

This profile scans all three pinned source files with seeded reservoir sampling,
keeping candidate pools of 100,000 train / 20,000 validation / 20,000 test.
After annotation validation and duplicate/group checks, an annotation-blind
whole-group hash sample selects exactly 40,000 official train / 5,000 validation /
5,000 test records. If a valid-record target cannot be met, M1 fails explicitly.
Approximately 20% of the 40,000 official train rows becomes calibration; actual
train/calibration counts depend on intact groups. Total valid records remain 50,000.

All four pretrained model/loss candidates train for one epoch at batch size 4.
M3 runs a matched standard continuation and FGSM continuation for one epoch,
plus all nine text-attack families. M4 compares raw/temperature/MC5/MC10.
M5/M6 retain all configured comparisons and controlled-profile limitations.
This is a bounded experiment, not the full parameter sweep in `full.json`.

Evaluation caps are configured before training: M2 validation 1,000; M3
validation 200; M4 calibration 500 and validation 250; frozen final test 1,000.
Each module saves its cohort counts and record IDs. Groups are never split to
fill a cap. M5/M6 consume the selected M4 validation cohort. The remaining
processed records are retained, but results must not be described as evaluation
on all 5,000 test rows. Calibration and test remain separate from training and
validation selection. Change evaluation caps before a new run, never in response
to test scores.

**Four hours is not guaranteed.** The runner currently uses one GPU even on
T4 x2; all comparison models and continuations run sequentially. No 50k GPU
benchmark has been executed here. Check observed training throughput before
committing to a session deadline. Checkpoint resume is supported. One epoch is
an initial experiment, not evidence of convergence. A larger dataset alone does
not establish useful detection accuracy.

Model artifacts occupy substantial disk space even on small datasets: the completed
local pretrained run retained approximately **18 GiB**. The notebook puts disposable
Hugging Face downloads under `/tmp` rather than saving them with notebook outputs.
Check current Kaggle storage/session allowances before larger sweeps. Do not create
a second ZIP of all checkpoints inside the same workspace unless enough disk remains.

MC counts can be expanded to `[5, 10, 20, 30, 50]` for a later run. Use a new run name
when changing any settings. The uncapped `full.json` configuration is not a safe
default for Kaggle because this transparent implementation materializes records and
MC arrays in memory. Start bounded and scale according to observed memory/runtime.

## Commands after setup

From `/kaggle/working`, the complete initial run is:

```bash
python -u -m review2.pipeline \
  --config review2/configs/kaggle_review2_50k.json \
  --output review2/outputs/kaggle_review2_50k \
  --evaluate-test
```

The notebook's execution cell improves on this command by skipping completed stages
after validating their manifests. If M1/M2 are complete and M3 was interrupted:

```bash
python -u -m review2.pipeline \
  --config review2/configs/kaggle_review2_50k.json \
  --output review2/outputs/kaggle_review2_50k \
  --stages m3 m4 m5 m6 --evaluate-test
```

If all development stages completed and only final evaluation remains:

```bash
python -u -m review2.pipeline \
  --config review2/configs/kaggle_review2_50k.json \
  --output review2/outputs/kaggle_review2_50k --test-only
```

## Outputs and resume

Artifacts live at `/kaggle/working/review2/outputs/kaggle_review2_50k/`.
The notebook displays status, detection comparisons, reliability selection and the
calibration SVG. Its full console is also saved under `/kaggle/working`.

Across sessions, attach the previous saved output and copy the complete run directory
back to the same path before running. Keep the same code/configuration. Retain
`best.pt`, `last.pt`, exported models, configs and all manifest-covered files; altering
or deleting them invalidates downstream lineage checks. Do not rerun completed
upstream stages just to resume a downstream stage: changed timing artifacts can
change their hashes. The provided notebook handles stage selection automatically.

M6 profiles are controlled fixtures until M7/M8 validation measurements exist. No
enforcement, RAG or real privacy guarantee is implemented by this notebook.

Notebook Python cells and archive integrity were checked locally. It has not been
executed against a Kaggle account/GPU in this session.

Platform references: [Kaggle kernel configuration](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md),
[NVIDIA's Kaggle accelerator setup](https://docs.nvidia.com/datascience/deployment/nightly/platforms/kaggle/).
