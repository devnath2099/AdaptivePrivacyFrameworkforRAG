# Implementation status

The fresh implementation is confined to this directory. No M5–M9 implementation
or auxiliary dataset has been added.

| Work | Status |
|---|---|
| Acquire and inspect actual PIILO release | Completed: publisher-linked Kaggle v2, 6,807 labelled documents |
| M1 validation, canonical records, provenance, grouped splits | Completed on the actual release |
| Nested 25/50/75/100% training membership | Created; fixed held-out splits |
| M2 DeBERTa baseline / learning-curve runner | Implemented |
| M3 FGSM training and clean/adversarial comparisons | Implemented, including clean continued-training control |
| M4 deterministic calibration and MC-dropout comparison | Implemented |
| Unit/integration tests | See `reports/tests.xml`; tiny random networks are test fixtures only |
| Pretrained real-data execution smoke | Passed through M2, M3 and M4; one optimization step per trained variant |
| Full learning-curve, robustness and uncertainty research experiments | Not run locally: CPU-only PyTorch, no CUDA GPU |
| Final test evaluation | Not opened |
| GPU workflow | `run_gpu.ipynb` and portable source bundle |

No final detector-quality, robustness-improvement, privacy-risk or RAG-performance
claim follows from the smoke run. The full experiment commands have no hidden
training caps. Their measured outputs must be obtained on suitable compute before
M1–M4 can be described as experimentally established.

The first smoke attempt exposed an installed tokenizer API difference and failed
before training; it was fixed and subsequent execution passed. Its incomplete
directory is retained for transparency. Reports and run manifests distinguish
checks from research results. The fixed split has no STREET_ADDRESS support in
calibration/validation; rare-category conclusions remain limited by the source.

The final pinned-model smoke artifacts are in `runs/smoke_verified/`. Its status
is `passed`. The original split hashes are checked again during packaging without
opening test annotations for model evaluation.

Subsequent Kaggle baseline failure: validation document 8447, native token 336
exceeds the model's subword window budget. The original encoder rejected this
case, which the small smoke sample did not include. The encoder now retains a
bounded head/tail representation and reports every affected token and omitted
subword count while preserving native annotations and one label per token.
The notebook checks all development-split tokenization before training. This
fix does not establish that a complete GPU training run succeeds.
