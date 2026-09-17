# Review-2 migration

The user request limits implementation to M1–M6 and overrides the handoff's older
recommendation to build M7–M9 prototypes. Folder names are functional identifiers;
no thesis title has been finalized.

The complete original handoff is preserved as `TECHNICAL_HANDOFF.txt` in this
directory. Its instructions are architectural reference material; the user's
explicit M1–M6-only scope takes precedence over its wider prototype recommendation.

Inspected before implementation: the root `src`, `configs`, `tests`, `scripts`,
`updated_pipeline`, `validated_pipeline`, the source caches, previous outputs,
and the complete supplied Technical Handoff. Existing modifications to
`datasetcount.py` and `ui/streamlit_trace_app.py` were left intact.

| Existing component | Disposition in review2 |
|---|---|
| validated/common.py hashing and atomic JSON output | Adapted in common.py; add code-content hash and stage file hashes |
| validated/corpus.py duplicate union and isolation checks | Adapted to typed spans, official splits, transitive groups and quarantine |
| validated/trainer.py checkpoint/RNG/optimizer state | Adapted for token classification, validation F1, early stopping |
| validated/adversarial.py raw-embedding FGSM | Adapted for masked BIO token loss, special-token exclusion |
| validated/calibration.py positive temperature, MC dropout | Adapted to externally provided hard BIO labels and detached logits |
| Existing Python environment and pretrained model cache | Reused; no new virtual environment |
| Existing QA corpora | Supported only as optional application/control inputs, never PII supervision |
| Snorkel labels, sensitivity/intent/threat heads and checkpoints | Incompatible with the new supervision contract; not reused |
| Old risk-to-policy if/else engine | Replaced by candidate enumeration and constrained minimization |
| Existing UI, cloud mechanisms, enforcement/RAG | Outside this request; left unchanged |

## Preparation choices made explicit

PIIBench release `713f76a486e83c498ab173076abc9cca63918ae3` is pinned.
Its `label_mapping.json` is retained in acquisition provenance. Never substitute
the paper's entity count for the downloaded release. Canonical types are the union
of the release inventory and observed annotations, with optional explicit aliases.

Some records contain WordPiece continuation markers. With
`wordpiece_alignment=true`, remove only the `##` prefix while locating the original
character offsets, and require the continuation to be contiguous. With
`repair_orphan_i=true`, interpret an orphan `I-X` as a new `B-X`, keeping X unchanged.
Count repairs, preserve source labels, and quarantine remaining misalignments.
These flags are visible in every saved configuration and manifest. No value is
invented to fill a source annotation gap.

Official split conflicts in a connected duplicate/group component quarantine the
whole component. Official test/validation records are never reassigned to train.
Calibration is carved out of official training components by seeded source-aware
hash. Near-duplicate paraphrases without source groups are a residual limitation.

The smoke and feasibility samples use bounded official-file prefixes. They are
not representative benchmark samples, and cannot support thesis performance claims.
The full configuration removes that cap. No new annotation protocol or weak labels
are introduced.

M5 category tiers and M6 profiles are explicitly documented design/test assumptions.
The handoff does not supply authoritative risk labels or actual M7/M8 measurements.
The code accepts replaceable mappings and measured validation profiles; the included
controlled profiles exercise the optimizer but establish no privacy guarantee.
