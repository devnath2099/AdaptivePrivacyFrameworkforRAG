# Review 1: Deep Learning-Based Adaptive Privacy Management for Cloud-Based RAG

**Phase 1, Review 1** (~15% of Phase 1): M1 (Multi-Domain Privacy & Evidence
Integration) + M2 (Snorkel-Based Privacy Label Synthesis), plus validation
and a Streamlit demonstration UI.

This is an M.Tech research prototype, not a production system. It runs on a
CPU laptop / Colab / Kaggle and has no microservices, Docker, or REST APIs.

## 1. Architecture (Review 1 scope)

```
Four datasets (HealthCareMagic-100k, FiQA-2018, HotpotQA, Natural Questions)
        |
   M1: Multi-Domain Privacy & Evidence Integration
        |  Dataset Loading -> Schema Harmonization -> Cleaning/Normalization
        |  -> Domain Assignment -> Deduplication -> Semantic Evidence Extraction
        v
   Unified dataset D  +  Evidence E_A = (eta, delta, rho, epsilon)
        |
   M2: Snorkel-Based Privacy Label Synthesis
        |  Privacy Dimensions -> Labeling Functions -> LF Matrix (Lambda)
        |  -> Generative Model -> Posterior Inference
        v
   Weak-label matrix L_w in [0,1]^(n x K), one per privacy dimension
```

M3-M9 (DeBERTa representation, adversarial robustness, uncertainty
quantification, Dempster-Shafer aggregation, conformal policy selection,
explainability, UI/deployment) are **not** implemented in Review 1.

## 2. M1 -- Multi-Domain Privacy & Evidence Integration

**Input:** heterogeneous raw records from the four configured datasets.

**Output:** a list of `UnifiedRecord` objects, each carrying:
- `record_id`, `domain`, `source_dataset`, `query_text`, `context_text`,
  `normalized_text`, `metadata`
- an `EvidenceBundle` E_A = (eta, delta, rho, epsilon):
  - **eta** -- named-entity evidence (spaCy NER; a rule-based
    capitalized-token heuristic is used only if the configured spaCy model
    isn't installed, and this is logged, never silent)
  - **delta** -- dependency/syntactic evidence (spaCy dependency parse;
    empty when the fallback heuristic above is active, since there is no
    real parser to fall back on)
  - **rho** -- regex/pattern evidence: email, phone, money, date, url,
    account-like numeric strings
  - **epsilon** -- contextual sentence embedding (SentenceTransformer
    `all-MiniLM-L6-v2` by default; a deterministic hashing-based
    pseudo-embedding is used only if the model can't be loaded, e.g. no
    internet access, and this is logged)

Evidence is **never** treated as a final privacy label -- it is only input
to M2's labeling functions.

Dataset loading tries the real HuggingFace dataset first; if that fails
(offline, gated dataset, rate-limited), each loader falls back to a small
deterministic **synthetic** generator so the rest of the pipeline stays
runnable. Every record is tagged `metadata.source_mode` = `"real"` or
`"synthetic_fallback"` so the two are never confused. In this sandboxed
evaluation environment (no HuggingFace Hub access), **all four datasets ran
via the synthetic fallback** -- this is expected and by design, not a bug.

Code: `src/m1_data_integration/{config,schemas,loaders,harmonizer,cleaner,
deduplicator,evidence,embeddings,pipeline}.py`

## 3. M2 -- Snorkel-Based Privacy Label Synthesis

**Input:** D + E_A from M1.

**Output:** `L_w`, a probabilistic weak-label matrix per privacy dimension,
with `L_w in [0,1]^(n x K)` and each row summing to 1 (for multi-class
dimensions) or independently in `[0,1]` per category (for the multi-label
`threat_content` dimension).

### 3.1 Privacy dimensions and label taxonomy (`configs/review1.yaml`)

| Dimension          | Type        | Labels |
|---------------------|-------------|--------|
| `entity_tags`         | multi-class | none, person, organization, location, contact_identifier |
| `sensitivity`         | multi-class | low, medium, high |
| `intent`               | multi-class | general_information, personal_information_request, medical_information_request, financial_information_request, identity_related_request |
| `disclosure_scope`     | multi-class | single_document, multi_document |
| `threat_content`       | **multi-label**, binary per category | re_identification, attribute_inference, membership_inference |

`threat_content`'s three categories are exactly as specified in the project
brief; no additional threat categories were introduced.

### 3.2 Labeling functions

Each dimension has 6-7 hand-written, semantically motivated labeling
functions (LFs) that read the evidence bundle and either vote for a class
or ABSTAIN (`src/m2_label_generation/labeling_functions.py`). Examples:
person/org/location entity votes and email/phone/account-id regex votes for
`entity_tags`; medical/financial/credential term matches for `sensitivity`;
domain- and question-pattern-based votes for `intent`; multi-hop/cross-doc
term matches and entity-count heuristics for `disclosure_scope`; and
term + evidence-combination votes for each `threat_content` category.

### 3.3 SynthesizeWeakLabels algorithm

Implemented per the authoritative spec (`src/m2_label_generation/lf_engine.py`,
`generative_model.py`, `weak_labels.py`):

1. Evidence is already attached to each document (from M1).
2. Build the LF matrix `Lambda in ({0..K-1} U {ABSTAIN})^(n x m)`.
3. Fit a generative model over the LFs to estimate LF reliability
   parameters and class priors.
4. Infer posteriors `nu_k = mu_k * PI_{j in O_i} P(Lambda(i,j) | y_i=k; theta_j)`
   and normalize: `L_w(i,k) = nu_k / sum_k' nu_k'`.
5. Return `L_w`.

**Generative model backend:** the real Snorkel `LabelModel` is used when the
`snorkel` package is installed. If it isn't (as in this sandboxed evaluation
environment), a documented from-scratch EM-style fallback
(`_fit_fallback` in `generative_model.py`) approximates the same posterior
formula -- this is explicitly a project-specific approximation, not a claim
to reproduce Snorkel's exact algorithm. Every diagnostics record and
run-summary line reports which backend (`snorkel_label_model` vs
`fallback_generative`) actually produced a given result.

**Numerical safeguard:** whenever the normalizing denominator would be zero,
`_safeguard_probs()` returns a uniform distribution over classes for that
row. This is purely a numerical safeguard, unrelated to Confident Learning
or Northcutt et al.

`threat_content` is handled as three independent binary
(`K=2`) SynthesizeWeakLabels runs (one per category), whose `P(category=1)`
columns are stacked into the final `(n x 3)` matrix.

## 4. Validation

`tests/test_m1.py`, `tests/test_m2.py`, `tests/test_integration.py` --
23 tests, all passing in this environment. Coverage includes: all four
datasets load; common schema fields exist; whitespace/Unicode/missing-value
normalization; duplicate removal; domain-label validity; evidence structure
and embedding-dimension consistency; every LF returns a valid class or
ABSTAIN; LF-matrix shape and label-space bounds; `L_w` shape, non-negativity,
row-sum-to-1 (multi-class) / [0,1] bounds (multi-label), and absence of
NaN/Inf; output present for all 5 dimensions; and M1-output-feeds-M2-input
integration.

Run:
```bash
PYTHONPATH=src pytest tests/ -v
```

## 5. Research-style diagnostics (no gold labels)

Since Review 1 has no human-annotated gold labels, only weak-supervision
diagnostics are reported (never an "accuracy" figure):
LF coverage / abstention / conflict per LF, per-dimension label distribution
under the current `L_w`, mean entropy and mean max-confidence of `L_w`, and a
count of "uncertain" records (max posterior below a configurable threshold,
default 0.6). See `src/m2_label_generation/diagnostics.py` and the printed
run summary from `scripts/run_review1.py`.

## 6. Inputs / Outputs

**Inputs:** `configs/review1.yaml` (dataset ids, sample limits, seed,
evidence/embedding settings, label taxonomy, output paths).

**Outputs** (written to `outputs/`):
- `m1_unified_dataset.jsonl` -- every unified record (incl. evidence) as JSON lines
- `m1_statistics.json` -- raw counts per dataset, domain distribution, dedup
  stats, evidence statistics
- `m2_lf_matrices/<dimension>_lambda_matrix.npy` -- LF matrix per dimension
- `m2_weak_labels/<dimension>_weak_labels.{npy,jsonl}` -- `L_w` per dimension,
  both as a NumPy array and a per-record JSON-lines file
- `m2_diagnostics.json` -- full diagnostics per dimension

## 7. Execution

Single entry point (M1 -> M2 -> validation-relevant artifact generation):
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm   # optional; falls back cleanly if skipped
python scripts/run_review1.py
```

Run tests:
```bash
PYTHONPATH=src pytest tests/ -v
```

Run the demonstration UI:
```bash
streamlit run ui/streamlit_app.py
```
The UI lets you pick one of the four datasets, a sample size, and optionally
enter a query manually or upload a `.txt`/`.csv` file. It runs M1, M2, or the
full pipeline; shows each stage's Pending -> Running -> Completed status;
displays the actual intermediate artifacts (raw/harmonized/cleaned record,
evidence, LF votes, LF matrix, coverage/conflict/abstention, generative-model
backend, posterior `L_w`); and includes a record-level inspection panel. The
UI (`ui/pipeline_bridge.py`) calls the exact same M1/M2 functions as
`scripts/run_review1.py` -- it contains no privacy-labeling logic of its own.

## 8. Configuration

All dataset ids, sample limits, seed, spaCy/embedding model names, regex
pattern list, and the full label taxonomy live in `configs/review1.yaml` --
nothing dataset- or label-specific is hard-coded in the source.

## 9. Reproducibility

`load_config()` seeds Python's `random` and NumPy from `configs/review1.yaml`
(`seed: 42` by default) at startup.

## 10. Research honesty / attribution

Established methods used here (not claimed as this project's contribution):
Snorkel / data programming (Ratner et al., 2016; Bach et al., 2019 --
Snorkel DryBell), spaCy's NLP pipeline (Honnibal et al.), regex pattern
matching, and SentenceTransformer contextual embeddings.

Project-specific design decisions (this project's own contribution, not from
any paper): the exact 5-dimension privacy taxonomy and label sets; the
specific labeling functions and their term lists; the synthetic-fallback
dataset generators; the from-scratch EM fallback approximation of the
generative model (used only when Snorkel isn't installed); the
hashing-based pseudo-embedding and rule-based NER/dependency fallbacks (used
only when their respective models aren't available); the uniform-distribution
numerical safeguard; and the overall M1/M2 artifact and UI structure.

No experimental results, accuracy figures, or test-pass claims are stated in
this README beyond what was actually executed and observed in this session
(23/23 tests passing; full pipeline run producing 142 records with the
sample config's limits, synthetic-fallback datasets, and fallback evidence/
embedding backends, since this sandboxed environment has no internet access
to HuggingFace Hub or the spaCy/SentenceTransformer model downloads, and
`snorkel` is not installed).

## 11. Project structure

```
configs/review1.yaml
src/
    m1_data_integration/   (config, schemas, loaders, harmonizer, cleaner,
                             deduplicator, evidence, embeddings, pipeline)
    m2_label_generation/   (taxonomy, labeling_functions, lf_engine,
                             generative_model, weak_labels, diagnostics, pipeline)
tests/  (test_m1.py, test_m2.py, test_integration.py, conftest.py)
scripts/run_review1.py     (single entry point)
ui/     (pipeline_bridge.py, streamlit_app.py)
outputs/                   (generated artifacts)
requirements.txt
LOC_REPORT.md
```
