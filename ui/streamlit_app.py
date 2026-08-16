"""Review 1 (M1 + M2) research demonstration UI.

Run with:
    streamlit run ui/streamlit_app.py

This file only renders. All computation is delegated to
`pipeline_bridge.py`, which in turn calls the exact M1/M2 modules used
by `scripts/run_review1.py`. No intermediate values are simulated or
hard-coded here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline_bridge as bridge  # noqa: E402

M1_STAGE_LABELS = {
    "dataset_loading": "1. Dataset Loading",
    "schema_harmonization": "2. Schema Harmonization",
    "cleaning_normalization": "3. Cleaning / Normalization",
    "domain_assignment": "4. Domain Assignment",
    "deduplication": "5. Deduplication",
    "semantic_evidence_extraction": "6. Semantic Evidence Extraction (eta, delta, rho, epsilon)",
}
M2_STAGE_LABELS = {
    "privacy_dimension_setup": "1. Privacy Dimension Setup",
    "labeling_functions": "2. Labeling Functions",
    "lf_matrix_construction": "3. LF Matrix Construction (Lambda)",
    "generative_model_fitting": "4. Snorkel Generative Model Fitting",
    "posterior_inference": "5. Posterior Inference (L_w)",
}

st.set_page_config(page_title="Review 1: Privacy M1-M2 Demo", layout="wide")
st.title("Adaptive Privacy Management -- Review 1 Demo (M1 + M2)")
st.caption("Multi-Domain Privacy & Evidence Integration -> Snorkel-Based Privacy Label Synthesis")

cfg = bridge.get_config()

# ---------------------------------------------------------------------------
# Sidebar: user controls
# ---------------------------------------------------------------------------
st.sidebar.header("Run configuration")
dataset_key = st.sidebar.selectbox(
    "Dataset", options=list(cfg.datasets.keys()),
    format_func=lambda k: f"{k} ({cfg.datasets[k]['domain']})",
)
n_samples = st.sidebar.slider("Sample size", min_value=0, max_value=60, value=15)

st.sidebar.markdown("**Optional: add one record manually**")
manual_text = st.sidebar.text_area("Enter a query / text", value="", height=80)

st.sidebar.markdown("**Optional: upload a text/CSV file**")
uploaded_file = st.sidebar.file_uploader("Upload .txt or .csv", type=["txt", "csv"])
uploaded_rows = None
if uploaded_file is not None:
    content = uploaded_file.read().decode("utf-8", errors="ignore")
    if uploaded_file.name.endswith(".csv"):
        import csv
        import io
        reader = csv.reader(io.StringIO(content))
        rows = [row[0] for row in reader if row]
        uploaded_rows = rows[:20]
    else:
        uploaded_rows = [line.strip() for line in content.splitlines() if line.strip()][:20]
    st.sidebar.caption(f"{len(uploaded_rows)} row(s) parsed from upload")

run_choice = st.sidebar.radio("Execute", ["M1 only", "M2 only (needs M1 result)", "Full M1 -> M2 pipeline"])
run_clicked = st.sidebar.button("Run", type="primary")

if "m1" not in st.session_state:
    st.session_state.m1 = None
if "m2" not in st.session_state:
    st.session_state.m2 = None


# ---------------------------------------------------------------------------
# Stage tracker rendering helper
# ---------------------------------------------------------------------------
def render_stage_tracker(stage_labels: dict, title: str):
    st.markdown(f"**{title}**")
    placeholders = {name: st.empty() for name in stage_labels}
    for name, label in stage_labels.items():
        placeholders[name].markdown(f"- ⏳ Pending -- {label}")

    def on_stage(stage, status, payload=None):
        icon = {"running": "🔄", "completed": "✅"}.get(status, "⏳")
        placeholders[stage].markdown(f"- {icon} {status.capitalize()} -- {stage_labels.get(stage, stage)}")

    return on_stage


# ---------------------------------------------------------------------------
# Run pipeline on button click
# ---------------------------------------------------------------------------
if run_clicked:
    if run_choice in ("M1 only", "Full M1 -> M2 pipeline"):
        st.subheader("M1 execution")
        on_stage_m1 = render_stage_tracker(M1_STAGE_LABELS, "M1 stage status")
        m1_out = bridge.run_m1_for_ui(cfg, dataset_key, n_samples, manual_text or None,
                                       uploaded_rows, on_stage=on_stage_m1)
        st.session_state.m1 = m1_out
        st.session_state.m2 = None  # invalidate stale M2 result

    if run_choice in ("M2 only (needs M1 result)", "Full M1 -> M2 pipeline"):
        if st.session_state.m1 is None:
            st.error("Run M1 first (select 'M1 only' or 'Full pipeline').")
        else:
            st.subheader("M2 execution")
            on_stage_m2 = render_stage_tracker(M2_STAGE_LABELS, "M2 stage status")
            kept = st.session_state.m1["kept_records"]
            if not kept:
                st.warning("No records survived M1 (all duplicates or empty). Increase sample size.")
            else:
                m2_out = bridge.run_m2_for_ui(cfg, kept, on_stage=on_stage_m2)
                st.session_state.m2 = m2_out

# ---------------------------------------------------------------------------
# M1 results
# ---------------------------------------------------------------------------
m1 = st.session_state.m1
if m1 is not None:
    st.divider()
    st.header("M1 outputs")
    c1, c2, c3 = st.columns(3)
    c1.metric("Raw records", len(m1["all_unified_records"]))
    c2.metric("Duplicates removed", m1["dedup_stats"]["duplicates_removed"])
    c3.metric("Records kept (-> M2)", len(m1["kept_records"]))

    st.write("**Domain distribution**")
    st.bar_chart(m1["domain_distribution"])

    with st.expander("Sample raw -> harmonized -> cleaned record"):
        if m1["all_unified_records"]:
            sample = m1["all_unified_records"][0]
            st.write("Harmonized + cleaned record:")
            st.json(sample.to_dict())

    with st.expander("Semantic evidence (eta / delta / rho / epsilon) for first kept record"):
        if m1["kept_records"] and m1["kept_records"][0].evidence:
            st.json(m1["kept_records"][0].evidence.to_dict())

# ---------------------------------------------------------------------------
# M2 results
# ---------------------------------------------------------------------------
m2 = st.session_state.m2
if m2 is not None and m1 is not None:
    st.divider()
    st.header("M2 outputs")

    tabs = st.tabs(list(m2["dimension_results"].keys()))
    for tab, (dim, res) in zip(tabs, m2["dimension_results"].items()):
        with tab:
            diag = m2["diagnostics"][dim]
            cols = st.columns(4)
            cols[0].metric("Records", diag["n_records"])
            cols[1].metric("Mean entropy", f"{diag['mean_entropy']:.3f}")
            cols[2].metric("Mean confidence", f"{diag['mean_max_confidence']:.3f}")
            cols[3].metric("Uncertain records", diag["n_uncertain_records"])

            st.write("**Generative model backend:**", diag["generative_model_method"])
            st.write("**Label distribution (argmax of L_w)**")
            st.bar_chart(diag["label_distribution"])

            st.write("**LF coverage**")
            st.json(diag["lf_coverage"])
            st.write("**LF conflict / abstention**")
            st.json({"conflict": diag["lf_conflict"], "abstention": diag["lf_abstention"]})

# ---------------------------------------------------------------------------
# Record-level inspection panel
# ---------------------------------------------------------------------------
if m1 is not None and m1["kept_records"]:
    st.divider()
    st.header("Record-level inspection")
    kept = m1["kept_records"]
    options = {f"{i}: {r.record_id} ({r.domain})": i for i, r in enumerate(kept)}
    choice = st.selectbox("Select a processed record", options=list(options.keys()))
    idx = options[choice]
    record = kept[idx]

    dim_results = m2["dimension_results"] if m2 is not None else None
    view = bridge.record_inspection_view(record, idx if dim_results else None, dim_results)

    st.subheader("Text")
    st.write("Query:", view["query_text"])
    if view["context_text"]:
        st.write("Context/Answer:", view["context_text"])
    st.write("Normalized text:", view["normalized_text"])

    st.subheader("Evidence (E_A)")
    st.json(view["evidence"])

    if dim_results is not None:
        st.subheader("Weak privacy labels (L_w) for this record")
        for dim, probs in view["weak_labels"].items():
            st.write(f"**{dim}**")
            st.bar_chart(probs)
        with st.expander("Raw LF votes for this record"):
            st.json(view["lf_votes"])
    else:
        st.info("Run M2 to see this record's weak privacy labels and LF votes.")
