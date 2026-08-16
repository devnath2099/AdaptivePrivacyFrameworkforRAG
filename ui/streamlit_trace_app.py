"""Review 1 pipeline TRACE view -- small batch, before/after per stage.

Run with:
    streamlit run ui/streamlit_trace_app.py

Unlike streamlit_app.py (which shows aggregate metrics/charts per module),
this page picks a small batch of records and walks them through M1 -> M2
one stage at a time, showing exactly what each record looked like BEFORE
and AFTER every stage. All computation is delegated to
pipeline_bridge.run_batch_trace_for_ui, which calls the same underlying
M1/M2 functions used everywhere else -- this file only renders.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline_bridge as bridge  # noqa: E402

STAGE_ICONS = ["1", "2", "3", "4", "5", "6", "7"]

st.set_page_config(page_title="Review 1: Pipeline Trace", layout="wide")
st.title("Deep Learning Based Adaptive Privacy Framework for RAG - Module 1 and Module 2")
st.caption("Follow a small batch of records through every M1 -> M2 stage in order.")

cfg = bridge.get_config()

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.header("Batch configuration")
dataset_key = st.sidebar.selectbox(
    "Dataset", options=list(cfg.datasets.keys()),
    format_func=lambda k: f"{k} ({cfg.datasets[k]['domain']})",
)
batch_size = st.sidebar.slider("Batch size (small, for readability)", min_value=1, max_value=6, value=3)

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
        rows = [row[0] for row in csv.reader(io.StringIO(content)) if row]
        uploaded_rows = rows[:batch_size]
    else:
        uploaded_rows = [l.strip() for l in content.splitlines() if l.strip()][:batch_size]
    st.sidebar.caption(f"{len(uploaded_rows)} row(s) parsed from upload")

run_clicked = st.sidebar.button("Run trace", type="primary")

if "trace" not in st.session_state:
    st.session_state.trace = None

st.sidebar.caption(
    "Note: with such a small batch, LF coverage and the generative model's "
    "fit are illustrative of the *mechanics* only -- not representative of "
    "full-scale (40+ record) statistical quality."
)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if run_clicked:
    progress_area = st.empty()
    stage_status = {}

    def on_stage(stage, status, payload=None):
        stage_status[stage] = status
        lines = [f"- {'done' if s == 'completed' else 'running...'} {k}" for k, s in stage_status.items()]
        progress_area.markdown("**Pipeline progress**\n\n" + "\n".join(lines))

    with st.spinner("Running batch through M1 -> M2..."):
        trace = bridge.run_batch_trace_for_ui(
            cfg, dataset_key, batch_size, manual_text or None, uploaded_rows, on_stage=on_stage
        )
    progress_area.empty()
    st.session_state.trace = trace

trace = st.session_state.trace

# ---------------------------------------------------------------------------
# Render: one block per stage, before/after side by side, per record
# ---------------------------------------------------------------------------
if trace is None:
    st.info("Configure a batch in the sidebar and click **Run trace**.")
else:
    stages = trace["stages"]
    n_records_initial = len(stages[0]["after"])

    st.success(f"Traced {n_records_initial} record(s) through {len(stages)} pipeline stages.")

    for i, stage in enumerate(stages):
        icon = STAGE_ICONS[i] if i < len(STAGE_ICONS) else str(i + 1)
        with st.expander(f"Stage {icon}: {stage['name']}", expanded=(i < 2)):
            if stage.get("stats"):
                st.caption(f"Stage stats: {stage['stats']}")

            before_list = stage["before"]
            after_list = stage["after"]
            n = max(len(before_list), len(after_list))

            for idx in range(n):
                b = before_list[idx] if idx < len(before_list) else None
                a = after_list[idx] if idx < len(after_list) else None
                label = None
                if isinstance(a, dict):
                    label = a.get("record_id") or a.get("query_text")
                elif isinstance(b, dict):
                    label = b.get("record_id") or b.get("query_text")
                st.markdown(f"**Record {idx + 1}**" + (f" -- `{label}`" if label else ""))

                col_before, col_after = st.columns(2)
                with col_before:
                    st.caption("BEFORE")
                    if b is None:
                        st.write("_(none -- this is the first stage)_")
                    else:
                        st.json(b, expanded=False)
                with col_after:
                    st.caption("AFTER")
                    if a is None:
                        st.write("_(record did not survive this stage)_")
                    else:
                        st.json(a, expanded=False)
                st.divider()

    st.header("Final state")
    final_records = trace["final_records"]
    st.write(f"{len(final_records)} of {n_records_initial} record(s) reached the end of the pipeline "
             f"(the rest were removed as duplicates in Stage 3).")
    if final_records:
        options = {f"{i}: {r.record_id}": i for i, r in enumerate(final_records)}
        choice = st.selectbox("Inspect one record's full final state", options=list(options.keys()))
        idx = options[choice]
        view = bridge.record_inspection_view(final_records[idx], idx, trace["dimension_results"])
        st.json(view)