"""
Simple Streamlit UI for the underwriting review queue. Lets a reviewer
pull the next ambiguous application, see the AI's draft recommendation
and the precedent it's based on, and approve/deny/request more info --
resuming the real LangGraph interrupt, not a simulation of one.

Also shows clear (auto-decided) cases, with a free batch button to
process them -- a real bank needs an auditable record of every decision,
not just the ones a human reviewed.

Run with: streamlit run app.py
"""

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import pandas as pd

from src.features import compute_features
from src.routing import classify_complexity
from src.review_queue import get_pending, get_decided, start_application, resume_application

st.set_page_config(page_title="Underwriting review queue", layout="wide")
st.title("Underwriting review queue")

DATA_PATH = "data/broome_visions_hmda_2024.csv"

# Threshold validated via threshold_analysis.py against all 312 real eval
# cases -- 0.78 was the empirical optimum, not the naive 0.5 default.
APPROVAL_RECOMMENDATION_THRESHOLD = 0.78


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH).to_dict(orient="records")


rows = load_data()
pending = get_pending()
decided = get_decided()
# [-1] not [1] -- thread_ids can be "app_3" (CLI scripts) or
# "streamlit_app_3" (this UI); -1 reliably grabs the trailing id either way.
already_touched_ids = {int(tid.split("_")[-1]) for tid in list(pending.keys()) + list(decided.keys())}

# --- Pull next ambiguous application ---
st.subheader("Pull a new application for review")
col1, col2 = st.columns([3, 1])
with col1:
    st.write("Pulling an ambiguous application makes real API calls (embedding + 2 LLM calls, a few cents).")
with col2:
    if st.button("Pull next ambiguous application", type="primary"):
        next_id = None
        for i, row in enumerate(rows):
            if i in already_touched_ids:
                continue
            if classify_complexity(compute_features(row)) == "ambiguous":
                next_id = i
                break
        if next_id is None:
            st.warning("No more unprocessed ambiguous applications found.")
        else:
            with st.spinner(f"Running application {next_id} through the graph..."):
                result = start_application(next_id, rows[next_id])
            if result["status"] == "paused":
                st.success(f"Application {next_id} is now pending review.")
            else:
                st.info(f"Application {next_id} was auto-decided (not ambiguous): {result['assigned_queue']}")
            st.rerun()

st.divider()

# --- Process all clear cases (free, no API calls) ---
st.subheader("Process clear cases")
st.write(
    "Clear cases never touch the AI -- this is free and processes every "
    "untouched clear case immediately, the way a real bank would auto-clear "
    "easy decisions without a human looking at each one. Every one is still "
    "logged below for the audit trail."
)
if st.button("Process all clear cases (free)"):
    count = 0
    for i, row in enumerate(rows):
        if i in already_touched_ids:
            continue
        if classify_complexity(compute_features(row)) in ("clear_approve", "clear_deny"):
            start_application(i, row)
            count += 1
    st.success(f"Processed {count} clear cases -- zero API cost, all logged below.")
    st.rerun()

st.divider()

# --- Pending queue ---
st.subheader(f"Pending review ({len(pending)})")

if not pending:
    st.write("No applications currently pending review.")
else:
    thread_ids = list(pending.keys())
    selected = st.selectbox("Select an application to review", thread_ids)

    if selected:
        payload = pending[selected]
        st.markdown(f"### {selected}")

        c1, c2 = st.columns(2)
        with c1:
            st.metric("AI approval likelihood", f"{payload['ai_approval_likelihood']:.0%}")
        with c2:
            recommendation = "Approve" if payload["ai_approval_likelihood"] >= APPROVAL_RECOMMENDATION_THRESHOLD else "Deny"
            st.write("**AI recommendation:**", recommendation)

        st.write("**Decision summary:**")
        st.write(payload["decision_summary"])

        if payload["adverse_action_reasons"]:
            st.write("**Draft adverse action reasons:**")
            for r in payload["adverse_action_reasons"]:
                st.write(f"- {r}")

        st.write("**Supporting evidence:**")
        for e in payload["supporting_evidence"]:
            st.write(f"- {e}")

        st.divider()
        st.write("**Your decision:**")
        notes = st.text_area("Notes (optional)", key=f"notes_{selected}")

        b1, b2, b3 = st.columns(3)
        with b1:
            if st.button("Approve", type="primary", key=f"approve_{selected}"):
                result = resume_application(selected, "approved", notes)
                st.success(f"Resumed. Final queue: {result['assigned_queue']}")
                st.rerun()
        with b2:
            if st.button("Deny", key=f"deny_{selected}"):
                result = resume_application(selected, "denied", notes)
                st.success(f"Resumed. Final queue: {result['assigned_queue']}")
                st.rerun()
        with b3:
            if st.button("Needs more info", key=f"more_info_{selected}"):
                result = resume_application(selected, "needs_more_info", notes)
                st.success(f"Resumed. Final queue: {result['assigned_queue']}")
                st.rerun()

st.divider()

# --- Full decision log: auto-decided AND human-reviewed ---
n_auto = sum(1 for d in decided.values() if not d.get("required_human_review", True))
n_human = len(decided) - n_auto
st.subheader(f"All decisions ({len(decided)} total -- {n_auto} auto-decided, {n_human} human-reviewed)")
if decided:
    log_rows = [
        {
            "thread_id": tid,
            "required_human_review": d.get("required_human_review", True),
            "final_action": d["decision"]["final_action"],
            "assigned_queue": d["assigned_queue"],
            "notes": d["decision"]["notes"],
        }
        for tid, d in decided.items()
    ]
    st.dataframe(pd.DataFrame(log_rows), use_container_width=True)
else:
    st.write("No decisions recorded yet.")