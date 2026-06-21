"""
The comprehensive final eval -- deferred on purpose until now, since the
full pipeline (including the real human-in-the-loop interrupt) needed to
exist for this to be a complete test. Runs ambiguous applications through
the real graph, auto-resuming each interrupt with a decision that mirrors
the AI's own recommendation -- NOT a real human judgment, just a
consistent automated baseline so these numbers are comparable to the
earlier 20-case pilot.

This also stress-tests the interrupt mechanism itself: every ambiguous
case pauses and resumes exactly once. If any pause/resume cycle fails,
that's a reliability problem worth knowing about, independent of
prediction accuracy.

Uses MemorySaver, not SqliteSaver -- persistence across a process restart
was already proven separately via start_review.py / resume_review.py.
This script tests reliability of the interrupt/resume mechanics at
scale, which an in-memory checkpointer validates just as well within one
continuous run.

LIMIT starts small to confirm the auto-resume loop itself works
correctly (this is new code) before committing to the full run.
"""

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from src.graph import build_graph
from src.features import compute_features
from src.routing import classify_complexity

LIMIT = 1000  # raise to 1000 (higher than the real ambiguous count) for the full run

df = pd.read_csv("data/broome_visions_hmda_2024.csv")
rows = df.to_dict(orient="records")

checkpointer = MemorySaver()
graph = build_graph(checkpointer)

results = []
interrupt_failures = 0

for i, row in enumerate(rows):
    features = compute_features(row)
    complexity = classify_complexity(features)
    if complexity != "ambiguous":
        continue
    if len(results) >= LIMIT:
        break

    actual_outcome = "originated" if row["action_taken"] == 1 else "denied"
    config = {"configurable": {"thread_id": f"eval_{i}"}}

    initial_state = {
        "application_id": i,
        "raw_application": row,
        "features": None,
        "risk_assessment": None,
        "complexity": None,
        "investigation_findings": None,
        "compliance_summary": None,
        "assigned_queue": None,
        "human_decision": None,
        "audit_trail": [],
    }

    paused = graph.invoke(initial_state, config=config)

    if "__interrupt__" not in paused:
        interrupt_failures += 1
        print(f"  WARNING: application {i} did not pause as expected, skipping")
        continue

    payload = paused["__interrupt__"][0].value
    likelihood = payload["ai_approval_likelihood"]
    auto_decision = {
        "underwriter_id": "eval_auto_baseline",
        "final_action": "approved" if likelihood >= 0.5 else "denied",
        "notes": "automated baseline for eval -- not a real human judgment",
        "decided_at": None,
    }

    final_state = graph.invoke(Command(resume=auto_decision), config=config)

    findings = final_state["investigation_findings"]
    precedent_outcomes = [d["outcome"] for d in findings["similar_past_decisions"]]
    n_originated = sum(1 for o in precedent_outcomes if o == "originated")
    n_denied = len(precedent_outcomes) - n_originated
    if n_originated > n_denied:
        precedent_majority = "originated"
    elif n_denied > n_originated:
        precedent_majority = "denied"
    else:
        precedent_majority = "tied"
    precedent_misleading = precedent_majority != "tied" and precedent_majority != actual_outcome

    predicted_outcome = "originated" if auto_decision["final_action"] == "approved" else "denied"
    correct = predicted_outcome == actual_outcome

    results.append({
        "application_id": i,
        "actual_outcome": actual_outcome,
        "predicted_outcome": predicted_outcome,
        "approval_likelihood": likelihood,
        "correct": correct,
        "precedent_misleading": precedent_misleading,
        "assigned_queue": final_state["assigned_queue"],
    })

    if len(results) % 25 == 0:
        print(f"  processed {len(results)}...")

results_df = pd.DataFrame(results)
results_df.to_csv("eval_results_final.csv", index=False)

print()
print(f"Total ambiguous cases evaluated: {len(results_df)}")
print(f"Interrupt/resume failures: {interrupt_failures}")
print()
print("Overall accuracy:", round(results_df["correct"].mean(), 3))
print()
print("Accuracy split by whether precedent was misleading for that case:")
print(results_df.groupby("precedent_misleading")["correct"].agg(["mean", "count"]))
print()
print("Confusion matrix (rows=actual, columns=predicted):")
print(pd.crosstab(results_df["actual_outcome"], results_df["predicted_outcome"]))
print()
print("Final assigned queue breakdown:")
print(results_df["assigned_queue"].value_counts())