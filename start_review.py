"""
Starts an application through the graph. Clear cases get decided
immediately. Ambiguous cases pause at the human_review interrupt -- this
script then exits completely. The application sits paused in
checkpoints.db until resume_review.py is run, even if that happens in a
totally separate process, hours later, after a full restart.

Usage:
    python3 start_review.py             # uses the first ambiguous application
    python3 start_review.py 42          # uses application row 42 specifically
"""

from dotenv import load_dotenv
load_dotenv()

import sys
import pandas as pd
from langgraph.checkpoint.sqlite import SqliteSaver
from src.graph import build_graph
from src.features import compute_features
from src.routing import classify_complexity

df = pd.read_csv("data/broome_visions_hmda_2024.csv")
rows = df.to_dict(orient="records")

if len(sys.argv) > 1:
    i = int(sys.argv[1])
    row = rows[i]
else:
    i, row = next((i, r) for i, r in enumerate(rows) if classify_complexity(compute_features(r)) == "ambiguous")

with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
    graph = build_graph(checkpointer)
    thread_id = f"app_{i}"
    config = {"configurable": {"thread_id": thread_id}}

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

    result = graph.invoke(initial_state, config=config)

    if "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        print(f"PAUSED for human review.  thread_id = {thread_id}")
        print()
        print("Compliance summary for underwriter:")
        print(f"  {payload['decision_summary']}")
        print(f"  AI approval likelihood: {payload['ai_approval_likelihood']}")
        print(f"  Adverse action reasons (if denying): {payload['adverse_action_reasons']}")
        print(f"  Supporting evidence: {payload['supporting_evidence']}")
        print()
        print(f"To resume:  python3 resume_review.py {thread_id} approved")
        print(f"       or:  python3 resume_review.py {thread_id} denied \"reason notes\"")
    else:
        print(f"No human review needed -- auto-decided. Assigned queue: {result['assigned_queue']}")