"""
Resumes a paused application with a human underwriter's decision. This is
meant to be run as a completely separate process invocation from
start_review.py -- if this works, the checkpoint genuinely persisted to
disk (checkpoints.db), not just in the memory of a still-running process.

Usage:
    python3 resume_review.py app_5 approved
    python3 resume_review.py app_5 denied "DTI inconsistent with stated income"
"""

from dotenv import load_dotenv
load_dotenv()

import sys
from datetime import datetime, timezone
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from src.graph import build_graph

if len(sys.argv) < 3:
    print('Usage: python3 resume_review.py <thread_id> <approved|denied|needs_more_info> ["notes"]')
    sys.exit(1)

thread_id = sys.argv[1]
final_action = sys.argv[2]
notes = sys.argv[3] if len(sys.argv) > 3 else None

if final_action not in ("approved", "denied", "needs_more_info"):
    print(f"final_action must be one of: approved, denied, needs_more_info (got '{final_action}')")
    sys.exit(1)

decision = {
    "underwriter_id": "demo_reviewer",
    "final_action": final_action,
    "notes": notes,
    "decided_at": datetime.now(timezone.utc).isoformat(),
}

with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
    graph = build_graph(checkpointer)
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(Command(resume=decision), config=config)

print(f"Resumed and completed.")
print(f"Final assigned queue: {result['assigned_queue']}")
print(f"Human decision recorded: {result['human_decision']}")