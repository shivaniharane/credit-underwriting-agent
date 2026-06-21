"""
Shared logic for managing the pending-review queue. The Streamlit app
uses this instead of duplicating graph-invocation logic that's already
proven correct in start_review.py / resume_review.py.

Design note: LangGraph's checkpointer persists EXECUTION state (what's
needed to resume a specific thread) -- it's not designed to answer "give
me a list of everything currently pending." That's a different concern,
so this module keeps a simple, separate JSON-based queue alongside it.
This is a common, legitimate pattern (a lightweight read-model next to
the source of truth), not a workaround.
"""

import json
import os
from datetime import datetime, timezone

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from src.graph import build_graph

PENDING_PATH = "pending_reviews.json"
DECIDED_PATH = "decided_reviews.json"
CHECKPOINT_DB = "checkpoints.db"


def _load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _save_json(path: str, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def get_pending() -> dict:
    return _load_json(PENDING_PATH)


def get_decided() -> dict:
    return _load_json(DECIDED_PATH)


def start_application(application_id: int, row: dict) -> dict:
    """Runs one application through the real graph. If it pauses for
    review, records it in the pending queue and returns the interrupt
    payload. If it's a clear case, it's already fully decided -- nothing
    to queue."""
    with SqliteSaver.from_conn_string(CHECKPOINT_DB) as checkpointer:
        graph = build_graph(checkpointer)
        thread_id = f"streamlit_app_{application_id}"
        config = {"configurable": {"thread_id": thread_id}}

        initial_state = {
            "application_id": application_id,
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
        pending = get_pending()
        pending[thread_id] = payload
        _save_json(PENDING_PATH, pending)
        return {"status": "paused", "thread_id": thread_id, "payload": payload}
    else:
        # Clear case -- decided automatically, no human needed. Still
        # record it: a real bank needs an auditable record of EVERY
        # decision, not just the ones that needed a human reviewer.
        final_action = "approved" if result["complexity"] == "clear_approve" else "denied"
        decision = {
            "underwriter_id": None,
            "final_action": final_action,
            "notes": "Auto-decided by policy threshold -- did not require human review.",
            "decided_at": datetime.now(timezone.utc).isoformat(),
        }
        decided = get_decided()
        decided[thread_id] = {
            "required_human_review": False,
            "decision": decision,
            "assigned_queue": result["assigned_queue"],
            "compliance_summary": result["compliance_summary"],
        }
        _save_json(DECIDED_PATH, decided)
        return {"status": "auto_decided", "thread_id": thread_id, "assigned_queue": result["assigned_queue"]}


def resume_application(thread_id: str, final_action: str, notes: str = None) -> dict:
    """Resumes a paused application with a human decision. Removes it
    from the pending queue and records it in the decided log."""
    decision = {
        "underwriter_id": "streamlit_reviewer",
        "final_action": final_action,
        "notes": notes,
        "decided_at": datetime.now(timezone.utc).isoformat(),
    }
    with SqliteSaver.from_conn_string(CHECKPOINT_DB) as checkpointer:
        graph = build_graph(checkpointer)
        config = {"configurable": {"thread_id": thread_id}}
        result = graph.invoke(Command(resume=decision), config=config)

    pending = get_pending()
    payload = pending.pop(thread_id, {})
    _save_json(PENDING_PATH, pending)

    decided = get_decided()
    decided[thread_id] = {
        "required_human_review": True,
        "decision": decision,
        "assigned_queue": result["assigned_queue"],
        "original_payload": payload,
    }
    _save_json(DECIDED_PATH, decided)

    return {"thread_id": thread_id, "assigned_queue": result["assigned_queue"], "decision": decision}