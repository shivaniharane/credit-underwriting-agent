"""
The LangGraph for the underwriting agent -- now with a real human-in-the-
loop pause.

Flow:
    START
      -> compute_features            (real: features.py + routing.py)
      -> [conditional branch on complexity]
           clear_approve / clear_deny -> compliance_writer   (deterministic, no AI)
           ambiguous                  -> investigate -> assess_risk -> compliance_writer (LLM)
      -> [conditional: does this case need a human?]
           ambiguous                  -> human_review (REAL INTERRUPT) -> route_decision
           clear_approve / clear_deny -> route_decision directly (no human needed)
      -> route_decision               (real: logs the final outcome)
      -> END

build_graph() now REQUIRES a checkpointer argument -- without one,
interrupt() cannot persist state, and resuming a paused application would
be impossible. This is what makes the pause survive even a full process
restart, not just a single Python session.
"""

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt

from src.state import UnderwritingState
from src.features import compute_features
from src.routing import classify_complexity
from src.assessor import assess_risk
from src.investigation import investigate
from src.compliance_writer import compliance_writer


def compute_features_node(state: UnderwritingState) -> dict:
    """REAL. Computes features and classifies complexity in one step, so
    both the features and the routing decision land in the audit trail."""
    features = compute_features(state["raw_application"])
    complexity = classify_complexity(features)
    return {
        "features": features,
        "complexity": complexity,
        "audit_trail": state["audit_trail"] + [
            {"step": "compute_features", "complexity": complexity}
        ],
    }


def route_by_complexity(state: UnderwritingState) -> str:
    """Conditional edge function. Pure routing -- no state changes, just
    reads what compute_features_node already decided."""
    return state["complexity"]


def human_review(state: UnderwritingState) -> dict:
    """REAL. The actual human-in-the-loop pause. Only reached by ambiguous
    cases (see route_after_compliance_writer below).

    Note: per LangGraph's interrupt() semantics, any code before the
    interrupt() call re-runs when the node resumes -- this function has
    nothing before it, so that's not a concern here."""
    decision = interrupt({
        "application_id": state["application_id"],
        "decision_summary": state["compliance_summary"]["decision_summary"],
        "adverse_action_reasons": state["compliance_summary"]["adverse_action_reasons"],
        "supporting_evidence": state["compliance_summary"]["supporting_evidence"],
        "ai_approval_likelihood": state["risk_assessment"]["approval_likelihood"],
    })
    return {
        "human_decision": decision,
        "audit_trail": state["audit_trail"] + [
            {"step": "human_review", "final_action": decision["final_action"]}
        ],
    }


def route_after_compliance_writer(state: UnderwritingState) -> str:
    """Conditional edge function. Ambiguous cases need a human; clear
    cases were already deterministically decided and skip straight to
    logging the outcome."""
    return "human_review" if state["complexity"] == "ambiguous" else "route_decision"


def route_decision(state: UnderwritingState) -> dict:
    """REAL. Logs the final outcome. For clear cases this is the
    deterministic decision from routing.py; for ambiguous cases this is
    the human underwriter's actual decision, not just the AI's
    recommendation."""
    complexity = state["complexity"]
    if complexity != "ambiguous":
        outcome = "approved" if complexity == "clear_approve" else "denied"
        queue = f"auto_{outcome}_log"
    else:
        queue = f"underwriter_{state['human_decision']['final_action']}_log"
    return {
        "assigned_queue": queue,
        "audit_trail": state["audit_trail"] + [{"step": "route_decision", "queue": queue}],
    }


def build_graph(checkpointer):
    graph = StateGraph(UnderwritingState)

    graph.add_node("compute_features", compute_features_node)
    graph.add_node("investigate", investigate)
    graph.add_node("assess_risk", assess_risk)
    graph.add_node("compliance_writer", compliance_writer)
    graph.add_node("human_review", human_review)
    graph.add_node("route_decision", route_decision)

    graph.add_edge(START, "compute_features")
    graph.add_conditional_edges(
        "compute_features",
        route_by_complexity,
        {
            "ambiguous": "investigate",
            "clear_approve": "compliance_writer",
            "clear_deny": "compliance_writer",
        },
    )
    graph.add_edge("investigate", "assess_risk")
    graph.add_edge("assess_risk", "compliance_writer")
    graph.add_conditional_edges(
        "compliance_writer",
        route_after_compliance_writer,
        {
            "human_review": "human_review",
            "route_decision": "route_decision",
        },
    )
    graph.add_edge("human_review", "route_decision")
    graph.add_edge("route_decision", END)

    return graph.compile(checkpointer=checkpointer)