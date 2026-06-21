"""
The real Investigation node -- precedent search over historical decisions,
replacing investigate_stub in graph.py.

Design notes:

- Builds a query description in the SAME format as build_precedent_index.py's
  describe_precedent(), minus the outcome sentence -- we don't know the
  outcome of the application being investigated, that's the whole point.
  If the text format in build_precedent_index.py ever changes, update
  describe_application() here to match, or the two texts stop being
  comparable in embedding space.

- Leave-one-out: requests one extra result and drops any hit whose id
  matches the current application_id, rather than relying on a Qdrant-side
  filter -- simpler to verify is actually correct, and doesn't depend on
  getting a less-common part of Qdrant's filter API exactly right.

- consistency_note is built with plain code, not another LLM call -- same
  principle as routing.py: don't spend an LLM call on something a few
  lines of Python can already tell you.
"""

from openai import OpenAI
from qdrant_client import QdrantClient

from src.state import UnderwritingState

QDRANT_PATH = "qdrant_data"
COLLECTION_NAME = "precedent_decisions"
TOP_K = 4

_PURPOSE_LABELS = {1: "Home purchase", 2: "Home improvement", 32: "Cash-out refinance"}


def describe_application(features: dict) -> str:
    """Same format as build_precedent_index.py's describe_precedent(),
    deliberately without an outcome sentence -- this describes an
    application that hasn't been decided yet."""
    purpose = _PURPOSE_LABELS.get(features["loan_purpose"], f"Other ({features['loan_purpose']})")
    ltv = "not reported" if features["ltv_missing"] else f"{features['loan_to_value_pct']}%"
    income = "not reported" if features["income"] is None else f"${features['income']:,.0f}/year"
    return (
        f"{purpose} loan application. "
        f"Loan amount ${features['loan_amount']:,.0f}, "
        f"debt-to-income ratio {features['debt_to_income_pct']}%, "
        f"loan-to-value ratio {ltv}, "
        f"applicant income {income}."
    )


def investigate(state: UnderwritingState) -> dict:
    """REAL. Replaces investigate_stub in graph.py."""
    client_oa = OpenAI()
    query_text = describe_application(state["features"])
    query_vector = client_oa.embeddings.create(
        model="text-embedding-3-small", input=[query_text]
    ).data[0].embedding

    client_qd = QdrantClient(path=QDRANT_PATH)
    raw_hits = client_qd.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=TOP_K + 1,  # +1 in case the application itself is in the index
    ).points

    current_id = state["application_id"]
    hits = [h for h in raw_hits if h.id != current_id][:TOP_K]

    similar_past_decisions = [
        {
            "outcome": h.payload["outcome"],
            "summary": h.payload["summary"],
            "similarity_score": round(h.score, 4),
        }
        for h in hits
    ]

    denied_count = sum(1 for d in similar_past_decisions if d["outcome"] == "denied")
    originated_count = len(similar_past_decisions) - denied_count
    consistency_note = (
        f"Found {len(similar_past_decisions)} similar past applications: "
        f"{originated_count} originated, {denied_count} denied."
    )

    findings = {
        "similar_past_decisions": similar_past_decisions,
        "consistency_note": consistency_note,
    }

    return {
        "investigation_findings": findings,
        "audit_trail": state["audit_trail"] + [{"step": "investigate", "stub": False}],
    }