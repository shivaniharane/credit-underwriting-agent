"""
State schema for the credit underwriting LangGraph.

This defines the shape of data that flows through every node in the graph.
LangGraph persists this exact structure via the checkpointer, which is what
makes interrupt()/resume work -- including across a process restart.
"""

from typing import TypedDict, Optional, Literal


class ApplicationFeatures(TypedDict):
    """Deterministic, pre-decision features computed from the raw HMDA row.

    Every field here is something genuinely known BEFORE a credit decision
    is made. Fields like interest_rate, denial_reason, and action_taken are
    deliberately excluded -- they only exist *after* a decision and would
    leak the answer straight into the input. See features.py for the full
    leakage rationale, verified against this dataset.
    """
    loan_amount: float                   # in $000s, per HMDA convention
    loan_purpose: int                    # 1=purchase, 2=home improvement, 32=cash-out refi
    loan_term_months: Optional[float]
    income: Optional[float]              # applicant-reported, in $000s
    debt_to_income_pct: Optional[float]  # parsed from band/exact string -> numeric
    loan_to_value_pct: Optional[float]
    ltv_missing: bool                    # True if LTV/property_value wasn't reported
    property_value: Optional[float]


class RiskAssessment(TypedDict):
    precedent_alignment: str             # does this agree/disagree with similar past decisions, and why
    reasoning: str
    approval_likelihood: float           # 0-1, model's estimate
    primary_factors: list[str]           # e.g. ["high DTI", "low LTV"]

class InvestigationFindings(TypedDict):
    similar_past_decisions: list[dict]   # retrieved precedent cases (vector RAG)
    consistency_note: str                # is this decision consistent with similar past ones?


class ComplianceSummary(TypedDict):
    decision_summary: str
    adverse_action_reasons: list[str]    # required under ECOA/Reg B if declining
    supporting_evidence: list[str]


class HumanDecision(TypedDict):
    underwriter_id: Optional[str]
    final_action: Literal["approved", "denied", "needs_more_info"]
    notes: Optional[str]
    decided_at: Optional[str]


class UnderwritingState(TypedDict):
    """Top-level state object. This is what LangGraph checkpoints at every
    step and what gets serialized when interrupt() pauses the graph."""
    application_id: str
    raw_application: dict                # original HMDA row, kept for audit trail
    features: Optional[ApplicationFeatures]
    risk_assessment: Optional[RiskAssessment]
    complexity: Optional[Literal["clear_approve", "clear_deny", "ambiguous"]]
    investigation_findings: Optional[InvestigationFindings]
    compliance_summary: Optional[ComplianceSummary]
    assigned_queue: Optional[str]
    human_decision: Optional[HumanDecision]
    audit_trail: list[dict]
