"""
The real Compliance Writer node -- merge point for all three complexity
paths (clear_approve, clear_deny, ambiguous), replacing
compliance_writer_stub in graph.py.

Design rules:

- clear_approve and clear_deny stay fully deterministic, no LLM call.
  These thresholds (see routing.py) are specific and known, so the
  adverse-action reasons can be generated directly from the threshold
  that was actually crossed -- more traceable and auditable than an LLM
  paraphrase of the same fact, and keeps the clear-case path genuinely
  zero-cost end to end (feature engineering -> routing -> compliance
  writer, no AI anywhere).
  THRESHOLDS BELOW MUST STAY IN SYNC WITH routing.py -- duplicated here
  rather than imported because this is explanatory text, not the actual
  decision logic (routing.py already made the decision; this only
  explains it).

- ambiguous cases use a real LLM call to synthesize risk_assessment and
  investigation_findings into a draft summary for the human underwriter.
  Explicitly framed as a DRAFT for review, not a final decision -- the
  human confirms, overrides, or requests more information. ECOA/Reg B
  requires specific, individualized reasons for any actual denial, which
  is why adverse_action_reasons is only populated when the recommendation
  leans toward denial, and is empty otherwise.
"""

from openai import OpenAI
from pydantic import BaseModel, Field

from src.state import UnderwritingState

# Must match routing.py's classify_complexity thresholds exactly.
CLEAR_APPROVE_DTI_MAX = 20
CLEAR_APPROVE_LTV_MAX = 70
CLEAR_DENY_DTI_MIN = 50
CLEAR_DENY_LTV_MIN = 100

_PURPOSE_LABELS = {1: "Home purchase", 2: "Home improvement", 32: "Cash-out refinance"}


def _clear_approve_summary(features: dict) -> dict:
    return {
        "decision_summary": (
            "Application meets policy thresholds for automatic approval -- "
            "debt-to-income ratio and loan-to-value ratio are both comfortably "
            "within approved ranges."
        ),
        "adverse_action_reasons": [],
        "supporting_evidence": [
            f"Debt-to-income ratio of {features['debt_to_income_pct']}% is at or below the "
            f"{CLEAR_APPROVE_DTI_MAX}% policy threshold for automatic approval.",
            f"Loan-to-value ratio of {features['loan_to_value_pct']}% is at or below the "
            f"{CLEAR_APPROVE_LTV_MAX}% policy threshold for automatic approval.",
        ],
    }


def _clear_deny_summary(features: dict) -> dict:
    reasons = []
    evidence = []
    dti = features["debt_to_income_pct"]
    ltv = features["loan_to_value_pct"]
    ltv_missing = features["ltv_missing"]

    if dti is not None and dti >= CLEAR_DENY_DTI_MIN:
        reasons.append("debt-to-income ratio")
        evidence.append(
            f"Debt-to-income ratio of {dti}% exceeds the {CLEAR_DENY_DTI_MIN}% policy threshold."
        )
    if not ltv_missing and ltv is not None and ltv > CLEAR_DENY_LTV_MIN:
        reasons.append("collateral")
        evidence.append(
            f"Loan-to-value ratio of {ltv}% exceeds {CLEAR_DENY_LTV_MIN}%, meaning the loan "
            f"amount exceeds the property's value -- insufficient collateral to secure the loan."
        )

    return {
        "decision_summary": (
            "Application does not meet policy thresholds for approval and is "
            "automatically declined."
        ),
        "adverse_action_reasons": reasons,
        "supporting_evidence": evidence,
    }


class ComplianceSummarySchema(BaseModel):
    """OpenAI is constrained to return exactly this shape -- mirrors
    ComplianceSummary in state.py. Used only for the ambiguous path."""
    decision_summary: str = Field(
        description="2-3 sentence plain-English summary for the underwriter, synthesizing "
                    "the risk assessment and precedent into a clear recommendation"
    )
    adverse_action_reasons: list[str] = Field(
        description="If the recommendation leans toward denial, list specific reasons "
                    "(e.g. 'debt-to-income ratio', 'credit history') the underwriter can use "
                    "if they agree and deny. Empty list if the recommendation leans approve."
    )
    supporting_evidence: list[str] = Field(
        description="Concrete facts backing the summary -- specific numbers from this "
                    "application and/or specific precedent cases referenced"
    )


SYSTEM_PROMPT = """You are a compliance writer preparing a draft summary for a human \
underwriter reviewing an ambiguous mortgage application. Synthesize the risk assessment \
and precedent research into a clear, concise summary the underwriter can quickly review \
and act on.

This is a DRAFT for human review, not a final decision -- the underwriter will confirm, \
override, or request more information. If the assessment leans toward denial, draft \
specific adverse action reasons the underwriter can use if they agree: lenders are \
required under the Equal Credit Opportunity Act to give specific, individualized reasons \
for any denial, generic reasons are not acceptable. If the assessment leans toward \
approval, leave adverse_action_reasons empty.

Base your summary only on the information provided below -- do not invent details."""


def build_user_prompt(features: dict, risk_assessment: dict, investigation_findings: dict) -> str:
    purpose = _PURPOSE_LABELS.get(features["loan_purpose"], f"Other ({features['loan_purpose']})")
    ltv = "not reported" if features["ltv_missing"] else f"{features['loan_to_value_pct']}%"
    income = "not reported" if features["income"] is None else f"${features['income']:,.0f}/year"

    return f"""Prepare a compliance summary for this ambiguous application:

Loan purpose: {purpose}
Loan amount: ${features['loan_amount']:,.0f}
Debt-to-income ratio: {features['debt_to_income_pct']}%
Loan-to-value ratio: {ltv}
Applicant annual income: {income}

Underwriting assessment:
- Approval likelihood: {risk_assessment['approval_likelihood']}
- Precedent alignment: {risk_assessment['precedent_alignment']}
- Reasoning: {risk_assessment['reasoning']}
- Primary factors: {', '.join(risk_assessment['primary_factors'])}

{investigation_findings['consistency_note']}"""


def compliance_writer(state: UnderwritingState) -> dict:
    """REAL. Replaces compliance_writer_stub in graph.py."""
    complexity = state["complexity"]

    if complexity == "clear_approve":
        summary = _clear_approve_summary(state["features"])
    elif complexity == "clear_deny":
        summary = _clear_deny_summary(state["features"])
    else:
        client = OpenAI()
        completion = client.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_user_prompt(
                        state["features"], state["risk_assessment"], state["investigation_findings"]
                    ),
                },
            ],
            response_format=ComplianceSummarySchema,
        )
        parsed = completion.choices[0].message.parsed
        summary = {
            "decision_summary": parsed.decision_summary,
            "adverse_action_reasons": parsed.adverse_action_reasons,
            "supporting_evidence": parsed.supporting_evidence,
        }

    return {
        "compliance_summary": summary,
        "audit_trail": state["audit_trail"] + [{"step": "compliance_writer", "stub": False}],
    }