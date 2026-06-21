"""
The real Underwriting Assessor node -- the first LLM call in the graph,
replacing assess_risk_stub from graph.py.

Design rules carried over from earlier in the project:

- Only the computed `features` dict and investigation_findings go into the
  prompt, never the raw application row.

- Structured output via Pydantic: don't trust the model's own uncalibrated
  self-report, and don't hope it returns valid JSON.

- NEW: precedent_alignment is a REQUIRED field, not a soft instruction.
  Testing showed the model would receive real precedent (e.g. "4 of 4
  similar applications originated") and simply not engage with it in its
  free-text reasoning, despite being told to. A system-prompt instruction
  alone wasn't enough -- the schema itself now forces explicit engagement.

- NEW: field order changed on purpose. OpenAI's structured outputs
  generate fields in the order they're defined in the schema. precedent_
  alignment and reasoning now come BEFORE approval_likelihood, so the
  model has to think through the precedent and articulate its reasoning
  before it's allowed to commit to a number -- not pick a number first
  and rationalize it afterward.

- This node only ever runs for "ambiguous" cases -- see graph.py's
  conditional edges. Clear cases never reach an LLM call at all.
"""

from openai import OpenAI
from pydantic import BaseModel, Field

from src.state import UnderwritingState


class AssessmentSchema(BaseModel):
    """OpenAI is constrained to return exactly this shape, IN THIS ORDER --
    mirrors RiskAssessment in state.py. See module docstring for why the
    field order matters here."""
    precedent_alignment: str = Field(
        description="State explicitly whether this assessment agrees or disagrees with how "
                    "similar past applications were resolved, and why. Name the actual outcome "
                    "split (e.g. '4 of 4 similar applications were approved') and say whether "
                    "you are following or deviating from that pattern."
    )
    reasoning: str = Field(
        description="2-3 sentences explaining the assessment, building on the precedent "
                    "alignment above and the application's own numbers"
    )
    approval_likelihood: float = Field(
        description="0.0 to 1.0, estimated likelihood this application should be approved"
    )
    primary_factors: list[str] = Field(
        description="short phrases naming the factors that most influenced this assessment, "
                    "e.g. 'consistent with similar approved applications' or "
                    "'debt-to-income ratio higher than similar denied cases'"
    )


SYSTEM_PROMPT = """You are an underwriting assistant helping a human loan \
officer assess a mortgage application that automated rules could not \
confidently approve or deny on their own -- it has been flagged as \
genuinely ambiguous. Your job is to reason carefully about the \
application's features AND how this specific lender has actually handled \
similar applications in the past, then produce a structured assessment \
for the underwriter to review.

You MUST explicitly engage with the precedent provided below in the \
precedent_alignment field -- name the actual outcome split among similar \
past applications and state whether your assessment follows or deviates \
from that pattern. Ground your reasoning primarily in this real \
precedent, not on generic industry conventions (such as standard DTI \
thresholds) that may not reflect how this particular lender actually \
behaves. If the precedent conflicts with a generic rule of thumb, trust \
the precedent.

You are NOT making the final decision. A human underwriter always \
reviews ambiguous cases before any action is taken. Base your reasoning \
only on the information provided below -- do not invent or assume \
information you were not given."""


_PURPOSE_LABELS = {1: "Home purchase", 2: "Home improvement", 32: "Cash-out refinance"}


def build_user_prompt(features: dict, investigation_findings: dict) -> str:
    """Turns the computed features dict AND real precedent into the text
    the model sees. Nothing from the raw application row is added here."""
    purpose = _PURPOSE_LABELS.get(features["loan_purpose"], f"Other ({features['loan_purpose']})")
    ltv = "not reported" if features["ltv_missing"] else f"{features['loan_to_value_pct']}%"
    prop_val = "not reported" if features["property_value"] is None else f"${features['property_value']:,.0f}"
    term = "not reported" if features["loan_term_months"] is None else f"{features['loan_term_months']:.0f} months"
    income = "not reported" if features["income"] is None else f"${features['income']:,.0f}/year"

    precedent_lines = "\n".join(
        f"- {d['outcome'].upper()}: {d['summary']}"
        for d in investigation_findings["similar_past_decisions"]
    ) or "(no comparable past applications found)"

    return f"""Assess this mortgage application:

Loan amount: ${features['loan_amount']:,.0f}
Loan purpose: {purpose}
Loan term: {term}
Applicant annual income: {income}
Debt-to-income ratio: {features['debt_to_income_pct']}%
Loan-to-value ratio: {ltv}
Property value: {prop_val}

{investigation_findings['consistency_note']}
Similar past applications from this lender:
{precedent_lines}"""


def assess_risk(state: UnderwritingState) -> dict:
    """REAL. Replaces assess_risk_stub in graph.py."""
    client = OpenAI()  # reads OPENAI_API_KEY from the environment automatically

    completion = client.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(state["features"], state["investigation_findings"])},
        ],
        response_format=AssessmentSchema,
    )
    parsed = completion.choices[0].message.parsed

    assessment = {
        "precedent_alignment": parsed.precedent_alignment,
        "reasoning": parsed.reasoning,
        "approval_likelihood": parsed.approval_likelihood,
        "primary_factors": parsed.primary_factors,
    }

    return {
        "risk_assessment": assessment,
        "audit_trail": state["audit_trail"] + [{"step": "assess_risk", "stub": False}],
    }