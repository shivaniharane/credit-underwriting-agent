"""
Deterministic conditional-branch logic for the underwriting graph.

This is plain Python, not an LLM call -- it decides which of three buckets
an application falls into BEFORE any AI agent gets involved:

  clear_approve  -- confident enough to auto-approve, skip the agent entirely
  clear_deny     -- confident enough to auto-deny, skip the agent entirely
  ambiguous      -- genuinely unclear, route to the Underwriting Assessor
                    and eventually to a human underwriter

WHY THE THRESHOLDS ARE ASYMMETRIC (this is the important design decision):

Thresholds were tuned against the actual 392-row Broome/Visions dataset,
not picked from a textbook. An early version used the same kind of
threshold for both directions and produced a clear_approve bucket that was
wrong 34% of the time (28/83) -- DTI and LTV alone can't see the #1 real
denial reason (credit history), since HMDA never reports actual credit
scores, only which scoring model was used.

In lending, a false clear_approve is far more costly than an unnecessary
ambiguous flag: wrongly skipping review on a bad loan is real financial
exposure, while sending an extra case to human review just costs a little
time. So clear_approve is deliberately strict (requires LTV to be known,
not missing, plus low DTI and low LTV both) while clear_deny is looser.
Verified result on this dataset: clear_approve 14/14 correct (100%),
clear_deny 65/66 correct (~98%).

This asymmetry -- and the one remaining clear_deny error -- is exactly the
kind of thing the eval harness (built later) should keep monitoring as
more data comes in, not something to consider "solved" by one tuning pass.
"""

from typing import Literal

Complexity = Literal["clear_approve", "clear_deny", "ambiguous"]


def classify_complexity(features: dict) -> Complexity:
    """Route a single application's computed features into one of three
    buckets. `features` is expected to be the output of
    features.compute_features()."""
    dti = features.get("debt_to_income_pct")
    ltv = features.get("loan_to_value_pct")
    ltv_missing = features.get("ltv_missing", True)

    # Can't make any confident call without knowing DTI at all.
    if dti is None:
        return "ambiguous"

    # clear_deny: looser threshold is acceptable here -- the downside of a
    # false clear_deny is one case denied without the extra ambiguous-path
    # review, not a funded bad loan.
    if dti >= 50 or (not ltv_missing and ltv > 100):
        return "clear_deny"

    # clear_approve: deliberately strict. Requires LTV to be known (not
    # missing) because an unknown LTV means we have less information to be
    # confident about, and requires both numbers to be comfortably low.
    if dti <= 20 and (not ltv_missing) and ltv <= 70:
        return "clear_approve"

    return "ambiguous"
