"""
Deterministic feature engineering for the underwriting agent.

Design rule carried over from the project plan: anything that can be
computed with code, should be -- the LLM nodes downstream reason over
these computed features, not raw application rows. This keeps the agent's
job to judgment and synthesis, not arithmetic it's unreliable at.

LEAKAGE WARNING: fields like interest_rate, rate_spread, total_loan_costs,
purchaser_type, and denial_reason are excluded on purpose. They are only
populated *after* a decision is made -- verified directly against this
dataset: interest_rate is populated for 100% of originated loans (272/272)
and 0% of denied loans (0/120). Including a field like that as an input
would let a model "predict" the outcome by reading a field that already
encodes it.

PROTECTED CHARACTERISTICS: applicant_race, applicant_ethnicity, applicant_sex,
and applicant_age are deliberately excluded from this feature set. They exist
in HMDA specifically so regulators can audit for disparate impact, not so a
model can use them to decide. Any fairness check on this system should pull
those fields straight from the raw application for a separate offline audit
-- never into the live decision path.
"""

import pandas as pd
from typing import Optional


# DTI is reported as band-strings below 36% and above 50%, and as exact
# integer strings (e.g. "42") between 36% and 49%. Verified against the
# actual dataset -- these are the only formats present in this file.
_DTI_BAND_MIDPOINTS = {
    "<20%": 15.0,
    "20%-<30%": 25.0,
    "30%-<36%": 33.0,
    "50%-60%": 55.0,
    ">60%": 65.0,
}


def parse_dti(raw: object) -> Optional[float]:
    """Convert HMDA's mixed band/exact debt-to-income string to a float."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    raw = str(raw).strip()
    if raw in ("NA", "Exempt", "nan"):
        return None
    if raw in _DTI_BAND_MIDPOINTS:
        return _DTI_BAND_MIDPOINTS[raw]
    try:
        return float(raw)
    except ValueError:
        return None


def parse_numeric(raw: object) -> Optional[float]:
    """Generic numeric parser for HMDA fields that use 'NA'/'Exempt' as
    string sentinels instead of true nulls."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    raw_str = str(raw).strip()
    if raw_str in ("NA", "Exempt", "nan", ""):
        return None
    try:
        return float(raw_str)
    except ValueError:
        return None


def compute_features(row: dict) -> dict:
    """Build the ApplicationFeatures dict for one raw HMDA row.

    `row` is expected to be a single row of the filtered Broome/Visions
    dataset, as a dict (e.g. from df.to_dict(orient='records')[i]).
    """
    property_value = parse_numeric(row.get("property_value"))
    ltv = parse_numeric(row.get("loan_to_value_ratio"))

    # HMDA reports `income` in thousands of dollars, but loan_amount and
    # property_value are already in full dollars -- verified directly
    # against this dataset (loan-to-income ratios only make sense, 0.6x to
    # 3.7x, once income is multiplied by 1000). Normalizing here means
    # every downstream node works in one consistent unit instead of every
    # consumer having to remember and re-apply this conversion themselves.
    raw_income = parse_numeric(row.get("income"))
    income_usd = raw_income * 1000 if raw_income is not None else None

    return {
        "loan_amount": parse_numeric(row.get("loan_amount")),
        "loan_purpose": int(row["loan_purpose"]),
        "loan_term_months": parse_numeric(row.get("loan_term")),
        "income": income_usd,
        "debt_to_income_pct": parse_dti(row.get("debt_to_income_ratio")),
        "loan_to_value_pct": ltv,
        "ltv_missing": ltv is None,
        "property_value": property_value,
    }


# Fields known to leak the outcome -- never pass these into a decision-making
# node. Kept here as an explicit, importable list so any future node can be
# checked against it rather than relying on someone remembering the rule.
LEAKAGE_FIELDS = [
    "action_taken", "action_taken_name",
    "interest_rate", "rate_spread",
    "total_loan_costs", "total_points_and_fees", "origination_charges",
    "discount_points", "lender_credits",
    "purchaser_type",
    "denial_reason-1", "denial_reason-2", "denial_reason-3", "denial_reason-4",
]

# Protected characteristics -- excluded from decision-path features, reserved
# for a separate offline fairness audit only.
PROTECTED_FIELDS = [
    "applicant_race-1", "applicant_race-2", "applicant_race-3",
    "applicant_race-4", "applicant_race-5",
    "applicant_ethnicity-1", "applicant_ethnicity-2", "applicant_ethnicity-3",
    "applicant_ethnicity-4", "applicant_ethnicity-5",
    "applicant_sex", "applicant_age",
    "derived_race", "derived_ethnicity", "derived_sex",
]
