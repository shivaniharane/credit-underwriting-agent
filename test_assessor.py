"""
Test the real Underwriting Assessor against a handful of real ambiguous
applications before running it against all 312 -- catches a broken API
key, a bad prompt, or a schema mismatch cheaply instead of expensively.
"""

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from src.features import compute_features
from src.routing import classify_complexity
from src.assessor import assess_risk

df = pd.read_csv("data/broome_visions_hmda_2024.csv")
rows = df.to_dict(orient="records")

tested = 0
for row in rows:
    features = compute_features(row)
    complexity = classify_complexity(features)
    if complexity != "ambiguous":
        continue

    state = {
        "application_id": "test",
        "raw_application": row,
        "features": features,
        "complexity": complexity,
        "risk_assessment": None,
        "investigation_findings": None,
        "compliance_summary": None,
        "assigned_queue": None,
        "human_decision": None,
        "audit_trail": [],
    }
    result = assess_risk(state)

    print("FEATURES:", features)
    print("ASSESSMENT:", result["risk_assessment"])
    print("-" * 60)

    tested += 1
    if tested >= 3:
        break