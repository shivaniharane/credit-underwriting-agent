"""
Tests the real investigate() node against a few real ambiguous
applications -- confirms precedent search returns sensible results and,
critically, that leave-one-out actually excludes the application from
matching itself.
"""

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from src.features import compute_features
from src.routing import classify_complexity
from src.investigation import investigate

df = pd.read_csv("data/broome_visions_hmda_2024.csv")
rows = df.to_dict(orient="records")

tested = 0
for i, row in enumerate(rows):
    features = compute_features(row)
    complexity = classify_complexity(features)
    if complexity != "ambiguous":
        continue

    state = {
        "application_id": i,  # must match the Qdrant point ID format: plain int = row index
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
    result = investigate(state)
    findings = result["investigation_findings"]

    own_outcome = "denied" if row["action_taken"] == 3 else "originated"
    print(f"Application {i} (its own actual outcome: {own_outcome})")
    print(f"  {findings['consistency_note']}")
    for d in findings["similar_past_decisions"]:
        print(f"    score={d['similarity_score']}  outcome={d['outcome']}  {d['summary']}")
    print("-" * 60)

    tested += 1
    if tested >= 3:
        break