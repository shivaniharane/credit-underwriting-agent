"""
Free re-analysis of eval_results_final.csv -- no new API calls. Checks
whether a different decision threshold (instead of the default 0.5)
would meaningfully rebalance accuracy between approvals and denials.
"""

import pandas as pd

df = pd.read_csv("eval_results_final.csv")

print("Actual denial rate in this set:", round((df["actual_outcome"] == "denied").mean(), 3))
print()

for threshold in [0.75, 0.78, 0.80, 0.82, 0.85]:
    predicted = df["approval_likelihood"].apply(lambda p: "originated" if p >= threshold else "denied")
    correct = (predicted == df["actual_outcome"])
    denied_mask = df["actual_outcome"] == "denied"
    originated_mask = df["actual_outcome"] == "originated"
    denial_recall = correct[denied_mask].mean()
    approval_recall = correct[originated_mask].mean()
    overall = correct.mean()
    print(f"threshold={threshold}  overall_acc={overall:.3f}  denial_recall={denial_recall:.3f}  approval_recall={approval_recall:.3f}")