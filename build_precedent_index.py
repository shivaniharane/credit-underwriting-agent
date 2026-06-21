"""
Builds the precedent index: embeds every historical, already-decided
application in the dataset and stores it in a local Qdrant collection.

Run this once before using the real investigate() node. Re-run it if the
underlying dataset changes -- or, as now, if the text representation
changes and the index needs rebuilding with richer information.

UPDATED: precedent summaries for denied applications now include the
actual denial reason (e.g. "mortgage insurance denied"), not just the
binary outcome. Eval testing showed that when similar past applications
were superficially similar (DTI/LTV) but denied for an unrelated reason,
the model would still treat them as supporting evidence -- accuracy on
"misleading precedent" cases was 0% across a 20-case pilot. Surfacing the
actual reason lets the model judge whether a precedent's denial is even
relevant to the current application's risk profile, instead of treating
every denial as equally informative.
"""

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from openai import OpenAI
from qdrant_client import QdrantClient, models

from src.features import compute_features

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536  # text-embedding-3-small's native output size
QDRANT_PATH = "qdrant_data"
COLLECTION_NAME = "precedent_decisions"

_PURPOSE_LABELS = {1: "Home purchase", 2: "Home improvement", 32: "Cash-out refinance"}

# Verified against FFIEC's HMDA code sheet -- the full set of denial
# reason codes, not just the ones observed in this dataset's first 4
# reason slots. Code 10 ("not applicable") is used for non-denied loans
# and filtered out, since it doesn't describe anything.
_DENIAL_REASON_LABELS = {
    1: "debt-to-income ratio",
    2: "employment history",
    3: "credit history",
    4: "collateral",
    5: "insufficient cash for down payment/closing costs",
    6: "unverifiable information",
    7: "credit application incomplete",
    8: "mortgage insurance denied",
    9: "other",
    10: "not applicable",
}


def get_denial_reasons(row: dict) -> list[str]:
    """Extracts and labels the actual denial reason codes for a denied
    application. HMDA allows up to 4 reasons per denial
    (denial_reason-1 through -4)."""
    reasons = []
    for i in range(1, 5):
        raw = row.get(f"denial_reason-{i}")
        if raw is None or pd.isna(raw):
            continue
        try:
            code = int(raw)
        except (ValueError, TypeError):
            continue
        if code == 10:
            continue
        label = _DENIAL_REASON_LABELS.get(code)
        if label:
            reasons.append(label)
    return reasons


def describe_precedent(features: dict, row: dict) -> str:
    """Builds the text that gets embedded for one historical application.
    Includes the real outcome AND, for denials, the real reason -- this
    describes the PAST, which is what makes including the outcome safe in
    the first place (see investigation.py for the current-application
    query text, which deliberately has no outcome)."""
    purpose = _PURPOSE_LABELS.get(features["loan_purpose"], f"Other ({features['loan_purpose']})")
    ltv = "not reported" if features["ltv_missing"] else f"{features['loan_to_value_pct']}%"
    income = "not reported" if features["income"] is None else f"${features['income']:,.0f}/year"

    if row["action_taken"] == 1:
        outcome = "approved/originated"
    else:
        reasons = get_denial_reasons(row)
        outcome = "denied" + (f" (reason: {', '.join(reasons)})" if reasons else "")

    return (
        f"{purpose} loan application. "
        f"Loan amount ${features['loan_amount']:,.0f}, "
        f"debt-to-income ratio {features['debt_to_income_pct']}%, "
        f"loan-to-value ratio {ltv}, "
        f"applicant income {income}. "
        f"Outcome: {outcome}."
    )


def build_index():
    df = pd.read_csv("data/broome_visions_hmda_2024.csv")
    rows = df.to_dict(orient="records")

    texts, payloads, ids = [], [], []
    for i, row in enumerate(rows):
        features = compute_features(row)
        text = describe_precedent(features, row)
        texts.append(text)
        payloads.append({
            "application_id": i,
            "summary": text,
            "outcome": "originated" if row["action_taken"] == 1 else "denied",
            "features": features,
        })
        ids.append(i)

    client_oa = OpenAI()
    print(f"Embedding {len(texts)} historical applications...")

    all_vectors = []
    for start in range(0, len(texts), 100):
        chunk = texts[start:start + 100]
        response = client_oa.embeddings.create(model=EMBEDDING_MODEL, input=chunk)
        all_vectors.extend([item.embedding for item in response.data])
        print(f"  embedded {min(start + 100, len(texts))}/{len(texts)}")

    client_qd = QdrantClient(path=QDRANT_PATH)
    if client_qd.collection_exists(COLLECTION_NAME):
        client_qd.delete_collection(COLLECTION_NAME)
    client_qd.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=EMBEDDING_DIM, distance=models.Distance.COSINE),
    )
    client_qd.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            models.PointStruct(id=ids[i], vector=all_vectors[i], payload=payloads[i])
            for i in range(len(ids))
        ],
    )
    print(f"Indexed {len(ids)} precedent decisions into Qdrant at ./{QDRANT_PATH}")


if __name__ == "__main__":
    build_index()

    print()
    print("Sanity check -- querying for cases similar to application 0:")
    df = pd.read_csv("data/broome_visions_hmda_2024.csv")
    row0 = df.iloc[0].to_dict()
    features0 = compute_features(row0)
    query_text = describe_precedent(features0, row0)

    client_oa = OpenAI()
    query_vector = client_oa.embeddings.create(model=EMBEDDING_MODEL, input=[query_text]).data[0].embedding

    client_qd = QdrantClient(path=QDRANT_PATH)
    hits = client_qd.query_points(collection_name=COLLECTION_NAME, query=query_vector, limit=4).points
    for hit in hits:
        print(f"  id={hit.id}  score={hit.score:.4f}  outcome={hit.payload['outcome']}  summary={hit.payload['summary']}")