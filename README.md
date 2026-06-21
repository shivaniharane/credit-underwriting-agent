# Credit Underwriting Agent

A LangGraph agent that supports mortgage underwriting decisions, with a
human-in-the-loop pause that's legally grounded, not just a design choice:
ECOA / Regulation B requires lenders to give specific, individualized
reasons when denying credit, which means an automated decision still needs
a human checkpoint before it's final.

Built on real 2024 HMDA data for Visions Federal Credit Union, Broome
County, NY (392 applications: 272 originated, 120 denied).

## Status: in progress

Built so far:
- `src/features.py` — turns a raw HMDA application row into clean,
  decision-safe features (parses banded debt-to-income strings, flags
  missing loan-to-value data, explicitly excludes any field that would
  leak the outcome or use a protected characteristic as a decision input)
- `src/routing.py` — deterministic clear_approve / clear_deny / ambiguous
  classification, tuned against this dataset (see file for the full
  rationale on why the thresholds are intentionally asymmetric)
- `src/state.py` — the LangGraph state schema everything else builds on

Not yet built: the LangGraph skeleton wiring these together, the LLM
nodes (Underwriting Assessor, Investigation, Compliance Writer), the
human-in-the-loop interrupt/resume with Postgres checkpointing, the
precedent-search RAG layer, and the eval harness.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Project structure

```
credit-underwriting-agent/
├── data/
│   └── broome_visions_hmda_2024.csv   # working dataset
├── src/
│   ├── state.py       # LangGraph state schema
│   ├── features.py    # raw row -> decision-safe features
│   └── routing.py     # deterministic complexity classification
├── requirements.txt
└── README.md
```
