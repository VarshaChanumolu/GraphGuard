# GraphGuard

A production-grade fraud detection system built on the [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) dataset (590,540 transactions, 3.5% fraud rate). Instead of treating transactions as independent rows, GraphGuard links them into a graph via shared card and device identifiers and uses a Graph Neural Network to surface fraud rings that row-by-row models miss ; with full MLflow experiment tracking, SHAP + LIME explainability, a Tableau Public dashboard, and a text-to-SQL guardrail layer for natural-language querying.

**[Live Dashboard →](https://public.tableau.com/app/profile/sree.varsha.chanumolu/viz/GraphGuard/GraphGuardProductionFraudDetectionSystem)**

---

## Pipeline

| Stage | What it does | Key tools |
|---|---|---|
| 1 ; Ingestion | Raw Kaggle CSVs → PostgreSQL (entity-relationship columns) + Parquet (full 435-column feature set) | Pandas, SQLAlchemy, PostgreSQL |
| 2 ; Graph construction | Transaction graph via shared card/device entities, time-windowed to 7-day bursts to distinguish fraud rings from legitimate repeat customers | NetworkX |
| 3 ; Baseline model | Random Forest on tabular + graph features; with vs. without graph features comparison | Scikit-learn |
| 4 ; GNN model | 2-layer GraphSAGE via mini-batch neighbor sampling; validation tracking, LR scheduling, early stopping, best-checkpoint restore | PyTorch, PyTorch Geometric |
| 5 ; Experiment tracking | All runs logged with params, per-epoch metrics, and final test scores | MLflow |
| 6 ; Explainability | SHAP (global importance + local drivers) and LIME (local, top-30 SHAP-feature subspace) on the deployed model | SHAP, LIME |
| 7 ; Dashboard | Four-panel Tableau Public dashboard: fraud score distribution, model comparison, SHAP importance, GNN training curve | Tableau Public |
| 7 ; Guardrails | Natural-language → SQL via Groq LLM; sqlglot AST validation blocks all non-SELECT statements; read-only Postgres role as defense in depth | Groq API, sqlglot, PostgreSQL |

---

## Results

| Model | AUC | PR-AUC | Notes |
|---|---|---|---|
| Random Forest ; tabular only | 0.8674 | 0.4431 | Baseline, no graph features |
| Random Forest ; with graph features | 0.8688 | 0.4478 | +0.0047 PR-AUC from graph construction |
| GraphSAGE v1 | 0.8282 | 0.1707 | Initial attempt ; flat loss, no validation split |
| GraphSAGE v2 (best val checkpoint) | 0.8441 | 0.3813 | Fixed training; val PR-AUC 0.4677 at epoch 14; test gap consistent with temporal drift |

**Honest finding:** the Random Forest with graph features is the stronger production model on the held-out test set. The GNN learns real structure (val PR-AUC matched the baseline at epoch 14) but generalizes less well to the most recent time window ; a documented pattern in fraud detection where fraud strategies evolve over time.

---

## Graph construction design decisions

Two entity columns form edges, one is deliberately excluded:

- **`card1`** (card identifier) ; kept; 1.21x fraud/legit degree ratio after time-windowing
- **`device_key`** (device type + device info) ; kept; 5.38x ratio, the strongest signal
- **`addr1`** (billing region) ; dropped; came back at 0.89x (legit transactions showed *more* addr1-based connectivity than fraud), confirmed as noise not signal
- **`P_emaildomain`** ; excluded from the start; only ~20 distinct values, each shared by tens of thousands of transactions ("this transaction used Gmail" is not a fraud signal)

Time-windowing (7-day buckets) is critical: without it, a legitimate customer reusing a card across 6 months looks identical to a fraud ring burning through a stolen card in 3 days. The `--diagnose` flag runs a per-column ablation to verify signal before building the full graph.

---

## Explainability findings

SHAP global importance (top features): `V91`, `V69`, `C13`, `V70`, `C5` ; Vesta's pre-engineered counting and behavioral features dominate, which explains why the Random Forest baseline was hard to beat even with graph features added.

SHAP/LIME local agreement: consistent 2/5 agreement on `C1` and `V258` across all three highest-confidence fraud cases. The remaining divergence is a documented LIME limitation in high-dimensional corlinear feature spaces (410 features, many correlated V-columns) ; not a bug, and documented in the code.

---

## Guardrails

The text-to-SQL layer uses two independent defenses:

1. **App layer:** sqlglot parses every LLM-generated query into an AST and rejects anything that isn't a pure `SELECT` ; catches multi-statement injections (`SELECT ...; DROP TABLE ...`) that naive string matching misses
2. **Database layer:** all queries execute under a `graphguard_readonly` role with `SELECT`-only grants ; even if the app layer were bypassed, the database itself has no write permissions

Adversarial test results: 6/6 injection attempts blocked (direct DELETE, prompt injection, UPDATE disguised as a question, nested DROP, TRUNCATE, INSERT injection). 4/4 natural-language happy-path queries generated valid, executable SQL.

---

## Setup

### 1. Get the data
Download `train_transaction.csv` and `train_identity.csv` from the [IEEE-CIS Fraud Detection competition](https://www.kaggle.com/competitions/ieee-fraud-detection/data) (free, requires Kaggle account + competition join). Place both in `data/raw/`.

### 2. Database
Install PostgreSQL locally (free). Then:
```bash
# In psql as postgres superuser:
CREATE USER graphguard WITH PASSWORD 'graphguard_dev';
CREATE DATABASE graphguard OWNER graphguard;
CREATE ROLE graphguard_readonly LOGIN PASSWORD 'readonly_dev';
GRANT CONNECT ON DATABASE graphguard TO graphguard_readonly;
\c graphguard
GRANT USAGE ON SCHEMA public TO graphguard_readonly;
GRANT SELECT ON transactions, identity TO graphguard_readonly;
```

### 3. Environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in:
```
PG_HOST=localhost
PG_PORT=5432
PG_DB=graphguard
PG_USER=graphguard
PG_PASSWORD=graphguard_dev
GROQ_API_KEY=your_key_here  # free at console.groq.com
```

### 4. Run the pipeline
```bash
python src/ingest.py
python src/graph_build.py
python src/baseline.py
python src/gnn.py
python src/log_historical_runs.py  # backfill MLflow with all runs
python src/explain.py
python src/export_tableau.py
python src/guardrail.py --setup
python src/guardrail.py --test
python src/guardrail.py            # interactive query mode
```

View experiment tracking:
```bash
python -m mlflow ui  # open http://localhost:5000
```

---

## Project structure

```
graphguard/
├── data/
│   ├── raw/                    # Kaggle CSVs (not committed)
│   └── processed/              # Parquet, graph pickle, model artifacts
├── reports/
│   └── tableau/                # CSVs for Tableau dashboard + spec
├── src/
│   ├── config.py               # DB connection from .env
│   ├── ingest.py               # Stage 1: ingestion
│   ├── graph_build.py          # Stage 2: graph construction + diagnosis
│   ├── baseline.py             # Stage 3: Random Forest baseline
│   ├── gnn.py                  # Stage 4: GraphSAGE
│   ├── log_historical_runs.py  # Stage 5: MLflow backfill
│   ├── explain.py              # Stage 6: SHAP + LIME
│   ├── export_tableau.py       # Stage 7: dashboard data export
│   └── guardrail.py            # Stage 7: text-to-SQL + guardrails
├── tests/
│   └── generate_synthetic_sample.py
├── .env.example
├── .gitignore
└── requirements.txt
```

---

## Stack

PostgreSQL · NetworkX · Scikit-learn · PyTorch · PyTorch Geometric · MLflow · SHAP · LIME · Groq API · sqlglot · Tableau Public · Pandas · NumPy · SQLAlchemy
