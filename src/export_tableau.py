"""Stage 7 (Part A): export pipeline results to CSV for Tableau Public.

Produces four flat CSV files in reports/tableau/:
  1. transactions_scored.csv  -- every test transaction with its fraud
                                  probability, graph features, and core fields
  2. model_comparison.csv     -- per-model AUC / PR-AUC for the comparison view
  3. shap_global.csv          -- top-30 SHAP feature importances for the bar chart
  4. gnn_training.csv         -- per-epoch GNN training curve (from MLflow history)

Tableau Public is a GUI tool -- these CSVs are the data source. The
dashboard build instructions are in reports/tableau/DASHBOARD_SPEC.md.

Usage:
    python src/export_tableau.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import shap

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline import GRAPH_FEATURE_COLS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DATA_PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"
TABLEAU_DIR = Path(__file__).resolve().parent.parent / "reports" / "tableau"


def export_transactions_scored(model, X_test, y_test) -> None:
    proba = model.predict_proba(X_test)[:, 1]
    df = X_test[["TransactionAmt"] + [c for c in GRAPH_FEATURE_COLS if c in X_test.columns]].copy()
    df["fraud_probability"] = proba
    df["actual_label"] = y_test.values
    df["predicted_fraud"] = (proba >= 0.5).astype(int)
    df["correct"] = (df["predicted_fraud"] == df["actual_label"]).astype(int)
    df = df.reset_index(drop=True)
    df.index.name = "row_id"
    out = TABLEAU_DIR / "transactions_scored.csv"
    df.to_csv(out)
    log.info("Exported %s scored transactions to %s", len(df), out)


def export_model_comparison() -> None:
    exp = mlflow.get_experiment_by_name("graphguard-fraud-detection")
    if exp is None:
        log.warning("No MLflow experiment found -- run log_historical_runs.py first. Skipping model_comparison.csv.")
        return
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    rows = []
    for _, run in runs.iterrows():
        name = run.get("tags.mlflow.runName", "unknown")
        auc = run.get("metrics.test_auc", None)
        pr_auc = run.get("metrics.test_pr_auc", None)
        stage = run.get("tags.stage", "")
        if pd.notna(auc):
            rows.append({"run_name": name, "stage": stage, "test_auc": round(auc, 4), "test_pr_auc": round(pr_auc, 4)})
    df = pd.DataFrame(rows)
    out = TABLEAU_DIR / "model_comparison.csv"
    df.to_csv(out, index=False)
    log.info("Exported model comparison (%s runs) to %s", len(df), out)


def export_shap_global(model, X_test) -> None:
    log.info("Computing SHAP values for Tableau export (sample of 2000)...")
    sample = X_test.sample(n=min(2000, len(X_test)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(sample)
    vals = shap_values.values[:, :, 1] if shap_values.values.ndim == 3 else shap_values.values
    mean_abs = np.abs(vals).mean(axis=0)
    rows = [
        {
            "feature": name,
            "mean_abs_shap": round(float(v), 6),
            "is_graph_feature": name in GRAPH_FEATURE_COLS,
        }
        for name, v in sorted(zip(sample.columns, mean_abs), key=lambda x: -x[1])[:30]
    ]
    df = pd.DataFrame(rows)
    out = TABLEAU_DIR / "shap_global.csv"
    df.to_csv(out, index=False)
    log.info("Exported top-30 SHAP importances to %s", out)


def export_gnn_training() -> None:
    exp = mlflow.get_experiment_by_name("graphguard-fraud-detection")
    if exp is None:
        log.warning("No MLflow experiment found -- skipping gnn_training.csv.")
        return
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    gnn_run = runs[runs["tags.mlflow.runName"] == "gnn_v2_tuned"]
    if gnn_run.empty:
        log.warning("gnn_v2_tuned run not found in MLflow -- skipping gnn_training.csv.")
        return
    run_id = gnn_run.iloc[0]["run_id"]
    client = mlflow.tracking.MlflowClient()
    rows = []
    for metric in ["train_loss", "val_auc", "val_pr_auc"]:
        history = client.get_metric_history(run_id, metric)
        for h in history:
            rows.append({"epoch": h.step, "metric": metric, "value": round(h.value, 6)})
    df = pd.DataFrame(rows).sort_values(["metric", "epoch"])
    out = TABLEAU_DIR / "gnn_training.csv"
    df.to_csv(out, index=False)
    log.info("Exported GNN training curve (%s data points) to %s", len(df), out)


def write_dashboard_spec() -> None:
    spec = """# GraphGuard — Tableau Dashboard Spec

Connect each sheet to the CSV in the same folder as this file.

## Sheet 1: Model Comparison (model_comparison.csv)
**Chart type:** Grouped bar chart
**Rows:** run_name  **Columns:** test_auc, test_pr_auc (side by side)
**Color:** stage (baseline = teal, gnn = coral)
**Title:** "Model Performance: AUC and PR-AUC by Run"
**Note:** PR-AUC is the more honest metric at 3.5% fraud rate -- label it prominently.

## Sheet 2: Fraud Score Distribution (transactions_scored.csv)
**Chart type:** Histogram of fraud_probability, colored by actual_label (0=legit, 1=fraud)
**Title:** "Fraud Probability Distribution -- Legitimate vs Actual Fraud"
**Insight to highlight:** Fraud transactions cluster near probability 1.0;
a bimodal distribution here confirms the model separates the classes well.

## Sheet 3: SHAP Global Feature Importance (shap_global.csv)
**Chart type:** Horizontal bar chart, sorted descending by mean_abs_shap
**Color:** is_graph_feature (True = different color to highlight graph contribution)
**Title:** "Global Feature Importance (Mean |SHAP Value|)"
**Note:** Graph features (degree, clustering_coeff, component_size) should be
visually distinct -- these are the features the graph construction stage added.

## Sheet 4: GNN Training Curve (gnn_training.csv)
**Chart type:** Line chart
**Filter:** metric IN ('val_auc', 'val_pr_auc', 'train_loss') -- show all three
**X-axis:** epoch  **Y-axis:** value  **Color:** metric
**Mark the best checkpoint epoch (14) with a reference line.**
**Title:** "GNN Training Curve -- Val PR-AUC peaked at epoch 14 (0.4677)"

## Dashboard Assembly
Arrange all four sheets on a single dashboard in a 2x2 grid.
Add a text box title: "GraphGuard: Production Fraud Detection System"
Add a subtitle: "IEEE-CIS dataset | 590,540 transactions | 3.5% fraud rate"
Publish to Tableau Public (File -> Save to Tableau Public).
"""
    out = TABLEAU_DIR / "DASHBOARD_SPEC.md"
    out.write_text(spec)
    log.info("Wrote dashboard spec to %s", out)


def run() -> None:
    TABLEAU_DIR.mkdir(parents=True, exist_ok=True)
    artifacts = joblib.load(DATA_PROCESSED / "baseline_model.joblib")
    model, X_test, y_test = artifacts["model"], artifacts["X_test"], artifacts["y_test"]

    export_transactions_scored(model, X_test, y_test)
    export_model_comparison()
    export_shap_global(model, X_test)
    export_gnn_training()
    write_dashboard_spec()

    log.info("=" * 70)
    log.info("All Tableau exports complete. Files saved to reports/tableau/")
    log.info("Open Tableau Public Desktop, connect to each CSV, follow DASHBOARD_SPEC.md")
    log.info("=" * 70)


if __name__ == "__main__":
    run()
