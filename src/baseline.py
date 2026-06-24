"""Stage 3: baseline model.

The whole point of this script is one comparison: train the same Random
Forest twice -- once on tabular features only, once with Stage 2's graph
features (degree, clustering coefficient, component size) added -- and see
whether the graph actually earns its keep. The diagnostic ratios from
Stage 2 are a proxy; this is the real test.

Uses a time-based train/test split (sorted by transaction_day), not a
random split -- a random split would leak future information into
training, which doesn't reflect how this model would actually be deployed
(train on the past, predict on what comes next).

Usage:
    python src/baseline.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import OrdinalEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DATA_PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"

# Kept as categorical features (label-encoded). DeviceInfo is deliberately
# excluded -- too high-cardinality to encode usefully, and its signal is
# already captured indirectly through the graph's device_key degree feature.
CATEGORICAL_COLS = ["ProductCD", "card4", "card6", "DeviceType", "P_emaildomain", "R_emaildomain"]

# component_id is excluded even when "graph features" are included -- it's
# an arbitrary integer label, not a meaningful magnitude. component_size,
# degree, and clustering_coeff are the features that actually carry signal.
GRAPH_FEATURE_COLS = ["degree", "clustering_coeff", "component_size"]

NON_FEATURE_COLS = {"TransactionID", "transactionid", "isFraud", "isfraud", "TransactionDT", "transaction_day"}


def load_merged_features() -> pd.DataFrame:
    txn = pd.read_parquet(DATA_PROCESSED / "transactions_full.parquet")
    graph = pd.read_parquet(DATA_PROCESSED / "graph_features.parquet")
    graph = graph.rename(columns={"transactionid": "TransactionID"})
    df = txn.merge(graph, on="TransactionID", how="left")
    # Transactions with zero graph connections never appear in graph_features
    # (an isolated node still exists in the graph, but degree=0 nodes were
    # still written -- this fillna covers the rare merge mismatch case).
    for col in GRAPH_FEATURE_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    log.info("Loaded %s rows, %s columns after merging graph features", *df.shape)
    return df


def prepare_features(df: pd.DataFrame, include_graph_features: bool) -> tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    y = df["isFraud"].astype(int)

    cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns]
    for col in cat_cols:
        df[col] = df[col].astype(str)

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS and c not in cat_cols]
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]  # drop any remaining stray text columns
    if not include_graph_features:
        feature_cols = [c for c in feature_cols if c not in GRAPH_FEATURE_COLS]

    X = df[feature_cols + cat_cols].copy()
    return X, y


def time_based_split(df: pd.DataFrame, test_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_sorted = df.sort_values("transaction_day" if "transaction_day" in df.columns else "TransactionDT")
    split_idx = int(len(df_sorted) * (1 - test_frac))
    train, test = df_sorted.iloc[:split_idx], df_sorted.iloc[split_idx:]
    log.info("Time-based split: %s train, %s test (test = most recent %.0f%%)", len(train), len(test), test_frac * 100)
    return train, test


def train_and_evaluate(train_df: pd.DataFrame, test_df: pd.DataFrame, include_graph_features: bool, cat_cols: list[str]) -> dict:
    X_train, y_train = prepare_features(train_df, include_graph_features)
    X_test, y_test = prepare_features(test_df, include_graph_features)

    num_cols = [c for c in X_train.columns if c not in cat_cols]
    all_nan_cols = [c for c in num_cols if X_train[c].notna().sum() == 0]
    if all_nan_cols:
        log.warning("Dropping all-NaN columns (no signal possible): %s", all_nan_cols)
        num_cols = [c for c in num_cols if c not in all_nan_cols]
        X_train = X_train.drop(columns=all_nan_cols)
        X_test = X_test.drop(columns=all_nan_cols)

    if cat_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_train[cat_cols] = enc.fit_transform(X_train[cat_cols])
        X_test[cat_cols] = enc.transform(X_test[cat_cols])

    imputer = SimpleImputer(strategy="median")
    X_train[num_cols] = imputer.fit_transform(X_train[num_cols])
    X_test[num_cols] = imputer.transform(X_test[num_cols])

    clf = RandomForestClassifier(n_estimators=200, max_depth=12, class_weight="balanced", n_jobs=-1, random_state=42)
    clf.fit(X_train, y_train)

    proba = clf.predict_proba(X_test)[:, 1]
    if y_test.nunique() < 2:
        log.warning("Test split has only one class present (%s fraud cases) -- AUC/PR-AUC are not meaningful here. "
                     "This shouldn't happen on the real dataset's scale; if it does, the split logic needs a look.",
                     int(y_test.sum()))
        auc, ap = float("nan"), float("nan")
    else:
        auc = roc_auc_score(y_test, proba)
        ap = average_precision_score(y_test, proba)  # PR-AUC -- more honest than accuracy at 3.5% fraud rate

    top_features = sorted(zip(X_train.columns, clf.feature_importances_), key=lambda x: -x[1])[:8]

    return {
        "auc": auc, "average_precision": ap, "n_features": X_train.shape[1], "top_features": top_features,
        "model": clf, "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
    }


def run() -> None:
    mlflow.set_experiment("graphguard-fraud-detection")
    df = load_merged_features()
    df = df[df["isFraud"].notna()]  # drop any unlabeled rows (test-set-style rows with no label)

    cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns]
    train_df, test_df = time_based_split(df)

    for include_graph in (False, True):
        label = "rf_with_graph_features" if include_graph else "rf_no_graph_features"
        log.info("Training %s graph features...", "WITH" if include_graph else "WITHOUT")
        with mlflow.start_run(run_name=label):
            mlflow.set_tag("stage", "3-baseline")
            mlflow.log_params({
                "model_type": "RandomForestClassifier", "n_estimators": 200, "max_depth": 12,
                "class_weight": "balanced", "include_graph_features": include_graph,
                "split": "time-based, 80/20",
            })
            result = train_and_evaluate(train_df, test_df, include_graph_features=include_graph, cat_cols=cat_cols)
            mlflow.log_metrics({"test_auc": result["auc"], "test_pr_auc": result["average_precision"]})
            mlflow.log_param("n_features", result["n_features"])
            if include_graph:
                with_graph = result
                # This is the better-performing model (Stage 4 confirmed the GNN
                # doesn't beat it on held-out data) -- persist it so Stage 6 can
                # load the actual trained model instead of retraining from scratch.
                DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
                joblib.dump({
                    "model": result["model"], "X_train": result["X_train"], "X_test": result["X_test"],
                    "y_train": result["y_train"], "y_test": result["y_test"], "cat_cols": cat_cols,
                }, DATA_PROCESSED / "baseline_model.joblib")
                log.info("Saved trained model + prepared data to %s", DATA_PROCESSED / "baseline_model.joblib")
            else:
                no_graph = result

    log.info("=" * 70)
    log.info("RESULT -- without graph features: AUC=%.4f  PR-AUC=%.4f  (%s features)",
              no_graph["auc"], no_graph["average_precision"], no_graph["n_features"])
    log.info("RESULT -- with graph features:    AUC=%.4f  PR-AUC=%.4f  (%s features)",
              with_graph["auc"], with_graph["average_precision"], with_graph["n_features"])
    auc_delta = with_graph["auc"] - no_graph["auc"]
    ap_delta = with_graph["average_precision"] - no_graph["average_precision"]
    log.info("DELTA -- AUC: %+.4f   PR-AUC: %+.4f", auc_delta, ap_delta)
    log.info("=" * 70)
    log.info("Top features WITH graph features included:")
    for name, importance in with_graph["top_features"]:
        marker = "  <- graph feature" if name in GRAPH_FEATURE_COLS else ""
        log.info("  %-20s %.4f%s", name, importance, marker)
    log.info("Logged to MLflow -- run `mlflow ui` to view")


if __name__ == "__main__":
    run()
