"""Stage 6: explainability.

Explains the Random Forest (with graph features) directly -- this is the
model Stage 3/4 confirmed actually performs better on held-out data, so
it's the one worth explaining, not the GNN. Being a tree ensemble, it gets
SHAP's exact, fast TreeExplainer -- no surrogate-model workaround needed,
which a black-box GNN would have required.

Two methods, deliberately both:
  - SHAP for global feature importance (what matters across all
    predictions) and consistent, game-theoretically grounded local
    explanations.
  - LIME for local explanations on the same handful of flagged
    transactions, as a sanity check -- if two different explanation
    methods agree on why a transaction was flagged, that's much more
    convincing than either one alone.

Note on LIME's discretize_continuous=False: LIME's default behavior bins
continuous features into quartiles computed once, globally, from the
training data, then reuses those same bin edges for every instance it
explains. The IEEE-CIS V-columns are heavily sparse (many transactions
are exactly 0 for a given V-column), so several quartile edges can
collapse onto 0 -- "is it zero or not" then becomes the dominant split
LIME finds for nearly every transaction, regardless of what's actually
distinctive about that specific case. Symptom: every local explanation
comes back with the same features and the same "<= 0.00" phrasing,
no matter how different the transactions actually are. Disabling
discretization makes LIME fit its local linear surrogate on the raw
continuous values instead, restoring real per-instance sensitivity.

Note on restricting LIME to the top-30 SHAP-ranked features: even after
fixing the discretization issue, SHAP/LIME agreement stayed near zero.
This dataset has 410 features, many of them (the V-columns especially)
highly correlated with each other -- Vesta built them as grouped,
correlated aggregations. LIME's local *linear* regression becomes
genuinely unstable in a high-dimensional, collinear space: many
different coefficient assignments fit the local perturbed samples
almost equally well, so which features "win" is close to arbitrary.
This is a documented LIME limitation, not a bug. Restricting LIME to
perturb only the top-30 SHAP-important features (holding the rest fixed
at each instance's own actual values) is the standard mitigation --
if agreement is still poor even in this much smaller, more tractable
space, that's real evidence of a genuine LIME limitation on this data,
not something to keep chasing further.

Usage:
    python src/explain.py
"""
from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # no display available when run from a script
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer

warnings.filterwarnings("ignore", message="X does not have valid feature names")  # LIME samples with raw arrays by design

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline import GRAPH_FEATURE_COLS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DATA_PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"

SHAP_SAMPLE_SIZE = 2000
N_LOCAL_EXPLANATIONS = 3


def load_artifacts() -> dict:
    path = DATA_PROCESSED / "baseline_model.joblib"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run src/baseline.py first to train and save the model.")
    return joblib.load(path)


def run_shap_global(model, X_test, feature_names: list[str]) -> shap.Explainer:
    log.info("Computing SHAP values (TreeExplainer) on a sample of %s test rows...", min(SHAP_SAMPLE_SIZE, len(X_test)))
    sample = X_test.sample(n=min(SHAP_SAMPLE_SIZE, len(X_test)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(sample)

    # Binary classifier -- TreeExplainer returns values for both classes;
    # we care about the "fraud" class (index 1).
    if shap_values.values.ndim == 3:
        fraud_shap = shap_values[:, :, 1]
    else:
        fraud_shap = shap_values

    mean_abs = np.abs(fraud_shap.values).mean(axis=0)
    ranked = sorted(zip(feature_names, mean_abs), key=lambda x: -x[1])[:15]
    log.info("Top 15 features by mean |SHAP value| (global importance):")
    for name, val in ranked:
        marker = "  <- graph feature" if name in GRAPH_FEATURE_COLS else ""
        log.info("  %-20s %.4f%s", name, val, marker)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure()
    shap.summary_plot(fraud_shap, sample, plot_type="bar", max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "shap_global_importance.png", dpi=120)
    plt.close()
    log.info("Saved global SHAP plot to %s", REPORTS_DIR / "shap_global_importance.png")

    top_30_names = [name for name, _ in sorted(zip(feature_names, mean_abs), key=lambda x: -x[1])[:30]]
    return explainer, sample, fraud_shap, top_30_names


def pick_interesting_cases(model, X_test, y_test, n: int) -> list[int]:
    """Highest-confidence true-positive fraud predictions -- the cases an
    analyst would actually want explained, not random rows."""
    proba = model.predict_proba(X_test)[:, 1]
    fraud_actual = (y_test == 1).values
    if fraud_actual.sum() == 0:
        log.warning("No actual fraud cases in this test set (synthetic-data quirk) -- "
                     "picking highest-confidence predictions instead, real or not.")
        candidates = np.argsort(-proba)[:n]
    else:
        fraud_indices = np.where(fraud_actual)[0]
        ranked = fraud_indices[np.argsort(-proba[fraud_indices])]
        candidates = ranked[:n]
    return list(candidates)


def make_reduced_predict_fn(model, full_columns: list[str], reduced_columns: list[str], background_row):
    """Wraps model.predict_proba so LIME can perturb only `reduced_columns`
    while every other feature stays fixed at this specific instance's own
    actual values. Necessary because the model needs all 410 features to
    predict, but full-410-feature LIME is unstable here -- see module
    docstring on why. background_row is the instance being explained, not
    a generic background, so "holding everything else fixed" means fixed
    at what actually happened in this transaction, not an artificial average.
    """
    reduced_idx = [full_columns.index(c) for c in reduced_columns]
    background_values = background_row.values

    def predict_fn(X_reduced: np.ndarray) -> np.ndarray:
        X_full = np.tile(background_values, (X_reduced.shape[0], 1))
        X_full[:, reduced_idx] = X_reduced
        return model.predict_proba(pd.DataFrame(X_full, columns=full_columns))

    return predict_fn


def explain_local_cases(model, explainer, X_train, X_test, y_test, case_idx: list[int], cat_cols: list[str], reduced_cols: list[str]) -> None:
    feature_names = list(X_train.columns)
    reduced_cat_indices = [reduced_cols.index(c) for c in cat_cols if c in reduced_cols]

    # Fit LIME on the top-30-by-SHAP-importance subspace, not all 410
    # features -- see module docstring for why full-dimensional LIME was
    # unstable on this dataset's highly correlated V-columns.
    lime_explainer = LimeTabularExplainer(
        X_train[reduced_cols].values, feature_names=reduced_cols, class_names=["legit", "fraud"],
        categorical_features=reduced_cat_indices, mode="classification", random_state=42,
        discretize_continuous=False,
    )

    for i, idx in enumerate(case_idx, start=1):
        row = X_test.iloc[idx]
        proba = model.predict_proba(row.to_frame().T)[0, 1]
        log.info("=" * 70)
        log.info("Case %s -- TransactionID row %s -- model fraud probability: %.4f", i, idx, proba)

        shap_row = explainer(row.to_frame().T)
        vals = shap_row.values[0, :, 1] if shap_row.values.ndim == 3 else shap_row.values[0]
        top_shap = sorted(zip(feature_names, vals), key=lambda x: -abs(x[1]))[:5]
        log.info("  SHAP top drivers:")
        for name, val in top_shap:
            marker = " <- graph feature" if name in GRAPH_FEATURE_COLS else ""
            log.info("    %-20s %+.4f%s", name, val, marker)

        predict_fn = make_reduced_predict_fn(model, feature_names, reduced_cols, row)
        lime_exp = lime_explainer.explain_instance(row[reduced_cols].values, predict_fn, num_features=5)
        log.info("  LIME top drivers (perturbing only the top-30 SHAP-ranked features):")
        for feature_desc, weight in lime_exp.as_list():
            log.info("    %-30s %+.4f", feature_desc, weight)

        shap_top_names = {name for name, _ in top_shap}
        lime_top_names = {desc.split()[0] for desc, _ in lime_exp.as_list()}
        overlap = shap_top_names & lime_top_names
        log.info("  Agreement: %s feature(s) appear in both methods' top 5 (%s)", len(overlap), ", ".join(overlap) or "none")


def run() -> None:
    artifacts = load_artifacts()
    model, X_train, X_test, y_train, y_test, cat_cols = (
        artifacts["model"], artifacts["X_train"], artifacts["X_test"],
        artifacts["y_train"], artifacts["y_test"], artifacts["cat_cols"],
    )
    feature_names = list(X_train.columns)

    explainer, sample, fraud_shap, top_30_names = run_shap_global(model, X_test, feature_names)

    case_idx = pick_interesting_cases(model, X_test, y_test, N_LOCAL_EXPLANATIONS)
    explain_local_cases(model, explainer, X_train, X_test, y_test, case_idx, cat_cols, top_30_names)

    log.info("=" * 70)
    log.info("Done. Global plot saved to reports/shap_global_importance.png")


if __name__ == "__main__":
    run()
