"""Stage 5 (part 1): backfill MLflow with the real experiment history from
Stages 3-4 -- run before MLflow tracking was wired into the scripts.

These are the actual numbers from the actual runs already done (Stage 3's
baseline, Stage 4's broken first GNN attempt, Stage 4's fixed second
attempt). Logging historical results after adopting a tracking tool is
normal practice, not a workaround -- this just makes sure the full
experimentation story (including the broken attempt and what fixed it)
lives in one place instead of scattered across terminal output.

Run once:
    python src/log_historical_runs.py
Then view it:
    mlflow ui
    -> open http://localhost:5000
"""
import mlflow

mlflow.set_experiment("graphguard-fraud-detection")

# --- Stage 3: Random Forest baseline ---
with mlflow.start_run(run_name="rf_no_graph_features"):
    mlflow.set_tag("stage", "3-baseline")
    mlflow.log_params({
        "model_type": "RandomForestClassifier", "n_estimators": 200, "max_depth": 12,
        "class_weight": "balanced", "include_graph_features": False, "n_features": 407,
        "split": "time-based, 80/20",
    })
    mlflow.log_metrics({"test_auc": 0.8674, "test_pr_auc": 0.4431})

with mlflow.start_run(run_name="rf_with_graph_features"):
    mlflow.set_tag("stage", "3-baseline")
    mlflow.log_params({
        "model_type": "RandomForestClassifier", "n_estimators": 200, "max_depth": 12,
        "class_weight": "balanced", "include_graph_features": True, "n_features": 410,
        "split": "time-based, 80/20",
    })
    mlflow.log_metrics({"test_auc": 0.8688, "test_pr_auc": 0.4478})

# --- Stage 4: GNN v1 -- initial attempt, before validation tracking existed ---
with mlflow.start_run(run_name="gnn_v1_initial"):
    mlflow.set_tag("stage", "4-gnn")
    mlflow.set_tag("notes", "Flat loss across epochs 2-5 -- underfitting, fixed in v2 with a real "
                            "validation split, LR scheduling, and softened class weights")
    mlflow.log_params({
        "model_type": "GraphSAGE-2layer", "hidden_channels": 64, "epochs": 5, "lr": 0.01,
        "lr_schedule": "none", "class_weight_ratio": 27.46, "class_weight_softening": "none",
        "early_stopping": False, "neighbors_per_hop": "[15, 10]",
        "split": "time-based, 80/20 (no separate val set)",
    })
    for epoch, loss in enumerate([0.5101, 0.4922, 0.4931, 0.4893, 0.4930], start=1):
        mlflow.log_metric("train_loss", loss, step=epoch)
    mlflow.log_metrics({"test_auc": 0.8282, "test_pr_auc": 0.1707})

# --- Stage 4: GNN v2 -- val split, softened weights, LR scheduling, early stopping ---
with mlflow.start_run(run_name="gnn_v2_tuned"):
    mlflow.set_tag("stage", "4-gnn")
    mlflow.set_tag("notes", "Val PR-AUC (0.4677) essentially matched the baseline, but held-out test "
                            "PR-AUC (0.3813) came in lower -- val/test gap on a chronological split is "
                            "consistent with fraud-pattern drift over time, not a training bug")
    mlflow.log_params({
        "model_type": "GraphSAGE-2layer", "hidden_channels": 64, "max_epochs": 15, "lr_initial": 0.01,
        "lr_schedule": "ReduceLROnPlateau(factor=0.5, patience=1)", "class_weight_ratio_raw": 27.43,
        "class_weight_ratio_softened": 5.24, "class_weight_softening": "sqrt",
        "early_stopping_patience": 3, "neighbors_per_hop": "[15, 10]",
        "split": "time-based, 70/10/20 (train/val/test)",
    })
    epoch_log = [
        (1, 0.3176, 0.8440, 0.3932, 0.01), (2, 0.3112, 0.8422, 0.3744, 0.01),
        (3, 0.3079, 0.8435, 0.3640, 0.005), (4, 0.2966, 0.8565, 0.4316, 0.005),
        (5, 0.2936, 0.8543, 0.4198, 0.005), (6, 0.2926, 0.8595, 0.4327, 0.005),
        (7, 0.2929, 0.8495, 0.4119, 0.005), (8, 0.2926, 0.8573, 0.4207, 0.0025),
        (9, 0.2819, 0.8613, 0.4472, 0.0025), (10, 0.2790, 0.8627, 0.4375, 0.0025),
        (11, 0.2775, 0.8660, 0.4646, 0.0025), (12, 0.2774, 0.8607, 0.4493, 0.0025),
        (13, 0.2759, 0.8630, 0.4494, 0.00125), (14, 0.2659, 0.8694, 0.4677, 0.00125),
        (15, 0.2649, 0.8676, 0.4567, 0.00125),
    ]
    for epoch, loss, val_auc, val_pr, lr in epoch_log:
        mlflow.log_metrics({"train_loss": loss, "val_auc": val_auc, "val_pr_auc": val_pr, "lr": lr}, step=epoch)
    mlflow.log_metric("best_checkpoint_epoch", 14)
    mlflow.log_metrics({"best_val_pr_auc": 0.4677, "test_auc": 0.8441, "test_pr_auc": 0.3813})

print("Logged 4 historical runs to MLflow (2 baseline, 2 GNN). Run `mlflow ui` and open http://localhost:5000 to view them.")
