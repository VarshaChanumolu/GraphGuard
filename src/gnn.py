"""Stage 4: GNN model.

Builds a 2-layer GraphSAGE (PyTorch Geometric), trained via mini-batch
neighbor sampling -- not full-batch -- since full-batch training on a
590K-node graph would be impractical on a CPU-only machine. Mini-batch
neighbor sampling is also the architecturally correct way to use
GraphSAGE -- it's literally what the method was designed for.

Fed the SAME full tabular feature set Stage 3's baseline used (not just
the graph summary stats), so the GNN has the same information advantage
the Random Forest had, plus the graph structure on top. Same time-based
train/test split as Stage 3, for a fair head-to-head comparison.

Baseline to beat (Stage 3, with graph features): AUC=0.8688, PR-AUC=0.4478

Usage:
    python src/gnn.py
"""
from __future__ import annotations

import logging
import pickle
import sys
from pathlib import Path

import mlflow
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline import CATEGORICAL_COLS, NON_FEATURE_COLS, load_merged_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DATA_PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"

BASELINE_AUC = 0.8688
BASELINE_PR_AUC = 0.4478

HIDDEN_CHANNELS = 64
MAX_EPOCHS = 15
EARLY_STOP_PATIENCE = 3
BATCH_SIZE = 512
NEIGHBORS_PER_HOP = [15, 10]  # 2-hop neighborhood sample sizes


def time_based_split_three(df: pd.DataFrame, val_frac: float = 0.1, test_frac: float = 0.2):
    sort_col = "transaction_day" if "transaction_day" in df.columns else "TransactionDT"
    df_sorted = df.sort_values(sort_col)
    n = len(df_sorted)
    test_start = int(n * (1 - test_frac))
    val_start = int(n * (1 - test_frac - val_frac))
    train, val, test = df_sorted.iloc[:val_start], df_sorted.iloc[val_start:test_start], df_sorted.iloc[test_start:]
    log.info("Three-way split: %s train, %s val, %s test (val and test are both held out of training)",
              len(train), len(val), len(test))
    return train, val, test


def build_pyg_data() -> Data:
    with open(DATA_PROCESSED / "transaction_graph.pkl", "rb") as f:
        G = pickle.load(f)
    node_list = list(G.nodes())
    node_to_idx = {tid: i for i, tid in enumerate(node_list)}
    log.info("Graph: %s nodes, %s edges", G.number_of_nodes(), G.number_of_edges())

    df = load_merged_features()
    df = df[df["isFraud"].notna()].copy()

    train_df, val_df, test_df = time_based_split_three(df)
    train_ids = set(train_df["TransactionID"])
    val_ids = set(val_df["TransactionID"])

    df_aligned = df.set_index("TransactionID").reindex(node_list)
    n_missing = df_aligned.isna().all(axis=1).sum()
    if n_missing:
        log.warning("%s graph nodes had no matching feature row -- check for a data mismatch", n_missing)

    cat_cols = [c for c in CATEGORICAL_COLS if c in df_aligned.columns]
    for col in cat_cols:
        df_aligned[col] = df_aligned[col].fillna("missing").astype(str)

    feature_cols = [c for c in df_aligned.columns if c not in NON_FEATURE_COLS and c not in cat_cols]
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df_aligned[c])]
    # Pandas nullable dtypes (Int64 etc.) use pd.NA, which SimpleImputer doesn't
    # reliably detect as missing -- force to plain float64/np.nan first.
    df_aligned[feature_cols] = df_aligned[feature_cols].astype("float64")

    train_mask_arr = pd.Series(node_list).isin(train_ids).values
    val_mask_arr = pd.Series(node_list).isin(val_ids).values
    is_train_row = train_mask_arr  # fit preprocessing on pure training rows only -- val and test both stay unseen

    if cat_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        enc.fit(df_aligned.loc[is_train_row, cat_cols])
        df_aligned[cat_cols] = enc.transform(df_aligned[cat_cols])

    all_nan_cols = [c for c in feature_cols if df_aligned.loc[is_train_row, c].notna().sum() == 0]
    if all_nan_cols:
        log.warning("Dropping all-NaN columns: %s", all_nan_cols)
        feature_cols = [c for c in feature_cols if c not in all_nan_cols]

    imputer = SimpleImputer(strategy="median")
    imputer.fit(df_aligned.loc[is_train_row, feature_cols])
    df_aligned[feature_cols] = imputer.transform(df_aligned[feature_cols])

    # Neural nets are far more sensitive to feature scale than tree models --
    # without this, card1 (~thousands) drowns out 0/1 flags and gradients blow up.
    scaler = StandardScaler()
    scaler.fit(df_aligned.loc[is_train_row, feature_cols])
    df_aligned[feature_cols] = scaler.transform(df_aligned[feature_cols])

    all_cols = feature_cols + cat_cols
    x = torch.tensor(df_aligned[all_cols].values, dtype=torch.float)
    y = torch.tensor(df_aligned["isFraud"].fillna(0).values, dtype=torch.long)

    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()]
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # explicit undirected
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    train_mask = torch.tensor(train_mask_arr, dtype=torch.bool)
    val_mask = torch.tensor(val_mask_arr, dtype=torch.bool)
    test_mask = ~(train_mask | val_mask)

    data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    log.info("Built PyG graph: x=%s, edge_index=%s, %s train / %s val / %s test nodes (%s features)",
              tuple(x.shape), tuple(edge_index.shape), train_mask.sum().item(), val_mask.sum().item(),
              test_mask.sum().item(), len(all_cols))
    return data


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int = 2):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        return self.lin(x)


def train_model(data: Data, class_weights: torch.Tensor) -> tuple[GraphSAGE, int]:
    model = GraphSAGE(in_channels=data.x.shape[1], hidden_channels=HIDDEN_CHANNELS)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    # Reduce LR when validation PR-AUC stalls -- ties the schedule to real
    # signal instead of guessing a decay schedule blind.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)

    train_loader = NeighborLoader(
        data, num_neighbors=NEIGHBORS_PER_HOP, batch_size=BATCH_SIZE,
        input_nodes=data.train_mask, shuffle=True,
    )

    best_val_pr_auc = -1.0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad()
            out = batch_forward(model, batch)
            seed_out = out[: batch.batch_size]
            seed_y = batch.y[: batch.batch_size]
            loss = F.cross_entropy(seed_out, seed_y, weight=class_weights)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        val_auc, val_pr_auc = evaluate_model(model, data, data.val_mask)
        scheduler.step(val_pr_auc)
        current_lr = optimizer.param_groups[0]["lr"]
        log.info("Epoch %s/%s -- avg loss %.4f -- val AUC=%.4f PR-AUC=%.4f -- lr=%.5f",
                  epoch, MAX_EPOCHS, total_loss / max(n_batches, 1), val_auc, val_pr_auc, current_lr)
        mlflow.log_metrics({
            "train_loss": total_loss / max(n_batches, 1), "val_auc": val_auc,
            "val_pr_auc": val_pr_auc, "lr": current_lr,
        }, step=epoch)

        if val_pr_auc > best_val_pr_auc:
            best_val_pr_auc = val_pr_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                log.info("No val PR-AUC improvement for %s epochs -- stopping early at epoch %s", EARLY_STOP_PATIENCE, epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        log.info("Restored best checkpoint (val PR-AUC=%.4f)", best_val_pr_auc)
    return model, epoch


def batch_forward(model: GraphSAGE, batch) -> torch.Tensor:
    return model(batch.x, batch.edge_index)


def evaluate_model(model: GraphSAGE, data: Data, mask: torch.Tensor) -> tuple[float, float]:
    model.eval()
    loader = NeighborLoader(
        data, num_neighbors=NEIGHBORS_PER_HOP, batch_size=BATCH_SIZE,
        input_nodes=mask, shuffle=False,
    )
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            out = batch_forward(model, batch)
            seed_out = out[: batch.batch_size]
            seed_y = batch.y[: batch.batch_size]
            probs = F.softmax(seed_out, dim=1)[:, 1]
            all_probs.append(probs)
            all_labels.append(seed_y)
    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()
    if len(set(labels.tolist())) < 2:
        log.warning("Split has only one class present -- AUC/PR-AUC are not meaningful here.")
        return float("nan"), float("nan")
    return roc_auc_score(labels, probs), average_precision_score(labels, probs)


def run() -> None:
    mlflow.set_experiment("graphguard-fraud-detection")
    data = build_pyg_data()

    n_pos = int((data.y[data.train_mask] == 1).sum())
    n_neg = int((data.y[data.train_mask] == 0).sum())
    raw_ratio = n_neg / max(n_pos, 1)
    # Raw imbalance ratio (~27x) makes weighted cross-entropy noisy and
    # unstable -- sqrt is a standard, principled way to soften this without
    # throwing away the imbalance signal entirely.
    softened_ratio = raw_ratio ** 0.5
    class_weights = torch.tensor([1.0, softened_ratio], dtype=torch.float)
    log.info("Class weight ratio -- raw=%.2f, softened (sqrt)=%.2f", raw_ratio, softened_ratio)

    with mlflow.start_run(run_name="gnn_run"):
        mlflow.set_tag("stage", "4-gnn")
        mlflow.log_params({
            "model_type": "GraphSAGE-2layer", "hidden_channels": HIDDEN_CHANNELS, "max_epochs": MAX_EPOCHS,
            "lr_initial": 0.01, "lr_schedule": "ReduceLROnPlateau(factor=0.5, patience=1)",
            "class_weight_ratio_raw": round(raw_ratio, 2), "class_weight_ratio_softened": round(softened_ratio, 2),
            "class_weight_softening": "sqrt", "early_stopping_patience": EARLY_STOP_PATIENCE,
            "neighbors_per_hop": str(NEIGHBORS_PER_HOP), "split": "time-based, 70/10/20 (train/val/test)",
        })

        model, stopped_epoch = train_model(data, class_weights)
        auc, ap = evaluate_model(model, data, data.test_mask)
        mlflow.log_metrics({"stopped_epoch": stopped_epoch, "test_auc": auc, "test_pr_auc": ap})

    log.info("=" * 70)
    log.info("GNN RESULT (held-out test, never seen during training or early stopping) -- AUC=%.4f  PR-AUC=%.4f", auc, ap)
    log.info("BASELINE (Stage 3)                                                       -- AUC=%.4f  PR-AUC=%.4f", BASELINE_AUC, BASELINE_PR_AUC)
    log.info("DELTA -- AUC: %+.4f   PR-AUC: %+.4f  (trained for %s epochs)", auc - BASELINE_AUC, ap - BASELINE_PR_AUC, stopped_epoch)
    log.info("=" * 70)
    log.info("Logged to MLflow -- run `mlflow ui` to view")

    torch.save(model.state_dict(), DATA_PROCESSED / "gnn_model.pt")
    log.info("Saved best-checkpoint model weights to %s", DATA_PROCESSED / "gnn_model.pt")


if __name__ == "__main__":
    run()
