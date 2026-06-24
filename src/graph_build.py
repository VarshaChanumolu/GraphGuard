"""Stage 2: graph construction.

Builds a transaction graph where nodes = transactions and edges connect
transactions that share an identifying entity (card, address, or device).

Design decision -- read this before changing ENTITY_COLS or MAX_GROUP_SIZE:
not every shared column is a good edge-forming signal. P_emaildomain has
only ~20 distinct values (gmail.com, yahoo.com, ...) each shared by tens
of thousands of transactions -- using it directly would collapse the graph
into a few enormous hairballs that say "this transaction used Gmail," not
"this transaction is linked to a fraud ring." So a shared value only forms
edges if its group size stays under MAX_GROUP_SIZE: specific enough to
mean something, not so common it's noise. Tune this constant and rerun if
you want to see the tradeoff -- it's a genuine judgment call, not a fixed
rule, and worth documenting as a design decision.

Usage:
    python src/graph_build.py
"""
from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PG

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DATA_PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"

ENTITY_COLS = ["card1", "device_key"]  # P_emaildomain and addr1 deliberately excluded -- see module docstring and the time-windowed diagnostic results (addr1 came back at 0.89x, confirming it's noise, not signal)
MAX_GROUP_SIZE = 50
TIME_WINDOW_DAYS = 7  # entities only link transactions within this many days of each other -- see diagnose_entity_columns for why


def load_core_data() -> pd.DataFrame:
    engine = create_engine(PG.sqlalchemy_url)
    txn = pd.read_sql("SELECT * FROM transactions", engine)
    ident = pd.read_sql("SELECT * FROM identity", engine)
    df = txn.merge(ident, on="transactionid", how="left")

    df["device_key"] = df["devicetype"].fillna("") + "_" + df["deviceinfo"].fillna("")
    df.loc[df["device_key"] == "_", "device_key"] = None
    df["time_bucket"] = df["transaction_day"] // TIME_WINDOW_DAYS

    log.info("Loaded %s transactions (identity merged, %s have a device match)",
              len(df), df["devicetype"].notna().sum())
    return df


def build_graph(df: pd.DataFrame, entity_cols: list[str] = ENTITY_COLS, max_group_size: int = MAX_GROUP_SIZE) -> nx.Graph:
    G = nx.Graph()

    nodes = [
        (tid, {
            "is_fraud": int(isfraud) if pd.notna(isfraud) else None,
            "amt": float(amt),
            "product_cd": pcd,
            "day": int(day) if pd.notna(day) else None,
        })
        for tid, isfraud, amt, pcd, day in zip(
            df["transactionid"], df["isfraud"], df["transactionamt"], df["productcd"], df["transaction_day"]
        )
    ]
    G.add_nodes_from(nodes)

    skipped_hub_groups = 0
    for col in entity_cols:
        if col not in df.columns:
            log.warning("Entity column %s not found, skipping", col)
            continue
        groups = df.dropna(subset=[col]).groupby([col, "time_bucket"])["transactionid"].apply(list)
        for txn_ids in groups:
            if len(txn_ids) < 2:
                continue
            if len(txn_ids) > max_group_size:
                skipped_hub_groups += 1
                continue
            for i in range(len(txn_ids)):
                for j in range(i + 1, len(txn_ids)):
                    a, b = txn_ids[i], txn_ids[j]
                    if G.has_edge(a, b):
                        G[a][b]["shared"].add(col)
                    else:
                        G.add_edge(a, b, shared={col})

    log.info(
        "Graph built: %s nodes, %s edges (%s oversized entity groups skipped as non-specific, threshold=%s)",
        G.number_of_nodes(), G.number_of_edges(), skipped_hub_groups, max_group_size,
    )
    return G


def compute_graph_features(G: nx.Graph) -> pd.DataFrame:
    log.info("Computing graph-theoretic features for %s nodes", G.number_of_nodes())
    degree = dict(G.degree())
    clustering = nx.clustering(G)

    comp_size_map: dict[int, int] = {}
    node_to_comp: dict[int, int] = {}
    for i, comp in enumerate(nx.connected_components(G)):
        comp_size_map[i] = len(comp)
        for node in comp:
            node_to_comp[node] = i

    rows = [
        {
            "transactionid": node,
            "degree": degree[node],
            "clustering_coeff": clustering[node],
            "component_id": node_to_comp[node],
            "component_size": comp_size_map[node_to_comp[node]],
        }
        for node in G.nodes()
    ]
    return pd.DataFrame(rows)


def save_outputs(G: nx.Graph, features: pd.DataFrame) -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    graph_path = DATA_PROCESSED / "transaction_graph.pkl"
    with open(graph_path, "wb") as f:
        pickle.dump(G, f)
    log.info("Saved graph to %s", graph_path)

    features_path = DATA_PROCESSED / "graph_features.parquet"
    features.to_parquet(features_path, index=False)
    log.info("Saved graph features to %s", features_path)


def diagnose_entity_columns(df: pd.DataFrame, entity_cols: list[str] = ENTITY_COLS, max_group_size: int = MAX_GROUP_SIZE) -> None:
    """Ablation: for each entity column on its own, how much fraud/legit
    degree separation does it provide? Run this before trusting the
    combined graph -- a column that looks fine in aggregate can be hiding
    a weak (or actively noisy) contribution behind a strong one.

    Entity-linking here is time-windowed (TIME_WINDOW_DAYS) -- two
    transactions only connect if they share the entity AND fall within
    the same window. Without this, "same card, six months apart" looks
    identical to "same card, same day," which conflates a legitimate
    repeat customer with an actual fraud ring.
    """
    fraud_mask = df["isfraud"] == 1
    legit_mask = df["isfraud"] == 0
    log.info("Per-column ablation (fraud rate: %.4f, max_group_size=%s, time_window=%sd)",
              fraud_mask.mean(), max_group_size, TIME_WINDOW_DAYS)
    log.info("%-14s %10s %14s %14s %8s %12s", "column", "n_groups", "fraud_avg_deg", "legit_avg_deg", "ratio", "pct_isolated")

    for col in entity_cols:
        if col not in df.columns:
            log.warning("  %s not found, skipping", col)
            continue
        sizes = df.groupby([col, "time_bucket"])[col].transform("size")
        qualifies = (sizes >= 2) & (sizes <= max_group_size)
        degree = pd.Series(np.where(qualifies, sizes - 1, 0), index=df.index)

        n_groups = df.loc[qualifies].groupby([col, "time_bucket"]).ngroups
        fraud_avg = degree[fraud_mask].mean()
        legit_avg = degree[legit_mask].mean()
        pct_isolated = (degree == 0).mean() * 100
        ratio = fraud_avg / legit_avg if legit_avg > 0 else float("nan")
        log.info("%-14s %10d %14.2f %14.2f %7.2fx %11.1f%%", col, n_groups, fraud_avg, legit_avg, ratio, pct_isolated)

    log.info("Read this as: a column with ratio close to 1.0x is contributing noise, not signal -- "
              "candidate to drop from ENTITY_COLS or tighten further.")


def run() -> None:
    df = load_core_data()
    G = build_graph(df)
    features = compute_graph_features(G)
    save_outputs(G, features)

    # Sanity check: if entity-linking is capturing real structure, fraud
    # transactions should sit in denser neighborhoods than legitimate ones.
    fraud_degrees = [d for n, d in G.degree() if G.nodes[n]["is_fraud"] == 1]
    legit_degrees = [d for n, d in G.degree() if G.nodes[n]["is_fraud"] == 0]
    if fraud_degrees and legit_degrees:
        avg_fraud = sum(fraud_degrees) / len(fraud_degrees)
        avg_legit = sum(legit_degrees) / len(legit_degrees)
        log.info(
            "Avg degree -- fraud: %.2f, legit: %.2f  (fraud > legit is the signal the GNN will exploit; "
            "if they're close, the entity-linking isn't capturing real structure yet)",
            avg_fraud, avg_legit,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnose", action="store_true", help="run the per-column ablation report instead of building the full graph")
    args = parser.parse_args()

    if args.diagnose:
        diagnose_entity_columns(load_core_data())
    else:
        run()
