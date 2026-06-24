"""Stage 1: data ingestion.

Design decision: the IEEE-CIS dataset has ~430 columns total, most of which
(the V1-V339 PCA-style features, plus the C/D/M blocks) are dense numeric
features meant for modeling, not for entity relationships. Cramming all of
them into Postgres would mean a 400+ column relational table, which is the
wrong tool for the job.

So this script splits the data on load:
  - CORE columns (the ones that define entity relationships: card, address,
    email, device) go to Postgres. This is what Stage 2 (graph construction)
    and the Tableau dashboard query against.
  - The FULL feature set (core + all C/D/M/V columns) goes to a Parquet file
    in data/processed/. This is what Stage 3/4 (baseline + GNN) train on —
    Parquet is far more efficient than a wide SQL table for that workload.

Usage:
    python src/ingest.py                  # uses data/raw/*.csv
    python src/ingest.py --synthetic       # generates + uses a fake sample
                                            # (for testing without the real
                                            # Kaggle download)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so `tests` is importable when run as a script
from config import PG

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DATA_RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
DATA_PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"

# Columns that define entity relationships -- these are what the graph in
# Stage 2 gets built from, so they're the ones worth a proper relational
# schema with indexes.
CORE_TRANSACTION_COLS = [
    "TransactionID", "isFraud", "TransactionDT", "TransactionAmt",
    "ProductCD", "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2", "dist1", "dist2", "P_emaildomain", "R_emaildomain",
]
CORE_IDENTITY_COLS = ["TransactionID", "DeviceType", "DeviceInfo"]


def load_raw(transaction_path: Path, identity_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    log.info("Loading %s", transaction_path.name)
    txn = pd.read_csv(transaction_path)
    log.info("Loading %s", identity_path.name)
    ident = pd.read_csv(identity_path)
    log.info("Loaded %s transactions, %s identity records", len(txn), len(ident))
    return txn, ident


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "isFraud" not in df.columns:
        df["isFraud"] = pd.NA  # test set has no label

    df["P_emaildomain"] = df.get("P_emaildomain", pd.Series(dtype=object)).fillna("unknown").str.lower()
    df["R_emaildomain"] = df.get("R_emaildomain", pd.Series(dtype=object)).fillna("unknown").str.lower()

    for col in ["card1", "card2", "card3", "card5", "addr1", "addr2"]:
        if col in df.columns:
            df[col] = df[col].astype("Int64")

    # TransactionDT is seconds-since-reference, not a real timestamp in this
    # dataset -- bucket into days, useful later for velocity / time-series features.
    df["transaction_day"] = (df["TransactionDT"] // (24 * 60 * 60)).astype("Int64")
    return df


def clean_identity(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["DeviceType", "DeviceInfo"]:
        if col in df.columns:
            df[col] = df[col].fillna("unknown")
    return df


def write_core_to_postgres(txn: pd.DataFrame, ident: pd.DataFrame) -> None:
    engine = create_engine(PG.sqlalchemy_url)

    core_txn_cols = [c for c in CORE_TRANSACTION_COLS if c in txn.columns] + ["transaction_day"]
    core_ident_cols = [c for c in CORE_IDENTITY_COLS if c in ident.columns]

    txn_core = txn[core_txn_cols].rename(columns=str.lower)
    ident_core = ident[core_ident_cols].rename(columns=str.lower)

    with engine.begin() as conn:
        log.info("Writing %s rows to transactions table", len(txn_core))
        txn_core.to_sql("transactions", conn, if_exists="replace", index=False)
        conn.execute(text("ALTER TABLE transactions ADD PRIMARY KEY (transactionid)"))

        log.info("Writing %s rows to identity table", len(ident_core))
        ident_core.to_sql("identity", conn, if_exists="replace", index=False)

        # Indexes on the columns Stage 2's graph construction will join/filter on
        for col in ["card1", "addr1", "p_emaildomain"]:
            conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_txn_{col} ON transactions ({col})"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_ident_txnid ON identity (transactionid)"))

    log.info("Postgres write complete: transactions(%s rows), identity(%s rows)", len(txn_core), len(ident_core))


def write_processed_parquet(txn: pd.DataFrame, ident: pd.DataFrame) -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    full = txn.merge(ident, on="TransactionID", how="left", suffixes=("", "_id"))
    out_path = DATA_PROCESSED / "transactions_full.parquet"
    full.to_parquet(out_path, index=False)
    log.info("Wrote full feature set (%s cols, %s rows) to %s", full.shape[1], full.shape[0], out_path)


def run(transaction_path: Path, identity_path: Path) -> None:
    txn, ident = load_raw(transaction_path, identity_path)
    txn = clean_transactions(txn)
    ident = clean_identity(ident)

    write_core_to_postgres(txn, ident)
    write_processed_parquet(txn, ident)

    fraud_rate = txn["isFraud"].mean() if txn["isFraud"].notna().any() else float("nan")
    log.info("Done. Fraud rate in this batch: %.4f", fraud_rate)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true", help="generate and use a fake sample instead of real Kaggle data")
    args = parser.parse_args()

    if args.synthetic:
        from tests.generate_synthetic_sample import generate
        txn_path, ident_path = generate(DATA_RAW)
    else:
        txn_path = DATA_RAW / "train_transaction.csv"
        ident_path = DATA_RAW / "train_identity.csv"
        if not txn_path.exists() or not ident_path.exists():
            raise FileNotFoundError(
                f"Expected {txn_path} and {ident_path}. "
                "Download them from Kaggle (see README) or run with --synthetic to test the pipeline first."
            )

    run(txn_path, ident_path)


if __name__ == "__main__":
    main()
