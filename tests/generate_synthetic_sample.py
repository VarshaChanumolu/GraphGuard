"""Generates a synthetic sample matching the IEEE-CIS schema, so the
pipeline can be built and tested before the real ~650MB Kaggle dataset
is downloaded. Swap in the real CSVs later -- the ingest script doesn't
change at all.

Includes a few intentionally-repeated card1/addr1/device combos so that
Stage 2 (graph construction) has actual shared-entity structure to find,
not just isolated nodes.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(seed=42)
random.seed(42)

N_TRANSACTIONS = 500
N_IDENTITY = 150  # IEEE-CIS only has identity records for a minority of transactions
FRAUD_RATE = 0.035  # matches the real dataset's ~3.5% imbalance


def generate(out_dir: Path) -> tuple[Path, Path]:
    # Deliberately a separate subfolder from where real Kaggle data lives,
    # so a --synthetic test run can never overwrite the real download.
    out_dir = out_dir / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)

    # A small pool of repeated entity values, so some transactions
    # genuinely share a card / address / email -- this is what makes
    # the graph in Stage 2 non-trivial.
    # Pool sizes are deliberately large relative to N_TRANSACTIONS so that
    # incidental entity-sharing among *legitimate* transactions is rare --
    # matching the real dataset's sparsity (~13,500 distinct card1 values
    # across 590K rows). If these pools are too small, every transaction
    # incidentally shares an entity with several others, and the planted
    # fraud ring below gets drowned out in noise rather than standing out.
    card1_pool = RNG.integers(1000, 9999, size=400)
    addr1_pool = RNG.integers(100, 599, size=200)
    email_pool = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "icloud.com", "unknown"]
    device_pool = [f"device_{i}" for i in range(25)]

    txn_ids = np.arange(2_987_000, 2_987_000 + N_TRANSACTIONS)
    is_fraud = RNG.choice([0, 1], size=N_TRANSACTIONS, p=[1 - FRAUD_RATE, FRAUD_RATE])

    transactions = pd.DataFrame({
        "TransactionID": txn_ids,
        "isFraud": is_fraud,
        "TransactionDT": np.sort(RNG.integers(86_400, 86_400 * 180, size=N_TRANSACTIONS)),
        "TransactionAmt": np.round(RNG.lognormal(mean=4.0, sigma=1.1, size=N_TRANSACTIONS), 2),
        "ProductCD": RNG.choice(["W", "C", "R", "H", "S"], size=N_TRANSACTIONS, p=[0.6, 0.15, 0.1, 0.1, 0.05]),
        "card1": RNG.choice(card1_pool, size=N_TRANSACTIONS),
        "card2": RNG.choice(list(RNG.integers(100, 600, size=30)) + [np.nan], size=N_TRANSACTIONS),
        "card3": RNG.choice([150.0, 185.0, np.nan], size=N_TRANSACTIONS, p=[0.85, 0.1, 0.05]),
        "card4": RNG.choice(["visa", "mastercard", "american express", "discover"], size=N_TRANSACTIONS),
        "card5": RNG.choice(list(RNG.integers(100, 240, size=20)) + [np.nan], size=N_TRANSACTIONS),
        "card6": RNG.choice(["debit", "credit"], size=N_TRANSACTIONS),
        "addr1": RNG.choice(list(addr1_pool) + [np.nan], size=N_TRANSACTIONS),
        "addr2": RNG.choice([87.0, 60.0, np.nan], size=N_TRANSACTIONS, p=[0.8, 0.15, 0.05]),
        "dist1": RNG.choice(list(RNG.integers(0, 500, size=50).astype(float)) + [np.nan] * 10, size=N_TRANSACTIONS),
        "dist2": [np.nan] * N_TRANSACTIONS,  # genuinely sparse in the real data too
        "P_emaildomain": RNG.choice(email_pool, size=N_TRANSACTIONS),
        "R_emaildomain": RNG.choice(email_pool + ["unknown"] * 3, size=N_TRANSACTIONS),
    })

    # Bias fraud rows toward sharing entities, like real fraud rings do --
    # otherwise the synthetic sample teaches nothing about why the graph helps.
    # Crucially, also cluster them in a TIGHT time window (a few days), not
    # spread across the full range -- this is what should distinguish a real
    # fraud ring from a legitimate customer who happens to reuse a card over
    # months. See the legit repeat-customer block below for the contrast case.
    fraud_idx = transactions.index[transactions["isFraud"] == 1]
    if len(fraud_idx) > 1:
        shared_card = RNG.choice(card1_pool[:5])
        shared_addr = RNG.choice(addr1_pool[:5])
        transactions.loc[fraud_idx, "card1"] = shared_card
        transactions.loc[fraud_idx, "addr1"] = shared_addr
        ring_start = RNG.integers(86_400 * 20, 86_400 * 150)
        transactions.loc[fraud_idx, "TransactionDT"] = ring_start + RNG.integers(0, 86_400 * 3, size=len(fraud_idx))

    # Legitimate repeat customers: a handful of card1 values reused by
    # ordinary (non-fraud) transactions, but spread across 60+ days -- the
    # pattern that should NOT be flagged as a ring once time-windowing is in
    # place, but WOULD look identical to the fraud ring under naive
    # any-time-in-180-days entity linking.
    legit_idx = transactions.index[transactions["isFraud"] == 0]
    repeat_customer_cards = RNG.choice(card1_pool[5:15], size=4, replace=False)
    for card in repeat_customer_cards:
        chosen = RNG.choice(legit_idx, size=min(5, len(legit_idx)), replace=False)
        transactions.loc[chosen, "card1"] = card
        spread_start = RNG.integers(86_400 * 5, 86_400 * 60)
        transactions.loc[chosen, "TransactionDT"] = spread_start + RNG.integers(0, 86_400 * 90, size=len(chosen))

    identity_txn_ids = RNG.choice(txn_ids, size=N_IDENTITY, replace=False)
    identity = pd.DataFrame({
        "TransactionID": np.sort(identity_txn_ids),
        "DeviceType": RNG.choice(["mobile", "desktop"], size=N_IDENTITY),
        "DeviceInfo": RNG.choice(device_pool, size=N_IDENTITY),
    })

    txn_path = out_dir / "train_transaction.csv"
    ident_path = out_dir / "train_identity.csv"
    transactions.to_csv(txn_path, index=False)
    identity.to_csv(ident_path, index=False)
    return txn_path, ident_path


if __name__ == "__main__":
    p1, p2 = generate(Path(__file__).resolve().parent.parent / "data" / "raw")
    print(f"Wrote {p1} and {p2}")
