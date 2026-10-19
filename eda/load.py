"""
Pure I/O loaders for the three heterogeneous raw sources described in the companion
feature-pipeline assignment's data_dictionary.md (or, if you don't have that document, from
your own Stage 0 read of the raw sources). No analysis logic belongs here — only loading and
basic type coercion, so loading can be unit-tested and reused independently of
`profile.py`, `hypotheses.py`, `time_checks.py`, and `leakage_checks.py`.

Fill in each function per your confirmed `eda_review.md`.
"""

import os
import sqlite3
import pandas as pd

# Automatically locate the project root directory (one level up from eda/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_customers(path: str = None) -> pd.DataFrame:
    """Load customers.csv.

    Read the CSV with pandas.
    Parse `signup_date` as a date.
    Return one row per customer_id, unmodified otherwise.
    """
    if path is None:
        path = os.path.join(PROJECT_ROOT, "data", "customers.csv")
    df = pd.read_csv(path)
    df["signup_date"] = pd.to_datetime(df["signup_date"]).dt.date
    return df


def load_transactions(path: str = None) -> pd.DataFrame:
    """Load the row-level `transactions` table from the SQLite database.

    Connect with sqlite3 (or pandas.read_sql) against `path`.
    Cast `txn_timestamp` to a real datetime (SQLite stores it as ISO-8601 text).
    Cast `is_returned` to a clean 0/1 int.
    Return the full row-level frame — do not aggregate here.
    """
    if path is None:
        path = os.path.join(PROJECT_ROOT, "data", "transactions.db")
    conn = sqlite3.connect(path)
    try:
        df = pd.read_sql_query("SELECT * FROM transactions", conn)
    finally:
        conn.close()

    df["txn_timestamp"] = pd.to_datetime(df["txn_timestamp"])
    df["is_returned"] = df["is_returned"].astype(int)
    return df


def load_monthly_agg(path: str = None) -> pd.DataFrame:
    """Load the pre-aggregated customer x month parquet extract.

    Read the parquet file with pandas.
    Parse `txn_month` as a date (first-of-month).
    Return as-is — remember customers/months with zero activity have no row.
    """
    if path is None:
        path = os.path.join(PROJECT_ROOT, "data", "transactions_monthly_agg.parquet")
    df = pd.read_parquet(path)
    df["txn_month"] = pd.to_datetime(df["txn_month"]).dt.date
    return df