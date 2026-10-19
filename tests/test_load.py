"""
Test skeletons for eda/load.py's loaders against the real provided files.

Fill these in against your confirmed eda_review.md and your actual loader functions — do not
leave them skipped in your submission.
"""

import datetime
import pandas as pd
import pytest

from eda.load import load_customers, load_monthly_agg, load_transactions


def test_load_customers_schema_and_row_count():
    """Load customers.csv and assert expected columns, dtypes, and a sane row count."""
    df = load_customers()

    assert not df.empty, "customers dataframe should not be empty."
    assert "customer_id" in df.columns, "Missing 'customer_id' column."
    assert "signup_date" in df.columns, "Missing 'signup_date' column."

    # Verify signup_date was parsed to a date object
    sample_date = df["signup_date"].dropna().iloc[0]
    assert isinstance(sample_date, datetime.date), f"Expected datetime.date, got {type(sample_date)}"
    assert len(df) > 0, "Customer row count must be greater than 0."


def test_load_transactions_schema_and_row_count():
    """Load transactions.db and assert expected columns, dtypes (txn_timestamp is a
    real datetime after loading, is_returned is 0/1 int), and a sane row count.
    """
    df = load_transactions()

    assert not df.empty, "transactions dataframe should not be empty."
    assert "customer_id" in df.columns, "Missing 'customer_id' column."
    assert "txn_timestamp" in df.columns, "Missing 'txn_timestamp' column."
    assert "is_returned" in df.columns, "Missing 'is_returned' column."

    # Verify timestamp and integer coercions
    assert pd.api.types.is_datetime64_any_dtype(df["txn_timestamp"]), "txn_timestamp must be datetime64."
    assert pd.api.types.is_integer_dtype(df["is_returned"]), "is_returned must be integer dtype."
    assert set(df["is_returned"].unique()).issubset({0, 1}), "is_returned must only contain 0 or 1."
    assert len(df) > 0, "Transactions row count must be greater than 0."


def test_load_monthly_agg_schema_and_row_count():
    """Load transactions_monthly_agg.parquet and assert expected columns, dtypes, and
    a sane row count.
    """
    df = load_monthly_agg()

    assert not df.empty, "monthly_agg dataframe should not be empty."
    assert "customer_id" in df.columns, "Missing 'customer_id' column."
    assert "txn_month" in df.columns, "Missing 'txn_month' column."

    # Verify txn_month was parsed to a date object
    sample_month = df["txn_month"].dropna().iloc[0]
    assert isinstance(sample_month, datetime.date), f"Expected datetime.date, got {type(sample_month)}"
    assert len(df) > 0, "Monthly aggregate row count must be greater than 0."