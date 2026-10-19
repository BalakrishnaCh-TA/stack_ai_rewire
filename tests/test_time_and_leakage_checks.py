"""
Test skeletons for eda/time_checks.py and eda/leakage_checks.py against constructed input.

Fill these in against your confirmed eda_review.md and your actual functions — do not leave
them skipped in your submission.
"""

import datetime
import pandas as pd
import pytest

from eda.leakage_checks import flag_label_circular_drivers
from eda.time_checks import compare_row_level_vs_monthly_agg, monthly_volume_trend


def test_monthly_volume_trend_detects_growth():
    """Construct a transactions frame with a clear month-over-month growth pattern and
    assert monthly_volume_trend reflects it (increasing txn_count/total_amount by month).
    """
    transactions = pd.DataFrame({
        "customer_id": [1, 1, 1, 1, 1, 1],
        "txn_timestamp": [
            "2024-01-05 10:00:00",
            "2024-02-10 10:00:00",
            "2024-02-15 10:00:00",
            "2024-03-01 10:00:00",
            "2024-03-12 10:00:00",
            "2024-03-20 10:00:00",
        ],
        "amount": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    })

    trend_df = monthly_volume_trend(transactions)

    assert len(trend_df) == 3, f"Expected 3 monthly aggregated rows, got {len(trend_df)}"
    counts = trend_df["txn_count"].tolist()
    assert counts == [1, 2, 3], f"Expected strictly increasing transaction counts [1, 2, 3], got {counts}"


def test_compare_row_level_vs_monthly_agg_detects_known_mismatch():
    """Construct a transactions frame and a monthly_agg frame that deliberately
    disagree for one month, and assert compare_row_level_vs_monthly_agg reports that
    month's discrepancy.
    """
    transactions = pd.DataFrame({
        "customer_id": [101, 101],
        "txn_timestamp": ["2024-01-10 10:00:00", "2024-01-20 10:00:00"],
        "amount": [50.0, 50.0],
    })

    # Parquet extract claims 3 transactions and $150 total for January
    monthly_agg = pd.DataFrame({
        "customer_id": [101],
        "txn_month": [datetime.date(2024, 1, 1)],
        "txn_count": [3],
        "total_amount": [150.0],
    })

    comparison_df = compare_row_level_vs_monthly_agg(transactions, monthly_agg)

    assert not comparison_df.empty
    jan_row = comparison_df[comparison_df["txn_month"] == datetime.date(2024, 1, 1)].iloc[0]
    assert jan_row["total_count_diff"] > 0, "Should report at least 1 record with count discrepancy."
    assert jan_row["total_amount_diff"] > 0, "Should report at least 1 record with amount discrepancy."


def test_compare_row_level_vs_monthly_agg_confirms_agreement_when_consistent():
    """Construct a transactions frame and a monthly_agg frame that agree everywhere,
    and assert the comparison reports no discrepancy — the check must not false-positive.
    """
    transactions = pd.DataFrame({
        "customer_id": [101, 101],
        "txn_timestamp": ["2024-01-10 10:00:00", "2024-01-20 10:00:00"],
        "amount": [50.0, 50.0],
    })

    monthly_agg = pd.DataFrame({
        "customer_id": [101],
        "txn_month": [datetime.date(2024, 1, 1)],
        "txn_count": [2],
        "total_amount": [100.0],
    })

    comparison_df = compare_row_level_vs_monthly_agg(transactions, monthly_agg)

    assert not comparison_df.empty
    jan_row = comparison_df[comparison_df["txn_month"] == datetime.date(2024, 1, 1)].iloc[0]
    assert jan_row["total_count_diff"] == 0, "Expected zero count discrepancies when sources agree."
    assert jan_row["total_amount_diff"] == 0, "Expected zero amount discrepancies when sources agree."


def test_flag_label_circular_drivers_flags_trailing_window_features():
    """Pass a candidate driver list including something like
    'days_since_last_txn' or 'trailing_90d_spend' and assert
    flag_label_circular_drivers flags it, with a reason tied to the label's own
    trailing-90-day definition.
    """
    candidate_drivers = ["tenure_months", "days_since_last_transaction", "trailing_90_day_spend"]

    res = flag_label_circular_drivers(candidate_drivers)

    assert res["has_circular_leakage"] is True
    assert "days_since_last_transaction" in res["flagged_drivers"]
    assert "trailing_90_day_spend" in res["flagged_drivers"]
    assert "trailing 90-day" in res["flagged_drivers"]["days_since_last_transaction"].lower()


def test_flag_label_circular_drivers_does_not_flag_legitimate_driver():
    """Pass a candidate driver list including something like 'tenure_days' (not
    definitionally tied to the label's trailing window) and assert it is NOT flagged —
    the check must not over-flag every strong result as leakage.
    """
    candidate_drivers = ["tenure_months", "payment_method", "signup_channel"]

    res = flag_label_circular_drivers(candidate_drivers)

    assert res["has_circular_leakage"] is False
    assert res["n_flagged_drivers"] == 0
    assert len(res["flagged_drivers"]) == 0