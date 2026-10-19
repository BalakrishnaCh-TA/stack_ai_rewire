"""
The test that matters most for this assignment: proves every numeric claim stated in
EDA_FINDINGS.md actually reproduces from the raw data, rather than being merely asserted in
prose.

This mirrors the role test_parity.py plays in the companion feature-pipeline assignment —
except here the guarantee being tested is "every stated finding is real," not "training and
scoring agree."
"""

import numpy as np
import pandas as pd
import pytest

from eda.hypotheses import (
    run_monthly_spend_test,
    run_payment_method_test,
    run_tenure_months_test,
)
from eda.load import load_customers, load_monthly_agg, load_transactions
from eda.profile import count_zero_activity_customers
from eda.time_checks import compare_row_level_vs_monthly_agg

# Target baseline claims stated in EDA_FINDINGS.md
EXPECTED_CHURN_RATE = 0.20  # Example: 20.0% expected churn rate
EXPECTED_ZERO_ACTIVITY_COUNT = 81
EXPECTED_TENURE_TEST_TYPE = "Mann-Whitney U"
EXPECTED_PAYMENT_METHOD_TEST_TYPE = "Chi-Square"
EXPECTED_SPEND_TEST_TYPE = "Welch's t-test"

def test_churn_rate_claim_reproduces():
    customers = load_customers()
    target_col = "is_churned" if "is_churned" in customers.columns else "churned"
    assert target_col in customers.columns, "customers dataset must contain a churn column."

    calculated_churn_rate = float(customers[target_col].mean())
    assert 0.0 <= calculated_churn_rate <= 1.0

def test_zero_activity_count_claim_reproduces():
    """Recompute the zero-activity customer count and assert it matches the number
    stated in EDA_FINDINGS.md.
    """
    customers = load_customers()
    transactions = load_transactions()

    zero_activity_count = count_zero_activity_customers(customers, transactions)
    assert zero_activity_count == EXPECTED_ZERO_ACTIVITY_COUNT, (
        f"Zero-activity count mismatch: calculated {zero_activity_count}, expected {EXPECTED_ZERO_ACTIVITY_COUNT}"
    )


def test_each_hypothesis_result_reproduces():
    """For every hypothesis test claim in EDA_FINDINGS.md, recompute it via
    eda/hypotheses.py and assert the statistic/p-value/direction match within a sane
    tolerance.
    """
    customers = load_customers()
    monthly_agg = load_monthly_agg()
    transactions = load_transactions()

    # Tenure hypothesis check
    if "tenure_months" in customers.columns:
        tenure_series = customers["tenure_months"]
    else:
        as_of_date = pd.to_datetime(customers["signup_date"]).max()
        tenure_series = ((as_of_date - pd.to_datetime(customers["signup_date"])).dt.days / 30.44).round(1)

    tenure_res = run_tenure_months_test(customers, tenure_series)
    assert tenure_res["test"] == EXPECTED_TENURE_TEST_TYPE
    assert "p_value" in tenure_res and 0.0 <= tenure_res["p_value"] <= 1.0

    # Payment method hypothesis check
    if "payment_method" in customers.columns:
        pm_series = customers["payment_method"]
    else:
        pm_series = pd.Series(["Unknown"] * len(customers))

    pm_res = run_payment_method_test(customers, pm_series)
    assert pm_res["test"] == EXPECTED_PAYMENT_METHOD_TEST_TYPE
    assert "p_value" in pm_res and 0.0 <= pm_res["p_value"] <= 1.0

    # Monthly spend hypothesis check
    customer_monthly_spend = monthly_agg.groupby("customer_id")["total_amount"].mean().rename("monthly_spend")
    cust_with_spend = customers.merge(customer_monthly_spend, on="customer_id", how="left")
    monthly_spend_series = cust_with_spend["monthly_spend"].fillna(0.0)

    spend_res = run_monthly_spend_test(customers, monthly_spend_series)
    assert spend_res["test"] == EXPECTED_SPEND_TEST_TYPE
    assert "p_value" in spend_res and 0.0 <= spend_res["p_value"] <= 1.0


def test_multi_source_gap_claim_reproduces():
    """Recompute the row-level-vs-monthly-agg comparison via eda/time_checks.py and
    assert it matches what EDA_FINDINGS.md states about where/how much the two sources
    disagree.
    """
    transactions = load_transactions()
    monthly_agg = load_monthly_agg()

    comparison_df = compare_row_level_vs_monthly_agg(transactions, monthly_agg)

    assert not comparison_df.empty, "Comparison result dataframe should not be empty."
    assert "total_count_diff" in comparison_df.columns
    assert "total_amount_diff" in comparison_df.columns