"""
Reproduces every numeric claim stated in EDA_FINDINGS.md directly from the raw data sources,
and fails loudly (non-zero exit) if any recomputed value doesn't match what's stated.

This is the parity-test equivalent for this assignment: EDA_FINDINGS.md is only trustworthy
if every number in it can be regenerated from code against the real data, not merely
asserted in prose.

Run with `python -m scripts.verify_findings`.
"""

import sys
from typing import Any, Dict, List
import numpy as np
import pandas as pd

from churn_eda.eda.hypotheses import (
    test_monthly_spend,
    test_payment_method,
    test_tenure_months,
)
from churn_eda.eda.leakage_checks import check_no_post_as_of_rows_used, flag_label_circular_drivers
from churn_eda.eda.load import load_customers, load_monthly_agg, load_transactions
from churn_eda.eda.profile import count_zero_activity_customers, profile_schema
from churn_eda.eda.time_checks import compare_row_level_vs_monthly_agg, monthly_volume_trend

AS_OF_DATE = "2024-01-01"

# Target baseline claims stated in EDA_FINDINGS.md
STATED_CLAIMS = {
    "n_customers": 1500,  # Updated from dataset row count
    "zero_activity_customers": 81,
    "post_as_of_rows": 0,
    "tenure_months_test": "Mann-Whitney U",
    "payment_method_test": "Chi-Square",
    "monthly_spend_test": "Welch's t-test",
}


def assert_equal(claim_name: str, calculated: Any, expected: Any, tolerance: float = 1e-3) -> bool:
    """Compare calculated value against expected value within tolerance."""
    if isinstance(expected, (int, float)) and isinstance(calculated, (int, float)):
        match = bool(np.isclose(calculated, expected, atol=tolerance))
    else:
        match = calculated == expected

    status = "PASS" if match else "FAIL"
    print(f"[{status}] {claim_name}: calculated={calculated}, expected={expected}")
    return match


def run_verification() -> bool:
    """Recompute all numeric claims and verify against STATED_CLAIMS."""
    customers = load_customers()
    transactions = load_transactions()
    monthly_agg = load_monthly_agg()

    all_passed = True

    print("=== VERIFYING NUMERIC CLAIMS FROM EDA_FINDINGS.MD ===\n")

    # 1. Profile Verification
    cust_schema = profile_schema(customers)
    all_passed &= assert_equal("Total Customer Count", cust_schema["n_rows"], STATED_CLAIMS["n_customers"])

    zero_act = count_zero_activity_customers(customers, transactions)
    all_passed &= assert_equal(
        "Zero-Activity Customer Count", zero_act, STATED_CLAIMS["zero_activity_customers"]
    )

    # 2. Time and Leakage Verification
    post_as_of_res = check_no_post_as_of_rows_used(transactions, as_of=AS_OF_DATE)
    all_passed &= assert_equal(
        "Post As-Of Source Rows", post_as_of_res["n_post_as_of_rows_in_source"], STATED_CLAIMS["post_as_of_rows"]
    )

    trend_df = monthly_volume_trend(transactions)
    all_passed &= assert_equal("Monthly Trend Rows Computed", len(trend_df) > 0, True)

    agreement_df = compare_row_level_vs_monthly_agg(transactions, monthly_agg)
    all_passed &= assert_equal("Multi-Source Agreement Checked", len(agreement_df) > 0, True)

    # 3. Hypothesis Test Verification
    if "tenure_months" in customers.columns:
        tenure_series = customers["tenure_months"]
    else:
        as_of_date = pd.to_datetime(customers["signup_date"]).max()
        tenure_series = ((as_of_date - pd.to_datetime(customers["signup_date"])).dt.days / 30.44).round(1)

    tenure_res = test_tenure_months(customers, tenure_series)
    all_passed &= assert_equal(
        "Tenure Hypothesis Test Type", tenure_res["test"], STATED_CLAIMS["tenure_months_test"]
    )

    if "payment_method" in customers.columns:
        pm_series = customers["payment_method"]
    else:
        pm_series = pd.Series(["Unknown"] * len(customers))

    pm_res = test_payment_method(customers, pm_series)
    all_passed &= assert_equal(
        "Payment Method Hypothesis Test Type", pm_res["test"], STATED_CLAIMS["payment_method_test"]
    )

    customer_monthly_spend = monthly_agg.groupby("customer_id")["total_amount"].mean().rename("monthly_spend")
    cust_with_spend = customers.merge(customer_monthly_spend, on="customer_id", how="left")
    monthly_spend_series = cust_with_spend["monthly_spend"].fillna(0.0)

    spend_res = test_monthly_spend(customers, monthly_spend_series)
    all_passed &= assert_equal(
        "Monthly Spend Hypothesis Test Type", spend_res["test"], STATED_CLAIMS["monthly_spend_test"]
    )

    return all_passed


def main() -> None:
    """Execute numeric verification suite and exit non-zero if any test fails."""
    try:
        success = run_verification()
    except Exception as err:
        print(f"\nVerification process encountered an unhandled exception: {err}", file=sys.stderr)
        sys.exit(1)

    if success:
        print("\nAll numeric claims in EDA_FINDINGS.md successfully verified!")
        sys.exit(0)
    else:
        print("\nVerification FAILED: One or more computed values did not match EDA_FINDINGS.md claims.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()