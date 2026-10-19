"""
Runs every hypothesis test function in eda/hypotheses.py for the candidate drivers named in
eda_review.md, and prints/saves a results table.

Run with `python -m scripts.run_hypothesis_tests`.
"""

import sys
from typing import List, Dict, Any
import pandas as pd

from churn_eda.eda.load import load_customers, load_monthly_agg, load_transactions
from churn_eda.eda.hypotheses import (
    run_monthly_spend_test,
    run_payment_method_test,
    run_tenure_months_test,
)


def run_all_tests() -> pd.DataFrame:
    """Load raw data, derive candidate drivers, run hypothesis tests, and aggregate results."""
    customers = load_customers()
    transactions = load_transactions()
    monthly_agg = load_monthly_agg()

    # Calculate tenure in months if not present in customers CSV
    if "tenure_months" in customers.columns:
        tenure_series = customers["tenure_months"]
    else:
        as_of_date = pd.to_datetime(customers["signup_date"]).max()
        tenure_series = (
            (as_of_date - pd.to_datetime(customers["signup_date"])).dt.days / 30.44
        ).round(1)

    # Derive monthly spend per customer if not present
    if "monthly_spend" in customers.columns:
        monthly_spend_series = customers["monthly_spend"]
    else:
        customer_monthly_spend = (
            monthly_agg.groupby("customer_id")["total_amount"].mean().rename("monthly_spend")
        )
        customers_with_spend = customers.merge(
            customer_monthly_spend, on="customer_id", how="left"
        )
        monthly_spend_series = customers_with_spend["monthly_spend"].fillna(0.0)

    # Extract payment method series
    if "payment_method" in customers.columns:
        payment_method_series = customers["payment_method"]
    elif "payment_method" in transactions.columns:
        first_payment = (
            transactions.groupby("customer_id")["payment_method"].first().rename("payment_method")
        )
        customers_with_payment = customers.merge(
            first_payment, on="customer_id", how="left"
        )
        payment_method_series = customers_with_payment["payment_method"].fillna("Unknown")
    else:
        payment_method_series = pd.Series(["Unknown"] * len(customers))

    # Execute hypothesis test suite
    results: List[Dict[str, Any]] = [
        run_tenure_months_test(customers, tenure_series),
        run_payment_method_test(customers, payment_method_series),
        run_monthly_spend_test(customers, monthly_spend_series),
    ]

    results_df = pd.DataFrame(results)
    return results_df


def main() -> None:
    """Execute hypothesis tests and print formatted findings."""
    print("Running Hypothesis Tests across candidate churn drivers...\n")

    try:
        results_df = run_all_tests()
    except Exception as err:
        print(f"Error running hypothesis tests: {err}", file=sys.stderr)
        sys.exit(1)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    print("=== HYPOTHESIS TEST RESULTS ===")
    print(results_df.to_string(index=False))
    print("\nNote: Null findings (p_value >= 0.05) are valid statistical results and reported above.")


if __name__ == "__main__":
    main()