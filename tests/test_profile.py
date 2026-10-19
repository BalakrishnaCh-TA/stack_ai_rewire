"""
Test skeletons for eda/profile.py against small constructed DataFrames with a known shape,
so each profiling function's correctness doesn't depend on eyeballing the real dataset.

Fill these in against your confirmed eda_review.md and your actual profiling functions — do
not leave them skipped in your submission.
"""

import pandas as pd
import pytest

from churn_eda.eda.profile import (
    count_outliers,
    count_zero_activity_customers,
    profile_distribution,
    profile_missingness,
    profile_schema,
)


def test_profile_schema_reports_expected_columns_and_dtypes():
    """Build a small DataFrame with known columns/dtypes and assert profile_schema
    reports them correctly.
    """
    df = pd.DataFrame({
        "customer_id": [1, 2, 3],
        "amount": [10.5, 20.0, 30.2],
        "is_active": [True, False, True],
    })

    schema = profile_schema(df)

    assert schema["n_rows"] == 3
    assert schema["columns"] == ["customer_id", "amount", "is_active"]
    assert "int" in schema["dtypes"]["customer_id"]
    assert "float" in schema["dtypes"]["amount"]
    assert "bool" in schema["dtypes"]["is_active"]


def test_profile_missingness_does_not_conflate_zero_activity_with_missing():
    """Construct a customers frame and a transactions frame where some customers have
    zero matching rows, and assert count_zero_activity_customers reports them as
    zero-activity, not as a null value in some column.
    """
    customers = pd.DataFrame({"customer_id": [101, 102, 103, 104]})
    transactions = pd.DataFrame({
        "customer_id": [101, 101, 102],
        "amount": [50.0, 25.0, 100.0],
    })

    missingness = profile_missingness(customers)
    assert missingness["customer_id"] == 0, "No actual null values exist in customers table."

    zero_activity = count_zero_activity_customers(customers, transactions)
    assert zero_activity == 2, f"Expected 2 zero-activity customers (103, 104), got {zero_activity}."


def test_profile_distribution_detects_right_skew():
    """Construct a right-skewed series (e.g. mostly small values, one large outlier)
    and assert profile_distribution reports skew_direction == 'right'.
    """
    # Heavily right-skewed data: majority low values with a few extreme high values
    right_skewed_series = pd.Series([1, 2, 1, 3, 2, 1, 2, 100, 250, 500])

    dist = profile_distribution(right_skewed_series)

    assert dist["skew_direction"] == "right", f"Expected 'right' skew, got '{dist['skew_direction']}'"
    assert dist["mean"] > dist["median"], "For right-skewed data, mean should exceed median."


def test_count_outliers_flags_known_outlier():
    """Construct a series with one deliberately extreme value and assert
    count_outliers flags exactly that value under the stated rule.
    """
    series = pd.Series([10.0, 12.0, 11.0, 13.0, 10.5, 12.5, 500.0])

    outlier_res = count_outliers(series, rule="iqr", multiplier=1.5)

    assert outlier_res["rule"] == "iqr"
    assert outlier_res["n_outliers"] == 1, f"Expected 1 outlier (500.0), got {outlier_res['n_outliers']}."
    assert series.max() > outlier_res["bound_high"]