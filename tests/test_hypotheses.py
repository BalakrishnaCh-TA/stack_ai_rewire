"""
Test skeletons for each hypothesis function in eda/hypotheses.py, against constructed input
with a known answer.

Per skill.md's Workflow: each hypothesis function needs, at minimum, a case with a clear
engineered difference between groups (should come back significant) and a case with no real
difference (should come back non-significant, reported honestly, not omitted).
"""

import numpy as np
import pandas as pd
import pytest

from churn_eda.eda.hypotheses import (
    run_monthly_spend_test,
    run_payment_method_test,
    run_tenure_months_test,
)


def test_hypothesis_detects_engineered_difference():
    """Construct two groups with a clear, engineered difference and assert the
    relevant hypothesis function reports a significant result in the correct direction.
    """
    np.random.seed(42)
    n = 200

    # Churned group has significantly lower tenure (mean ~ 5 months)
    # Non-churned group has higher tenure (mean ~ 25 months)
    churned_tenure = np.random.exponential(scale=5, size=n // 2)
    not_churned_tenure = np.random.exponential(scale=25, size=n // 2)

    customers = pd.DataFrame({"is_churned": [1] * (n // 2) + [0] * (n // 2)})
    driver_values = pd.Series(np.concatenate([churned_tenure, not_churned_tenure]))

    result = run_tenure_months_test(customers, driver_values)

    assert result["p_value"] < 0.05, f"Expected p-value < 0.05, got {result['p_value']}"
    assert result["higher_group"] == "not_churned", (
        f"Expected non-churned group to have higher values, got {result['higher_group']}"
    )


def test_hypothesis_reports_null_result_honestly():
    """Construct two groups drawn from the same distribution (no real difference) and
    assert the relevant hypothesis function reports a non-significant result rather than a
    false positive.
    """
    np.random.seed(42)
    n = 200

    # Both groups drawn from the exact same normal distribution
    churned_spend = np.random.normal(loc=50, scale=10, size=n // 2)
    not_churned_spend = np.random.normal(loc=50, scale=10, size=n // 2)

    customers = pd.DataFrame({"is_churned": [1] * (n // 2) + [0] * (n // 2)})
    driver_values = pd.Series(np.concatenate([churned_spend, not_churned_spend]))

    result = run_monthly_spend_test(customers, driver_values)

    assert result["p_value"] >= 0.05, f"Expected non-significant p-value (>= 0.05), got {result['p_value']}"


def test_hypothesis_uses_test_matching_variable_shape():
    """For a driver eda_review.md designates as skewed/non-normal, assert the
    function actually uses a test appropriate to that shape (e.g. Mann-Whitney U), not a
    t-test applied without checking the normality assumption.
    """
    customers = pd.DataFrame({"is_churned": [1, 1, 0, 0]})
    driver_values = pd.Series([1.0, 2.0, 10.0, 20.0])

    result = run_tenure_months_test(customers, driver_values)

    assert result["test"] == "Mann-Whitney U", (
        f"Expected 'Mann-Whitney U' test for skewed continuous data, got '{result['test']}'"
    )