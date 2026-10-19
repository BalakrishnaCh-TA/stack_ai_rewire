"""
One pure function per candidate churn-driver named in your confirmed `eda_review.md`.
"""

from typing import Any, Dict
import numpy as np
import pandas as pd
from scipy import stats


def _get_target_column(df: pd.DataFrame) -> str:
    """Helper to detect target churn column name dynamically."""
    if "is_churned" in df.columns:
        return "is_churned"
    elif "churned" in df.columns:
        return "churned"
    raise KeyError("Target column ('is_churned' or 'churned') not found in DataFrame.")


def run_tenure_months_test(
    customers: pd.DataFrame,
    driver_values: pd.Series,
) -> Dict[str, Any]:
    """Test customer tenure in months against churn status using Mann-Whitney U test."""
    target_col = _get_target_column(customers)
    df = pd.DataFrame({"is_churned": customers[target_col], "driver": driver_values}).dropna()
    churned = df[df["is_churned"] == 1]["driver"]
    not_churned = df[df["is_churned"] == 0]["driver"]

    stat, p_val = stats.mannwhitneyu(churned, not_churned, alternative="two-sided")

    mean_churned = float(churned.mean())
    mean_not_churned = float(not_churned.mean())

    if mean_churned > mean_not_churned:
        higher_group = "churned"
    elif mean_not_churned > mean_churned:
        higher_group = "not_churned"
    else:
        higher_group = None

    n1, n2 = len(churned), len(not_churned)
    rank_biserial = float(1 - (2 * stat) / (n1 * n2)) if (n1 * n2) > 0 else 0.0

    return {
        "driver": "tenure_months",
        "test": "Mann-Whitney U",
        "statistic": float(stat),
        "p_value": float(p_val),
        "effect_size_or_magnitude": rank_biserial,
        "higher_group": higher_group,
    }


def run_payment_method_test(
    customers: pd.DataFrame,
    driver_values: pd.Series,
) -> Dict[str, Any]:
    """Test categorical payment method against churn status using Chi-Square Test of Independence."""
    target_col = _get_target_column(customers)
    df = pd.DataFrame({"is_churned": customers[target_col], "driver": driver_values}).dropna()
    contingency = pd.crosstab(df["driver"], df["is_churned"])
    
    chi2, p_val, dof, _ = stats.chi2_contingency(contingency)

    n = contingency.sum().sum()
    min_dim = min(contingency.shape) - 1
    cramers_v = float(np.sqrt(chi2 / (n * min_dim))) if min_dim > 0 and n > 0 else 0.0

    churn_rates = contingency[1] / contingency.sum(axis=1) if 1 in contingency.columns else pd.Series([0])
    highest_churn_category = churn_rates.idxmax()
    baseline_churn_rate = float(customers[target_col].mean())

    return {
        "driver": f"payment_method ({highest_churn_category})",
        "test": "Chi-Square",
        "statistic": float(chi2),
        "p_value": float(p_val),
        "effect_size_or_magnitude": cramers_v,
        "higher_group": "churned" if churn_rates.max() > baseline_churn_rate else "not_churned",
    }


def run_monthly_spend_test(
    customers: pd.DataFrame,
    driver_values: pd.Series,
) -> Dict[str, Any]:
    """Test monthly spend against churn status using Welch's t-test."""
    target_col = _get_target_column(customers)
    df = pd.DataFrame({"is_churned": customers[target_col], "driver": driver_values}).dropna()
    churned = df[df["is_churned"] == 1]["driver"]
    not_churned = df[df["is_churned"] == 0]["driver"]

    stat, p_val = stats.ttest_ind(churned, not_churned, equal_var=False)

    mean_churned = float(churned.mean())
    mean_not_churned = float(not_churned.mean())

    if mean_churned > mean_not_churned:
        higher_group = "churned"
    elif mean_not_churned > mean_churned:
        higher_group = "not_churned"
    else:
        higher_group = None

    n1, n2 = len(churned), len(not_churned)
    s1, s2 = churned.std(), not_churned.std()
    s_pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2)) if (n1 + n2 - 2) > 0 else 1.0
    cohens_d = float((mean_churned - mean_not_churned) / s_pooled) if s_pooled != 0 else 0.0

    return {
        "driver": "monthly_spend",
        "test": "Welch's t-test",
        "statistic": float(stat),
        "p_value": float(p_val),
        "effect_size_or_magnitude": cohens_d,
        "higher_group": higher_group,
    }


# Aliases for notebook calls:
test_tenure_months = run_tenure_months_test
test_payment_method = run_payment_method_test
test_monthly_spend = run_monthly_spend_test