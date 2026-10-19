"""
Schema, missingness, distribution-shape, and outlier profiling for the three raw sources.

Every function here should return something concrete an evaluator (or
`scripts/verify_findings.py`) can check against the real data — not a prose-only summary.
"""

from typing import Dict, Any
import numpy as np
import pandas as pd


def profile_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """Return schema information including column names, dtypes, and row count."""
    return {
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "n_rows": int(len(df)),
    }


def profile_missingness(df: pd.DataFrame) -> Dict[str, int]:
    """Return per-column null counts.

    Do not conflate a customer legitimately absent from transactions.db (zero activity) with
    an actually-missing value in a column that IS present — see `count_zero_activity_customers`
    below for the former.
    """
    return {col: int(null_count) for col, null_count in df.isnull().sum().items()}


def count_zero_activity_customers(customers: pd.DataFrame, transactions: pd.DataFrame) -> int:
    """Count customers with zero matching rows in `transactions`. Compute this
    directly — do not hardcode a number from any prior document.
    """
    active_customer_ids = set(transactions["customer_id"].unique())
    all_customer_ids = set(customers["customer_id"].unique())
    zero_activity_customers = all_customer_ids - active_customer_ids
    return int(len(zero_activity_customers))


def profile_distribution(series: pd.Series) -> Dict[str, Any]:
    """Return distribution summary statistics for a numeric column.
    
    Uses skewness statistic to classify skew direction (skew > 0.5: right,
    skew < -0.5: left, between -0.5 and 0.5: roughly symmetric).
    """
    clean_s = series.dropna()
    if clean_s.empty:
        return {
            "mean": None,
            "median": None,
            "skew_direction": "unknown",
            "min": None,
            "max": None,
        }

    mean_val = float(clean_s.mean())
    median_val = float(clean_s.median())
    skew_val = float(clean_s.skew())

    if skew_val > 0.5:
        skew_direction = "right"
    elif skew_val < -0.5:
        skew_direction = "left"
    else:
        skew_direction = "roughly symmetric"

    return {
        "mean": mean_val,
        "median": median_val,
        "skew_direction": skew_direction,
        "min": float(clean_s.min()),
        "max": float(clean_s.max()),
    }


def count_outliers(series: pd.Series, rule: str = "iqr", multiplier: float = 1.5) -> Dict[str, Any]:
    """Count outliers using an explicit statistical rule (IQR or Z-Score)."""
    clean_s = series.dropna()

    if rule.lower() == "iqr":
        q1 = float(clean_s.quantile(0.25))
        q3 = float(clean_s.quantile(0.75))
        iqr = q3 - q1
        bound_low = q1 - (multiplier * iqr)
        bound_high = q3 + (multiplier * iqr)
    elif rule.lower() == "zscore":
        mean = float(clean_s.mean())
        std = float(clean_s.std())
        bound_low = mean - (multiplier * std)
        bound_high = mean + (multiplier * std)
    else:
        raise ValueError(f"Unsupported outlier rule: {rule}. Choose 'iqr' or 'zscore'.")

    n_outliers = int(((clean_s < bound_low) | (clean_s > bound_high)).sum())

    return {
        "rule": rule,
        "bound_low": bound_low,
        "bound_high": bound_high,
        "n_outliers": n_outliers,
    }