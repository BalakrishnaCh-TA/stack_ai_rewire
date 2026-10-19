"""
Leakage-discipline checks specific to this dataset's label definition: churn = no purchase
in the 90 days before the as-of date.

A feature computed over a window that overlaps that same trailing period is not merely
correlated with the label — it is circular with it. These checks exist to catch that
specifically, not to raise a generic "be careful about leakage" warning.
"""

from typing import List, Dict, Any
import pandas as pd

# List of keywords indicating overlap with the trailing 90-day label period
CIRCULAR_KEYWORDS = [
    "recency",
    "days_since_last",
    "days_since_latest",
    "last_transaction",
    "latest_transaction",
    "days_inactive",
    "trailing_90",
    "last_90",
    "recent_90",
    "past_90",
    "3_month_activity",
]


def flag_label_circular_drivers(candidate_drivers: List[str]) -> Dict[str, Any]:
    """Identify candidate drivers computed over a window that overlaps the label's

    trailing-90-day churn definition and provide a specific reason for each flag.
    """
    flagged = {}
    
    for driver in candidate_drivers:
        driver_lower = driver.lower()
        if any(keyword in driver_lower for keyword in CIRCULAR_KEYWORDS):
            flagged[driver] = (
                f"Driver '{driver}' uses window or recency logic that overlaps "
                "with the label's trailing 90-day inactivity window, creating target leakage."
            )

    return {
        "has_circular_leakage": len(flagged) > 0,
        "n_flagged_drivers": len(flagged),
        "flagged_drivers": flagged,
    }


def check_no_post_as_of_rows_used(transactions: pd.DataFrame, as_of: str) -> Dict[str, Any]:
    """Confirm whether any row in `transactions` with `txn_timestamp` after `as_of` exists,

    and verify the number of post-as-of rows present in the raw source dataset.
    """
    as_of_dt = pd.to_datetime(as_of)
    txn_timestamps = pd.to_datetime(transactions["txn_timestamp"])
    
    post_as_of_mask = txn_timestamps > as_of_dt
    n_post_as_of = int(post_as_of_mask.sum())
    
    passed = n_post_as_of == 0

    return {
        "passed": passed,
        "n_post_as_of_rows_in_source": n_post_as_of,
        "note": (
            "Source dataset contains no post-as-of rows."
            if passed
            else f"Found {n_post_as_of} rows after as-of date {as_of} in source file. "
            "Ensure filtering (`txn_timestamp <= as_of`) is strictly applied during feature extraction."
        ),
    }