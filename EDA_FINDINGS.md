# Reproducible Findings & Metric Claims

This document lists the baseline metrics, driver statistics, and data claims established during EDA for automated test parity verification.

## Stated Claims & Expected Values

- **Total Customer Count**: `1500`
- **Baseline Churn Rate**: `~20.0%`
- **Zero-Activity Customers**: `81` customers have zero transactions in the trailing window.
- **Post As-Of Date Rows**: `0` (Source transaction log contains no invalid post-cutoff timestamps).

## Hypothesis Test Specifications

1. **Tenure Months**: Evaluated via Mann-Whitney U test due to non-normal right skew.
2. **Payment Method**: Evaluated via Chi-Square Test of Independence with Cramér's V effect size.
3. **Monthly Spend**: Evaluated via Welch's t-test (relaxing equal variance assumptions).

## Leakage Classifications
- **Legitimate Drivers**: `tenure_months`, `payment_method`, `monthly_spend`, `signup_channel`, `account_tier`.
- **Flagged Leakage Drivers**: `days_since_last_transaction`, `trailing_90_day_spend`.