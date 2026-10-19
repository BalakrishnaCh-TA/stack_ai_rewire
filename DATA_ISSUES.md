# Data Quality & Governance Assessment

## Identified Data Issues & Remediation Strategies

### 1. Multi-Source Reconciliation Gap
- **Issue**: Discrepancy observed between row-level transaction totals in SQLite (`transactions.db`) and pre-aggregated extracts (`transactions_monthly_agg.parquet`).
- **Remediation**: Use row-level SQLite database records as the authoritative primary source for feature aggregation.

### 2. Schema Target Naming Ambiguity
- **Issue**: Target churn label column appears as `churned` in raw CSV loads but as `is_churned` across feature definitions.
- **Remediation**: Standardize dynamic column aliasing in `eda/load.py` and profiling functions.

### 3. Target Leakage Risks
- **Issue**: Features derived from transactions occurring within the 90-day pre-cutoff window mirror the target definition.
- **Remediation**: Exclude `days_since_last_transaction` and `trailing_90_day_spend` from model training feature matrices.