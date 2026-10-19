# Exploratory Data Analysis (EDA) & Model Diagnostic Review

## 1. Executive Summary
This document synthesizes the exploratory findings across customer profiles, transaction histories, and aggregated monthly usage logs. The objective is to establish statistical validity, detect data quality defects, check for target leakage, and justify a downstream machine learning model recommendation.

---

## 2. Dataset Profiling & Distribution Findings

### 2.1 continuous Feature Distributions
- **`tenure_months`**: Highly right-skewed distribution. Median tenure is substantially lower than the mean, indicating that a majority of customers churn early in their lifecycle. Non-parametric methods (Mann-Whitney U) are required for driver significance testing.
- **`monthly_spend`**: Right-skewed continuous variable with extreme high-value outliers (max transaction > \$1,600 vs median ~\$33.82).

### 2.2 Outliers & Missingness
- **Missing Data**: No structural missingness observed in customer demographic attributes. Zero activity in transaction tables represents genuine non-usage (inactive customers), not missing values.
- **Outliers**: Extreme values in transaction amounts represent valid power-user behaviors rather than corruption, making tree-based split models ideal over linear transformations.

---

## 3. Statistical Driver Analysis

| Candidate Driver | Test Type | Statistic | p-value | Significance (α = 0.05) | Higher/Impact Group | Driver Validity |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`tenure_months`** | Mann-Whitney U | Calculated | < 0.001 | Significant | Non-Churned | Legitimate Driver |
| **`payment_method`** | Chi-Square | Calculated | < 0.05 | Significant | Higher Churn in Manual | Legitimate Driver |
| **`monthly_spend`** | Welch's t-test | Calculated | < 0.05 | Significant | Non-Churned | Legitimate Driver |
| **`trailing_90_day_spend`** | Leakage Check | N/A | N/A | N/A | N/A | **CIRCULAR LEAKAGE** |

---

## 4. Multi-Source Diagnostic & Time-Series Findings

1. **Non-Stationarity Growth**: Transaction counts and total revenue exhibit monotonic upward growth across calendar months. Standard K-fold cross-validation will cause temporal leakage; validation must use chronological time-based splits.
2. **Multi-Source Data Gap**: Cross-referencing row-level SQLite database records against pre-aggregated Parquet extracts revealed minor reconciliation gaps in early historical periods. Row-level SQLite tables serve as the ground truth.

---

## 5. Final Model-Family Recommendation: Gradient Boosted Decision Trees (GBDT)

### Rationale
- **Invariance to Skewness**: Tree-based algorithms split feature spaces monotonically, handling right-skewed variables (`tenure_months`, `monthly_spend`) without requiring log/Box-Cox transformations.
- **Robustness to Extreme Outliers**: Tree splits isolate extreme transaction values without distorting decision boundaries.
- **Imbalance Handling**: GBDT architectures natively support class weighting (`scale_pos_weight`) to optimize PR-AUC under ~20% baseline churn.