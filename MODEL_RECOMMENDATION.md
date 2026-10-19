# Downstream Model Family Recommendation

## Recommended Architecture: Gradient Boosted Decision Trees (GBDT)

We formally recommend **Gradient Boosted Decision Trees (e.g., XGBoost, LightGBM, or CatBoost)** for downstream churn prediction modeling.

### Core Justifications

1. **Monotonic Split Invariance**: Continuous features such as `tenure_months` and `monthly_spend` are heavily right-skewed with long tails. GBDT decision splits depend solely on value rank ordering, eliminating the need for feature normalization or power transformations.
2. **Tabular & Categorical Optimization**: GBDTs natively handle high-cardinality nominal categoricals (e.g., `payment_method`, `region`) alongside continuous metrics.
3. **Class Imbalance Optimization**: The baseline churn rate is ~20%. GBDT frameworks natively optimize non-differentiable ranking metrics (PR-AUC, ROC-AUC) via class weight adjustments.

### Alternatives Evaluated & Rejected
- **Linear / Logistic Regression**: Rejected due to high risk of underfitting non-linear interactions, sensitivity to right-skewed features, and requirement for manual spline/polynomial transformations.