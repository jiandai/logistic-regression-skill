# Interpretation: Logistic Regression Coefficient for 'Disease Duration' = -1.58

## What the coefficient means

This coefficient is on the **log-odds scale** for the **standardised** feature (StandardScaler applied). The skill (Step 6) states:

> Coefficients are log-odds on the standardised scale — use them to rank feature importance, not for absolute clinical magnitude.
> Positive log-odds → feature increases probability of the positive class.
> Negative log-odds → feature decreases it.

## Direction

The coefficient is **negative (−1.58)**, so longer Disease Duration **decreases** the probability of the positive outcome.

## Odds ratio conversion (from `references/metrics-guide.md`)

```python
import numpy as np
odds_ratio = np.exp(-1.58)  # ≈ 0.206
```

| Metric | Value |
|---|---|
| Coefficient (log-odds, standardised scale) | −1.58 |
| Odds ratio exp(−1.58) | ≈ 0.206 |
| Change in odds per 1 SD increase in Disease Duration | ≈ −79.4% |
| Direction | Negative predictor |
| Relative feature importance | High (large absolute value) |

## Plain-language interpretation

Each one-standard-deviation increase in Disease Duration multiplies the odds of the positive outcome by approximately **0.21** — roughly a **79% reduction in odds**. Disease Duration is a **strong negative predictor**: patients with longer disease duration are substantially less likely to experience the positive outcome.

## Important caveats (from the skill)

- "One unit" here means **one standard deviation** on the original scale (not one year or one month) — the StandardScaler was applied in preprocessing.
- Use this coefficient to **rank feature importance** and understand direction, not to report absolute clinical magnitude for a specific patient.
- Always report alongside the odds ratio confidence interval (bootstrap or Wald) for regulatory submissions.
