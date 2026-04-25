# Interpretation of Logistic Regression Coefficient: Disease Duration = -1.58

In logistic regression, a coefficient represents the change in the **log-odds** of the outcome for a one-unit increase in that predictor, holding all other variables constant.

## Direction (Negative Sign)
The negative sign indicates an **inverse relationship**: as Disease Duration increases, the predicted probability of the outcome decreases.

## Odds Ratio Conversion
```
OR = e^(-1.58) ≈ 0.206
```
Each one-unit increase in Disease Duration multiplies the odds of the outcome by ~0.206 — a roughly **79.4% reduction** in odds per unit increase.

## Magnitude
A coefficient of -1.58 is large in absolute terms, making Disease Duration a **strong negative predictor**. Patients with longer disease duration are substantially less likely to experience the positive outcome.

## Caveats
- The unit of measurement matters (year vs. month changes the interpretation).
- Assumes all other variables are held constant.
- Statistical significance (p-value, confidence interval) should also be assessed.
- Does not imply causality — only association.

| Metric | Value |
|---|---|
| Coefficient (log-odds) | -1.58 |
| Odds Ratio (e^-1.58) | ~0.206 |
| Change in odds per unit | ~-79.4% |
| Direction | Negative |
| Strength | Large effect |
