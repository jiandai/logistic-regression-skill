# Metrics Interpretation Guide

This reference is loaded when the user asks "is my model good?", "which metric
should I focus on?", or when choosing between competing evaluation criteria.

---

## Quick interpretation thresholds

| Metric | Poor | Acceptable | Good | Excellent |
|---|---|---|---|---|
| AUC-ROC | < 0.60 | 0.60–0.75 | 0.75–0.90 | > 0.90 |
| F1 Score | < 0.50 | 0.50–0.70 | 0.70–0.85 | > 0.85 |
| Brier Score | > 0.25 | 0.15–0.25 | 0.05–0.15 | < 0.05 |
| CV AUC SD | > 0.08 | 0.05–0.08 | 0.02–0.05 | < 0.02 |

A model can have high AUC but poor calibration (high Brier). Always report both.

---

## Choosing the right primary metric

### Use AUC-ROC when:
- The cost of FP and FN are symmetric
- Comparing models trained on different class-imbalance datasets
- You need a threshold-independent summary
- Regulatory submissions require discrimination statistics (e.g. FDA/ICH E9R1)

### Use F1 / Precision-Recall AUC when:
- Positive class is rare (< 20%)
- AUC-ROC is misleadingly optimistic (common with severe imbalance)
- Business/clinical cost of FP ≠ FN (adjust beta in F-beta score)

### Use Sensitivity (Recall) as primary when:
- False negatives are catastrophic (missed disease, missed fraud)
- E.g. cancer screening, sepsis prediction, safety signals
- Accept lower specificity to capture all true positives

### Use Precision (PPV) as primary when:
- False positives are costly (unnecessary surgery, expensive intervention)
- E.g. confirmatory diagnostics, high-cost treatment allocation

### Use Brier score when:
- You need calibrated probabilities, not just rankings
- Probabilities will feed a downstream decision model
- Communicating "this patient has a 73% chance of response" to clinicians

---

## Diagnosing common problems

### High accuracy, low AUC
- Usually caused by class imbalance and a majority-class-biased model
- Fix: check confusion matrix — if most predictions are one class, set
  `class_weight='balanced'` and re-evaluate with F1 and AUC

### Good CV AUC, poor test AUC (> 0.05 gap)
- Likely overfitting to training folds
- Fix: lower C (increase regularisation); check for feature leakage

### Low Brier score with low AUC
- Impossible — miscalculation. Re-check that `y_proba` is the positive class
  column (index 1 from `predict_proba`)

### CV AUC SD > 0.08
- High variance across folds; model is unstable
- Fix: increase n_splits, check for small dataset, consider simpler model or
  stronger regularisation

### Calibration curve far from diagonal
- Model probabilities are not trustworthy as absolute estimates
- Fix: wrap with `sklearn.calibration.CalibratedClassifierCV` using
  `method='isotonic'` (n > 1000) or `method='sigmoid'` (n < 1000)

---

## Regulatory / clinical reporting note

For clinical trial endpoints or regulatory submissions:
- Report AUC with 95% CI (bootstrap, n=1000 resamples)
- Report sensitivity and specificity at the pre-specified operating threshold
- Document the threshold selection rationale in the SAP
- Consider the Youden index (sensitivity + specificity − 1) for threshold
  selection when FP/FN costs are equal
- ICH E9(R1) requires pre-specified estimands — document target population,
  intercurrent events strategy, and summary measure

---

## Odds ratio conversion

To convert log-odds coefficients to odds ratios for reporting:

```python
import numpy as np
odds_ratios = np.exp(coef_df["log_odds"])
# OR > 1 → positive association with outcome
# OR < 1 → protective / negative association
# OR = 1 → no effect
```

A one-unit increase in a standardised feature multiplies the odds of the
positive outcome by `exp(coef)`.
