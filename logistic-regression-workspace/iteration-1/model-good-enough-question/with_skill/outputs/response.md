# Assessment: AUC = 0.72, Brier = 0.19 — Is this good enough for a clinical trial?

**Short answer: Acceptable, but not ready for clinical trial use without further work.**

## Metric Ratings (from `references/metrics-guide.md`)

| Metric | Your Value | Band | Rating |
|---|---|---|---|
| AUC-ROC | 0.72 | 0.60–0.75 | **Acceptable** (threshold for "Good" is 0.75) |
| Brier Score | 0.19 | 0.15–0.25 | **Calibration concern flagged** — skill flags Brier > 0.15 |

Neither metric reaches the "Good" tier.

## Why this matters clinically

**AUC = 0.72:** The model ranks a random positive above a random negative 72% of the time — better than chance, but the 0.75 threshold is where discrimination becomes reliably useful. Below 0.75, patient stratification or enrichment in a trial may introduce meaningful noise. FDA/ICH E9(R1) regulatory submissions require AUC reported with 95% bootstrap CI.

**Brier = 0.19:** The skill explicitly flags Brier > 0.15 as a calibration concern. The threshold band for "Acceptable" calibration is 0.15–0.25; for "Good" it requires < 0.15. If probability estimates feed a downstream decision model or are communicated to clinicians, uncalibrated probabilities at 0.19 are a material issue.

## What is missing before a full verdict

1. CV AUC ± SD — stability check (flag if SD > 0.05)
2. Sensitivity/Specificity at the operating threshold
3. Confusion matrix and F1 score
4. Class balance (if positive class < 20%, also need Precision-Recall AUC)
5. AUC 95% CI (bootstrap, n=1000) — required for regulatory submissions
6. Calibration curve — visual confirmation of probability reliability

## Recommendations

1. **Target AUC ≥ 0.75** before clinical use — add features, tune regularisation C, check for data leakage.
2. **Reduce Brier below 0.15** using `CalibratedClassifierCV`:
   ```python
   from sklearn.calibration import CalibratedClassifierCV
   calibrated = CalibratedClassifierCV(pipe, method='isotonic', cv=5)  # n > 1000
   # or method='sigmoid' for smaller datasets
   ```
3. **Report the full metric battery**: sensitivity, specificity, PPV, NPV, F1, CV AUC ± SD, confusion matrix.
4. **Pre-specify the operating threshold** in the SAP using the Youden index if FP/FN costs are equal.
5. **Bootstrap the AUC 95% CI** (n=1000 resamples) for any regulatory submission.
6. **Consult a biostatistician** — the model's role in the trial (enrichment, stratification, endpoint) must be pre-specified under ICH E9(R1) estimand requirements.
