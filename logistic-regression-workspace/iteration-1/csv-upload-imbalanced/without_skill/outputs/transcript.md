# Transcript: csv-upload-imbalanced WITHOUT skill

## Task
I have a CSV with patient data and a column 'adverse_event' that is 1 for 12% of rows. Build a logistic regression classifier.

## Steps

1. Generated synthetic patient data with 11.8% adverse_event rate (same seed).
2. Split 80/20 non-stratified.
3. Applied StandardScaler to all features.
4. Trained LogisticRegression with default settings (no class_weight).
5. Computed accuracy (90%) and AUC (0.771). Reported as primary metric.
6. Listed coefficient table (unsorted, no odds ratios).

## Observations
- Accuracy = 90% looks great but is misleading: confusion matrix shows [179,1],[19,1] — model predicts negative almost always, correctly classifying 179/180 negatives but only 1/20 positives.
- Class imbalance NOT flagged.
- class_weight='balanced' NOT set.
- F1 for positive class = 0.09 (very poor).

## Results
Accuracy=0.900 (misleading), AUC=0.771, F1_positive=0.09. Model essentially ignores positive class.
