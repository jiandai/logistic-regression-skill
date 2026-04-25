# Transcript: synthetic-clinical-data WITHOUT skill

## Task
Build a logistic regression model to predict treatment response. Use synthetic clinical data with features: Age, Baseline SLEDAI, CRP, Anti-dsDNA, C3, Prior Biologics, Disease Duration, Renal Involvement.

## Steps

1. Generated synthetic data with n=600 rows using same seed and feature distributions.
2. Split 80/20 train/test (non-stratified).
3. Applied StandardScaler to all features together (including binary) — no ColumnTransformer separation.
4. Trained LogisticRegression(max_iter=1000) with default settings (no class_weight consideration).
5. Computed accuracy (0.675) and AUC (0.696). Reported classification report.
6. Listed coefficient table (not sorted by abs value, no odds ratios computed).
7. Saved basic ROC curve plot.

## Results
Accuracy=0.675, AUC=0.696
Missing: Brier score, CV AUC±SD, confusion matrix visualization, sorted coefficient table
