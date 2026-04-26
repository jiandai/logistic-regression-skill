---
name: logistic-regression
description: >
  This is the skill for logistic regression and binary classification. Invoke it
  the moment a user: says "logistic regression" or "logreg" or "binary classifier";
  has a dataset where the target column has exactly two values (yes/no, 0/1,
  churned/not churned, responded/not) and wants a model; asks to predict a binary
  outcome (hospitalization, churn, fraud, clinical response, default); mentions AUC,
  Brier score, sensitivity, or specificity in a classification context; or needs to
  handle class imbalance in a two-class prediction problem. If the user wants to
  classify observations into one of two groups, this skill applies. Do not invoke
  for linear regression, multiclass classification, or time-series forecasting.
compatibility:
  python: ">=3.9"
  packages: [scikit-learn, pandas, numpy, matplotlib]
---

# Logistic Regression Skill

Guide Claude through a reproducible, production-quality binary classification
workflow: data intake → preprocessing → modelling → evaluation → interpretation.
The output is always a trained model plus a full diagnostic report.

---

## Step 0 — Understand the task before writing any code

Ask (or infer from context) these four things:

1. **Target column** — which column is the binary outcome? Confirm 0/1 encoding
   or string labels (e.g. "Yes"/"No").
2. **Data source** — uploaded CSV, synthetic data, or an existing DataFrame?
3. **Class balance** — roughly equal, or is one class rare (< 20% of rows)?
   Rare positive class → set `class_weight='balanced'` and prioritise AUC/F1
   over accuracy.
4. **Evaluation priority** — sensitivity (minimise false negatives, e.g. disease
   detection) or precision (minimise false positives, e.g. fraud alerts)?

If data is not provided, generate a realistic synthetic dataset using
`sklearn.datasets.make_classification` with domain-appropriate feature names.

---

## Step 1 — Data loading & exploration

```python
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv("data.csv")          # or build synthetic data
print(df.shape, df.dtypes)
print(df["target"].value_counts(normalize=True))  # check balance
print(df.isnull().sum())              # check missingness
```

**Handle before modelling:**
- Drop or impute columns with > 35% missing values.
- Encode categorical features with `pd.get_dummies(drop_first=True)`.
- Do NOT impute the target column — drop those rows.

---

## Step 2 — Train / test split

Always stratify on the target to preserve class balance in both sets.

```python
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
```

Use 20% test for n > 500. Use 30% for smaller datasets (n < 500).

---

## Step 3 — Preprocessing

Apply `StandardScaler` to continuous features. Fit on train, transform both.
Binary/dummy columns do not need scaling — separate them first.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

continuous_cols = X_train.select_dtypes(include="number").columns.tolist()

preprocessor = ColumnTransformer([
    ("scale", StandardScaler(), continuous_cols)
], remainder="passthrough")
```

---

## Step 4 — Model training

Wrap in a `Pipeline` so the preprocessor travels with the model.

```python
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ("pre", preprocessor),
    ("clf", LogisticRegression(
        C=1.0,                  # L2 regularisation strength (tune if needed)
        solver="lbfgs",
        max_iter=1000,
        class_weight=None,      # set "balanced" if class imbalance > 3:1
        random_state=42
    ))
])

pipe.fit(X_train, y_train)
```

**When to tune C:**
- If the model is over-fitting (train AUC >> test AUC) → lower C (e.g. 0.1).
- If the model is under-fitting (both AUCs are low) → raise C (e.g. 10).
- Quick grid search: `C in [0.01, 0.1, 1, 10, 100]` with 5-fold CV.

---

## Step 5 — Evaluation

Run the full diagnostic battery. See `references/metrics-guide.md` for
interpretation thresholds and clinical/business guidance.

```python
from sklearn.metrics import (
    roc_auc_score, classification_report,
    confusion_matrix, brier_score_loss
)
from sklearn.model_selection import cross_val_score, StratifiedKFold

y_pred  = pipe.predict(X_test)
y_proba = pipe.predict_proba(X_test)[:, 1]

# Core metrics
auc    = roc_auc_score(y_test, y_proba)
brier  = brier_score_loss(y_test, y_proba)
report = classification_report(y_test, y_pred, output_dict=True)
cm     = confusion_matrix(y_test, y_pred)

# Cross-validation stability
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_aucs = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
print(f"CV AUC: {cv_aucs.mean():.3f} ± {cv_aucs.std():.3f}")
```

**Always report all of the following:**

| Metric | Description |
|---|---|
| AUC-ROC | Discrimination (threshold-free) |
| Sensitivity | True positive rate at 0.5 |
| Specificity | True negative rate at 0.5 |
| PPV / Precision | Of predicted positives, how many are correct |
| NPV | Of predicted negatives, how many are correct |
| F1 | Harmonic mean of precision + recall |
| Brier score | Calibration quality (lower = better; 0 = perfect) |
| CV AUC mean ± SD | Generalisation stability |

---

## Step 6 — Coefficient interpretation

Extract coefficients from the fitted pipeline. These are log-odds on the
standardised scale — use them to rank feature importance, not for absolute
clinical magnitude.

```python
coef = pipe.named_steps["clf"].coef_[0]
features = pipe.named_steps["pre"].get_feature_names_out()
coef_df = (pd.DataFrame({"feature": features, "log_odds": coef})
             .assign(abs_coef=lambda d: d.log_odds.abs())
             .sort_values("abs_coef", ascending=False))
print(coef_df.to_string(index=False))
```

Positive log-odds → feature increases probability of the positive class.
Negative log-odds → feature decreases it.

---

## Step 7 — Output & visualisation

Run `scripts/run_pipeline.py` for the full automated pipeline with plots.
It accepts `--data <path>` and `--target <column>` arguments and writes:

```
outputs/
├── model_report.json     # all metrics as structured JSON
├── roc_curve.png
├── calibration_plot.png
├── confusion_matrix.png
└── coefficients.png
```

For interactive dashboards in Claude.ai, use the `frontend-design` skill to
render the JSON report as an HTML artefact.

---

## Decision checklist before delivering results

- [ ] Reported AUC, F1, Brier score, and CV AUC ± SD
- [ ] Confusion matrix shown (not just accuracy)
- [ ] Class balance checked; `class_weight` set if ratio > 3:1
- [ ] No data leakage (scaler fit only on train, not full dataset)
- [ ] Coefficient table presented with correct sign interpretation
- [ ] If Brier > 0.15, flagged calibration concern
- [ ] If CV AUC SD > 0.05, flagged instability concern

---

## Reference files

- **`references/metrics-guide.md`** — Interpretation thresholds for each metric,
  when to prefer AUC vs F1, and clinical/business decision guidance. Read this
  when the user asks "is this model good?" or "which metric should I use?"
- **`scripts/run_pipeline.py`** — Executable end-to-end pipeline. Run directly
  when the user provides a CSV; outputs structured JSON + four diagnostic plots.
