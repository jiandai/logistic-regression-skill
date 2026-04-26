# Logistic Regression Skill

A Claude AI skill for building, evaluating, and interpreting binary classification models using logistic regression. Designed for reproducible, production-quality workflows with emphasis on clinical and healthcare applications — but applicable to any binary prediction task.

## What It Does

When invoked, this skill guides Claude through a complete seven-step binary classification workflow:

1. **Understand the task** — clarify the target column, data source, class balance, and evaluation priority
2. **Load and explore data** — validate inputs, inspect distributions, identify missing values
3. **Split data** — stratified train/test split (80/20 for n > 500, 70/30 otherwise) to prevent leakage
4. **Preprocess** — `StandardScaler` on continuous features (fit on train only), one-hot encoding for categoricals
5. **Train the model** — `LogisticRegression` in a `Pipeline` with automatic class imbalance detection
6. **Evaluate** — full diagnostic metrics suite with 5-fold cross-validation
7. **Interpret coefficients** — log-odds explanation and conversion to odds ratios

### Class Imbalance Handling

When the positive class rate is below 20% or above 80%, the skill automatically sets `class_weight='balanced'` and deprioritizes accuracy in favor of AUC and F1.

## Repository Structure

```
logistic-regression-skill/
├── logistic-regression/
│   ├── SKILL.md                    # Skill definition and full workflow guide
│   ├── evals/
│   │   └── evals.json              # 4 evaluation scenarios with assertions
│   ├── references/
│   │   └── metrics-guide.md        # Metric interpretation thresholds and guidance
│   └── scripts/
│       └── run_pipeline.py         # Standalone end-to-end pipeline script
└── logistic-regression-workspace/
    ├── trigger-eval.json           # 20 trigger/no-trigger example queries
    └── iteration-1/                # Benchmark results and evaluation feedback
```

## Metrics Evaluated

| Metric | Poor | Acceptable | Good | Excellent |
|--------|------|-----------|------|-----------|
| AUC-ROC | < 0.60 | 0.60–0.75 | 0.75–0.90 | > 0.90 |
| F1 Score | < 0.50 | 0.50–0.70 | 0.70–0.85 | > 0.85 |
| Brier Score | > 0.25 | 0.15–0.25 | 0.05–0.15 | < 0.05 |
| CV AUC SD | > 0.08 | 0.05–0.08 | 0.02–0.05 | < 0.02 |

See `logistic-regression/references/metrics-guide.md` for full thresholds, metric selection guidance, and clinical/regulatory reporting context.

## Running the Pipeline Script

The included script runs the full pipeline from the command line without needing to invoke the skill interactively.

**Requirements:** Python 3.9+, with `scikit-learn`, `pandas`, `numpy`, and `matplotlib` installed.

```bash
python logistic-regression/scripts/run_pipeline.py \
  --data <path/to/data.csv> \
  --target <target_column_name> \
  [--output <output_directory>]
```

**Example:**

```bash
python logistic-regression/scripts/run_pipeline.py \
  --data patients.csv \
  --target treatment_response \
  --output results/
```

**Outputs** written to the output directory (default: `./outputs`):

| File | Description |
|------|-------------|
| `model_report.json` | All metrics and diagnostic flags |
| `roc_curve.png` | ROC curve with AUC annotation |
| `calibration_plot.png` | Predicted probability calibration |
| `confusion_matrix.png` | TP/TN/FP/FN visualization |
| `coefficients.png` | Top 12 features by absolute coefficient |

The script emits diagnostic flags when:
- Brier score > 0.15 (calibration concern)
- CV AUC standard deviation > 0.05 (model instability)
- AUC < 0.75 (weak discrimination)

## Skill Trigger Conditions

Claude invokes this skill when a user request involves:

- Binary classification or logistic regression (churn prediction, disease diagnosis, response prediction)
- Requests to build, train, or evaluate a binary classifier on tabular data
- Coefficient or odds ratio interpretation for logistic regression output
- Questions about whether a binary classification model is "good enough"

The skill is **not** triggered for multi-class classification, regression tasks, clustering, or general statistics questions.

## Evaluation

The skill is validated against 4 scenarios defined in `logistic-regression/evals/evals.json`:

| Scenario | Focus | Assertions |
|----------|-------|-----------|
| `synthetic-clinical-data` | Full end-to-end pipeline | 7 |
| `csv-upload-imbalanced` | Class imbalance detection and handling | 4 |
| `model-good-enough-question` | Metric interpretation and calibration | 4 |
| `coefficient-interpretation` | Log-odds and odds ratio explanation | 4 |

Benchmark results in `logistic-regression-workspace/iteration-1/` show 100% assertion pass rate with the skill active, versus 42–57% pass rate without it.
