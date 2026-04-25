#!/usr/bin/env python3
"""
Logistic Regression Pipeline Script
Usage: python run_pipeline.py --data <path.csv> --target <column> [--output <dir>]

Runs the full binary classification workflow and writes:
  outputs/model_report.json
  outputs/roc_curve.png
  outputs/calibration_plot.png
  outputs/confusion_matrix.png
  outputs/coefficients.png
"""

import argparse
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score
)
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    average_precision_score, precision_recall_curve,
    confusion_matrix, classification_report,
    brier_score_loss, log_loss
)
from sklearn.calibration import calibration_curve

# ── Plotting style ─────────────────────────────────────────────────────────
TEAL   = "#0d9488"
ROSE   = "#e11d48"
AMBER  = "#d97706"
SLATE  = "#475569"
INK    = "#1e293b"
BG     = "#f8fafc"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "axes.edgecolor": "#cbd5e1", "axes.labelcolor": INK,
    "xtick.color": SLATE, "ytick.color": SLATE,
    "grid.color": "#e2e8f0", "grid.linewidth": 0.5,
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlecolor": INK,
})


def load_and_validate(path: str, target: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if target not in df.columns:
        sys.exit(f"Error: column '{target}' not found. Available: {list(df.columns)}")
    initial_rows = len(df)
    df = df.dropna(subset=[target])
    if len(df) < initial_rows:
        print(f"  Dropped {initial_rows - len(df)} rows with missing target.")
    return df


def preprocess(df: pd.DataFrame, target: str):
    y = df[target]
    X = df.drop(columns=[target])

    # encode categoricals
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
        print(f"  One-hot encoded: {cat_cols}")

    # fill remaining missing with median
    X = X.fillna(X.median(numeric_only=True))

    # encode binary target if string
    if y.dtype == object:
        vals = y.unique()
        if len(vals) != 2:
            sys.exit(f"Target has {len(vals)} unique values — must be exactly 2.")
        mapping = {vals[0]: 0, vals[1]: 1}
        print(f"  Target encoded: {mapping}")
        y = y.map(mapping)

    return X, y.astype(int)


def build_pipeline(X_train, class_weight):
    num_cols = X_train.select_dtypes(include="number").columns.tolist()
    pre = ColumnTransformer(
        [("scale", StandardScaler(), num_cols)], remainder="passthrough"
    )
    clf = LogisticRegression(
        C=1.0, solver="lbfgs", max_iter=1000,
        class_weight=class_weight, random_state=42
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def evaluate(pipe, X_test, y_test):
    y_pred  = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    report = classification_report(y_test, y_pred, output_dict=True)

    metrics = {
        "accuracy":    round((tp+tn)/(tp+tn+fp+fn), 4),
        "auc_roc":     round(roc_auc_score(y_test, y_proba), 4),
        "avg_precision": round(average_precision_score(y_test, y_proba), 4),
        "sensitivity": round(tp/(tp+fn), 4) if (tp+fn) > 0 else None,
        "specificity": round(tn/(tn+fp), 4) if (tn+fp) > 0 else None,
        "ppv":         round(report["1"]["precision"], 4),
        "npv":         round(report["0"]["precision"], 4),
        "f1":          round(report["1"]["f1-score"], 4),
        "brier_score": round(brier_score_loss(y_test, y_proba), 4),
        "log_loss":    round(log_loss(y_test, y_proba), 4),
    }
    cm = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    return metrics, cm, y_proba


def plot_roc(y_test, y_proba, auc, outdir):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.plot(fpr, tpr, color=TEAL, lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], color=SLATE, lw=1, ls="--", label="Random")
    ax.fill_between(fpr, tpr, alpha=0.06, color=TEAL)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve"); ax.legend(fontsize=9); ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "roc_curve.png"), dpi=150)
    plt.close(fig)


def plot_calibration(y_test, y_proba, brier, outdir):
    frac_pos, mean_pred = calibration_curve(y_test, y_proba, n_bins=8)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.plot([0, 1], [0, 1], color=SLATE, lw=1, ls="--", label="Perfect")
    ax.plot(mean_pred, frac_pos, "o-", color=TEAL, lw=2, ms=7,
            label=f"Model (Brier={brier:.4f})")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve"); ax.legend(fontsize=9); ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "calibration_plot.png"), dpi=150)
    plt.close(fig)


def plot_confusion(cm_dict, outdir):
    labels = [["TN", "FP"], ["FN", "TP"]]
    vals   = [[cm_dict["tn"], cm_dict["fp"]], [cm_dict["fn"], cm_dict["tp"]]]
    colors = [["#dcfce7", "#fee2e2"], ["#fee2e2", "#dcfce7"]]
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.set_xlim(0, 2); ax.set_ylim(0, 2); ax.axis("off")
    for r in range(2):
        for c in range(2):
            rect = FancyBboxPatch((c+0.05, 1-r+0.05), 0.9, 0.9,
                                  boxstyle="round,pad=0.04",
                                  facecolor=colors[r][c], edgecolor="white", lw=2)
            ax.add_patch(rect)
            ax.text(c+0.5, 1.5-r, str(vals[r][c]),
                    ha="center", va="center", fontsize=22, fontweight="bold",
                    color=INK)
            ax.text(c+0.5, 1.5-r-0.28, labels[r][c],
                    ha="center", va="center", fontsize=9, color=SLATE)
    ax.set_title("Confusion Matrix", pad=10)
    for i, lbl in enumerate(["Neg", "Pos"]):
        ax.text(-0.08, 1.5-i, lbl, ha="right", va="center", fontsize=9, color=SLATE)
        ax.text(i+0.5, -0.05, lbl, ha="center", va="top", fontsize=9, color=SLATE)
    ax.text(-0.08, 2.08, "Actual →", ha="right", fontsize=8, color=SLATE)
    ax.text(1.0, -0.18, "Predicted →", ha="center", fontsize=8, color=SLATE)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_coefficients(pipe, feature_names, outdir):
    coef = pipe.named_steps["clf"].coef_[0]
    try:
        feat = pipe.named_steps["pre"].get_feature_names_out()
    except Exception:
        feat = feature_names

    df_coef = (pd.DataFrame({"feature": feat, "coef": coef})
               .reindex(pd.Series(coef).abs().sort_values(ascending=False).index)
               .head(12))

    colors = [TEAL if c >= 0 else ROSE for c in df_coef["coef"]]
    fig, ax = plt.subplots(figsize=(6, max(3.5, len(df_coef) * 0.38)))
    bars = ax.barh(df_coef["feature"], df_coef["coef"],
                   color=colors, edgecolor="white", height=0.6)
    ax.axvline(0, color=SLATE, lw=0.8)
    ax.set_xlabel("Log-odds coefficient (standardised)")
    ax.set_title("Feature Coefficients")
    ax.grid(axis="x", alpha=0.5)
    for bar, val in zip(bars, df_coef["coef"]):
        ax.text(val + (0.03 if val >= 0 else -0.03), bar.get_y() + bar.get_height()/2,
                f"{val:+.3f}", va="center", ha="left" if val >= 0 else "right",
                fontsize=8, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "coefficients.png"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   required=True,  help="Path to CSV file")
    parser.add_argument("--target", required=True,  help="Name of target column")
    parser.add_argument("--output", default="outputs", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    print(f"\n{'='*55}")
    print(f"  Logistic Regression Pipeline")
    print(f"  Data: {args.data}  |  Target: {args.target}")
    print(f"{'='*55}\n")

    # Load
    df = load_and_validate(args.data, args.target)
    X, y = preprocess(df, args.target)
    print(f"  Dataset: {X.shape[0]} rows × {X.shape[1]} features")
    print(f"  Class balance: {y.mean():.1%} positive\n")

    # Split
    test_size = 0.20 if len(df) > 500 else 0.30
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    # Balance detection
    pos_rate = y_train.mean()
    class_weight = "balanced" if pos_rate < 0.2 or pos_rate > 0.8 else None
    if class_weight:
        print(f"  ⚠  Imbalanced classes ({pos_rate:.1%} positive) → class_weight='balanced'")

    # Train
    pipe = build_pipeline(X_train, class_weight)
    pipe.fit(X_train, y_train)
    print("  Model trained.\n")

    # Evaluate
    metrics, cm, y_proba = evaluate(pipe, X_test, y_test)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_aucs = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
    metrics["cv_auc_mean"] = round(float(cv_aucs.mean()), 4)
    metrics["cv_auc_std"]  = round(float(cv_aucs.std()), 4)

    # Flags
    flags = []
    if metrics["brier_score"] > 0.15:
        flags.append("⚠  Calibration concern: Brier > 0.15 — consider CalibratedClassifierCV")
    if metrics["cv_auc_std"] > 0.05:
        flags.append("⚠  Instability: CV AUC SD > 0.05 — consider stronger regularisation")
    if metrics["auc_roc"] < 0.75:
        flags.append("⚠  Weak discrimination: AUC < 0.75 — review features and data quality")

    # Save JSON
    report = {
        "model": "LogisticRegression",
        "data": {"path": args.data, "target": args.target,
                 "n_train": len(X_train), "n_test": len(X_test),
                 "pos_rate_train": round(float(y_train.mean()), 4)},
        "metrics": metrics,
        "confusion_matrix": cm,
        "flags": flags,
    }
    with open(os.path.join(args.output, "model_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # Plots
    plot_roc(y_test, y_proba, metrics["auc_roc"], args.output)
    plot_calibration(y_test, y_proba, metrics["brier_score"], args.output)
    plot_confusion(cm, args.output)
    plot_coefficients(pipe, X.columns.tolist(), args.output)

    # Summary
    print(f"  {'─'*45}")
    print(f"  {'AUC-ROC':<22} {metrics['auc_roc']:.4f}")
    print(f"  {'Accuracy':<22} {metrics['accuracy']:.4f}")
    print(f"  {'F1 Score':<22} {metrics['f1']:.4f}")
    print(f"  {'Brier Score':<22} {metrics['brier_score']:.4f}")
    print(f"  {'CV AUC':<22} {metrics['cv_auc_mean']:.4f} ± {metrics['cv_auc_std']:.4f}")
    print(f"  {'─'*45}")
    for flag in flags:
        print(f"  {flag}")
    print(f"\n  Outputs saved to: {args.output}/")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
