
# ============================================================
# FINAL LOGISTIC REGRESSION MODEL
# UCI Default of Credit Card Clients
# ============================================================
#
# Pipeline:
# 1. Load cleaned data
# 2. Feature engineering
# 3. Train / Validation / Test split
# 4. Preprocessing:
#       - Signed Log Transform
#       - RobustScaler
#       - One-Hot Encoding
# 5. Logistic Regression
# 6. Threshold selection on Validation
#       based on business cost
# 7. Final evaluation on untouched Test
# 8. Model interpretation
# 9. Calibration
# 10. Risk groups
#
# Important:
# Test set is NOT used for model selection or threshold selection.
# ============================================================


# ============================================================
# 0. IMPORT LIBRARIES
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    RobustScaler,
    FunctionTransformer
)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss
)

from sklearn.calibration import calibration_curve


# ============================================================
# 1. SETTINGS
# ============================================================

RANDOM_STATE = 42

DATA_PATH = "studybuuild_3/default of credit card clients_cleaned.xlsx"

OUTPUT_DIR = "final_model_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# Business costs
# ------------------------------------------------------------
# FN = actual default but model predicts non-default
# FP = actual non-default but model predicts default
#
# Because missing a real defaulter is more costly,
# FN is assigned a higher cost.
#
# Change these values if your business assumption changes.
# ------------------------------------------------------------

COST_FP = 1
COST_FN = 5


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_excel(DATA_PATH)

print("=" * 70)
print("DATA LOADED")
print("=" * 70)

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

# Number of months with payment delay
pay_cols = [
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6"
]

df["N_DELAY_MONTHS"] = (df[pay_cols] > 0).sum(axis=1)


# ============================================================
# 4. DEFINE X AND y
# ============================================================

TARGET = "default payment next month"

X = df.drop(columns=["ID", TARGET])
y = df[TARGET]


print("\n" + "=" * 70)
print("TARGET DISTRIBUTION")
print("=" * 70)

print(y.value_counts())
print("\nPercent:")
print((y.value_counts(normalize=True) * 100).round(2))


# ============================================================
# 5. TRAIN / VALIDATION / TEST SPLIT
# ============================================================
#
# 60% Train
# 20% Validation
# 20% Test
#
# Train:
#     model fitting
#
# Validation:
#     threshold selection
#
# Test:
#     final unbiased evaluation
#
# ============================================================

X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.25,
    stratify=y_temp,
    random_state=RANDOM_STATE
)


print("\n" + "=" * 70)
print("DATA SPLIT")
print("=" * 70)

print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)


# ============================================================
# 6. DEFINE FEATURE GROUPS
# ============================================================

categorical_features = [
    "SEX",
    "EDUCATION",
    "MARRIAGE"
]

log_features = [
    "LIMIT_BAL",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6"
]

scaled_features = [
    "AGE",
    "N_DELAY_MONTHS"
]

numeric_passthrough = [
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6"
]


# ============================================================
# 7. SIGNED LOG TRANSFORM
# ============================================================

def signed_log_transform(X):
    """
    Signed logarithmic transformation:

        sign(x) * log(1 + |x|)

    This preserves the sign while reducing the influence
    of highly skewed values and outliers.
    """

    X = np.asarray(X, dtype=float)

    return np.sign(X) * np.log1p(np.abs(X))


log_transformer = Pipeline(
    steps=[
        (
            "signed_log",
            FunctionTransformer(
                signed_log_transform,
                feature_names_out="one-to-one"
            )
        ),
        (
            "robust_scaler",
            RobustScaler()
        )
    ]
)


# ============================================================
# 8. PREPROCESSING PIPELINE
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore",
                dtype=int
            ),
            categorical_features
        ),

        (
            "log_numeric",
            log_transformer,
            log_features
        ),

        (
            "scaled_numeric",
            RobustScaler(),
            scaled_features
        ),

        (
            "numeric",
            "passthrough",
            numeric_passthrough
        )
    ],
    remainder="drop"
)


# ============================================================
# 9. FINAL LOGISTIC REGRESSION MODEL
# ============================================================
#
# class_weight="balanced" is used because the target is
# imbalanced (~22% defaults vs ~78% non-defaults).
#
# C is not manually tuned here because CV showed that
# Logistic and LASSO variants had practically identical
# performance.
#
# ============================================================

model = LogisticRegression(
    class_weight="balanced",
    solver="liblinear",
    max_iter=2000,
    random_state=RANDOM_STATE
)


# ============================================================
# 10. COMPLETE MODEL PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("model", model)
    ]
)


# ============================================================
# 11. TRAIN MODEL
# ============================================================

print("\n" + "=" * 70)
print("TRAINING FINAL LOGISTIC REGRESSION")
print("=" * 70)

pipeline.fit(X_train, y_train)

print("Model trained successfully.")


# ============================================================
# 12. VALIDATION PROBABILITIES
# ============================================================

y_val_proba = pipeline.predict_proba(X_val)[:, 1]


# ============================================================
# 13. VALIDATION PERFORMANCE AT THRESHOLD = 0.50
# ============================================================

default_threshold = 0.50

y_val_pred_05 = (
    y_val_proba >= default_threshold
).astype(int)

print("\n" + "=" * 70)
print("VALIDATION PERFORMANCE — THRESHOLD 0.50")
print("=" * 70)

print(
    classification_report(
        y_val,
        y_val_pred_05,
        digits=4
    )
)

print("Confusion Matrix:")
print(confusion_matrix(y_val, y_val_pred_05))

print(
    "ROC-AUC:",
    round(roc_auc_score(y_val, y_val_proba), 4)
)

print(
    "PR-AUC:",
    round(average_precision_score(y_val, y_val_proba), 4)
)


# ============================================================
# 14. THRESHOLD ANALYSIS
# ============================================================
#
# We do NOT select threshold based only on F1.
#
# Because FN is more expensive:
#
#     Total Cost =
#         FP * COST_FP
#         +
#         FN * COST_FN
#
# ============================================================

thresholds = np.arange(0.10, 0.71, 0.01)

threshold_results = []

for threshold in thresholds:

    y_pred = (
        y_val_proba >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val,
        y_pred
    ).ravel()

    precision = precision_score(
        y_val,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        y_pred,
        zero_division=0
    )

    total_cost = (
        fp * COST_FP
        +
        fn * COST_FN
    )

    threshold_results.append({
        "Threshold": threshold,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Total_Cost": total_cost
    })


threshold_df = pd.DataFrame(threshold_results)


# ============================================================
# 15. SELECT BEST THRESHOLD
# ============================================================

best_row = threshold_df.loc[
    threshold_df["Total_Cost"].idxmin()
]

BEST_THRESHOLD = best_row["Threshold"]


print("\n" + "=" * 70)
print("THRESHOLD SELECTION")
print("=" * 70)

print(
    f"Business cost assumption: "
    f"FP={COST_FP}, FN={COST_FN}"
)

print(
    f"\nBest threshold: {BEST_THRESHOLD:.2f}"
)

print(
    best_row[
        [
            "Threshold",
            "Precision",
            "Recall",
            "F1",
            "FP",
            "FN",
            "TP",
            "TN",
            "Total_Cost"
        ]
    ]
)


# ============================================================
# 16. PLOT — PRECISION / RECALL / F1 VS THRESHOLD
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    threshold_df["Threshold"],
    threshold_df["Precision"],
    label="Precision"
)

plt.plot(
    threshold_df["Threshold"],
    threshold_df["Recall"],
    label="Recall"
)

plt.plot(
    threshold_df["Threshold"],
    threshold_df["F1"],
    label="F1"
)

plt.axvline(
    BEST_THRESHOLD,
    linestyle="--",
    label=f"Selected threshold = {BEST_THRESHOLD:.2f}"
)

plt.xlabel("Classification Threshold")
plt.ylabel("Score")
plt.title("Precision, Recall and F1 vs Threshold")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "01_precision_recall_f1_vs_threshold.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# 17. PLOT — BUSINESS COST VS THRESHOLD
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    threshold_df["Threshold"],
    threshold_df["Total_Cost"]
)

plt.axvline(
    BEST_THRESHOLD,
    linestyle="--",
    label=f"Selected threshold = {BEST_THRESHOLD:.2f}"
)

plt.xlabel("Classification Threshold")
plt.ylabel("Total Misclassification Cost")
plt.title(
    f"Business Cost vs Threshold "
    f"(FP={COST_FP}, FN={COST_FN})"
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "02_business_cost_vs_threshold.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# 18. FINAL VALIDATION PREDICTIONS
# ============================================================

y_val_pred = (
    y_val_proba >= BEST_THRESHOLD
).astype(int)


print("\n" + "=" * 70)
print("FINAL VALIDATION RESULT")
print("=" * 70)

print(
    classification_report(
        y_val,
        y_val_pred,
        digits=4
    )
)

print("Confusion Matrix:")
print(confusion_matrix(y_val, y_val_pred))


# ============================================================
# 19. ROC CURVE — VALIDATION
# ============================================================

fpr, tpr, _ = roc_curve(
    y_val,
    y_val_proba
)

roc_auc = roc_auc_score(
    y_val,
    y_val_proba
)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"Logistic Regression (AUC = {roc_auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Validation")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "03_ROC_curve_validation.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# 20. PRECISION-RECALL CURVE — VALIDATION
# ============================================================

precision_curve, recall_curve, _ = precision_recall_curve(
    y_val,
    y_val_proba
)

pr_auc = average_precision_score(
    y_val,
    y_val_proba
)

baseline = y_val.mean()

plt.figure(figsize=(8, 6))

plt.plot(
    recall_curve,
    precision_curve,
    label=f"Logistic Regression (PR-AUC = {pr_auc:.3f})"
)

plt.axhline(
    baseline,
    linestyle="--",
    label=f"Baseline = {baseline:.3f}"
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve — Validation")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "04_precision_recall_curve_validation.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# 21. CONFUSION MATRIX — VALIDATION
# ============================================================

cm_val = confusion_matrix(
    y_val,
    y_val_pred
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_val,
    display_labels=["Non-default", "Default"]
)

fig, ax = plt.subplots(figsize=(7, 6))

disp.plot(
    ax=ax,
    values_format="d"
)

ax.set_title(
    f"Confusion Matrix — Validation\n"
    f"Threshold = {BEST_THRESHOLD:.2f}"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "05_confusion_matrix_validation.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# 22. CALIBRATION CURVE — VALIDATION
# ============================================================

prob_true, prob_pred = calibration_curve(
    y_val,
    y_val_proba,
    n_bins=10,
    strategy="quantile"
)

brier = brier_score_loss(
    y_val,
    y_val_proba
)

plt.figure(figsize=(8, 6))

plt.plot(
    prob_pred,
    prob_true,
    marker="o",
    label=f"Logistic Regression"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect calibration"
)

plt.xlabel("Mean Predicted Probability")
plt.ylabel("Observed Default Rate")
plt.title(
    f"Calibration Curve — Validation\n"
    f"Brier Score = {brier:.4f}"
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "06_calibration_curve_validation.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# 23. MODEL COEFFICIENTS
# ============================================================

feature_names = (
    pipeline
    .named_steps["preprocessing"]
    .get_feature_names_out()
)

coefficients = (
    pipeline
    .named_steps["model"]
    .coef_[0]
)

coef_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients
})

coef_df["Absolute_Coefficient"] = (
    coef_df["Coefficient"].abs()
)

coef_df = coef_df.sort_values(
    "Absolute_Coefficient",
    ascending=False
)


print("\n" + "=" * 70)
print("TOP MODEL COEFFICIENTS")
print("=" * 70)

print(
    coef_df.head(20).to_string(index=False)
)


# ============================================================
# 24. COEFFICIENT PLOT
# ============================================================

top_n = 20

plot_df = (
    coef_df
    .head(top_n)
    .sort_values("Coefficient")
)

plt.figure(figsize=(10, 8))

plt.barh(
    plot_df["Feature"],
    plot_df["Coefficient"]
)

plt.axvline(
    0,
    linestyle="--"
)

plt.xlabel("Logistic Regression Coefficient")
plt.ylabel("Feature")
plt.title(
    f"Top {top_n} Features by Absolute Coefficient"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "07_top_coefficients.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# 25. ODDS RATIOS
# ============================================================

coef_df["Odds_Ratio"] = np.exp(
    coef_df["Coefficient"]
)

odds_df = coef_df[
    [
        "Feature",
        "Coefficient",
        "Odds_Ratio"
    ]
].copy()


print("\n" + "=" * 70)
print("TOP ODDS RATIOS")
print("=" * 70)

print(
    odds_df.head(20).to_string(index=False)
)


# ============================================================
# 26. ODDS RATIO PLOT
# ============================================================

plot_or = (
    odds_df
    .assign(
        Distance_From_1=lambda x:
        (x["Odds_Ratio"] - 1).abs()
    )
    .sort_values(
        "Distance_From_1",
        ascending=False
    )
    .head(top_n)
    .sort_values("Odds_Ratio")
)

plt.figure(figsize=(10, 8))

plt.barh(
    plot_or["Feature"],
    plot_or["Odds_Ratio"]
)

plt.axvline(
    1,
    linestyle="--"
)

plt.xlabel("Odds Ratio")
plt.ylabel("Feature")
plt.title(
    f"Top {top_n} Odds Ratios"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "08_odds_ratios.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# 27. RISK GROUPS
# ============================================================
#
# IMPORTANT:
#
# Binary classification threshold and risk-group boundaries
# are NOT the same concept.
#
# Here we use illustrative probability bands:
#
# Low:    < 0.30
# Medium: 0.30 - <0.60
# High:   >= 0.60
#
# These boundaries can later be optimized using business
# requirements and calibration.
# ============================================================

def assign_risk_group(probability):

    if probability < 0.37:
        return "Low"

    elif probability < 0.7:
        return "Medium"

    else:
        return "High"


risk_groups = pd.Series(
    y_val_proba
).apply(assign_risk_group)


risk_distribution = (
    risk_groups
    .value_counts()
    .reindex(["Low", "Medium", "High"])
    .fillna(0)
)

risk_percentage = (
    risk_distribution
    / len(risk_groups)
    * 100
)


print("\n" + "=" * 70)
print("VALIDATION RISK GROUP DISTRIBUTION")
print("=" * 70)

risk_table = pd.DataFrame({
    "Count": risk_distribution.astype(int),
    "Percentage": risk_percentage.round(2)
})

print(risk_table)


# ============================================================
# 28. RISK GROUP BAR CHART
# ============================================================

plt.figure(figsize=(8, 6))

bars = plt.bar(
    risk_distribution.index,
    risk_distribution.values
)

plt.xlabel("Risk Group")
plt.ylabel("Number of Customers")
plt.title("Customer Risk Group Distribution — Validation")

for bar in bars:

    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{int(height)}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "09_risk_group_distribution.png"
    ),
    dpi=300
)

plt.show()

# ============================================================
# 29. VALIDATION RISK GROUP PERFORMANCE
# ============================================================

risk_table = pd.DataFrame({
    "Risk_Group": risk_groups.values,
    "Actual_Default": y_val.values
})

risk_summary = (
    risk_table
    .groupby("Risk_Group")
    .agg(
        Count=("Actual_Default", "size"),
        Default_Count=("Actual_Default", "sum"),
        Default_Rate=("Actual_Default", "mean")
    )
    .reindex(["Low", "Medium", "High"])
)

risk_summary["Percentage"] = (
    risk_summary["Count"] / len(risk_groups) * 100
)

risk_summary["Default_Rate"] = (
    risk_summary["Default_Rate"] * 100
)

print("\n" + "=" * 70)
print("VALIDATION RISK GROUP PERFORMANCE")
print("=" * 70)

print(risk_summary.round(2))
# ============================================================
# 29. FINAL MODEL FIT ON TRAIN + VALIDATION
# ============================================================
#
# After selecting:
#
#     - model
#     - threshold
#
# we can use Train + Validation to train the final model.
#
# Test remains completely untouched.
#
# ============================================================

X_train_val = pd.concat(
    [X_train, X_val],
    axis=0
)

y_train_val = pd.concat(
    [y_train, y_val],
    axis=0
)


final_pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("model", LogisticRegression(
            class_weight="balanced",
            solver="liblinear",
            max_iter=2000,
            random_state=RANDOM_STATE
        ))
    ]
)


print("\n" + "=" * 70)
print("FITTING FINAL MODEL ON TRAIN + VALIDATION")
print("=" * 70)

final_pipeline.fit(
    X_train_val,
    y_train_val
)

print("Final model fitted.")


# ============================================================
# 30. FINAL TEST PREDICTIONS
# ============================================================

y_test_proba = (
    final_pipeline
    .predict_proba(X_test)[:, 1]
)

y_test_pred = (
    y_test_proba >= BEST_THRESHOLD
).astype(int)


# ============================================================
# 31. FINAL TEST EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

print(
    classification_report(
        y_test,
        y_test_pred,
        digits=4
    )
)

cm_test = confusion_matrix(
    y_test,
    y_test_pred
)

print("Confusion Matrix:")
print(cm_test)

test_roc_auc = roc_auc_score(
    y_test,
    y_test_proba
)

test_pr_auc = average_precision_score(
    y_test,
    y_test_proba
)

test_brier = brier_score_loss(
    y_test,
    y_test_proba
)

print(
    f"\nROC-AUC : {test_roc_auc:.4f}"
)

print(
    f"PR-AUC  : {test_pr_auc:.4f}"
)

print(
    f"Brier   : {test_brier:.4f}"
)

print(
    f"Threshold: {BEST_THRESHOLD:.2f}"
)


# ============================================================
# 32. FINAL TEST COST
# ============================================================

tn, fp, fn, tp = cm_test.ravel()

test_cost = (
    fp * COST_FP
    +
    fn * COST_FN
)

print(
    f"Total Test Cost: {test_cost}"
)


# ============================================================
# 33. FINAL TEST CONFUSION MATRIX
# ============================================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_test,
    display_labels=["Non-default", "Default"]
)

fig, ax = plt.subplots(figsize=(7, 6))

disp.plot(
    ax=ax,
    values_format="d"
)

ax.set_title(
    f"Final Test Confusion Matrix\n"
    f"Threshold = {BEST_THRESHOLD:.2f}"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "10_final_test_confusion_matrix.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# 34. FINAL TEST ROC CURVE
# ============================================================

fpr_test, tpr_test, _ = roc_curve(
    y_test,
    y_test_proba
)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr_test,
    tpr_test,
    label=f"ROC-AUC = {test_roc_auc:.3f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Final Test ROC Curve")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "11_final_test_ROC_curve.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# 35. FINAL TEST PRECISION-RECALL CURVE
# ============================================================

precision_test, recall_test, _ = precision_recall_curve(
    y_test,
    y_test_proba
)

test_baseline = y_test.mean()

plt.figure(figsize=(8, 6))

plt.plot(
    recall_test,
    precision_test,
    label=f"PR-AUC = {test_pr_auc:.3f}"
)

plt.axhline(
    test_baseline,
    linestyle="--",
    label=f"Baseline = {test_baseline:.3f}"
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Final Test Precision-Recall Curve")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "12_final_test_precision_recall_curve.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# 36. SAVE THRESHOLD ANALYSIS
# ============================================================

threshold_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "threshold_analysis_validation.csv"
    ),
    index=False
)



# ============================================================
# 37. SAVE COEFFICIENTS
# ============================================================

coef_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "logistic_coefficients.csv"
    ),
    index=False
)


# ============================================================
# 38. SAVE TEST PREDICTIONS
# ============================================================

test_results = X_test.copy()

test_results["Actual_Default"] = y_test.values

test_results["Predicted_Probability"] = y_test_proba

test_results["Predicted_Default"] = y_test_pred

test_results["Risk_Group"] = (
    pd.Series(y_test_proba)
    .apply(assign_risk_group)
    .values
)

test_results.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "final_test_predictions.csv"
    ),
    index=False
)


# ============================================================
# 39. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL MODEL SUMMARY")
print("=" * 70)

print(
    """
Model:
    Logistic Regression

Preprocessing:
    - Signed Log Transform
    - RobustScaler
    - One-Hot Encoding

Feature Engineering:
    - N_DELAY_MONTHS

Class Imbalance:
    - class_weight='balanced'

Threshold Selection:
    - Validation set
    - Business cost based

Final Training:
    - Train + Validation

Final Evaluation:
    - Untouched Test set
"""
)

print(
    f"Selected Threshold : {BEST_THRESHOLD:.2f}"
)

print(
    f"Test ROC-AUC       : {test_roc_auc:.4f}"
)

print(
    f"Test PR-AUC        : {test_pr_auc:.4f}"
)

print(
    f"Test Brier Score   : {test_brier:.4f}"
)

print(
    f"Test TN            : {tn}"
)

print(
    f"Test FP            : {fp}"
)

print(
    f"Test FN            : {fn}"
)

print(
    f"Test TP            : {tp}"
)

print(
    f"Test Total Cost    : {test_cost}"
)

print("\nOutput files saved in:")
print(OUTPUT_DIR)

print("\nDone.")