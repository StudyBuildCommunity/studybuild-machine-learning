
# ============================================================
# FINAL LOGISTIC REGRESSION MODEL + 5-FOLD CROSS-VALIDATION
# UCI Default of Credit Card Clients
# ============================================================

# Pipeline:
# 1. Load cleaned data
# 2. Feature engineering
# 3. Train / Validation / Test split
# 4. Preprocessing:
#       - Signed Log Transform
#       - RobustScaler
#       - One-Hot Encoding
# 5. 5-Fold Stratified Cross-Validation
#       - Select best C using PR-AUC
# 6. Logistic Regression with best C
# 7. Threshold selection on Validation
#       based on business cost
# 8. Final model training on Train + Validation
# 9. Final evaluation on untouched Test
# 10. Model interpretation
# 11. Calibration
# 12. Risk groups
# ============================================================


# ============================================================
# 0. IMPORT LIBRARIES
# ============================================================

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV
)

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

DATA_PATH = (
    "studybuuild_3/default of credit card clients_cleaned.xlsx"
)

OUTPUT_DIR = "final_logistic_cv_outputs"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ------------------------------------------------------------
# Business costs
# ------------------------------------------------------------

# FN = actual default but model predicts non-default
# FP = actual non-default but model predicts default

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

df["N_DELAY_MONTHS"] = (
    df[pay_cols] > 0
).sum(axis=1)


# ============================================================
# 4. DEFINE X AND y
# ============================================================

TARGET = "default payment next month"

X = df.drop(
    columns=["ID", TARGET]
)

y = df[TARGET]


print("\n" + "=" * 70)
print("TARGET DISTRIBUTION")
print("=" * 70)

print(y.value_counts())

print("\nPercent:")
print(
    (y.value_counts(normalize=True) * 100).round(2)
)


# ============================================================
# 5. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

# 60% Train
# 20% Validation
# 20% Test

# Train:
#     Used for model fitting and Cross-Validation
#
# Validation:
#     Used for threshold selection
#
# Test:
#     Used only for final evaluation

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

    Preserves the sign while reducing the influence
    of highly skewed values and outliers.
    """

    X = np.asarray(
        X,
        dtype=float
    )

    return (
        np.sign(X)
        * np.log1p(np.abs(X))
    )


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
# 9. BASE LOGISTIC REGRESSION PIPELINE
# ============================================================

# C controls the amount of regularization.
#
# Larger C:
#     weaker regularization
#
# Smaller C:
#     stronger regularization
#
# C will be selected using 5-Fold Cross-Validation.

base_model = LogisticRegression(

    class_weight="balanced",

    solver="liblinear",

    max_iter=2000,

    random_state=RANDOM_STATE
)


pipeline = Pipeline(
    steps=[

        (
            "preprocessing",
            preprocessor
        ),

        (
            "model",
            base_model
        )
    ]
)


# ============================================================
# 10. 5-FOLD CROSS-VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("5-FOLD CROSS-VALIDATION")
print("=" * 70)

# StratifiedKFold preserves the class distribution
# in each fold.

cv = StratifiedKFold(

    n_splits=5,

    shuffle=True,

    random_state=RANDOM_STATE
)


# Candidate values for C

param_grid = {

    "model__C": [
        0.001,
        0.01,
        0.1,
        1,
        10,
        100
    ]
}


# PR-AUC / Average Precision is used because
# the target is imbalanced.

grid_search = GridSearchCV(

    estimator=pipeline,

    param_grid=param_grid,

    scoring="average_precision",

    cv=cv,

    n_jobs=-1,

    refit=True,

    return_train_score=True
)


print("\nRunning 5-Fold CV...")
print("Scoring metric: Average Precision (PR-AUC)")

grid_search.fit(
    X_train,
    y_train
)


print("\n" + "=" * 70)
print("CROSS-VALIDATION RESULTS")
print("=" * 70)

print(
    "Best C:",
    grid_search.best_params_["model__C"]
)

print(
    "Best CV PR-AUC:",
    round(
        grid_search.best_score_,
        4
    )
)


# ============================================================
# 11. CV RESULTS TABLE
# ============================================================

cv_results = pd.DataFrame(
    grid_search.cv_results_
)

cv_results_table = cv_results[
    [
        "param_model__C",
        "mean_test_score",
        "std_test_score",
        "mean_train_score",
        "std_train_score"
    ]
].copy()

cv_results_table.columns = [
    "C",
    "Mean_CV_PR_AUC",
    "Std_CV_PR_AUC",
    "Mean_Train_PR_AUC",
    "Std_Train_PR_AUC"
]

cv_results_table = cv_results_table.sort_values(
    "Mean_CV_PR_AUC",
    ascending=False
)

print("\nCV Results:")
print(
    cv_results_table.round(4).to_string(
        index=False
    )
)

cv_results_table.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "logistic_cv_results.csv"
    ),
    index=False
)


# ============================================================
# 12. BEST PIPELINE
# ============================================================

# GridSearchCV refit=True means that the best pipeline
# is automatically fitted on the complete training set.

best_pipeline = grid_search.best_estimator_

BEST_C = grid_search.best_params_[
    "model__C"
]


print("\n" + "=" * 70)
print("BEST LOGISTIC MODEL")
print("=" * 70)

print(
    "Selected C:",
    BEST_C
)

print(
    "Best CV PR-AUC:",
    round(
        grid_search.best_score_,
        4
    )
)


# ============================================================
# 13. VALIDATION PROBABILITIES
# ============================================================

y_val_proba = (
    best_pipeline
    .predict_proba(X_val)[:, 1]
)


# ============================================================
# 14. VALIDATION PERFORMANCE AT THRESHOLD = 0.50
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

print(
    confusion_matrix(
        y_val,
        y_val_pred_05
    )
)

print(
    "ROC-AUC:",
    round(
        roc_auc_score(
            y_val,
            y_val_proba
        ),
        4
    )
)

print(
    "PR-AUC:",
    round(
        average_precision_score(
            y_val,
            y_val_proba
        ),
        4
    )
)


# ============================================================
# 15. THRESHOLD ANALYSIS
# ============================================================

# Threshold selection is SEPARATE from Cross-Validation.
#
# CV selected C.
#
# Validation selects the classification threshold.
#
# Business cost:
#
# Total Cost =
#       FP * COST_FP
#       +
#       FN * COST_FN

thresholds = np.arange(
    0.10,
    0.71,
    0.01
)

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


threshold_df = pd.DataFrame(
    threshold_results
)


# ============================================================
# 16. SELECT BEST THRESHOLD
# ============================================================

best_row = threshold_df.loc[
    threshold_df["Total_Cost"].idxmin()
]

BEST_THRESHOLD = best_row[
    "Threshold"
]


print("\n" + "=" * 70)
print("THRESHOLD SELECTION")
print("=" * 70)

print(
    f"Business cost assumption: "
    f"FP={COST_FP}, FN={COST_FN}"
)

print(
    f"\nBest threshold: "
    f"{BEST_THRESHOLD:.2f}"
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
# 17. PRECISION / RECALL / F1 VS THRESHOLD
# ============================================================

plt.figure(
    figsize=(10, 6)
)

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
    label=(
        f"Selected threshold = "
        f"{BEST_THRESHOLD:.2f}"
    )
)

plt.xlabel(
    "Classification Threshold"
)

plt.ylabel(
    "Score"
)

plt.title(
    "Precision, Recall and F1 vs Threshold"
)

plt.legend()

plt.grid(
    alpha=0.3
)

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
# 18. BUSINESS COST VS THRESHOLD
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    threshold_df["Threshold"],
    threshold_df["Total_Cost"]
)

plt.axvline(
    BEST_THRESHOLD,
    linestyle="--",
    label=(
        f"Selected threshold = "
        f"{BEST_THRESHOLD:.2f}"
    )
)

plt.xlabel(
    "Classification Threshold"
)

plt.ylabel(
    "Total Misclassification Cost"
)

plt.title(
    f"Business Cost vs Threshold "
    f"(FP={COST_FP}, FN={COST_FN})"
)

plt.legend()

plt.grid(
    alpha=0.3
)

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
# 19. FINAL VALIDATION PREDICTIONS
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

print(
    confusion_matrix(
        y_val,
        y_val_pred
    )
)


# ============================================================
# 20. ROC CURVE — VALIDATION
# ============================================================

fpr, tpr, _ = roc_curve(
    y_val,
    y_val_proba
)

val_roc_auc = roc_auc_score(
    y_val,
    y_val_proba
)

plt.figure(
    figsize=(8, 6)
)

plt.plot(
    fpr,
    tpr,
    label=(
        f"Logistic Regression "
        f"(AUC = {val_roc_auc:.3f})"
    )
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random classifier"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curve — Validation"
)

plt.legend()

plt.grid(
    alpha=0.3
)

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
# 21. PRECISION-RECALL CURVE — VALIDATION
# ============================================================

precision_curve, recall_curve, _ = (
    precision_recall_curve(
        y_val,
        y_val_proba
    )
)

val_pr_auc = average_precision_score(
    y_val,
    y_val_proba
)

baseline = y_val.mean()


plt.figure(
    figsize=(8, 6)
)

plt.plot(
    recall_curve,
    precision_curve,
    label=(
        f"Logistic Regression "
        f"(PR-AUC = {val_pr_auc:.3f})"
    )
)

plt.axhline(
    baseline,
    linestyle="--",
    label=(
        f"Baseline = "
        f"{baseline:.3f}"
    )
)

plt.xlabel(
    "Recall"
)

plt.ylabel(
    "Precision"
)

plt.title(
    "Precision-Recall Curve — Validation"
)

plt.legend()

plt.grid(
    alpha=0.3
)

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
# 22. CONFUSION MATRIX — VALIDATION
# ============================================================

cm_val = confusion_matrix(
    y_val,
    y_val_pred
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_val,
    display_labels=[
        "Non-default",
        "Default"
    ]
)

fig, ax = plt.subplots(
    figsize=(7, 6)
)

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
# 23. CALIBRATION CURVE — VALIDATION
# ============================================================

prob_true, prob_pred = calibration_curve(
    y_val,
    y_val_proba,
    n_bins=10,
    strategy="quantile"
)

val_brier = brier_score_loss(
    y_val,
    y_val_proba
)


plt.figure(
    figsize=(8, 6)
)

plt.plot(
    prob_pred,
    prob_true,
    marker="o",
    label="Logistic Regression"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect calibration"
)

plt.xlabel(
    "Mean Predicted Probability"
)

plt.ylabel(
    "Observed Default Rate"
)

plt.title(
    f"Calibration Curve — Validation\n"
    f"Brier Score = {val_brier:.4f}"
)

plt.legend()

plt.grid(
    alpha=0.3
)

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
# 24. MODEL COEFFICIENTS
# ============================================================

feature_names = (
    best_pipeline
    .named_steps["preprocessing"]
    .get_feature_names_out()
)

coefficients = (
    best_pipeline
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
    coef_df.head(20).to_string(
        index=False
    )
)


# ============================================================
# 25. COEFFICIENT PLOT
# ============================================================

top_n = 20

plot_df = (
    coef_df
    .head(top_n)
    .sort_values(
        "Coefficient"
    )
)

plt.figure(
    figsize=(10, 8)
)

plt.barh(
    plot_df["Feature"],
    plot_df["Coefficient"]
)

plt.axvline(
    0,
    linestyle="--"
)

plt.xlabel(
    "Logistic Regression Coefficient"
)

plt.ylabel(
    "Feature"
)

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
# 26. ODDS RATIOS
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
    odds_df.head(20).to_string(
        index=False
    )
)


# ============================================================
# 27. ODDS RATIO PLOT
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
    .sort_values(
        "Odds_Ratio"
    )
)

plt.figure(
    figsize=(10, 8)
)

plt.barh(
    plot_or["Feature"],
    plot_or["Odds_Ratio"]
)

plt.axvline(
    1,
    linestyle="--"
)

plt.xlabel(
    "Odds Ratio"
)

plt.ylabel(
    "Feature"
)

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
# 28. RISK GROUPS
# ============================================================

# IMPORTANT:
#
# Binary classification threshold and risk-group boundaries
# are NOT the same concept.
#
# Risk groups are defined here using:
#
# Low:    < 0.37
# Medium: 0.37 - <0.70
# High:   >= 0.70
#
# These are descriptive probability bands.


def assign_risk_group(probability):

    if probability < 0.37:

        return "Low"

    elif probability < 0.70:

        return "Medium"

    else:

        return "High"


risk_groups = pd.Series(
    y_val_proba
).apply(
    assign_risk_group
)


risk_distribution = (
    risk_groups
    .value_counts()
    .reindex(
        [
            "Low",
            "Medium",
            "High"
        ]
    )
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

    "Count":
    risk_distribution.astype(int),

    "Percentage":
    risk_percentage.round(2)
})

print(
    risk_table
)


# ============================================================
# 29. RISK GROUP PERFORMANCE
# ============================================================

risk_table_performance = pd.DataFrame({

    "Risk_Group":
    risk_groups.values,

    "Actual_Default":
    y_val.values
})


risk_summary = (
    risk_table_performance
    .groupby(
        "Risk_Group"
    )
    .agg(

        Count=(
            "Actual_Default",
            "size"
        ),

        Default_Count=(
            "Actual_Default",
            "sum"
        ),

        Default_Rate=(
            "Actual_Default",
            "mean"
        )
    )
    .reindex(
        [
            "Low",
            "Medium",
            "High"
        ]
    )
)

risk_summary["Percentage"] = (
    risk_summary["Count"]
    / len(risk_groups)
    * 100
)

risk_summary["Default_Rate"] = (
    risk_summary["Default_Rate"]
    * 100
)


print("\n" + "=" * 70)
print("VALIDATION RISK GROUP PERFORMANCE")
print("=" * 70)

print(
    risk_summary.round(2)
)


# ============================================================
# 30. FINAL MODEL FIT ON TRAIN + VALIDATION
# ============================================================

# After selecting:
#
#     1. Best C using CV
#     2. Best threshold using Validation
#
# We train the final model using:
#
#     Train + Validation
#
# Test remains completely untouched.


X_train_val = pd.concat(
    [
        X_train,
        X_val
    ],
    axis=0
)

y_train_val = pd.concat(
    [
        y_train,
        y_val
    ],
    axis=0
)


final_pipeline = Pipeline(

    steps=[

        (
            "preprocessing",
            preprocessor
        ),

        (
            "model",

            LogisticRegression(

                C=BEST_C,

                class_weight="balanced",

                solver="liblinear",

                max_iter=2000,

                random_state=RANDOM_STATE
            )
        )
    ]
)


print("\n" + "=" * 70)
print("FITTING FINAL MODEL ON TRAIN + VALIDATION")
print("=" * 70)

final_pipeline.fit(
    X_train_val,
    y_train_val
)

print(
    "Final model fitted."
)


# ============================================================
# 31. FINAL TEST PREDICTIONS
# ============================================================

y_test_proba = (
    final_pipeline
    .predict_proba(X_test)[:, 1]
)

y_test_pred = (
    y_test_proba >= BEST_THRESHOLD
).astype(int)


# ============================================================
# 32. FINAL TEST EVALUATION
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

print(
    f"Selected C: {BEST_C}"
)


# ============================================================
# 33. FINAL TEST COST
# ============================================================

tn, fp, fn, tp = (
    cm_test.ravel()
)

test_cost = (
    fp * COST_FP
    +
    fn * COST_FN
)

print(
    f"Total Test Cost: {test_cost}"
)


# ============================================================
# 34. FINAL TEST CONFUSION MATRIX
# ============================================================

disp = ConfusionMatrixDisplay(

    confusion_matrix=cm_test,

    display_labels=[
        "Non-default",
        "Default"
    ]
)

fig, ax = plt.subplots(
    figsize=(7, 6)
)

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
        "09_final_test_confusion_matrix.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# 35. FINAL TEST ROC CURVE
# ============================================================

fpr_test, tpr_test, _ = roc_curve(
    y_test,
    y_test_proba
)


plt.figure(
    figsize=(8, 6)
)

plt.plot(
    fpr_test,
    tpr_test,
    label=(
        f"ROC-AUC = "
        f"{test_roc_auc:.3f}"
    )
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "Final Test ROC Curve"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "10_final_test_ROC_curve.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# 36. FINAL TEST PRECISION-RECALL CURVE
# ============================================================

precision_test, recall_test, _ = (
    precision_recall_curve(
        y_test,
        y_test_proba
    )
)

test_baseline = y_test.mean()


plt.figure(
    figsize=(8, 6)
)

plt.plot(
    recall_test,
    precision_test,
    label=(
        f"PR-AUC = "
        f"{test_pr_auc:.3f}"
    )
)

plt.axhline(
    test_baseline,
    linestyle="--",
    label=(
        f"Baseline = "
        f"{test_baseline:.3f}"
    )
)

plt.xlabel(
    "Recall"
)

plt.ylabel(
    "Precision"
)

plt.title(
    "Final Test Precision-Recall Curve"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "11_final_test_precision_recall_curve.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# 37. SAVE THRESHOLD ANALYSIS
# ============================================================

threshold_df.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "threshold_analysis_validation.csv"
    ),

    index=False
)


# ============================================================
# 38. SAVE COEFFICIENTS
# ============================================================

final_feature_names = (
    final_pipeline
    .named_steps["preprocessing"]
    .get_feature_names_out()
)

final_coefficients = (
    final_pipeline
    .named_steps["model"]
    .coef_[0]
)

final_coef_df = pd.DataFrame({

    "Feature":
    final_feature_names,

    "Coefficient":
    final_coefficients
})

final_coef_df["Absolute_Coefficient"] = (
    final_coef_df["Coefficient"].abs()
)

final_coef_df = final_coef_df.sort_values(
    "Absolute_Coefficient",
    ascending=False
)

final_coef_df["Odds_Ratio"] = np.exp(
    final_coef_df["Coefficient"]
)

final_coef_df.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "final_logistic_coefficients.csv"
    ),

    index=False
)


# ============================================================
# 39. SAVE TEST PREDICTIONS
# ============================================================

test_results = X_test.copy()

test_results["Actual_Default"] = (
    y_test.values
)

test_results["Predicted_Probability"] = (
    y_test_proba
)

test_results["Predicted_Default"] = (
    y_test_pred
)

test_results["Risk_Group"] = (

    pd.Series(
        y_test_proba
    )
    .apply(
        assign_risk_group
    )
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
# 40. Q9 — THRESHOLD STABILITY ANALYSIS
# ============================================================

comparison_thresholds = [
    0.30,
    float(BEST_THRESHOLD)
]

threshold_comparison = []


for threshold in comparison_thresholds:

    y_val_pred_temp = (
        y_val_proba >= threshold
    ).astype(int)

    tn_temp, fp_temp, fn_temp, tp_temp = (
        confusion_matrix(
            y_val,
            y_val_pred_temp
        ).ravel()
    )

    precision_temp = precision_score(
        y_val,
        y_val_pred_temp,
        zero_division=0
    )

    recall_temp = recall_score(
        y_val,
        y_val_pred_temp,
        zero_division=0
    )

    threshold_comparison.append({

        "Threshold":
        threshold,

        "Precision":
        precision_temp,

        "Recall":
        recall_temp,

        "TN":
        tn_temp,

        "FP":
        fp_temp,

        "FN":
        fn_temp,

        "TP":
        tp_temp
    })


threshold_comparison_df = pd.DataFrame(
    threshold_comparison
)


print("\n" + "=" * 70)
print("Q9 - THRESHOLD STABILITY ANALYSIS")
print("=" * 70)

print(
    threshold_comparison_df.round(3)
)


threshold_comparison_df.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "threshold_stability_analysis.csv"
    ),

    index=False
)


# ============================================================
# 41. FINAL MODEL SUMMARY
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

Cross-Validation:

    - Stratified 5-Fold Cross-Validation
    - Hyperparameter: C
    - Scoring: Average Precision (PR-AUC)

Threshold Selection:

    - Validation set
    - Business cost based
    - FP = 1
    - FN = 5

Final Training:

    - Train + Validation

Final Evaluation:

    - Untouched Test set
"""
)

print(
    f"Selected C        : {BEST_C}"
)

print(
    f"Best CV PR-AUC    : "
    f"{grid_search.best_score_:.4f}"
)

print(
    f"Selected Threshold : "
    f"{BEST_THRESHOLD:.2f}"
)

print(
    f"Test ROC-AUC      : "
    f"{test_roc_auc:.4f}"
)

print(
    f"Test PR-AUC       : "
    f"{test_pr_auc:.4f}"
)

print(
    f"Test Brier Score  : "
    f"{test_brier:.4f}"
)

print(
    f"Test TN           : {tn}"
)

print(
    f"Test FP           : {fp}"
)

print(
    f"Test FN           : {fn}"
)

print(
    f"Test TP           : {tp}"
)

print(
    f"Test Total Cost   : {test_cost}"
)

print(
    "\nOutput files saved in:"
)

print(
    OUTPUT_DIR
)

print(
    "\nDone."
)
