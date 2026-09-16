# ============================================================
# CREDIT CARD DEFAULT PREDICTION
# Logistic Regression + LASSO+ smote+ apply features
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score
)
#from imblearn.over_sampling import SMOTE

# ============================================================
# 0. READ DATA
# ============================================================

path = "studybuuild_3/default of credit card clients_cleaned.xlsx"

df = pd.read_excel(path)

print("Original shape:", df.shape)

# ============================================================
# 0.5. FEATURE ENGINEERING
# ============================================================

pay_cols = [
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6"
]

bill_cols = [
    f"BILL_AMT{i}" for i in range(1, 7)
]

payamt_cols = [
    f"PAY_AMT{i}" for i in range(1, 7)
]


# Number of months with positive payment delay
df["N_DELAY_MONTHS"] = (
    df[pay_cols] > 0
).sum(axis=1)


# Average payment delay status
df["AVG_DELAY"] = (
    df[pay_cols].mean(axis=1)
)


# Maximum payment delay status
df["MAX_DELAY"] = (
    df[pay_cols].max(axis=1)
)


# Total payment / total bill ratio
df["PAYMENT_RATIO"] = (
    df[payamt_cols].sum(axis=1)
    /
    df[bill_cols].sum(axis=1).replace(0, 1)
)


# Average bill / credit limit
df["UTILIZATION"] = (
    df[bill_cols].mean(axis=1)
    /
    df["LIMIT_BAL"].replace(0, 1)
)


# Change in bill amount between month 1 and month 6
df["BILL_TREND"] = (
    df["BILL_AMT1"]
    -
    df["BILL_AMT6"]
)


# print("\n==============================")
# print("ENGINEERED FEATURES")
# print("==============================")

# engineered_cols = [
#     "N_DELAY_MONTHS",
#     "AVG_DELAY",
#     "MAX_DELAY",
#     "PAYMENT_RATIO",
#     "UTILIZATION",
#     "BILL_TREND"
# ]

# print(engineered_cols)

# print("\nEngineered feature summary:")
# print(
#     df[engineered_cols].describe()
# )

# ============================================================
# 1. DEFINE TARGET AND FEATURES
# ============================================================

target = "default payment next month"

# Remove target and ID
X = df.drop(columns=[target, "ID"])
y = df[target]

# print("\nTarget distribution:")
# print(y.value_counts())

# print("\nTarget percentage:")
# print(y.value_counts(normalize=True) * 100)


# ============================================================
# 2. DEFINE FEATURE GROUPS
# ============================================================

# Continuous numerical variables
continuous_cols = (
    [
        "LIMIT_BAL",
        "AGE",
        "N_DELAY_MONTHS",
        "AVG_DELAY",
        "MAX_DELAY",
        "PAYMENT_RATIO",
        "UTILIZATION",
        "BILL_TREND"
    ]
    +
    [f"BILL_AMT{i}" for i in range(1, 7)]
    +
    [f"PAY_AMT{i}" for i in range(1, 7)]
)
# Variables selected for Signed Log Transformation
log_cols = (
    ["LIMIT_BAL"] +
    [f"BILL_AMT{i}" for i in range(1, 7)] +
    [f"PAY_AMT{i}" for i in range(1, 7)]
)

# Categorical variables
categorical_cols = [
    "SEX",
    "EDUCATION",
    "MARRIAGE"
]


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain shape:", X_train.shape)
print("Test shape :", X_test.shape)


# ============================================================
# 4. ONE-HOT ENCODING
# ============================================================

X_train = pd.get_dummies(
    X_train,
    columns=categorical_cols,
    drop_first=True,
    dtype=int
)

X_test = pd.get_dummies(
    X_test,
    columns=categorical_cols,
    drop_first=True,
    dtype=int
)

# Make sure Train and Test have identical columns
X_test = X_test.reindex(
    columns=X_train.columns,
    fill_value=0
)

print("\nShape after One-Hot Encoding:")
print("Train:", X_train.shape)
print("Test :", X_test.shape)

# ============================================================
# 5. SIGNED LOG TRANSFORMATION
# ============================================================

def signed_log(data, columns):

    data = data.copy()

    for col in columns:

        data[col] = (
            np.sign(data[col])
            * np.log1p(np.abs(data[col]))
        )

    return data


# Create copies for the log-transformed version
X_train_log = signed_log(
    X_train,
    log_cols
)

X_test_log = signed_log(
    X_test,
    log_cols
)

# ============================================================
# 6. ROBUST SCALING
# ============================================================


scale_cols = (
    [
        "LIMIT_BAL",
        "AGE",
        "N_DELAY_MONTHS",
        "AVG_DELAY",
        "MAX_DELAY",
        "PAYMENT_RATIO",
        "UTILIZATION",
        "BILL_TREND"
    ]
    +
    [f"BILL_AMT{i}" for i in range(1, 7)]
    +
    [f"PAY_AMT{i}" for i in range(1, 7)]
)
print("\nColumns that will be scaled:")
print(scale_cols)

scaler = RobustScaler()

# Fit ONLY on training data
scaler.fit(X_train_log[scale_cols])

# Transform training data
X_train_log[scale_cols] = scaler.transform(
    X_train_log[scale_cols]
)

# Transform test data using the TRAIN scaler
X_test_log[scale_cols] = scaler.transform(
    X_test_log[scale_cols]
)

# # ============================================================

# ============================================================
# LOGISTIC REGRESSION
# ============================================================

model_logistic = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)

model_logistic.fit(
    X_train_log,
    y_train
)




# ============================================================
# LASSO HYPERPARAMETER TUNING
# ============================================================

C_values = [0.01, 0.05, 0.1, 0.2, 0.5, 1, 2]

lasso_results = []

for C in C_values:

    model = LogisticRegression(
        C=C,
        class_weight="balanced",
        l1_ratio=1,
        solver="saga",
        max_iter=5000,
        random_state=42
    )

    model.fit(X_train_log, y_train)

    y_prob = model.predict_proba(X_test_log)[:, 1]

    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    non_zero = np.sum(model.coef_[0] != 0)

    lasso_results.append({
        "C": C,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
        "Non_Zero_Coefficients": non_zero
    })


lasso_results_df = pd.DataFrame(lasso_results)

print("\n==============================")
print("LASSO HYPERPARAMETER TUNING")
print("==============================")

print(lasso_results_df.to_string(index=False))

# ============================================================
# 9. PREDICTION
# ============================================================

# Logistic Regression predictions
y_pred_logistic = model_logistic.predict(
    X_test_log
)

y_prob_logistic = model_logistic.predict_proba(
    X_test_log
)[:, 1]


# LASSO predictions
y_pred_lasso = model_lasso.predict(
    X_test_log
)

y_prob_lasso = model_lasso.predict_proba(
    X_test_log
)[:, 1]

# ============================================================
# 
# ============================================================
print("\nFeature count before modeling:")
print("Number of features:", X.shape[1])
print("\nShape after One-Hot Encoding:")
print("Train:", X_train.shape)
print("Test :", X_test.shape)

# ============================================================
# 10. MODEL EVALUATION
# ============================================================

print("\n==============================")
print("LOGISTIC REGRESSION")
print("==============================")

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred_logistic
    )
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred_logistic
    )
)

print(
    "ROC-AUC:",
    round(
        roc_auc_score(
            y_test,
            y_prob_logistic
        ),
        4
    )
)


print("\n==============================")
print("LASSO LOGISTIC REGRESSION")
print("==============================")

print("\nSelected C:", selected_C)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred_lasso
    )
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred_lasso
    )
)

print(
    "ROC-AUC:",
    round(
        roc_auc_score(
            y_test,
            y_prob_lasso
        ),
        4
    )
)


# ============================================================
# 11. LASSO COEFFICIENTS
# ============================================================

lasso_coefficients = pd.DataFrame({
    "Feature": X_train_log.columns,
    "Coefficient": model_lasso.coef_[0]
})

lasso_coefficients["Absolute_Coefficient"] = (
    lasso_coefficients["Coefficient"].abs()
)

lasso_coefficients = lasso_coefficients.sort_values(
    "Absolute_Coefficient",
    ascending=False
)

print("\n==============================")
print("LASSO COEFFICIENTS")
print("==============================")

print(
    lasso_coefficients.to_string(
        index=False
    )
)


# ============================================================
# 12. REFERENCE CATEGORIES
# ============================================================

print("\n==============================")
print("REFERENCE CATEGORIES")
print("==============================")

print("SEX reference category       : SEX = 1")
print("EDUCATION reference category : EDUCATION = 0")
print("MARRIAGE reference category  : MARRIAGE = 0")

# ============================================================
# 14. ODDS RATIOS
# ============================================================

# Convert Logistic Regression coefficients to Odds Ratios
lasso_coefficients["Odds_Ratio"] = np.exp(
    lasso_coefficients["Coefficient"]
)

print("\n==============================")
print("LASSO COEFFICIENTS + ODDS RATIOS")
print("==============================")

print(
    lasso_coefficients
    .sort_values(
        "Absolute_Coefficient",
        ascending=False
    )
    .to_string(index=False)
)

# ============================================================
# 15. PRECISION-RECALL AUC (PR-AUC)
# ============================================================

# PR-AUC for Logistic Regression
pr_auc_logistic = average_precision_score(
    y_test,
    y_prob_logistic
)

# PR-AUC for LASSO
pr_auc_lasso = average_precision_score(
    y_test,
    y_prob_lasso
)

print("\n==============================")
print("PRECISION-RECALL AUC (PR-AUC)")
print("==============================")

print(
    "Logistic Regression PR-AUC:",
    round(pr_auc_logistic, 4)
)

print(
    "LASSO PR-AUC:",
    round(pr_auc_lasso, 4)
)

# Baseline PR-AUC
# For an imbalanced dataset, the baseline is the
# proportion of positive cases in the test set.

baseline_pr_auc = y_test.mean()

print(
    "Baseline PR-AUC:",
    round(baseline_pr_auc, 4)
)


# ============================================================
# 16. PRECISION-RECALL CURVE
# ============================================================

precision_logistic, recall_logistic, thresholds_logistic = (
    precision_recall_curve(
        y_test,
        y_prob_logistic
    )
)

precision_lasso, recall_lasso, thresholds_lasso = (
    precision_recall_curve(
        y_test,
        y_prob_lasso
    )
)


# ============================================================
# 17. PLOT PRECISION-RECALL CURVES
# ============================================================

plt.figure(figsize=(8, 6))

plt.plot(
    recall_logistic,
    precision_logistic,
    label=f"Logistic Regression (PR-AUC = {pr_auc_logistic:.3f})"
)

plt.plot(
    recall_lasso,
    precision_lasso,
    label=f"LASSO (PR-AUC = {pr_auc_lasso:.3f})"
)

plt.axhline(
    y=baseline_pr_auc,
    linestyle="--",
    label=f"Baseline = {baseline_pr_auc:.3f}"
)

plt.xlabel("Recall")
plt.ylabel("Precision")

plt.title(
    "Figure 6: Precision-Recall Curve"
)

plt.legend()

plt.grid()

plt.tight_layout()

plt.show()
# ============================================================
# 18. THRESHOLD ANALYSIS
# ============================================================

thresholds_to_check = np.arange(
    0.10,
    0.71,
    0.05
)

threshold_results = []

for threshold in thresholds_to_check:

    y_pred_threshold = (
        y_prob_lasso >= threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        y_pred_threshold,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred_threshold,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred_threshold,
        zero_division=0
    )

    threshold_results.append({
        "Threshold": threshold,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    })


threshold_results = pd.DataFrame(
    threshold_results
)


print("\n==============================")
print("THRESHOLD ANALYSIS - LASSO")
print("==============================")

print(
    threshold_results.to_string(
        index=False
    )
)
# ============================================================
# 19. PRECISION / RECALL / F1 VS THRESHOLD
# ============================================================

plt.figure(figsize=(9, 6))

plt.plot(
    threshold_results["Threshold"],
    threshold_results["Precision"],
    marker="o",
    label="Precision"
)

plt.plot(
    threshold_results["Threshold"],
    threshold_results["Recall"],
    marker="o",
    label="Recall"
)

plt.plot(
    threshold_results["Threshold"],
    threshold_results["F1"],
    marker="o",
    label="F1"
)

plt.xlabel("Threshold")
plt.ylabel("Score")

plt.title(
    "Figure 7: Precision, Recall and F1 vs Threshold"
)

plt.legend()

plt.grid()

plt.tight_layout()

plt.show()
# ============================================================
# 20. BEST THRESHOLD BASED ON F1
# ============================================================

best_threshold_row = threshold_results.loc[
    threshold_results["F1"].idxmax()
]

print("\n==============================")
print("BEST THRESHOLD BASED ON F1")
print("==============================")

print(
    best_threshold_row
)
