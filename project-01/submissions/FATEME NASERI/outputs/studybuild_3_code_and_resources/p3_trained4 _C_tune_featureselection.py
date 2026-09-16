# ============================================================
# CREDIT CARD DEFAULT PREDICTION
# Logistic Regression + LASSO+ c tunning + threshold
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
# 1.FEATURE ENGINEERING
# ============================================================

pay_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]

df["N_DELAY_MONTHS"] = (
    df[pay_cols] > 0
).sum(axis=1)

print("\nFeature count before modeling:")
print("Number of features:", df.shape[1] - 1)
# ============================================================
# 1.5. DEFINE TARGET AND FEATURES
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
    ["LIMIT_BAL", "AGE", "N_DELAY_MONTHS"] +
    [f"BILL_AMT{i}" for i in range(1, 7)] +
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
# 3.TRAIN / VALIDATION / TEST SPLIT
# ============================================================


X = df.drop(
    columns=["ID", "default payment next month"]
)

y = df["default payment next month"]


# First split:
# 80% → temporary train
# 20% → final test

X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)


# Second split:
# From the remaining 80%:
# 75% → train = 60% of total
# 25% → validation = 20% of total

X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.25,
    stratify=y_temp,
    random_state=42
)


print("\n==============================")
print("DATA SPLIT")
print("==============================")

print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

print("\nTarget distribution:")

print(
    "\nTrain:"
)
print(
    y_train.value_counts(normalize=True)
)

print(
    "\nValidation:"
)
print(
    y_val.value_counts(normalize=True)
)

print(
    "\nTest:"
)
print(
    y_test.value_counts(normalize=True)
)

# ============================================================
# 4. ONE-HOT ENCODING
# ============================================================

categorical_cols = [
    "SEX",
    "EDUCATION",
    "MARRIAGE"
]

X_train = pd.get_dummies(
    X_train,
    columns=categorical_cols,
    drop_first=True,
    dtype=int
)

X_val = pd.get_dummies(
    X_val,
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


# Make sure validation and test have exactly
# the same columns as training

X_val = X_val.reindex(
    columns=X_train.columns,
    fill_value=0
)

X_test = X_test.reindex(
    columns=X_train.columns,
    fill_value=0
)


print("\n==============================")
print("AFTER ONE-HOT ENCODING")
print("==============================")

print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)


# ============================================================
# 5. SIGNED LOG TRANSFORMATION
# ============================================================

log_cols = (
    ["LIMIT_BAL"] +
    [f"BILL_AMT{i}" for i in range(1, 7)] +
    [f"PAY_AMT{i}" for i in range(1, 7)]
)


def signed_log_transform(
    df,
    columns
):

    df = df.copy()

    for col in columns:

        df[col] = (
            np.sign(df[col]) *
            np.log1p(np.abs(df[col]))
        )

    return df


X_train_log = signed_log_transform(
    X_train,
    log_cols
)

X_val_log = signed_log_transform(
    X_val,
    log_cols
)

X_test_log = signed_log_transform(
    X_test,
    log_cols
)

# ============================================================
# 6. ROBUST SCALING
# ============================================================


scale_cols = (
    ["LIMIT_BAL", "AGE", "N_DELAY_MONTHS"] +
    [f"BILL_AMT{i}" for i in range(1, 7)] +
    [f"PAY_AMT{i}" for i in range(1, 7)]
)


scaler = RobustScaler()

# FIT ONLY ON TRAIN
X_train_log[scale_cols] = scaler.fit_transform(
    X_train_log[scale_cols]
)

# TRANSFORM VALIDATION
X_val_log[scale_cols] = scaler.transform(
    X_val_log[scale_cols]
)

# TRANSFORM TEST
X_test_log[scale_cols] = scaler.transform(
    X_test_log[scale_cols]
)


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

# Probability prediction
y_prob_logistic = model_logistic.predict_proba(
    X_test_log
)[:, 1]

# Prediction at default threshold = 0.50
y_pred_logistic = (
    y_prob_logistic >= 0.50
).astype(int)

# ROC-AUC
roc_auc_logistic = roc_auc_score(
    y_test,
    y_prob_logistic
)

# PR-AUC
pr_auc_logistic = average_precision_score(
    y_test,
    y_prob_logistic
)

print("\n==============================")
print("LOGISTIC REGRESSION")
print("==============================")
print(f"ROC-AUC: {roc_auc_logistic:.4f}")
print(f"PR-AUC:  {pr_auc_logistic:.4f}")


# ============================================================
# LASSO - COMPARE C = 0.1 AND C = 1
# ============================================================

C_values = [0.1, 1.0]

lasso_models = {}
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

    model.fit(
        X_train_log,
        y_train
    )

    y_prob = model.predict_proba(
        X_test_log
    )[:, 1]

    y_pred = (
        y_prob >= 0.50
    ).astype(int)

    roc_auc = roc_auc_score(
        y_test,
        y_prob
    )

    pr_auc = average_precision_score(
        y_test,
        y_prob
    )

    non_zero = np.sum(
        model.coef_[0] != 0
    )

    lasso_models[C] = model

    lasso_results.append({
        "C": C,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
        "Non_Zero_Coefficients": non_zero
    })


lasso_results_df = pd.DataFrame(
    lasso_results
)

print("\n==============================")
print("LASSO: C COMPARISON")
print("==============================")

print(
    lasso_results_df.to_string(
        index=False
    )
)


# ============================================================
# PR-AUC COMPARISON
# ============================================================

print("\n==============================")
print("PRECISION-RECALL AUC")
print("==============================")

print(
    f"Logistic Regression PR-AUC: "
    f"{pr_auc_logistic:.4f}"
)

for C in C_values:

    y_prob = lasso_models[C].predict_proba(
        X_test_log
    )[:, 1]

    pr_auc = average_precision_score(
        y_test,
        y_prob
    )

    print(
        f"LASSO C={C} PR-AUC: "
        f"{pr_auc:.4f}"
    )

print(
    f"Baseline PR-AUC: "
    f"{y_test.mean():.4f}"
)


# ============================================================
# PRECISION-RECALL CURVES
# ============================================================

plt.figure(figsize=(8, 6))

# Logistic
precision, recall, _ = precision_recall_curve(
    y_test,
    y_prob_logistic
)

plt.plot(
    recall,
    precision,
    label=f"Logistic (PR-AUC={pr_auc_logistic:.4f})"
)


# LASSO
for C in C_values:

    y_prob = lasso_models[C].predict_proba(
        X_test_log
    )[:, 1]

    precision, recall, _ = precision_recall_curve(
        y_test,
        y_prob
    )

    pr_auc = average_precision_score(
        y_test,
        y_prob
    )

    plt.plot(
        recall,
        precision,
        label=f"LASSO C={C} (PR-AUC={pr_auc:.4f})"
    )


plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves")
plt.legend()
plt.grid()
plt.show()


# ============================================================
# THRESHOLD ANALYSIS FUNCTION
# ============================================================

thresholds = np.arange(
    0.10,
    0.71,
    0.05
)


def threshold_analysis(
    y_true,
    y_prob
):

    results = []

    for threshold in thresholds:

        y_pred = (
            y_prob >= threshold
        ).astype(int)

        precision = precision_score(
            y_true,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_true,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0
        )

        results.append({
            "Threshold": threshold,
            "Precision": precision,
            "Recall": recall,
            "F1": f1
        })

    return pd.DataFrame(results)


# ============================================================
# THRESHOLD ANALYSIS - ALL THREE MODELS
# ============================================================

threshold_logistic = threshold_analysis(
    y_test,
    y_prob_logistic
)

threshold_lasso_01 = threshold_analysis(
    y_test,
    lasso_models[0.1].predict_proba(
        X_test_log
    )[:, 1]
)

threshold_lasso_1 = threshold_analysis(
    y_test,
    lasso_models[1.0].predict_proba(
        X_test_log
    )[:, 1]
)


# ============================================================
# LOGISTIC REGRESSION - THRESHOLD CURVE
# ============================================================

plt.figure(figsize=(8, 6))

plt.plot(
    threshold_logistic["Threshold"],
    threshold_logistic["Precision"],
    marker="o",
    label="Precision"
)

plt.plot(
    threshold_logistic["Threshold"],
    threshold_logistic["Recall"],
    marker="o",
    label="Recall"
)

plt.plot(
    threshold_logistic["Threshold"],
    threshold_logistic["F1"],
    marker="o",
    label="F1"
)

plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Logistic Regression - Threshold Analysis")
plt.legend()
plt.grid()
plt.show()

#_________________\print("\nLOGISTIC")
print(threshold_logistic.to_string(index=False))

print("\nLASSO C=0.1")
print(threshold_lasso_01.to_string(index=False))

print("\nLASSO C=1.0")
print(threshold_lasso_1.to_string(index=False))
# ============================================================
# CONFUSION MATRIX - ALL THREE MODELS
# ============================================================

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# ------------------------------------------------------------
# 1. Logistic Regression
# ------------------------------------------------------------

y_pred_logistic = (
    y_prob_logistic >= 0.50
).astype(int)

cm_logistic = confusion_matrix(
    y_test,
    y_pred_logistic
)

print("\n==============================")
print("LOGISTIC REGRESSION")
print("==============================")
print(cm_logistic)


# ------------------------------------------------------------
# 2. LASSO C = 0.1
# ------------------------------------------------------------

y_prob_lasso_01 = lasso_models[0.1].predict_proba(
    X_test_log
)[:, 1]

y_pred_lasso_01 = (
    y_prob_lasso_01 >= 0.50
).astype(int)

cm_lasso_01 = confusion_matrix(
    y_test,
    y_pred_lasso_01
)

print("\n==============================")
print("LASSO C = 0.1")
print("==============================")
print(cm_lasso_01)


# ------------------------------------------------------------
# 3. LASSO C = 1.0
# ------------------------------------------------------------

y_prob_lasso_1 = lasso_models[1.0].predict_proba(
    X_test_log
)[:, 1]

y_pred_lasso_1 = (
    y_prob_lasso_1 >= 0.50
).astype(int)

cm_lasso_1 = confusion_matrix(
    y_test,
    y_pred_lasso_1
)

print("\n==============================")
print("LASSO C = 1.0")
print("==============================")
print(cm_lasso_1)


# ============================================================
# VISUALIZE ALL THREE CONFUSION MATRICES
# ============================================================

fig, axes = plt.subplots(
    1, 3,
    figsize=(15, 4)
)

ConfusionMatrixDisplay(
    confusion_matrix=cm_logistic,
    display_labels=["No Default", "Default"]
).plot(
    ax=axes[0],
    values_format="d"
)

axes[0].set_title("Logistic Regression")


ConfusionMatrixDisplay(
    confusion_matrix=cm_lasso_01,
    display_labels=["No Default", "Default"]
).plot(
    ax=axes[1],
    values_format="d"
)

axes[1].set_title("LASSO C=0.1")


ConfusionMatrixDisplay(
    confusion_matrix=cm_lasso_1,
    display_labels=["No Default", "Default"]
).plot(
    ax=axes[2],
    values_format="d"
)

axes[2].set_title("LASSO C=1.0")


plt.tight_layout()
plt.show()