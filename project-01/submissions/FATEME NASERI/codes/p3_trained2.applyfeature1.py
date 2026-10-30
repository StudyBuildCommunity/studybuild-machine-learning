# ============================================================
# CREDIT CARD DEFAULT PREDICTION
# Logistic Regression + LASSO
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV
)

from sklearn.preprocessing import RobustScaler

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve
)


# ============================================================
# 0. READ DATA
# ============================================================

path = "studybuuild_3/default of credit card clients_cleaned.xlsx"

df = pd.read_excel(path)

print("Original shape:", df.shape)


# ============================================================
# 1. FEATURE ENGINEERING
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


# Average payment delay
df["AVG_DELAY"] = (
    df[pay_cols].mean(axis=1)
)


# Maximum payment delay
df["MAX_DELAY"] = (
    df[pay_cols].max(axis=1)
)


# Total payment / total bill
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


# Change in bill amount
df["BILL_TREND"] = (
    df["BILL_AMT1"]
    -
    df["BILL_AMT6"]
)


engineered_cols = [
    "N_DELAY_MONTHS",
    "AVG_DELAY",
    "MAX_DELAY",
    "PAYMENT_RATIO",
    "UTILIZATION",
    "BILL_TREND"
]

print("\nEngineered features:")
print(engineered_cols)


# ============================================================
# 2. DEFINE X AND y
# ============================================================

target = "default payment next month"

X = df.drop(
    columns=[
        target,
        "ID"
    ]
)

y = df[target]


print("\nTarget distribution:")
print(y.value_counts())


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

X_test = pd.get_dummies(
    X_test,
    columns=categorical_cols,
    drop_first=True,
    dtype=int
)


# Ensure identical columns
X_test = X_test.reindex(
    columns=X_train.columns,
    fill_value=0
)


print("\nAfter One-Hot Encoding:")
print("Train:", X_train.shape)
print("Test :", X_test.shape)


# ============================================================
# 5. SIGNED LOG TRANSFORMATION
# ============================================================

log_cols = (
    ["LIMIT_BAL"]
    +
    [f"BILL_AMT{i}" for i in range(1, 7)]
    +
    [f"PAY_AMT{i}" for i in range(1, 7)]
)


def signed_log(data, columns):

    data = data.copy()

    for col in columns:

        data[col] = (
            np.sign(data[col])
            *
            np.log1p(np.abs(data[col]))
        )

    return data


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


scaler = RobustScaler()

scaler.fit(
    X_train_log[scale_cols]
)

X_train_log[scale_cols] = scaler.transform(
    X_train_log[scale_cols]
)

X_test_log[scale_cols] = scaler.transform(
    X_test_log[scale_cols]
)


print("\nPreprocessing completed.")


# ============================================================
# 7. LOGISTIC REGRESSION
# ============================================================

model_logistic = LogisticRegression(
    class_weight="balanced",
    l1_ratio=0,
    solver="lbfgs",
    max_iter=20000,
    random_state=42
)

model_logistic.fit(
    X_train_log,
    y_train
)


# Probability
y_prob_logistic = model_logistic.predict_proba(
    X_test_log
)[:, 1]


# ============================================================
# 8. LASSO - SELECT BEST C USING TRAINING DATA
# ============================================================

C_values = [
    0.01,
    0.1,
    1,
    10
]

lasso = LogisticRegression(
    class_weight="balanced",
    l1_ratio=1,
    solver="liblinear",
    max_iter=20000,
    random_state=42
)

cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

grid = GridSearchCV(
    estimator=lasso,
    param_grid={"C": C_values},
    scoring="average_precision",
    cv=cv,
    n_jobs=1,          # مهم: فعلاً parallel را خاموش می‌کنیم
    refit=True,
    error_score="raise"
)

print("\nStarting LASSO GridSearch...")

grid.fit(
    X_train_log,
    y_train
)

print("LASSO GridSearch completed.")

selected_C = grid.best_params_["C"]
model_lasso = grid.best_estimator_

print("\nSelected LASSO C:", selected_C)
print(
    "Best CV PR-AUC:",
    round(grid.best_score_, 4)
)

# Sanity check: confirm this is actually behaving like L1/LASSO

n_zeroed = int((model_lasso.coef_[0] == 0).sum())
print(f"Features zeroed by LASSO: {n_zeroed} / {len(model_lasso.coef_[0])}")
if n_zeroed == 0:
    print("*** WARNING: no coefficients were zeroed. On some scikit-learn")
    print("    versions, l1_ratio is ignored unless penalty='elasticnet' is")
    print("    also set, which would silently make this Ridge (L2) instead")
    print("    of LASSO (L1). If you see this warning, add penalty='l1'")
    print("    explicitly (and drop l1_ratio) and re-run.")


# ============================================================
# 9. TEST SET PREDICTIONS
# ============================================================

y_prob_lasso = model_lasso.predict_proba(
    X_test_log
)[:, 1]


# ============================================================
# 10. ROC-AUC
# ============================================================

roc_auc_logistic = roc_auc_score(
    y_test,
    y_prob_logistic
)

roc_auc_lasso = roc_auc_score(
    y_test,
    y_prob_lasso
)


# ============================================================
# 11. PR-AUC
# ============================================================

pr_auc_logistic = average_precision_score(
    y_test,
    y_prob_logistic
)

pr_auc_lasso = average_precision_score(
    y_test,
    y_prob_lasso
)

baseline_pr_auc = y_test.mean()


print("\n==============================")
print("FINAL MODEL PERFORMANCE")
print("==============================")

print("\nLogistic Regression")
print("ROC-AUC:", round(roc_auc_logistic, 4))
print("PR-AUC :", round(pr_auc_logistic, 4))

print("\nLASSO")
print("Selected C:", selected_C)
print("ROC-AUC:", round(roc_auc_lasso, 4))
print("PR-AUC :", round(pr_auc_lasso, 4))

print("\nPR-AUC Baseline:",
      round(baseline_pr_auc, 4))

# ==============================
# MODEL COEFFICIENTS
# ==============================

coefficients = pd.DataFrame({
    "Feature": X_train_log.columns,
    "Logistic_Coefficient": model_logistic.coef_[0],
    "LASSO_Coefficient": model_lasso.coef_[0]
})

coefficients["Abs_LASSO_Coefficient"] = (
    coefficients["LASSO_Coefficient"].abs()
)

coefficients = coefficients.sort_values(
    "Abs_LASSO_Coefficient",
    ascending=False
)

print("\n==============================")
print("MODEL COEFFICIENTS")
print("==============================")
print(
    coefficients[
        [
            "Feature",
            "Logistic_Coefficient",
            "LASSO_Coefficient"
        ]
    ].to_string(index=False)
)

# ============================================================
# 12. COMMON PRECISION-RECALL CURVE
# ============================================================

precision_logistic, recall_logistic, _ = (
    precision_recall_curve(
        y_test,
        y_prob_logistic
    )
)

precision_lasso, recall_lasso, _ = (
    precision_recall_curve(
        y_test,
        y_prob_lasso
    )
)


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
    baseline_pr_auc,
    linestyle="--",
    label=f"Baseline = {baseline_pr_auc:.3f}"
)

plt.xlabel("Recall")
plt.ylabel("Precision")

plt.title(
    "Precision-Recall Curve"
)

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

plt.show()