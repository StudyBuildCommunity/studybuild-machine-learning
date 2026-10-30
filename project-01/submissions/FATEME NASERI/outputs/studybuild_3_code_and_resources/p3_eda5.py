#-----------------compare models ,logistic regression--------#
#-----------------virtual in terminal------------------------#
# python3 -m venv .venv
# source .venv/bin/activate
# which python
#-------------------------------------------------------------

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score
)



# ============================================================
# 0. READ DATA
# ============================================================

path = "studybuuild_3/default of credit card clients_cleaned.xlsx"

df = pd.read_excel(path)

print("Original shape:", df.shape)


# ============================================================
# 1. DEFINE TARGET
# ============================================================

target = "default payment next month"

X = df.drop(columns=[target])

# ID is only an identifier and has no meaningful predictive meaning
X = X.drop(columns=["ID"])

y = df[target]

print("\nTarget distribution:")
print(y.value_counts())

print("\nTarget percentage:")
print(y.value_counts(normalize=True) * 100)


# ============================================================
# 1. DEFINE FEATURE GROUPS
# ============================================================

# Continuous numerical variables

continuous_cols = (
    ["LIMIT_BAL", "AGE"] +
    [f"BILL_AMT{i}" for i in range(1, 7)] +
    [f"PAY_AMT{i}" for i in range(1, 7)]
)


# Log-transformed variables

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
# 2. TRAIN / TEST SPLIT
# ============================================================

X = df.drop(columns=[target])
X = X.drop(columns=["ID"])

y = df[target]

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain:", X_train_raw.shape)
print("Test :", X_test_raw.shape)


# ============================================================
# 3. ONE-HOT ENCODING
# ============================================================

# ------------------------------------------------------------
# CHANGED:
#
# We use drop_first=True.
#
# This means one category from each categorical variable
# becomes the REFERENCE CATEGORY.
#
# Example:
#
# SEX:
#     SEX_1 -> Reference
#     SEX_2 -> Model feature
#
# EDUCATION:
#     EDUCATION_1 -> Reference
#     EDUCATION_2
#     EDUCATION_3
#     EDUCATION_4
#
# MARRIAGE:
#     MARRIAGE_1 -> Reference
#     MARRIAGE_2
#     MARRIAGE_3
#
# ------------------------------------------------------------

X_train_raw = pd.get_dummies(
    X_train_raw,
    columns=categorical_cols,
    drop_first=True,       # NEW
    dtype=int
)

X_test_raw = pd.get_dummies(
    X_test_raw,
    columns=categorical_cols,
    drop_first=True,       # NEW
    dtype=int
)


# ------------------------------------------------------------
# Make sure Train and Test have exactly the same columns
# ------------------------------------------------------------

X_test_raw = X_test_raw.reindex(
    columns=X_train_raw.columns,
    fill_value=0
)


print("\nShape after One-Hot Encoding:")
print("Train:", X_train_raw.shape)
print("Test :", X_test_raw.shape)


# ------------------------------------------------------------
# NEW: Check the categorical columns
# ------------------------------------------------------------

print("\nOne-Hot encoded columns:")

one_hot_cols = [
    col for col in X_train_raw.columns
    if (
        col.startswith("SEX_")
        or col.startswith("EDUCATION_")
        or col.startswith("MARRIAGE_")
    )
]

print(one_hot_cols)


# ============================================================
# 4. CREATE LOG-TRANSFORMED VERSION
# ============================================================

X_train_log = X_train_raw.copy()
X_test_log = X_test_raw.copy()


def signed_log(data, columns):

    data = data.copy()

    for col in columns:

        data[col] = (
            np.sign(data[col])
            * np.log1p(np.abs(data[col]))
        )

    return data


# ------------------------------------------------------------
# Apply Signed Log to:
#
# LIMIT_BAL
# BILL_AMT1 - BILL_AMT6
# PAY_AMT1 - PAY_AMT6
# ------------------------------------------------------------

X_train_log = signed_log(
    X_train_log,
    log_cols
)

X_test_log = signed_log(
    X_test_log,
    log_cols
)


# ============================================================
# 5. ROBUST SCALING
# ============================================================

# Columns that we want to scale
#
# We do NOT scale One-Hot categorical columns.
# We are also NOT scaling PAY_0 - PAY_6 yet.

scale_cols = (
    ["LIMIT_BAL", "AGE"] +
    [f"BILL_AMT{i}" for i in range(1, 7)] +
    [f"PAY_AMT{i}" for i in range(1, 7)]
)

print("\nColumns that will be scaled:")
print(scale_cols)


scaler_raw = RobustScaler()
scaler_log = RobustScaler()


# ------------------------------------------------------------
# Fit scaler ONLY on TRAIN data
# ------------------------------------------------------------

scaler_raw.fit(
    X_train_raw[scale_cols]
)

scaler_log.fit(
    X_train_log[scale_cols]
)


# ------------------------------------------------------------
# Transform TRAIN and TEST
# ------------------------------------------------------------

X_train_raw[scale_cols] = scaler_raw.transform(
    X_train_raw[scale_cols]
)

X_test_raw[scale_cols] = scaler_raw.transform(
    X_test_raw[scale_cols]
)


X_train_log[scale_cols] = scaler_log.transform(
    X_train_log[scale_cols]
)

X_test_log[scale_cols] = scaler_log.transform(
    X_test_log[scale_cols]
)


# ============================================================
# 6. LOGISTIC REGRESSION
# ============================================================




model_raw = LogisticRegression(
    class_weight="balanced",

    # Current default L2 regularization
    # Later we will tune C using Cross-Validation

    #penalty="l2",

    max_iter=1000,

    random_state=42
)


model_log = LogisticRegression(
    class_weight="balanced",

    #penalty="l2",

    max_iter=1000,

    random_state=42
)


# ============================================================
# 7. TRAIN MODELS
# ============================================================

model_raw.fit(
    X_train_raw,
    y_train
)

model_log.fit(
    X_train_log,
    y_train
)


# ============================================================
# 8. PREDICTION
# ============================================================

y_pred_raw = model_raw.predict(
    X_test_raw
)

y_pred_log = model_log.predict(
    X_test_log
)


# Probability of default = 1

y_prob_raw = model_raw.predict_proba(
    X_test_raw
)[:, 1]

y_prob_log = model_log.predict_proba(
    X_test_log
)[:, 1]


# ============================================================
# 9. MODEL EVALUATION
# ============================================================




print("\n==============================")
print("RAW FEATURES")
print("==============================")


print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred_raw
    )
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred_raw
    )
)


print(
    "ROC-AUC:",
    round(
        roc_auc_score(
            y_test,
            y_prob_raw
        ),
        4
    )
)


print("\n==============================")
print("LOG-TRANSFORMED FEATURES")
print("==============================")


print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred_log
    )
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred_log
    )
)


print(
    "ROC-AUC:",
    round(
        roc_auc_score(
            y_test,
            y_prob_log
        ),
        4
    )
)


# ============================================================
# 10. LASSO LOGISTIC REGRESSION
# ============================================================




model_lasso = LogisticRegression(
    class_weight="balanced",
    l1_ratio=1,          # جایگزین penalty="l1" -> یعنی L1 خالص (Lasso)
    C=0.1,               # هرچی کوچیک‌تر، فشار Regularization بیشتر
    solver="saga",       # فقط saga رسماً از l1_ratio/Elastic-Net پشتیبانی می‌کنه
    max_iter=2000,       # saga کندتر از liblinear همگرا می‌شه، iteration بیشتری لازمه
    random_state=42
)


model_lasso.fit(
    X_train_log,
    y_train
)


# ============================================================
# 11. LASSO PREDICTION
# ============================================================

y_pred_lasso = model_lasso.predict(
    X_test_log
)

y_prob_lasso = model_lasso.predict_proba(
    X_test_log
)[:, 1]


# ============================================================
# 12. LASSO EVALUATION
# ============================================================

print("\n==============================")
print("LASSO (L1)")
print("==============================")


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
# 13. LASSO COEFFICIENTS
# ============================================================

lasso_coefficients = pd.DataFrame({

    "Feature": X_train_log.columns,

    "Coefficient": model_lasso.coef_[0]

})


lasso_coefficients["Absolute_Coefficient"] = (
    lasso_coefficients["Coefficient"].abs()
)


print("\n==============================")
print("LASSO COEFFICIENTS")
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
# 14. CROSS VALIDATION FOR LASSO
# ============================================================


# ------------------------------------------------------------
# C values that we want to test
# ------------------------------------------------------------

C_values = [0.001, 0.01, 0.1, 1, 10, 100]

print("\nC values that will be tested:")
print(C_values)


# ------------------------------------------------------------
# Create 5-fold Cross Validation
# ------------------------------------------------------------

cv = StratifiedKFold(

    n_splits=5,

    shuffle=True,

    random_state=42

)


# ------------------------------------------------------------
# Test different C values
# ------------------------------------------------------------

cv_results = []

for C in C_values:

    model = LogisticRegression(

        C=C,

        class_weight="balanced",

        penalty="l1",

        solver="liblinear",

        max_iter=1000,

        random_state=42

    )

    scores = cross_val_score(

        model,

        X_train_log,

        y_train,

        cv=cv,

        scoring="roc_auc"

    )

    cv_results.append({

        "C": C,

        "Mean_ROC_AUC": scores.mean(),

        "Std_ROC_AUC": scores.std()

    })


# ------------------------------------------------------------
# Show Cross Validation results
# ------------------------------------------------------------

cv_results_df = pd.DataFrame(cv_results)

print("\n==============================")
print("LASSO CROSS VALIDATION")
print("==============================")

print(

    cv_results_df

    .sort_values(

        "Mean_ROC_AUC",

        ascending=False

    )

    .to_string(index=False)

)


# ============================================================
# 15. SELECT BEST C
# ============================================================

best_C = cv_results_df.loc[

    cv_results_df["Mean_ROC_AUC"].idxmax(),

    "C"

]

print("\nBest C:", best_C)


# ============================================================
# 16. TRAIN FINAL LASSO WITH BEST C
# ============================================================

model_lasso_cv = LogisticRegression(

    C=best_C,

    class_weight="balanced",

    penalty="l1",

    solver="liblinear",

    max_iter=1000,

    random_state=42

)

model_lasso_cv.fit(

    X_train_log,

    y_train

)


# ============================================================
# 17. FINAL LASSO PREDICTION
# ============================================================

y_pred_lasso_cv = model_lasso_cv.predict(

    X_test_log

)

y_prob_lasso_cv = model_lasso_cv.predict_proba(

    X_test_log

)[:, 1]


# ============================================================
# 18. FINAL LASSO EVALUATION
# ============================================================

print("\n==============================")
print("LASSO WITH BEST C")
print("==============================")

print("\nConfusion Matrix:")

print(

    confusion_matrix(

        y_test,

        y_pred_lasso_cv

    )

)

print("\nClassification Report:")

print(

    classification_report(

        y_test,

        y_pred_lasso_cv

    )

)

print(

    "ROC-AUC:",

    round(

        roc_auc_score(

            y_test,

            y_prob_lasso_cv

        ),

        4

    )

)


# ============================================================
# 19. LASSO COEFFICIENTS WITH BEST C
# ============================================================

lasso_cv_coefficients = pd.DataFrame({

    "Feature": X_train_log.columns,

    "Coefficient": model_lasso_cv.coef_[0]

})

lasso_cv_coefficients["Absolute_Coefficient"] = (

    lasso_cv_coefficients["Coefficient"].abs()

)

print("\n==============================")
print("LASSO COEFFICIENTS WITH BEST C")
print("==============================")

print(

    lasso_cv_coefficients

    .sort_values(

        "Absolute_Coefficient",

        ascending=False

    )

    .to_string(index=False)

)