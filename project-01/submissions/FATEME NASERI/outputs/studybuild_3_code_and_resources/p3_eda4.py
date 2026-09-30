
#-----------------train model ,logestic regression--------
#-----------------virtual in terminal--------------------
#python3 -m venv .venv
#source .venv/bin/activate
#which python

#-----------------------------------------------------------


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


# ============================================================
# 1. READ DATA
# ============================================================

path = "studybuuild_3/default of credit card clients_cleaned.xlsx"

df = pd.read_excel(path)

print("Original shape:", df.shape)


# ============================================================
# 2. DEFINE TARGET
# ============================================================

target = "default payment next month"

X = df.drop(columns=[target])
y = df[target]

print("\nTarget distribution:")
print(y.value_counts())

print("\nTarget percentage:")
print(y.value_counts(normalize=True) * 100)


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
# 4. SIGNED LOG TRANSFORMATION
# ============================================================
#
# Formula:
#
# sign(x) * log1p(abs(x))
#
# Advantages:
# - works with positive values
# - works with zero
# - works with negative values
# - reduces the effect of very large values
#
# ============================================================


def signed_log_transform(data, columns):

    data = data.copy()

    for col in columns:

        data[f"{col}_log"] = (
            np.sign(data[col])
            * np.log1p(np.abs(data[col]))
        )

    return data


# ------------------------------------------------------------
# Columns selected for transformation
# ------------------------------------------------------------

log_cols = (
    ["LIMIT_BAL"] +
    [f"BILL_AMT{i}" for i in range(1, 7)] +
    [f"PAY_AMT{i}" for i in range(1, 7)]
)

print("\nColumns receiving Signed Log Transformation:")
print(log_cols)


# ------------------------------------------------------------
# Apply transformation separately to Train and Test
# ------------------------------------------------------------

X_train = signed_log_transform(
    X_train,
    log_cols
)

X_test = signed_log_transform(
    X_test,
    log_cols
)


# ============================================================
# 5. ONE-HOT ENCODING
# ============================================================
#
# Categorical variables:
#
# SEX
# EDUCATION
# MARRIAGE
#
# Example:
#
# EDUCATION = 1, 2, 3, 4
#
# becomes:
#
# EDUCATION_1
# EDUCATION_2
# EDUCATION_3
# EDUCATION_4
#
# ============================================================

categorical_cols = [
    "SEX",
    "EDUCATION",
    "MARRIAGE"
]

print("\nCategorical columns:")
print(categorical_cols)


# ------------------------------------------------------------
# One-Hot Encoding
# ------------------------------------------------------------

X_train = pd.get_dummies(
    X_train,
    columns=categorical_cols,
    prefix=categorical_cols,
    dtype=int
)

X_test = pd.get_dummies(
    X_test,
    columns=categorical_cols,
    prefix=categorical_cols,
    dtype=int
)


# ------------------------------------------------------------
# Make Train and Test columns identical
# ------------------------------------------------------------

X_test = X_test.reindex(
    columns=X_train.columns,
    fill_value=0
)


# ============================================================
# 6. CHECK TRANSFORMED DATA
# ============================================================

print("\nFinal Train shape:", X_train.shape)
print("Final Test shape :", X_test.shape)


# ------------------------------------------------------------
# Check One-Hot columns
# ------------------------------------------------------------

print("\nOne-Hot encoded columns:")

one_hot_cols = [
    col for col in X_train.columns
    if (
        col.startswith("SEX_")
        or col.startswith("EDUCATION_")
        or col.startswith("MARRIAGE_")
    )
]

print(one_hot_cols)


# ------------------------------------------------------------
# Check Signed Log columns
# ------------------------------------------------------------

print("\nSigned Log columns:")

log_columns_created = [
    f"{col}_log"
    for col in log_cols
]

print(log_columns_created)


# ============================================================
# 7. SHOW EXAMPLES
# ============================================================

print("\nExample of original + transformed values:")

example_cols = [
    "LIMIT_BAL",
    "LIMIT_BAL_log",
    "BILL_AMT1",
    "BILL_AMT1_log",
    "PAY_AMT1",
    "PAY_AMT1_log"
]

print(
    X_train[example_cols].head()
)


# ============================================================
# 8. CHECK NEGATIVE VALUES AFTER TRANSFORMATION
# ============================================================

print("\nMinimum values after Signed Log:")

print(
    X_train[log_columns_created].min()
)

# bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]

# for col in bill_cols:
#     df[f"{col}_log"] = (
#         np.sign(df[col]) *
#         np.log1p(np.abs(df[col]))
#     )