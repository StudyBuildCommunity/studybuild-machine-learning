# ============================================================
# 5-FOLD CROSS-VALIDATION
# Logistic Regression vs LASSO
# ============================================================

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression


# ============================================================
# 1. LOAD DATA
# ============================================================

file_path = "studybuuild_3/default of credit card clients_cleaned.xlsx"

df = pd.read_excel(file_path)

print("Original dataset shape:", df.shape)


# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================

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
# 3. DEFINE X AND y
# ============================================================

X = df.drop(
    columns=["ID", "default payment next month"]
)

y = df["default payment next month"]


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================
# Test is kept completely outside Cross-Validation.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

print("\nTrain shape:", X_train.shape)
print("Test shape :", X_test.shape)


# ============================================================
# 5. DEFINE COLUMNS
# ============================================================

categorical_cols = [
    "SEX",
    "EDUCATION",
    "MARRIAGE"
]

log_cols = [
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

scale_cols = [
    "LIMIT_BAL",
    "AGE",
    "N_DELAY_MONTHS",
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

# Remaining numerical columns
numeric_cols = [
    col for col in X_train.columns
    if col not in categorical_cols
    and col not in log_cols
]


# ============================================================
# 6. SIGNED LOG TRANSFORMATION
# ============================================================

def signed_log_transform(X):
    return np.sign(X) * np.log1p(np.abs(X))


# ============================================================
# 7. PREPROCESSING PIPELINE
# ============================================================
# IMPORTANT:
# Every preprocessing step is fitted separately inside
# each CV fold. This prevents data leakage.


# Log transformation + RobustScaler
log_pipeline = Pipeline([
    (
        "log",
        FunctionTransformer(
            signed_log_transform,
            feature_names_out="one-to-one"
        )
    ),
    (
        "scaler",
        RobustScaler()
    )
])


# Scaling without log transformation
scale_pipeline = Pipeline([
    (
        "scaler",
        RobustScaler()
    )
])


preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore"
            ),
            categorical_cols
        ),

        (
            "log_numeric",
            log_pipeline,
            log_cols
        ),

        (
            "scaled_numeric",
            scale_pipeline,
            [
                col for col in scale_cols
                if col not in log_cols
            ]
        ),

        (
            "numeric",
            "passthrough",
            numeric_cols
        )
    ],
    remainder="drop"
)


# ============================================================
# 8. DEFINE MODELS
# ============================================================

models = {

    "Logistic Regression": LogisticRegression(
        class_weight="balanced",
        max_iter=5000,
        random_state=42
    ),

    "LASSO C=0.1": LogisticRegression(
        C=0.1,
        class_weight="balanced",
        l1_ratio=1,
        solver="saga",
        max_iter=5000,
        random_state=42
    ),

    "LASSO C=1": LogisticRegression(
        C=1.0,
        class_weight="balanced",
        l1_ratio=1,
        solver="saga",
        max_iter=5000,
        random_state=42
    )
}


# ============================================================
# 9. 5-FOLD STRATIFIED CROSS-VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# 10. EVALUATION METRICS
# ============================================================

scoring = {
    "PR_AUC": "average_precision",
    "ROC_AUC": "roc_auc"
}


# ============================================================
# 11. RUN CROSS-VALIDATION
# ============================================================

results = []

fold_scores = {}


for model_name, model in models.items():

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)

    # Complete pipeline:
    # preprocessing → model
    pipeline = Pipeline([
        (
            "preprocessing",
            preprocessor
        ),
        (
            "model",
            model
        )
    ])

    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    # Save individual fold scores
    fold_scores[model_name] = scores

    pr_auc = scores["test_PR_AUC"]
    roc_auc = scores["test_ROC_AUC"]

    # Print fold results
    print("\nPR-AUC per fold:")
    print(np.round(pr_auc, 4))

    print("\nROC-AUC per fold:")
    print(np.round(roc_auc, 4))

    # Print mean and std
    print("\nMean PR-AUC :", round(pr_auc.mean(), 4))
    print("Std PR-AUC  :", round(pr_auc.std(), 4))

    print("\nMean ROC-AUC:", round(roc_auc.mean(), 4))
    print("Std ROC-AUC :", round(roc_auc.std(), 4))

    # Save summary
    results.append({
        "Model": model_name,

        "PR_AUC_Mean": pr_auc.mean(),
        "PR_AUC_Std": pr_auc.std(),

        "ROC_AUC_Mean": roc_auc.mean(),
        "ROC_AUC_Std": roc_auc.std()
    })


# ============================================================
# 12. FINAL RESULTS TABLE
# ============================================================

cv_results = pd.DataFrame(results)

print("\n\n" + "=" * 70)
print("FINAL CROSS-VALIDATION SUMMARY")
print("=" * 70)

print(
    cv_results.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 13. BEST MODEL BASED ON PR-AUC
# ============================================================

best_model = cv_results.loc[
    cv_results["PR_AUC_Mean"].idxmax()
]

print("\n" + "=" * 70)
print("BEST MODEL BASED ON MEAN PR-AUC")
print("=" * 70)

print("Model     :", best_model["Model"])
print("Mean PR-AUC:", round(best_model["PR_AUC_Mean"], 4))
print("Std PR-AUC :", round(best_model["PR_AUC_Std"], 4))