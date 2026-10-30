# ============================================================
# DECISION TREE + CROSS-VALIDATION
# Credit Card Default Prediction
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    brier_score_loss,
    roc_curve,
    precision_recall_curve
)
from sklearn.calibration import calibration_curve


# ============================================================
# 1. SETTINGS
# ============================================================

DATA_PATH = "studybuuild_3/default of credit card clients_cleaned.xlsx"
OUTPUT_DIR = "decision_tree_cv_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET = "default payment next month"

RANDOM_STATE = 42

# Business costs
FP_COST = 1
FN_COST = 5


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_excel(DATA_PATH)

print("=" * 70)
print("DATASET")
print("=" * 70)

print("Shape:", df.shape)
print("\nTarget distribution:")
print(df[TARGET].value_counts())
print("\nTarget percentage:")
print(df[TARGET].value_counts(normalize=True))


# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

pay_cols = [
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6"
]

# Number of months with payment delay
df["N_DELAY_MONTHS"] = (df[pay_cols] > 0).sum(axis=1)

print("\n" + "=" * 70)
print("FEATURE ENGINEERING")
print("=" * 70)

print("Added feature: N_DELAY_MONTHS")

print("\nN_DELAY_MONTHS distribution:")
print(df["N_DELAY_MONTHS"].value_counts().sort_index())


# ============================================================
# 4. X AND y
# ============================================================

X = df.drop(columns=[TARGET])
y = df[TARGET]


# ============================================================
# 5. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

# First: 80% temporary data, 20% test
X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE
)

# Then split temporary data into 75% train and 25% validation
# Final result = 60% train / 20% validation / 20% test

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
# 6. PREPROCESSING
# ============================================================

categorical_cols = [
    "SEX",
    "EDUCATION",
    "MARRIAGE"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore"
            ),
            categorical_cols
        )
    ],
    remainder="passthrough"
)


# ============================================================
# 7. DECISION TREE PIPELINE
# ============================================================

tree = DecisionTreeClassifier(
    class_weight="balanced",
    random_state=RANDOM_STATE
)

pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", tree)
])


# ============================================================
# 8. CROSS-VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("CROSS-VALIDATION")
print("=" * 70)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)


# Parameters to test
param_grid = {
    "model__max_depth": [3, 4, 5, 6, 7, 8, 10, None],
    "model__min_samples_leaf": [5, 10, 20, 30, 50],
    "model__min_samples_split": [2, 10, 20, 50]
}


grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="average_precision",
    cv=cv,
    n_jobs=-1,
    refit=True,
    verbose=1
)

grid_search.fit(X_train, y_train)


# ============================================================
# 9. BEST CV RESULT
# ============================================================

print("\n" + "=" * 70)
print("BEST CROSS-VALIDATION RESULT")
print("=" * 70)

print("Best Parameters:")
print(grid_search.best_params_)

print("\nBest CV PR-AUC:")
print(round(grid_search.best_score_, 4))


# Save CV results
cv_results = pd.DataFrame(grid_search.cv_results_)

cv_results = cv_results.sort_values(
    by="mean_test_score",
    ascending=False
)

cv_results.to_csv(
    os.path.join(OUTPUT_DIR, "decision_tree_cv_results.csv"),
    index=False
)


# ============================================================
# 10. BEST MODEL
# ============================================================

best_model = grid_search.best_estimator_


# ============================================================
# 11. VALIDATION PROBABILITIES
# ============================================================

y_val_proba = best_model.predict_proba(X_val)[:, 1]

val_roc_auc = roc_auc_score(
    y_val,
    y_val_proba
)

val_pr_auc = average_precision_score(
    y_val,
    y_val_proba
)

print("\n" + "=" * 70)
print("VALIDATION PROBABILITY PERFORMANCE")
print("=" * 70)

print(f"Validation ROC-AUC : {val_roc_auc:.4f}")
print(f"Validation PR-AUC  : {val_pr_auc:.4f}")


# ============================================================
# 12. THRESHOLD SELECTION
# ============================================================

thresholds = np.arange(
    0.10,
    0.71,
    0.01
)

threshold_results = []

for threshold in thresholds:

    y_val_pred = (
        y_val_proba >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val,
        y_val_pred
    ).ravel()

    total_cost = (
        fp * FP_COST +
        fn * FN_COST
    )

    precision = precision_score(
        y_val,
        y_val_pred,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        y_val_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        y_val_pred,
        zero_division=0
    )

    threshold_results.append({
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "total_cost": total_cost
    })


threshold_df = pd.DataFrame(
    threshold_results
)

threshold_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "threshold_analysis.csv"
    ),
    index=False
)


# Select threshold with minimum business cost
best_threshold_row = threshold_df.loc[
    threshold_df["total_cost"].idxmin()
]

best_threshold = best_threshold_row["threshold"]

print("\n" + "=" * 70)
print("THRESHOLD SELECTION")
print("=" * 70)

print(
    f"Selected Threshold: {best_threshold:.2f}"
)

print(
    f"Validation Cost: "
    f"{best_threshold_row['total_cost']:.0f}"
)


# ============================================================
# 13. FINAL VALIDATION CLASSIFICATION
# ============================================================

y_val_pred = (
    y_val_proba >= best_threshold
).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_val,
    y_val_pred
).ravel()

val_precision = precision_score(
    y_val,
    y_val_pred,
    zero_division=0
)

val_recall = recall_score(
    y_val,
    y_val_pred,
    zero_division=0
)

val_f1 = f1_score(
    y_val,
    y_val_pred,
    zero_division=0
)

val_cost = (
    fp * FP_COST +
    fn * FN_COST
)

print("\n" + "=" * 70)
print("FINAL VALIDATION RESULT")
print("=" * 70)

print(
    classification_report(
        y_val,
        y_val_pred,
        zero_division=0
    )
)

print("Confusion Matrix:")
print(
    confusion_matrix(
        y_val,
        y_val_pred
    )
)

print(f"\nThreshold : {best_threshold:.2f}")
print(f"Precision : {val_precision:.4f}")
print(f"Recall    : {val_recall:.4f}")
print(f"F1        : {val_f1:.4f}")
print(f"Total Cost: {val_cost}")


# ============================================================
# 14. FEATURE IMPORTANCE
# ============================================================

preprocessor_fitted = best_model.named_steps[
    "preprocessing"
]

tree_fitted = best_model.named_steps[
    "model"
]

feature_names = (
    preprocessor_fitted
    .get_feature_names_out()
)

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": tree_fitted.feature_importances_
})

feature_importance = (
    feature_importance
    .sort_values(
        by="Importance",
        ascending=False
    )
)

feature_importance.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "feature_importance.csv"
    ),
    index=False
)

print("\n" + "=" * 70)
print("TOP DECISION TREE FEATURES")
print("=" * 70)

print(
    feature_importance.head(20).to_string(
        index=False
    )
)


# ============================================================
# 15. FINAL MODEL: TRAIN + VALIDATION
# ============================================================

X_train_val = pd.concat(
    [X_train, X_val]
)

y_train_val = pd.concat(
    [y_train, y_val]
)

final_model = grid_search.best_estimator_

final_model.fit(
    X_train_val,
    y_train_val
)


# ============================================================
# 16. FINAL TEST EVALUATION
# ============================================================

y_test_proba = final_model.predict_proba(
    X_test
)[:, 1]

y_test_pred = (
    y_test_proba >= best_threshold
).astype(int)


# Probability metrics
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


# Classification metrics
test_precision = precision_score(
    y_test,
    y_test_pred,
    zero_division=0
)

test_recall = recall_score(
    y_test,
    y_test_pred,
    zero_division=0
)

test_f1 = f1_score(
    y_test,
    y_test_pred,
    zero_division=0
)

tn, fp, fn, tp = confusion_matrix(
    y_test,
    y_test_pred
).ravel()

test_cost = (
    fp * FP_COST +
    fn * FN_COST
)

baseline_pr_auc = y_test.mean()


# ============================================================
# 17. FINAL TEST RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL DECISION TREE RESULTS")
print("=" * 70)

print("\nModel:")
print("    Decision Tree")

print("\nFeature Engineering:")
print("    - N_DELAY_MONTHS")

print("\nPreprocessing:")
print("    - One-Hot Encoding")
print("    - No scaling required for Decision Tree")

print("\nClass Imbalance:")
print("    - class_weight='balanced'")

print("\nCross-Validation:")
print("    - 5-fold Stratified CV")
print("    - Scoring = PR-AUC")

print("\nThreshold Selection:")
print("    - Validation set")
print("    - Business cost based")
print("    - FP cost = 1")
print("    - FN cost = 5")

print("\nFinal Training:")
print("    - Train + Validation")

print("\nFinal Evaluation:")
print("    - Untouched Test set")

print(f"\nSelected Threshold : {best_threshold:.2f}")

print("\n--- Test Metrics ---")
print(f"Test ROC-AUC       : {test_roc_auc:.4f}")
print(f"Test PR-AUC        : {test_pr_auc:.4f}")
print(f"Baseline PR-AUC    : {baseline_pr_auc:.4f}")
print(f"Test Brier Score   : {test_brier:.4f}")

print("\n--- Test Classification ---")
print(f"Precision          : {test_precision:.4f}")
print(f"Recall             : {test_recall:.4f}")
print(f"F1                 : {test_f1:.4f}")

print("\n--- Confusion Matrix ---")
print(f"TN = {tn}")
print(f"FP = {fp}")
print(f"FN = {fn}")
print(f"TP = {tp}")

print(f"\nTest Total Cost    : {test_cost}")


# ============================================================
# 18. ROC CURVE
# ============================================================

fpr, tpr, _ = roc_curve(
    y_test,
    y_test_proba
)

plt.figure(figsize=(7, 5))
plt.plot(
    fpr,
    tpr,
    label=f"Decision Tree (AUC = {test_roc_auc:.3f})"
)
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Decision Tree - ROC Curve")
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "decision_tree_roc_curve.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 19. PRECISION-RECALL CURVE
# ============================================================

precision_curve, recall_curve, _ = precision_recall_curve(
    y_test,
    y_test_proba
)

plt.figure(figsize=(7, 5))

plt.plot(
    recall_curve,
    precision_curve,
    label=f"PR-AUC = {test_pr_auc:.3f}"
)

plt.axhline(
    baseline_pr_auc,
    linestyle="--",
    label=f"Baseline = {baseline_pr_auc:.3f}"
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Decision Tree - Precision-Recall Curve")
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "decision_tree_pr_curve.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 20. CALIBRATION CURVE
# ============================================================

prob_true, prob_pred = calibration_curve(
    y_test,
    y_test_proba,
    n_bins=10,
    strategy="quantile"
)

plt.figure(figsize=(7, 5))

plt.plot(
    prob_pred,
    prob_true,
    marker="o",
    label="Decision Tree"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect Calibration"
)

plt.xlabel("Mean Predicted Probability")
plt.ylabel("Observed Default Rate")
plt.title("Decision Tree - Calibration Curve")
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "decision_tree_calibration.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 21. SAVE TEST PREDICTIONS
# ============================================================

test_predictions = X_test.copy()

test_predictions["Actual"] = y_test.values
test_predictions["Predicted_Probability"] = y_test_proba
test_predictions["Predicted_Class"] = y_test_pred

test_predictions.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "test_predictions.csv"
    ),
    index=False
)


# ============================================================
# 22. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(f"Best CV PR-AUC : {grid_search.best_score_:.4f}")
print(f"ROC-AUC        : {test_roc_auc:.4f}")
print(f"PR-AUC         : {test_pr_auc:.4f}")
print(f"Baseline       : {baseline_pr_auc:.4f}")
print(f"Brier          : {test_brier:.4f}")
print(f"Threshold      : {best_threshold:.2f}")
print(f"Precision      : {test_precision:.4f}")
print(f"Recall         : {test_recall:.4f}")
print(f"F1             : {test_f1:.4f}")
print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
print(f"Total Cost     : {test_cost}")

print("\nOutput files saved in:")
print(OUTPUT_DIR)

print("\nDone.")