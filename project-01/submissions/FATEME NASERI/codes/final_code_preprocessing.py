
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import StratifiedKFold, cross_val_score




#-------------------------------------step1------------------------------------------------------------
#-------------------------clean data------------------------

# choose header=1 , in order to find table headers on the second row not a first row
path = "studybuuild_3/default of credit card clients_raw.xls"
df = pd.read_excel(path,header=1)
print(df.columns.tolist())
print(df.shape)
#----------------
categorical_cols = ["SEX", "EDUCATION", "MARRIAGE"]
df["EDUCATION"] = df["EDUCATION"].replace({
    0: 4,
    5: 4,
    6: 4
})

df["MARRIAGE"] = df["MARRIAGE"].replace({
    0: 3
})
pay_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
bill_cols = [
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
    "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"
]

pay_amt_cols = [
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3",
    "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
]
   
#------------------------------simple graphs--------------------------------
#number of defaults with 0 and 1
#describing "default"
# print(df["default payment next month"].value_counts())
# print(df["default payment next month"].value_counts(normalize=True) * 100)
#--------------------------------------------------------------  
#target distribution plot
#bar chart for number of default situation of customers
# default_counts = df["default payment next month"].value_counts().sort_index() 
# ax = default_counts.plot(kind="bar")
# ax.bar_label(ax.containers[0])

# plt.xlabel("Default Payment Next Month")
# plt.ylabel("Number of Customers")
# plt.title("Distribution of Default Payment")
# plt.show()
#--------------------------------------------------------------  
#distribution of gender plot
plt.figure(1)
sex_counts = df["SEX"].value_counts().sort_index()

ax = sex_counts.plot(kind="bar")

ax.bar_label(ax.containers[0])

plt.xlabel("Sex")
plt.ylabel("Number of Customers")
plt.title("Customer Distribution by Sex")
plt.xticks([0, 1], ["Female", "Male"], rotation=0)

plt.show()

#--------------------------------------------------------------  
#default rate by gender

plt.figure(2)
sex_default = df.groupby("SEX")["default payment next month"].mean() * 100

ax = sex_default.plot(kind="bar")

ax.bar_label(ax.containers[0], fmt="%.1f%%")

plt.xlabel("Sex")
plt.ylabel("Default Rate (%)")
plt.title("Default Rate by Sex")
plt.xticks([0, 1], ["Female", "Male"], rotation=0)

plt.show()

#--------------------------------------------------------------  
#default rate by education

plt.figure(3)
education_default = (
    df.groupby("EDUCATION")["default payment next month"]
    .mean() * 100
)

ax = education_default.plot(kind="bar")

ax.bar_label(ax.containers[0], fmt="%.1f%%")

plt.xlabel("Education")
plt.ylabel("Default Rate (%)")
plt.title("Default Rate by Education Level")
plt.xticks(rotation=0)

plt.show()
#--------------------------------------------------------------  
#default rate by marriage status


# Select 300 random customers
df_plot = df.sample(n=300, random_state=42)

# Sort them by Education
df_plot = df_plot.sort_values("EDUCATION").reset_index(drop=True)

# Create X axis
df_plot["row"] = range(1, len(df_plot) + 1)

print(df_plot.shape)



plt.figure(4)
marriage_default = (
    df.groupby("MARRIAGE")["default payment next month"]
    .mean() * 100
)

ax = marriage_default.plot(kind="bar")

ax.bar_label(ax.containers[0], fmt="%.1f%%")

plt.xlabel("Marriage")
plt.ylabel("Default Rate (%)")
plt.title("Default Rate by Marriage Status")
plt.xticks(rotation=0)

plt.show()
#--------------------------------------------------------------  
#--------------------------------------------------------------  

# Sort by Education

fig, ax1 = plt.subplots(figsize=(14, 6))

ax1.plot(
    df_plot["row"],
    df_plot["EDUCATION"],
    color="orange",
    marker="o",
    linestyle="-",
    linewidth=1,
    markersize=3
)

ax1.set_xlabel("Selected Customers (sorted by Education)")
ax1.set_ylabel("Education", color="orange")
ax1.set_ylim(0.5, 4.5)
ax1.set_yticks([1, 2, 3, 4])

ax2 = ax1.twinx()

ax2.plot(
    df_plot["row"],
    df_plot["default payment next month"],
    color="black",
    marker="o",
    linestyle="-",
    linewidth=1,
    markersize=3
)

ax2.set_ylabel("Default", color="black")
ax2.set_ylim(-0.05, 1.05)
ax2.set_yticks([0, 1])
ax2.set_yticklabels(["No Default", "Default"])

plt.title("Figure 5: Education and Default for 300 Random Customers")

plt.tight_layout()
plt.show()

#---------------------------------------------------------------------------------------step1 finished
#--------------------------------------------step2----------------------------------------------------------
#------------------------ploting and default rate_______________________
# ----------------ploting considering target group----------------------
# target_counts = df["default payment next month"].value_counts().sort_index()
# fig, ax = plt.subplots(num=1, figsize=(7, 5))

# ax.bar(
#     ["No Default", "Default"],
#     target_counts
# )

# ax.bar_label(
#     ax.containers[0]
# )

# ax.set_xlabel("Default Status")
# ax.set_ylabel("Number of Customers")
# ax.set_title("Figure 1: Target Distribution")

# plt.tight_layout()
# plt.show()

# #--------------------------------------------------------------
# #--------------------------------------------------------------

# #limit ball distribution- credit specifies to people, by nominal 
# fig, ax = plt.subplots(num=2, figsize=(9, 5))

# ax.hist(
#     df["LIMIT_BAL"],
#     bins=30,
#     edgecolor="black"
# )

# ax.set_xlabel("Credit Limit")
# ax.set_ylabel("Number of Customers")
# ax.set_title("Figure 2: Credit Limit Distribution")

# plt.tight_layout()
# plt.show()
# #--------------------------------------------------------------
# #--------------------------------------------------------------
# # group customer in 5 group based on LIMIT-BAL 

# df["LIMIT_GROUP"] = pd.qcut(
#     df["LIMIT_BAL"],
#     q=5
# )
# limit_default = (
#     df.groupby("LIMIT_GROUP", observed=True)
#     ["default payment next month"]
#     .mean()
#     * 100
# )
# fig, ax = plt.subplots(num=3, figsize=(10, 5))

# limit_default.plot(
#     kind="bar",
#     ax=ax
# )

# ax.set_xlabel("Credit Limit Group")
# ax.set_ylabel("Default Rate (%)")
# ax.set_title("Figure 3: Default Rate by Credit Limit")

# plt.xticks(rotation=45)

# plt.tight_layout()
# plt.show()
# #--------------------------------------------------------------
# #-------------------------mean of bil-amt & pay-amt
# bill_cols = [
#     "BILL_AMT1",
#     "BILL_AMT2",
#     "BILL_AMT3",
#     "BILL_AMT4",
#     "BILL_AMT5",
#     "BILL_AMT6"
# ]

# pay_cols = [
#     "PAY_AMT1",
#     "PAY_AMT2",
#     "PAY_AMT3",
#     "PAY_AMT4",
#     "PAY_AMT5",
#     "PAY_AMT6"
# ]

# df["AVG_BILL"] = df[bill_cols].mean(axis=1)
# df["AVG_PAYMENT"] = df[pay_cols].mean(axis=1)

# df["MEDIAN_PAYMENT"] = df[pay_cols].median(axis=1)
# df["MEDIAN_BILL"] = df[bill_cols].median(axis=1)

# df[pay_cols].median(axis=1)
# df["MEDIAN_PAYMENT"].median()

# #-------------------------------------------------

# #Average Bill vs Average Payment

# fig, ax = plt.subplots(num=4, figsize=(9, 6))

# ax.scatter(
#     df["AVG_BILL"],
#     df["AVG_PAYMENT"],
#     alpha=0.2
# )

# ax.set_xlabel("Average Bill Amount")
# ax.set_ylabel("Average Payment Amount")
# ax.set_title("Figure 4: Average Bill vs Average Payment")
# plt.tight_layout()
# plt.show()

# #----------------------------------------------------
# #PAY_0 vs Default
# print(df["PAY_0"].value_counts().sort_index())
# pay0_default = (
#     df.groupby("PAY_0")["default payment next month"]
#     .mean()
#     * 100
# )
# fig, ax = plt.subplots(num=5, figsize=(10, 5))

# pay0_default.plot(
#     kind="bar",
#     ax=ax
# )

# ax.set_xlabel("PAY_0")
# ax.set_ylabel("Default Rate (%)")
# ax.set_title("Figure 5: Default Rate by Payment Status (PAY_0)")

# ax.bar_label(
#     ax.containers[0]
# )

# plt.xticks(rotation=0)
# plt.tight_layout()
# plt.show()
#-------------------------------------------------------------------finished step 2
#-------------------------------------------step3-----------------------------------------------
#-------------checking statistically for outlier and correlation checking for multicollinary-----
#IQR-OUTLIER DETECTION

outlier_cols = [
    "LIMIT_BAL",
    "AGE",
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

for col in outlier_cols:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[
        (df[col] < lower_bound) |
        (df[col] > upper_bound)
    ]

    print(
        f"{col}: {len(outliers)} outliers"
    )

#___________________correlation----------------------
numeric_cols = [
    "LIMIT_BAL",
    "AGE",
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
    "PAY_AMT6",
    "default payment next month"
]
corr = df[numeric_cols].corr()
#print(corr["default payment next month"].sort_values())
#-----------------------------------------------------------------------------step3 finished
#-------------------------------------------------------------------------------------------
#-------------------------------step4-----------------------------------------------------
#-----------------------dataframe saves ststistical features of numeric features-----------
# # --------------------------------------------------------------
# # Numerical columns
# # --------------------------------------------------------------

# numeric_cols = (
#     ["LIMIT_BAL", "AGE"] +
#     [f"BILL_AMT{i}" for i in range(1, 7)] +
#     [f"PAY_AMT{i}" for i in range(1, 7)]
# )

# results = []

# # --------------------------------------------------------------
# # Calculate IQR and outliers
# # --------------------------------------------------------------

# for col in numeric_cols:

#     mean = df[col].mean()
#     median = df[col].median()

#     q1 = df[col].quantile(0.25)
#     q3 = df[col].quantile(0.75)

#     iqr = q3 - q1

#     low = q1 - 1.5 * iqr
#     high = q3 + 1.5 * iqr

#     n_out = ((df[col] < low) | (df[col] > high)).sum()

#     results.append({
#         "ستون": col,
#         "میانگین": round(mean, 0),
#         "میانه": round(median, 0),
#         "Q1": round(q1, 0),
#         "Q3": round(q3, 0),
#         "IQR": round(iqr, 0),
#         "حد_پایین": round(low, 0),
#         "حد_بالا": round(high, 0),
#         "تعداد_اوت‌لایر": n_out,
#         "درصد": round(n_out / len(df) * 100, 1)
#     })

# # --------------------------------------------------------------
# # Create DataFrame
# # --------------------------------------------------------------

# summary_df = pd.DataFrame(results)

# print(summary_df.to_string(index=False))

# # --------------------------------------------------------------
# # Save in the same Excel file as a new sheet
# # --------------------------------------------------------------

# with pd.ExcelWriter(
#     path,
#     engine="openpyxl",
#     mode="a",
#     if_sheet_exists="replace"
# ) as writer:

#     summary_df.to_excel(
#         writer,
#         sheet_name="df_quantil",
#         index=False
#     )

# print("Sheet 'df_quantil' was successfully added to the Excel file.")
#--------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------step4 finished
#-----------------------------------------------------------------------------step 5
#--------------------apply logestic model and apply transformations-------------------

#===================logestic model============================
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

    penalty="l2",

    max_iter=1000,

    random_state=42
)


model_log = LogisticRegression(
    class_weight="balanced",

    penalty="l2",

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

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score
)


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
    C=1.0,               # هرچی کوچیک‌تر، فشار Regularization بیشتر
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
#_________________________________________________________________________step5 finished