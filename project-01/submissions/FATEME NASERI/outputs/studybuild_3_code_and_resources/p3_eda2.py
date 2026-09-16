#------------------ploting and default rate_______________________

#-----------------virtual in terminal--------------------
#python3 -m venv .venv
#source .venv/bin/activate
#which python

#-----------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt


# choose header=1 , in order to find table headers on the second row not a first row
path = "studybuuild_3/default of credit card clients_cleaned.xlsx"
df = pd.read_excel(path)

print(df.columns.tolist())
print(df.shape)

#--------------------------------------------------------------
#--------------------------------------------------------------

#default distribution ? 50/50 0r 40/60 0r less? less


target_counts = df["default payment next month"].value_counts().sort_index()

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


#--------------------------------------------------------------
#--------------------------------------------------------------

#limit ball distribution- credit specifies to people, by nominal 
fig, ax = plt.subplots(num=2, figsize=(9, 5))

ax.hist(
    df["LIMIT_BAL"],
    bins=30,
    edgecolor="black"
)

ax.set_xlabel("Credit Limit")
ax.set_ylabel("Number of Customers")
ax.set_title("Figure 2: Credit Limit Distribution")

plt.tight_layout()
plt.show()


#--------------------------------------------------------------
#--------------------------------------------------------------
# group customer in 5 group based on LIMIT-BAL 

df["LIMIT_GROUP"] = pd.qcut(
    df["LIMIT_BAL"],
    q=5
)
limit_default = (
    df.groupby("LIMIT_GROUP", observed=True)
    ["default payment next month"]
    .mean()
    * 100
)
fig, ax = plt.subplots(num=3, figsize=(10, 5))

limit_default.plot(
    kind="bar",
    ax=ax
)

ax.set_xlabel("Credit Limit Group")
ax.set_ylabel("Default Rate (%)")
ax.set_title("Figure 3: Default Rate by Credit Limit")

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

#--------------------------------------------------------------
#--------------------------------------------------------------
#-------------------------mean of bii-amt & pay-amt
bill_cols = [
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6"
]

pay_cols = [
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6"
]

df["AVG_BILL"] = df[bill_cols].mean(axis=1)
df["AVG_PAYMENT"] = df[pay_cols].mean(axis=1)

df["MEDIAN_PAYMENT"] = df[pay_cols].median(axis=1)
df["MEDIAN_BILL"] = df[bill_cols].median(axis=1)

df[pay_cols].median(axis=1)
df["MEDIAN_PAYMENT"].median()

#-------------------------------------------------

#Average Bill vs Average Payment

fig, ax = plt.subplots(num=4, figsize=(9, 6))

ax.scatter(
    df["AVG_BILL"],
    df["AVG_PAYMENT"],
    alpha=0.2
)

ax.set_xlabel("Average Bill Amount")
ax.set_ylabel("Average Payment Amount")
ax.set_title("Figure 4: Average Bill vs Average Payment")

plt.tight_layout()
plt.show()


#----------------------------------------------------
#----------------------------------------------------
#PAY_0 vs Default
print(df["PAY_0"].value_counts().sort_index())
pay0_default = (
    df.groupby("PAY_0")["default payment next month"]
    .mean()
    * 100
)
fig, ax = plt.subplots(num=5, figsize=(10, 5))

pay0_default.plot(
    kind="bar",
    ax=ax
)

ax.set_xlabel("PAY_0")
ax.set_ylabel("Default Rate (%)")
ax.set_title("Figure 5: Default Rate by Payment Status (PAY_0)")

ax.bar_label(
    ax.containers[0]
)

plt.xticks(rotation=0)

plt.tight_layout()
plt.show()
#
#----------------------------------------------------
#----------------------------------------------------
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

print(corr["default payment next month"].sort_values())