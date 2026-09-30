
#----------------clean data------------------------
#-----------------virtual in terminal--------------------
#python3 -m venv .venv
#source .venv/bin/activate
#which python

#-----------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt


# choose header=1 , in order to find table headers on the second row not a first row
path = "studybuuild_3/default of credit card clients_raw.xls"
df = pd.read_excel(path,header=1)

#file info
# print(df.head())
# print(df.shape)
# print(df.columns)
# print(df.dtypes)
# df.info()

# #read the file ending true
# print(df.tail())

# #missing values-just nulls
# df.isnull().sum()
# #null row?
# df.isnull().all(axis=1).sum()
# #is duplicate?
# df.duplicated().sum()
# df[df.duplicated()]
# # are ids unique in each row by num of unique rows =nunique
# df["ID"].nunique()
# df["ID"].duplicated().sum()

#----------------
# بشمار چند بار هر عدد در هر ستون تکرار شده و بعد بر اساس عدد ایندکس مرتب کن
categorical_cols = ["SEX", "EDUCATION", "MARRIAGE"]

# for col in categorical_cols:
#     print(f"\n--- {col} ---")
#     print(df[col].value_counts().sort_index())


#تعمیم دسته های اشتباه به گروه مجزا 0 
# marriage:0 , education:0,5,6   
df["EDUCATION"] = df["EDUCATION"].replace({
    0: 4,
    5: 4,
    6: 4
})

df["MARRIAGE"] = df["MARRIAGE"].replace({
    0: 3
})
# print(df["EDUCATION"].value_counts().sort_index())
# print(df["MARRIAGE"].value_counts().sort_index())

#-------------
# Repayment status variables contain negative and zero codes.
# These values are treated as valid category codes and are preserved.
# No cleaning is applied at this stage.
pay_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
# for col in pay_cols:
#     print(f"\n--- {col} ---")
#     print(df[col].value_counts().sort_index())
#--------------------------------------------------------------
#age
# print("Minimum age:", df["AGE"].min())
# print("Maximum age:", df["AGE"].max())

# print(df["AGE"].describe())
#--------------------------------------------------------------
#limit bal describe
# print(df["LIMIT_BAL"].describe())
# print("Negative values:", (df["LIMIT_BAL"] < 0).sum())
# print("Zero values:", (df["LIMIT_BAL"] == 0).sum())
#--------------------------------------------------------------
#bill columns describe -any negative? yes any zero? yes
bill_cols = [
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
    "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"
]

# for col in bill_cols:
#     print(f"\n--- {col} ---")
#     print(df[col].describe())
#     print("Negative values:", (df[col] < 0).sum())
#     print("Zero values:", (df[col] == 0).sum())
    
#--------------------------------------------------------------    
#pay-amt describe : how much he/she paid per month , negative value?
pay_amt_cols = [
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3",
    "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
]

# for col in pay_amt_cols:
#     print(f"\n--- {col} ---")
#     print(df[col].describe())
#     print("Negative values:", (df[col] < 0).sum())
#     print("Zero values:", (df[col] == 0).sum())
 
 #--------------------------------------------------------------
 #--------------------------------------------------------------
#save dataframe
# Save cleaned dataset

# output_path = "studybuuild_3/default of credit card clients_cleaned.xlsx"

# df.to_excel(output_path, index=False)

# print(f"Cleaned dataset saved to: {output_path}")
    
#--------------------------------------------------------------
#--------------------------------------------------------------   
#charts before EDA
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

#
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

#
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
#
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

#----------------------------------------------------------------------