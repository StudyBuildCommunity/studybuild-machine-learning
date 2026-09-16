
#-------------iqr& outliers____________________________
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

# --------------------------------------------------------------
# Numerical columns
# --------------------------------------------------------------

numeric_cols = (
    ["LIMIT_BAL", "AGE"] +
    [f"BILL_AMT{i}" for i in range(1, 7)] +
    [f"PAY_AMT{i}" for i in range(1, 7)]
)

results = []

# --------------------------------------------------------------
# Calculate IQR and outliers
# --------------------------------------------------------------

for col in numeric_cols:

    mean = df[col].mean()
    median = df[col].median()

    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)

    iqr = q3 - q1

    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr

    n_out = ((df[col] < low) | (df[col] > high)).sum()

    results.append({
        "ستون": col,
        "میانگین": round(mean, 0),
        "میانه": round(median, 0),
        "Q1": round(q1, 0),
        "Q3": round(q3, 0),
        "IQR": round(iqr, 0),
        "حد_پایین": round(low, 0),
        "حد_بالا": round(high, 0),
        "تعداد_اوت‌لایر": n_out,
        "درصد": round(n_out / len(df) * 100, 1)
    })

# --------------------------------------------------------------
# Create DataFrame
# --------------------------------------------------------------

summary_df = pd.DataFrame(results)

print(summary_df.to_string(index=False))

# --------------------------------------------------------------
# Save in the same Excel file as a new sheet
# --------------------------------------------------------------

with pd.ExcelWriter(
    path,
    engine="openpyxl",
    mode="a",
    if_sheet_exists="replace"
) as writer:

    summary_df.to_excel(
        writer,
        sheet_name="df_quantil",
        index=False
    )

print("Sheet 'df_quantil' was successfully added to the Excel file.")