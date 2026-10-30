import pandas as pd
import numpy as np

path = "studybuuild_3/default of credit card clients_cleaned.xlsx"

df = pd.read_excel(path)

pay_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]
payamt_cols = [f"PAY_AMT{i}" for i in range(1, 7)]

# ---------- تعریف فیچرهای جدید ----------
df["N_DELAY_MONTHS"] = (df[pay_cols] > 0).sum(axis=1)
df["AVG_DELAY"]      = df[pay_cols].mean(axis=1)
df["MAX_DELAY"]      = df[pay_cols].max(axis=1)
df["PAYMENT_RATIO"]  = df[payamt_cols].sum(axis=1) / df[bill_cols].sum(axis=1).replace(0, 1)
df["UTILIZATION"]    = df[bill_cols].mean(axis=1) / df["LIMIT_BAL"].replace(0, 1)
df["BILL_TREND"]     = df["BILL_AMT1"] - df["BILL_AMT6"]

# ---------- ذخیره در یه شیت جدید، بدون پاک کردن شیت‌های قبلی ----------
with pd.ExcelWriter(path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    df.to_excel(writer, sheet_name="df_engineered_features", index=False)

print("شیت df_engineered_features اضافه شد.")
print(df.shape)