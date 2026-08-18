"""
Generates a synthetic customer dataset in the style of the classic
'Mall_Customers.csv' dataset (CustomerID, Gender, Age, Annual Income, Spending Score).

If you already have a real dataset (e.g. downloaded from Kaggle), skip this script
and just place your CSV at data/Mall_Customers.csv with matching column names.
"""
import numpy as np
import pandas as pd

np.random.seed(42)
n = 200

# Create 5 natural customer archetypes so clustering has real structure to find
segments = [
    # (n_customers, age_range, income_range, spending_range)
    (40, (18, 30), (15, 40), (70, 95)),   # young, low income, high spenders
    (40, (25, 40), (60, 90), (60, 90)),   # mid age, high income, high spenders
    (40, (35, 55), (60, 90), (5, 35)),    # mid age, high income, low spenders (savers)
    (40, (45, 65), (15, 40), (5, 35)),    # older, low income, low spenders
    (40, (30, 50), (40, 60), (40, 60)),   # average across the board
]

rows = []
cid = 1
for count, age_r, inc_r, spend_r in segments:
    for _ in range(count):
        age = np.random.randint(*age_r)
        income = np.random.randint(*inc_r)
        spend = np.clip(np.random.randint(*spend_r) + np.random.randint(-8, 8), 1, 100)
        gender = np.random.choice(["Male", "Female"])
        rows.append([cid, gender, age, income, spend])
        cid += 1

df = pd.DataFrame(rows, columns=["CustomerID", "Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df["CustomerID"] = range(1, len(df) + 1)

df.to_csv("data/Mall_Customers.csv", index=False)
print(f"Saved {len(df)} rows to data/Mall_Customers.csv")
print(df.head())
