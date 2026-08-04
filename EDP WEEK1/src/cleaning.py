import pandas as pd
import numpy as np
data = {
    "Employee": ["Rahul", "Priya", "Amit", "Sneha", "Kiran", "Anjali"],
    "Age": [25, 30, np.nan, 28, 35, 27],
    "Experience": [2, 5, 3, np.nan, 10, 4],
    "Education_Years": [16, 18, 15, 17, 20, np.nan],
    "Salary": [35000, 60000, 42000, 50000, 90000, np.nan]
}

df = pd.DataFrame(data)

print("=" * 60)
print("ORIGINAL EMPLOYEE DATASET")
print("=" * 60)
print(df)
print("\nDataset Info")
print(df.info())

print("\nStatistical Summary")
print(df.describe())
print("\nMissing Values")
print(df.isnull().sum())

df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Experience"] = df["Experience"].fillna(df["Experience"].mean())
df["Education_Years"] = df["Education_Years"].fillna(df["Education_Years"].mean())
df["Salary"] = df["Salary"].fillna(df["Salary"].mean())

print("\n" + "=" * 60)
print("CLEANED DATASET")
print("=" * 60)
print(df)