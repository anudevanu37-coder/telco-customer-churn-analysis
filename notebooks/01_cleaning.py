"""
Step 1: Load and clean the Telco Customer Churn dataset.
"""
import pandas as pd

df = pd.read_csv('../data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# TotalCharges loads as text due to 11 blank entries (all tenure=0, day-0 customers)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

# Binary target for modeling
df['Churn_Binary'] = df['Churn'].map({'Yes': 1, 'No': 0})

print('Shape:', df.shape)
print('Nulls:', df.isnull().sum().sum())
print('Duplicates:', df.duplicated().sum())
print('Churn rate:', df['Churn_Binary'].mean().round(3))

df.to_csv('telco_churn_clean.csv', index=False)
