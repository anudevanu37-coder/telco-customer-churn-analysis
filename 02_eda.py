"""
Step 2: Exploratory data analysis on the cleaned churn dataset.
Answers the core business questions and saves supporting charts.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('telco_churn_clean.csv')
sns.set_style('whitegrid')

# --- Churn by contract type ---
print(df.groupby('Contract')['Churn_Binary'].mean().round(3))

# --- Churn by tenure, internet service, monthly charges ---
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

sns.barplot(data=df, x='Contract', y='Churn_Binary',
            order=['Month-to-month', 'One year', 'Two year'], ax=axes[0, 0])
axes[0, 0].set_title('Churn Rate by Contract Type')

sns.boxplot(data=df, x='Churn', y='tenure', ax=axes[0, 1])
axes[0, 1].set_title('Tenure — Churned vs Retained')

sns.barplot(data=df, x='InternetService', y='Churn_Binary', ax=axes[1, 0])
axes[1, 0].set_title('Churn Rate by Internet Service')

sns.boxplot(data=df, x='Churn', y='MonthlyCharges', ax=axes[1, 1])
axes[1, 1].set_title('Monthly Charges — Churned vs Retained')

plt.tight_layout()
plt.savefig('../images/eda_overview.png', dpi=120)

# --- Support add-ons & payment method ---
print(df.groupby('TechSupport')['Churn_Binary'].mean().round(3))
print(df.groupby('OnlineSecurity')['Churn_Binary'].mean().round(3))
print(df.groupby('PaymentMethod')['Churn_Binary'].mean().round(3))

# --- Correlation heatmap ---
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen', 'Churn_Binary']
plt.figure(figsize=(7, 5.5))
sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f', center=0)
plt.title('Correlation Heatmap (Numeric Features)')
plt.tight_layout()
plt.savefig('../images/corr_heatmap.png', dpi=120)
