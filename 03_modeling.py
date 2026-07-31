"""
Step 3: Feature engineering + churn prediction modeling.
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)

df = pd.read_csv('telco_churn_clean.csv')

# --- Feature engineering ---
model_df = df.drop(columns=['customerID', 'Churn'])

binary_map = {'Yes': 1, 'No': 0}
for col in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']:
    model_df[col] = model_df[col].map(binary_map)
model_df['gender'] = model_df['gender'].map({'Male': 1, 'Female': 0})

multi_cat_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                   'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                   'Contract', 'PaymentMethod']
model_df = pd.get_dummies(model_df, columns=multi_cat_cols, drop_first=True)

model_df['tenure_bucket'] = pd.cut(model_df['tenure'], bins=[-1, 12, 24, 48, 72],
                                    labels=['0-12mo', '12-24mo', '24-48mo', '48-72mo'])
model_df = pd.get_dummies(model_df, columns=['tenure_bucket'], drop_first=True)

# --- Train/test split ---
X = model_df.drop(columns=['Churn_Binary'])
y = model_df['Churn_Binary']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# --- Logistic Regression baseline ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_model = LogisticRegression(max_iter=1000, random_state=42)
log_model.fit(X_train_scaled, y_train)
log_preds = log_model.predict(X_test_scaled)

print('=== Logistic Regression ===')
print(classification_report(y_test, log_preds, target_names=['No Churn', 'Churn']))

# --- Random Forest (class-balanced) ---
rf_model = RandomForestClassifier(n_estimators=200, max_depth=10,
                                   random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

print('=== Random Forest (balanced) ===')
print(classification_report(y_test, rf_preds, target_names=['No Churn', 'Churn']))

# --- Feature importance ---
importances = pd.Series(rf_model.feature_importances_, index=X.columns) \
    .sort_values(ascending=False).head(10)

plt.figure(figsize=(8, 6))
importances.sort_values().plot(kind='barh')
plt.title('Top 10 Feature Importances (Random Forest)')
plt.tight_layout()
plt.savefig('../images/feature_importance.png', dpi=120)
