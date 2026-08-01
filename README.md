# Telco Customer Churn — Analysis & Prediction

A data analytics project exploring why telecom customers churn, and building a model to predict who's likely to leave next.

## Business Problem

Customer churn (customers cancelling their service) directly costs revenue. This project analyzes 7,043 customer records from a telecom provider to:

1. Understand **why** customers churn (exploratory analysis)
2. Predict **who** is likely to churn next (classification model)
3. Turn both into **actionable retention recommendations**

## Dataset

- **Source:** [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (Kaggle, IBM sample dataset)
- **Size:** 7,043 rows x 21 columns
- **Target variable:** Churn (Yes/No)
- **Features:** demographics, account info (contract, tenure, charges), and subscribed services (internet, streaming, tech support, etc.)

## Tools Used

- **Python** — pandas, numpy (cleaning & analysis)
- **matplotlib, seaborn** — visualization
- **scikit-learn** — Logistic Regression & Random Forest classification
- **Power BI** — interactive dashboard

## Project Structure

- README.md
- 01_cleaning.py — data loading and cleaning
- 02_eda.py — exploratory data analysis
- 03_modeling.py — feature engineering and modeling
- Telco_Churn_Dashboard.pbix — Power BI dashboard file
- Telco_Churn_Dashboard.pdf — static preview of the dashboard
- eda_overview.png, corr_heatmap.png, feature_importance.png — supporting charts

## 1. Data Cleaning

- TotalCharges was loaded as text due to 11 blank entries. Investigation showed all 11 belonged to customers with tenure = 0 (brand-new signups, not yet billed) — converted to numeric and filled with 0.
- No duplicate rows found.
- Added a binary Churn_Binary column (1 = Yes, 0 = No) for modeling.

## 2. Exploratory Data Analysis — Key Findings

| Finding | Detail |
|---|---|
| Overall churn rate | 26.5% (1,869 of 7,043 customers) |
| Contract type | Month-to-month churns 15x more than two-year contracts (42.7% vs 2.8%) |
| Tenure | Strongest single driver — churn drops from 47.4% in the first year to 9.5% after 4+ years |
| Internet service | Fiber optic customers churn more (41.9%) than DSL (19.0%) or no-internet customers (7.4%) |
| Support add-ons | Having Tech Support or Online Security cuts churn risk by roughly 3x |
| Payment method | Electronic check payers churn at 45.3% vs ~15-19% for automatic payment methods |

See eda_overview.png and corr_heatmap.png for supporting charts.

## 3. Feature Engineering

- Binary-encoded Yes/No columns, one-hot encoded multi-category columns
- Engineered a tenure_bucket feature (0-12mo / 12-24mo / 24-48mo / 48-72mo)
- Final model-ready dataset: 7,043 rows x 34 columns

## 4. Modeling — Predicting Churn

| Model | Accuracy | Precision (Churn) | Recall (Churn) | F1 (Churn) |
|---|---|---|---|---|
| Logistic Regression | 79.8% | 65.0% | 52.1% | 57.9% |
| Random Forest (class-balanced) | 76.4% | 54.0% | 73.5% | 62.3% |

Random Forest was chosen as the primary model since missing an actual churner costs more than a false alarm — it catches 73.5% of real churners vs. 52.1% for Logistic Regression.

### Top predictive features

Tenure, Total charges, Monthly charges, Two-year contract (protective), Fiber optic internet, Electronic check payment, Online security / tech support (protective). See feature_importance.png.

## 5. Dashboard

Built in Power BI Desktop (Telco_Churn_Dashboard.pbix) — a 2-page interactive report with KPI cards, churn breakdowns by contract, internet service, and payment method, a tenure trend view, and live slicers for filtering. See Telco_Churn_Dashboard.pdf for a static preview.

## 6. Business Recommendations

1. Target the first 12 months — nearly half of churn happens in year one.
2. Incentivize contract upgrades from month-to-month to annual plans.
3. Push autopay enrollment — electronic check payers churn 3x more.
4. Bundle support services — tech support/online security correlate with much lower churn.

## Future Improvements

- Hyperparameter tuning (GridSearchCV) for the Random Forest
- Try gradient boosting models (XGBoost/LightGBM)
- Deploy the model as a simple API or Streamlit app for live scoring
- A/B test retention interventions on the highest-risk segment
