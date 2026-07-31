# Telco Customer Churn — Analysis & Prediction

A data analytics project exploring why telecom customers churn, and building a
model to predict who's likely to leave next.

## Business Problem

Customer churn (customers cancelling their service) directly costs revenue.
This project analyzes 7,043 customer records from a telecom provider to:

1. Understand **why** customers churn (exploratory analysis)
2. Predict **who** is likely to churn next (classification model)
3. Turn both into **actionable retention recommendations**

## Dataset

- **Source:** [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (Kaggle, IBM sample dataset)
- **Size:** 7,043 rows × 21 columns
- **Target variable:** `Churn` (Yes/No)
- **Features:** demographics, account info (contract, tenure, charges), and
  subscribed services (internet, streaming, tech support, etc.)

## Tools Used

- **Python** — pandas, numpy (cleaning & analysis)
- **matplotlib, seaborn** — visualization
- **scikit-learn** — Logistic Regression & Random Forest classification
- **HTML/CSS** — interactive dashboard

## Project Structure

```
├── README.md
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # raw data
├── notebooks/
│   ├── 01_cleaning.py
│   ├── 02_eda.py
│   └── 03_modeling.py
├── dashboard/
│   └── churn_dashboard.html
└── images/
    ├── eda_overview.png
    ├── corr_heatmap.png
    └── feature_importance.png
```

## 1. Data Cleaning

- `TotalCharges` was loaded as text due to 11 blank entries. Investigation
  showed all 11 belonged to customers with `tenure = 0` (brand-new signups,
  not yet billed) — converted to numeric and filled with `0`, the accurate
  value for a day-zero customer.
- No duplicate rows found.
- Added a binary `Churn_Binary` column (1 = Yes, 0 = No) for modeling.

## 2. Exploratory Data Analysis — Key Findings

| Finding | Detail |
|---|---|
| **Overall churn rate** | 26.5% (1,869 of 7,043 customers) |
| **Contract type** | Month-to-month churns 15x more than two-year contracts (42.7% vs 2.8%) |
| **Tenure** | Strongest single driver — churn drops from **47.4%** in the first year to **9.5%** after 4+ years |
| **Internet service** | Fiber optic customers churn more (41.9%) than DSL (19.0%) or no-internet customers (7.4%) |
| **Support add-ons** | Having Tech Support or Online Security cuts churn risk by roughly 3x |
| **Payment method** | Electronic check payers churn at 45.3% vs ~15–19% for automatic payment methods |
| **Monthly charges** | Churned customers pay more per month on average than retained customers |

See `images/eda_overview.png` and `images/corr_heatmap.png` for the supporting charts.

## 3. Feature Engineering

- Binary-encoded Yes/No columns (`Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`, `gender`)
- One-hot encoded multi-category columns (`Contract`, `PaymentMethod`, `InternetService`, and all service add-ons)
- Engineered a `tenure_bucket` feature (0–12mo / 12–24mo / 24–48mo / 48–72mo) to capture the non-linear tenure effect
- Final model-ready dataset: 7,043 rows × 34 columns

## 4. Modeling — Predicting Churn

Two classifiers were trained on an 80/20 stratified train/test split:

| Model | Accuracy | Precision (Churn) | Recall (Churn) | F1 (Churn) |
|---|---|---|---|---|
| Logistic Regression | 79.8% | 65.0% | 52.1% | 57.9% |
| **Random Forest (class-balanced)** | 76.4% | 54.0% | **73.5%** | **62.3%** |

**Why Random Forest was chosen as the primary model:** in a churn use case,
missing an actual churner (false negative) is usually costlier than a false
alarm — a missed churner is lost revenue, while a false alarm is just an
unnecessary retention offer. The balanced Random Forest catches **73.5%** of
actual churners vs. 52.1% for plain Logistic Regression, at the cost of more
false positives — an acceptable trade-off if retention outreach is low-cost
(email/discount) rather than high-cost (e.g. a phone call).

### Top predictive features (Random Forest)
1. Tenure
2. Total charges
3. Monthly charges
4. Two-year contract (protective)
5. Fiber optic internet
6. Electronic check payment
7. Online security / tech support (protective)

See `images/feature_importance.png`.

## 5. Dashboard

An interactive HTML dashboard (`dashboard/churn_dashboard.html`) summarizes:
- Key KPIs (total customers, churn rate, revenue at risk, model recall)
- The retention window (churn rate by tenure bucket)
- Churn breakdowns by contract, internet service, payment method, and support add-ons
- Model comparison table

## 6. Business Recommendations

1. **Target the first 12 months.** Nearly half of churn happens in year one —
   proactive onboarding/check-ins during this window would likely have the
   biggest impact.
2. **Incentivize contract upgrades.** Offer discounts for switching from
   month-to-month to annual contracts, especially for new fiber customers.
3. **Push autopay enrollment.** Electronic check payers churn 3x more than
   autopay customers — a small incentive to switch payment methods could
   meaningfully reduce churn.
4. **Bundle support services.** Tech support and online security correlate
   with much lower churn — consider offering these as free trials to
   at-risk segments.

## Future Improvements

- Hyperparameter tuning (GridSearchCV) for the Random Forest
- Try gradient boosting models (XGBoost/LightGBM) for potentially better recall/precision balance
- Deploy the model as a simple API or Streamlit app for live scoring
- A/B test the recommended retention interventions on the highest-risk segment

---
**Author:** *[Your Name]* · Built as a portfolio project for data analyst / data science practice.
