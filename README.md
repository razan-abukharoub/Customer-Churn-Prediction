# Customer Churn Prediction

## 📌 Project Overview

Customer churn is a major challenge for telecom companies because losing existing customers can directly affect revenue and long-term growth.

This project develops a machine learning system to predict whether a telecom customer is likely to churn based on customer demographics, services, contract information, payment method, tenure, and billing information.

The project focuses not only on predictive performance, but also on model comparison, threshold optimization, and model explainability using SHAP.

---

## 🎯 Business Problem

The main objective is to identify customers who are at high risk of churn so that the company can take proactive retention actions.

For this reason, the project does not treat all prediction errors equally. Missing a customer who is likely to churn can be more costly than contacting an additional customer who would not churn.

Therefore, the final model uses an optimized classification threshold to improve churn detection.

---

## 📊 Dataset

The project uses IBM's Telco Customer Churn dataset.

- Rows: 7,043
- Input features: 19
- Target: `Churn`
- `No` → 0
- `Yes` → 1

The dataset contains information about:

- Customer demographics
- Partner and dependents
- Phone and internet services
- Additional services
- Contract type
- Payment method
- Tenure
- Monthly charges
- Total charges

---

## 🔍 Exploratory Data Analysis

Several factors showed important relationships with customer churn.

### Contract Type

Month-to-month customers have a substantially higher churn rate than customers with one-year or two-year contracts.

### Tenure

Customers who churn tend to have shorter tenure.

- Non-churned customers: 37.57 months average
- Churned customers: 17.98 months average

### Monthly Charges

Churned customers have higher average monthly charges.

- Non-churned customers: 61.27
- Churned customers: 74.44

### Internet Service

Fiber optic customers showed a higher churn rate than DSL customers and customers without internet service.

### Additional Services

Customers without services such as Online Security and Tech Support generally showed higher churn rates.

### Payment Method

Electronic check customers had the highest churn rate among the payment methods.

---

## 🧹 Data Preprocessing

The following preprocessing steps were performed:

1. Removed the `customerID` column because it is an identifier.
2. Converted `TotalCharges` from object/string to numeric.
3. Identified 11 blank values in `TotalCharges`.
4. Converted the blanks to missing values and replaced them with `0`.
5. Encoded the target variable:
   - `No → 0`
   - `Yes → 1`
6. Split the data into training and testing sets using an 80/20 stratified split.
7. Applied `StandardScaler` to numerical features for models that require scaling.
8. Applied `OneHotEncoder` to categorical features.
9. Used Scikit-learn Pipelines to keep preprocessing and modeling together and avoid data leakage.

---

## 🤖 Machine Learning Models

Six models were compared:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Decision Tree
4. Random Forest
5. XGBoost
6. LightGBM

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

ROC-AUC was used as the main metric for model comparison.

---

## 📈 Model Comparison

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.806 | 0.657 | 0.559 | 0.604 | 0.842 |
| KNN | 0.764 | 0.553 | 0.575 | 0.564 | 0.790 |
| Decision Tree | 0.798 | 0.635 | 0.567 | 0.599 | 0.830 |
| Random Forest | 0.784 | 0.618 | 0.484 | 0.543 | 0.820 |
| XGBoost | 0.801 | 0.654 | 0.527 | 0.584 | 0.841 |
| LightGBM | 0.796 | 0.646 | 0.508 | 0.569 | 0.838 |

---

## ⚙️ Hyperparameter Tuning

RandomizedSearchCV with 5-fold cross-validation was used to tune:

- Logistic Regression
- XGBoost
- LightGBM

The optimization metric was ROC-AUC.

### Best XGBoost Parameters

```text
n_estimators = 300
learning_rate = 0.03
max_depth = 2
min_child_weight = 3
subsample = 0.7
colsample_bytree = 1.0
