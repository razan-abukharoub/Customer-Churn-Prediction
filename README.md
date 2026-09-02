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
```
Tuned XGBoost achieved:

- Cross-validation ROC-AUC: **0.850**
- Test ROC-AUC: **0.847**

---

## 🎚️ Classification Threshold Optimization

The default classification threshold of 0.50 is not always the best choice for a business problem.

For churn prediction, missing a customer who is likely to churn can be costly. Therefore, different thresholds were evaluated using cross-validation predictions from the training data.

| Threshold | Precision | Recall | F1-score |
|---:|---:|---:|---:|
| 0.30 | 0.541 | 0.769 | 0.635 |
| 0.35 | 0.574 | 0.718 | **0.638** |
| 0.40 | 0.608 | 0.654 | 0.630 |
| 0.45 | 0.644 | 0.602 | 0.622 |
| 0.50 | 0.670 | 0.530 | 0.592 |
| 0.55 | 0.696 | 0.457 | 0.551 |
| 0.60 | 0.722 | 0.381 | 0.499 |

A threshold of **0.35** was selected because it achieved the highest F1-score among the evaluated thresholds while substantially improving recall.

---

## 🏆 Final Model

**Final Model: Tuned XGBoost + Classification Threshold = 0.35**

Final test performance:

| Metric | Score |
|---|---:|
| Accuracy | 0.784 |
| Precision | 0.571 |
| Recall | **0.738** |
| F1-score | **0.644** |
| ROC-AUC | **0.847** |

At the default threshold of 0.50:

- True Positives: 209
- False Negatives: 165

At the optimized threshold of 0.35:

- True Positives: 276
- False Negatives: 98

The optimized threshold detected **67 additional churners** in the test set, while increasing the number of false positives.

This trade-off can be useful when the cost of missing a potential churner is higher than the cost of contacting an additional customer.

---

## 🔎 Model Explainability with SHAP

SHAP (SHapley Additive exPlanations) was used to understand which features contribute most to the model's predictions.

The most important features included:

1. Contract - Two year
2. Payment Method - Electronic check
3. Tech Support - No
4. Paperless Billing - No
5. Multiple Lines - No
6. Total Charges
7. Streaming Movies - Yes
8. Streaming TV - Yes
9. Online Backup - No
10. Dependents - No

The SHAP analysis also showed that:

- Month-to-month contracts increase churn risk.
- Short tenure increases churn risk.
- Fiber optic internet service increases churn risk.
- Higher monthly charges increase churn risk.
- Lack of Online Security increases churn risk.
- Electronic check payment is associated with higher churn risk.
- Lack of Tech Support increases churn risk.

---

## 👤 High-Risk Customer Example

The model was used to examine an individual high-risk customer.

Predicted churn probability:

**91.6%**

Important characteristics included:

- Tenure: 1 month
- Contract: Month-to-month
- Internet Service: Fiber optic
- Monthly Charges: 95.10
- Payment Method: Electronic check
- Online Security: No
- Tech Support: No

The SHAP explanation showed that short tenure, month-to-month contract, fiber optic service, higher monthly charges, electronic check payment, and lack of additional services contributed strongly to the high-risk prediction.

---

## 💡 Business Recommendations

Based on the analysis, the telecom company could:

1. **Target month-to-month customers** with retention offers.
2. **Focus on new customers** during their early months.
3. **Promote additional services** such as Online Security and Tech Support.
4. **Investigate electronic check customers** and encourage alternative payment methods.
5. **Monitor customers with high monthly charges**, especially when combined with other risk factors.
6. **Use risk-based retention campaigns** to prioritize customers with high predicted churn probability.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- LightGBM
- SHAP
- Google Colab
- Jupyter Notebook

---

## 📁 Project Structure

```text
Customer-Churn-Prediction/
│
├── Customer_Churn_Prediction_Clean.ipynb
├── README.md
├── requirements.txt
│
└── data/
    └── Telco-Customer-Churn.csv
