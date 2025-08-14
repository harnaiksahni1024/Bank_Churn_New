# 🔍 Churn Prediction in Banking

## ✨ Overview

This project aims to predict whether bank customers will leave the service provider within the next six months. Churn prediction is crucial for banks to understand customer behavior and improve service quality to retain customers. This project utilizes neural network-based classifiers along with several traditional machine learning models.

---

## 🔎 Problem Statement

Banks face the challenge of customer churn, where customers leave for alternative service providers. Understanding the factors influencing customer churn is essential for targeted improvement efforts. The goal is to build a robust classifier to predict if a customer will leave the bank in the near future.

---

## 📊 Data Description

* **Source**: Kaggle
* **Size**: 10,000 records with 14 columns

| No. | Attribute       | Description                                       |
| --- | --------------- | ------------------------------------------------- |
| 1.  | CustomerId      | Unique customer ID                                |
| 2.  | Surname         | Customer's last name                              |
| 3.  | CreditScore     | Credit score indicating customer's credit history |
| 4.  | Geography       | Country of the customer                           |
| 5.  | Gender          | Gender of the customer                            |
| 6.  | Age             | Age of the customer                               |
| 7.  | Tenure          | Years the customer has been with the bank         |
| 8.  | NumOfProducts   | Number of products the customer owns              |
| 9.  | Balance         | Account balance                                   |
| 10. | HasCrCard       | Has credit card (1 = Yes, 0 = No)                 |
| 11. | EstimatedSalary | Estimated annual salary                           |
| 12. | IsActiveMember  | Active customer flag (1 = Yes, 0 = No)            |
| 13. | Exited          | Churn status (1 = churned, 0 = retained)          |

---

## 🔢 Process

### 1. Data Cleaning

* Removed irrelevant features (e.g., CustomerId, Surname)
* Retained only predictive attributes

### 2. Data Analysis

* Visualized class imbalance
* Noted high churn rates for:

  * Female customers (25%)
  * German customers (33%)
* Applied SMOTE to balance class distribution

### 3. Models Used

* Logistic Regression
* Decision Tree
* Random Forest
* K-Nearest Neighbors
* Support Vector Machine
* XGBoost
* Neural Networks (with and without dropout)

---

## 🔍 Features

* Data Cleaning & Preprocessing
* Univariate & Multivariate Analysis
* SMOTE Oversampling for Class Imbalance
* Feature Engineering
* Neural Network & Machine Learning Modeling
* Model Evaluation
* Model Serialization using `joblib`

---

## 📈 Key Insights

* Female customers churn more than male customers.
* German customers show higher churn probability.
* Active members are less likely to churn.
* Customers with fewer bank products are more likely to churn.

---

## 📊 Key Libraries Used

* `pandas`, `numpy`
* `matplotlib`, `seaborn`, `plotly`
* `scikit-learn`
* `xgboost`, `tensorflow`
* `joblib`

---

## 📚 Modeling Strategy

* Used **industry-standard ML pipelines**
* Stratified K-Fold Cross Validation
* SMOTE + class\_weight balancing
* Threshold tuning for imbalanced prediction
* Models saved in `/models` folder using joblib

---

### 📊 Model Performance Summary (Threshold tuned for ~75% Recall)

| Model                     | Precision | Recall   | F1 Score (Macro) | Accuracy | ROC AUC |
|---------------------------|-----------|----------|------------------|----------|---------|
| Decision Tree             | 0.4303    | 0.7740   | 0.5531           | 0.7455   | 0.8385  |
| K-Nearest Neighbors       | 0.4039    | 0.7592   | 0.5273           | 0.7230   | 0.8145  |
| Logistic Regression       | 0.4402    | 0.7690   | 0.5599           | 0.7540   | 0.8377  |
| Random Forest             | 0.4952    | 0.7592   | 0.5994           | 0.7935   | 0.8579  |
| XGBoost                   | 0.5058    | 0.7518   | 0.6047           | 0.8000   | 0.8639  |
| Support Vector Classifier | 0.4565    | 0.7740   | 0.5743           | 0.7665   | 0.8590  |


---

## 🚀 Final Model (Neural Network)

* **Optimizer**: Adam
* **Dropout**: 0.2
* **Best Metric**: Recall = **0.76 85** (highest among all models)

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 0.7191 |
| Precision | 0.4040 |
| Recall    | 0.7686 |
| F1 Score  | 0.5226 |
| AUC       | 0.83   |

> Chosen for its balance between precision and high recall, ideal for minimizing customer loss.

---

## 🚚 Recommendations

### 🔄 Dormant Member Engagement

* Launch campaigns to re-activate dormant accounts.
* Offer personalized offers or guidance.

### 💼 Product Diversification

* Encourage clients to use more products (51% use only one).
* Provide bundling discounts or multi-service rewards.

### 👥 Age-Based Targeting

* Older users churn more. Offer age-specific financial services.

### 📅 Tenure-Based Retention

* New customers (tenure < 2 years) churn more.
* Improve onboarding, loyalty bonuses in first 2 years.

---

## 🛠️ Technologies Used

* Python
* pandas, numpy
* scikit-learn
* XGBoost
* TensorFlow
* joblib (model saving)

---

## 📆 Dataset

* **File**: `resources/analytical_base_table.csv`
* **Target**: `Exited`
* **Features**: 13 selected attributes

---

## 📊 Conclusion

This end-to-end churn prediction project demonstrates how neural networks and traditional ML algorithms can be applied in a real-world banking scenario to retain customers and improve service strategies. A recall-oriented approach ensures that high-risk customers are accurately identified and proactively retained.

> A strategic application of AI can significantly improve customer retention, reduce churn costs, and optimize bank operations.
