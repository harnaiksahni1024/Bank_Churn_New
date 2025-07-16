Churn Prediction in Banking Using Neural Networks
Overview
This project aims to predict whether bank customers will leave the service provider within the next six months. Churn prediction is crucial for businesses like banks to understand customer behavior and improve service quality to retain customers. The project utilizes neural network-based classifiers to achieve this objective.

Problem Statement
Businesses like banks face the challenge of customer churn, where customers leave for alternative service providers. Understanding the factors influencing customer churn is essential for targeted improvement efforts. The objective is to build a neural network-based classifier that predicts whether a bank customer will leave within the next six months.

Data Description
The dataset, taken from Kaggle, contains 10,000 sample points with 14 distinct features. These are:

Number	Attribute	Description
1.	CustomerId	Unique ID assigned to each customer.
2.	Surname	Last name of the customer.
3.	CreditScore	Defines the credit history of the customer.
4.	Geography	Customer's location.
5.	Gender	Gender of the customer.
6.	Age	Age of the customer.
7.	Tenure	Number of years for which the customer has been with the bank.
8.	NumOfProducts	Number of products that a customer has purchased through the bank.
9.	Balance	Account balance.
10.	HasCrCard	Categorical variable indicating whether the customer has a credit card (1 for yes, 0 for no).
11.	EstimatedSalary	Estimated salary of the customer.
12.	IsActiveMember	Categorical variable indicating whether the customer is an active member of the bank (1 for yes, 0 for no).
13.	Exited	Categorical variable indicating whether the customer left the bank within six months (1 for yes, 0 for no).
14.	

Churn Modelling - How to predict if a bank’s customer will stay or leave the bank
Using a source of 10,000 bank records, we created an app to demonstrate the ability to apply machine learning models to predict the likelihood of customer churn. We accomplished this using the following steps:

1. Clean the data
By reading the dataset into a dataframe using pandas, we removed unnecessary data fields including individual customer IDs and names. This left us with a list of columns for Credit Score, Geography, Gender, Age, Length of time as a Bank customer, Balance, Number Of Bank Products Used, Has a Credit Card, Is an Active Member, Estimated Salary and Exited.

2. Analyze initial DataFrame
Utilizing Matplotlib, Seaborn and Pandas, we next analyzed the data. We can see that our dataset was imbalanced. The majority class, "Stays" (0), has around 80% data points and the minority class, "Exits" (1), has around 20% datapoints. To address this, we utilized SMOTE in our machine learning algorithms (Synthetic Minority Over-sampling Technique). More on that later on.

In percentage, female customers are more likely to leave the bank at 25%, compared to 16% of males.

The smallest number of customers are from Germany, and they are also the most likely to leave the bank. Almost one in three German customers in our sample left the bank.

3. Machine Learning using 6 different models
We tested seven different machine learning models (and used six in the final application) to predict customer churn, including Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbor, Support Vector Machine and XGBoost.

As mentioned earlier, we also used SMOTE to handle issues with the imbalanced data on and we also use class_weight parameter and it works well


## 🔍 Features

- Data cleaning and preprocessing
- Univariate and multivariate analysis
- Correlation and distribution visualization
- Customer segmentation insights
- Target variable behavior analysis

## 📈 Key Insights

Based on the analysis conducted in `bank_eda.ipynb`, here are some major findings:

- **Class Imbalance:** A strong imbalance in the target variable (`y`), with the majority of customers not subscribing to a term deposit.
- **Age Factor:** Older customers (especially above 60) are more likely to subscribe to term deposits.
- **Previous Contacts Matter:** Customers who were previously contacted and had a positive outcome (`poutcome = success`) show a higher likelihood of subscribing again.
- **Contact Type:** Contacting customers via **cellular** phone leads to more subscriptions compared to telephone.
- **Duration Bias:** Longer call duration correlates with higher success, though this is a post-event variable and must be handled carefully in predictive modeling.
- **Job Influence:** Certain professions (e.g., **retired**, **student**) have higher subscription rates compared to blue-collar or unemployed individuals.
- **Education Level:** Customers with higher education levels have slightly higher conversion rates.
- **Month of Contact:** Most conversions occur during the months of **March**, **May**, **August**, and **October**.

> These insights can be used to refine marketing strategies and optimize campaign timing, audience targeting, and communication channels.

## 📊 Key Libraries Used

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `plotly` (optional)
- `scikit-learn` (optional, for clustering or PCA)


This project focuses on building industry-level machine learning models to predict customer churn for a bank. Churn prediction helps businesses proactively identify customers who are likely to leave and take preventive actions.

---

## 📂 Dataset

- **Source**: `resources/analytical_base_table.csv`
- **Target Variable**: `Exited` (1 = churned, 0 = retained)
- **Features**: Demographic and account activity features.

---

## ⚙️ Modeling Approach

- Used **industry-standard ML pipelines** for consistent preprocessing and modeling.
- Applied **Stratified K-Fold Cross-Validation** to ensure balanced class distribution across folds.
- Addressed **class imbalance** using:
  - `class_weight='balanced'` where applicable.
  - Custom **threshold tuning** for each model post-prediction.
- All models were serialized using `joblib` and stored in the `models/` directory.

---

## 🧪 Models Evaluated

| Model                   | Precision | Recall   | F1 (Macro) | Accuracy | ROC AUC  |
|------------------------|-----------|----------|------------|----------|----------|
| Decision Tree          | 0.4974    | 0.6953   | 0.5799     | 0.7950   | 0.8426   |
| K-Nearest Neighbors    | 0.4512    | 0.6241   | 0.5237     | 0.7690   | 0.8020   |
| Logistic Regression    | 0.5130    | 0.6781   | 0.5841     | 0.8035   | 0.8382   |
| Random Forest          | 0.6250    | 0.6143   | 0.6196     | 0.8465   | 0.8618   |
| XGBoost                | 0.6028    | 0.6339   | 0.6180     | 0.8405   | 0.8639   |
| Support Vector Classifier | 0.5948 | 0.6241   | 0.6091     | 0.8370   | 0.8551   |

---

Model Evaluation Criterion
Recall is prioritized, especially when the cost of false negatives is high. In customer churn prediction, reducing false negatives is crucial as it minimizes missed opportunities to retain customers by providing incentives to prevent them from leaving.

Models
SGD Optimizer: Uses Stochastic Gradient Descent (SGD) to update model weights in small batches, aiming for gradual loss function minimization.

Adam Optimizer: Utilizes Adam optimizer, adapting learning rates for each parameter, leading to faster convergence and improved performance, especially on large datasets.

Adam Optimizer with Dropout (Dropout rate: 0.2): Incorporates dropout regularization, randomly removing 20% of neurons during training to prevent overfitting by reducing dependency on specific neurons.

Adam Optimizer with Hyperparameter Tuning: Employs grid search to find the best hyperparameter combination (batch size and learning rate) for optimizing model performance.

Balanced Data with SMOTE and Adam Optimizer (Oversampling): Addresses class imbalance by oversampling the minority class using SMOTE in combination with Adam optimizer during training, enhancing the model's ability to learn from minority class instances.

Balanced Data with SMOTE and Adam Optimizer (Oversampling)
Metric	Accuracy	Precision	Recall	F1 Score	AUC Score
Score	0.73625	0.413978	0.708589	0.522624	0.87
Final Model
After evaluating all considered models, we determined that the model incorporating the Adam optimizer and dropout rate of 0.2 (Model 3) exhibited the highest recall value. In the context of our specific problem statement, where correctly identifying customers likely to leave is crucial, recall serves as an important metric. Therefore, we select Model 3 as our final model for predicting values on the test set.

Metric	Score
Accuracy	0.73625
Precision	0.413978
Recall	0.708589
F1 Score	0.522624
AUC	0.83
Accuracy (0.73625): The model correctly predicted the customer churn for about 73.625% of the customers in the test set.
Precision (0.413978): When the model predicts a customer will churn, it is correct about 41.3978% of the time. The model may be overestimating customer churn, leading to unnecessary retention efforts.
Recall (0.708589): The model correctly identified 70.8589% of the customers who actually churned. This is crucial to identify as many churning customers as possible.
F1 Score (0.522624): An F1 score of 52.2624% indicates that there is room for improvement in achieving a better balance.
AUC (0.83): An AUC score of 0.83 on the test set indicates relatively good performance of the model in distinguishing between positive and negative instances. Therefore, the model has a high probability of ranking a randomly chosen positive instance higher than a randomly chosen negative instance.

## 🛠️ Technologies Used

- Python
- pandas, numpy
- scikit-learn
- XGBoost
- joblib (for saving models)

---

## 🧮 Evaluation Metrics

- **Precision**: Fraction of true churn predictions out of all churn predictions.
- **Recall**: Fraction of actual churners identified.
- **F1 Macro**: Harmonic mean of precision and recall across both classes.
- **Accuracy**: Overall correct predictions.
- **ROC AUC**: Model’s ability to distinguish churners vs. non-churners.

---

nsights and Recommendations
Engagement of Dormant Members
The bank may consider launching a campaign to re-engage dormant members and convert them into active clients. This could involve reaching out to them with exclusive deals, incentives, or personalized financial guidance to help them make the most out of their accounts.

Product Retention and Diversification
Encouraging customers to diversify their product holdings could be beneficial, especially considering that a significant proportion (51%) of customers only own one product. Implementing retention strategies to retain clients with multiple products, such as offering incentives or bundled services, could be effective.

Services Tailored to Age
Given the positive correlation between leaving a bank and age, the bank should consider offering age-specific services or incentives to retain customers across different age groups. Customizing services to cater to various life stages could enhance client retention.

Retention Strategies Based on Tenure
Customers with shorter tenures, specifically one year and zero years, exhibit higher rates of churn. Implementing promotions, personalized services, or onboarding programs targeted at acquiring and retaining customers during the early years of their banking relationship could mitigate churn.



