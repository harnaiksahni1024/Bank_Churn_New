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
