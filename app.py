import streamlit as st
import numpy as np
import joblib
import pandas as pd


dt_model = joblib.load(r"models/dtree.pkl")
rf_model = joblib.load(r"models/rforest_model.pkl")
lr_model = joblib.load(r"models/log_model.pkl")
xgb_model = joblib.load(r"models/xgb_model.pkl")
knn_model = joblib.load(r"models/knn_model.pkl")
svc_model = joblib.load(r"models/svc_model.pkl")
loaded_models = {
    'Decision Tree': dt_model,
    'K-Nearest Neighbors': knn_model,
    'Logistic Regression': lr_model,
    'Random Forest': rf_model,
    'XGBoost': xgb_model,
    'Support Vector Classifier':svc_model
}

# Decode prediction
def decode(pred):
    return 'Customer Exits' if pred == 1 else 'Customer Stays'

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")
st.title("🔍 Customer Churn Prediction (Multiple Models)")

with st.form("customer_form"):
    st.subheader("Enter Customer Details")

    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    tenure = st.slider("Tenure", 0, 10, 5)
    balance = st.number_input("Balance", min_value=0.0, value=50000.0)
    num_of_products = st.selectbox("Number of Products", [1, 2, 3, 4])
    has_cr_card = st.radio("Has Credit Card?", ["Yes", "No"])
    is_active = st.radio("Is Active Member?", ["Yes", "No"])
    est_salary = st.number_input("Estimated Salary", min_value=0.0, value=60000.0)

    submitted = st.form_submit_button("Predict")


if submitted:
    input_df = pd.DataFrame([{
        'CreditScore': credit_score,
        'Geography': geography,
        'Gender': gender,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_of_products,
        'HasCrCard': has_cr_card,
        'IsActiveMember': is_active,
        'EstimatedSalary': est_salary
    }])

    st.subheader("📊 Predictions")
    results = []

    for name, bundle in loaded_models.items():
        model = bundle['model']
        threshold = bundle['threshold']
        prob = model.predict_proba(input_df)[0][1]
        pred = 1 if prob >= threshold else 0
        results.append((name, decode(pred)))

    for model_name, prediction in results:
        st.write(f"**{model_name}:** {prediction}")
