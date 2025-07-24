import streamlit as st
import pickle
import numpy as np
import pandas as pd


with open("logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model_columns.pkl", "rb") as f:
    model_columns = pickle.load(f)

st.set_page_config(page_title="📉 Customer Churn Prediction", layout="centered")    
st.title("📉 Customer Churn Prediction App")
st.markdown("Provide customer details below to predict churn.")

gender = st.selectbox("Gender", ["Male", "Female"])
tenure = st.slider("Tenure (months)", 0, 72, 12)
Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
MonthlyCharges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)

# Create input dataframe
input_dict = {
    "gender": gender,
    "tenure": tenure,
    "Contract": Contract,
    "MonthlyCharges": MonthlyCharges,
}

input_df = pd.DataFrame([input_dict])

# Match training format using same dummy variables
input_df = pd.get_dummies(input_df)
input_df = input_df.reindex(columns=model_columns, fill_value=0)

# Predict
if st.button("Predict Churn"):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.warning("⚠️ This customer is likely to churn.")
    else:
        st.success("✅ This customer is likely to stay.")