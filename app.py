import streamlit as st
import pandas as pd
import joblib
from churn_predictor import preprocess_data

# Load model and training column names
model = joblib.load("churn_model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("📉 Customer Churn Prediction App")
st.write("Enter customer details to predict if they are likely to churn.")

# Input form
with st.form("churn_form"):
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly_charges = st.slider("Monthly Charges", 0.0, 150.0, 70.0)
    total_charges = st.slider("Total Charges", 0.0, 10000.0, 2000.0)

    # Submit button
    submitted = st.form_submit_button("Predict Churn")

# When form is submitted
if submitted:
    input_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [1 if senior == "Yes" else 0],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "TechSupport": [tech_support],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })

    # Preprocess the data
    _, X_input, _, _ = preprocess_data(input_data)

    # Match model columns
    for col in model_columns:
        if col not in X_input.columns:
            X_input[col] = 0
    X_input = X_input[model_columns]

    # Make prediction
    prediction = model.predict(X_input)
    probability = model.predict_proba(X_input)[0][1]  # Churn probability

    result = "Yes ✅" if prediction[0] == 1 else "No ❌"
    confidence = round(probability * 100, 2)

    # Display results
    st.subheader(f"**Will the customer churn? → {result}**")
    st.write(f"**Model Confidence:** {confidence}%")
