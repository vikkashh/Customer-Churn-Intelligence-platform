import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load('churn_rf_model.pkl')

st.title("Customer Churn Prediction")

st.markdown("Fill out the following details to predict if the customer is likely to churn.")

# ========== USER INPUT SECTION ==========

# Numeric Inputs
senior_citizen = st.selectbox("Is the customer a Senior Citizen?", [0, 1])
monthly_charges = st.number_input("Monthly Charges", min_value=0.0)
total_charges = st.number_input("Total Charges", min_value=0.0)

# Binary Encoded Inputs
gender = st.selectbox("Gender", ["Female", "Male"])
partner = st.selectbox("Has Partner?", ["Yes", "No"])
dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
phone_service = st.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.selectbox("Payment Method", [
    "Bank transfer (automatic)", 
    "Credit card (automatic)", 
    "Electronic check", 
    "Mailed check"
])
tenure_group = st.selectbox("Tenure Group", [
    "1-12", "13-24", "25-36", "37-48", "49-60", "61-72"
])

# ========== FEATURE VECTOR BUILDING ==========

# One-hot vector creation
def build_input_vector():
    cols = [
        'SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender_Female',
        'gender_Male', 'Partner_No', 'Partner_Yes', 'Dependents_No',
        'Dependents_Yes', 'PhoneService_No', 'PhoneService_Yes',
        'MultipleLines_No', 'MultipleLines_No phone service',
        'MultipleLines_Yes', 'InternetService_DSL',
        'InternetService_Fiber optic', 'InternetService_No',
        'OnlineSecurity_No', 'OnlineSecurity_No internet service',
        'OnlineSecurity_Yes', 'OnlineBackup_No',
        'OnlineBackup_No internet service', 'OnlineBackup_Yes',
        'DeviceProtection_No', 'DeviceProtection_No internet service',
        'DeviceProtection_Yes', 'TechSupport_No',
        'TechSupport_No internet service', 'TechSupport_Yes', 'StreamingTV_No',
        'StreamingTV_No internet service', 'StreamingTV_Yes',
        'StreamingMovies_No', 'StreamingMovies_No internet service',
        'StreamingMovies_Yes', 'Contract_Month-to-month', 'Contract_One year',
        'Contract_Two year', 'PaperlessBilling_No', 'PaperlessBilling_Yes',
        'PaymentMethod_Bank transfer (automatic)',
        'PaymentMethod_Credit card (automatic)',
        'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check',
        'tenure_group_1-12', 'tenure_group_13-24', 'tenure_group_25-36',
        'tenure_group_37-48', 'tenure_group_49-60', 'tenure_group_61-72'
    ]
    
    data = [0] * len(cols)

    # Numeric fields
    data[cols.index("SeniorCitizen")] = senior_citizen
    data[cols.index("MonthlyCharges")] = monthly_charges
    data[cols.index("TotalCharges")] = total_charges

    # One-hot encodings
    def set_col(option, prefix):
        for col in cols:
            if col.startswith(prefix) and col.endswith(option):
                data[cols.index(col)] = 1

    set_col(gender, "gender_")
    set_col(partner, "Partner_")
    set_col(dependents, "Dependents_")
    set_col(phone_service, "PhoneService_")
    set_col(multiple_lines, "MultipleLines_")
    set_col(internet_service, "InternetService_")
    set_col(online_security, "OnlineSecurity_")
    set_col(online_backup, "OnlineBackup_")
    set_col(device_protection, "DeviceProtection_")
    set_col(tech_support, "TechSupport_")
    set_col(streaming_tv, "StreamingTV_")
    set_col(streaming_movies, "StreamingMovies_")
    set_col(contract, "Contract_")
    set_col(paperless_billing, "PaperlessBilling_")
    set_col(payment_method, "PaymentMethod_")
    set_col(tenure_group, "tenure_group_")

    return np.array([data])

# ========== PREDICTION ==========

if st.button("Predict Churn"):
    features = build_input_vector()
    prediction = model.predict(features)

    if prediction[0] == 1:
        st.error("⚠ This customer is likely to churn.")
    else:
        st.success("✅ This customer is not likely to churn.")
