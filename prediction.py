import streamlit as st
import pandas as pd
import numpy as np
import joblib

from tensorflow.keras.models import load_model

# ==================================
# PAGE CONFIG
# ==================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📞",
    layout="wide"
)

# ==================================
# LOAD FILES
# ==================================

model = load_model("churn_model.h5")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")

# ==================================
# TITLE
# ==================================

st.title("📞 AI-Powered Customer Churn Prediction")
st.markdown(
    "Predict whether a telecom customer is likely to churn."
)

st.divider()

# ==================================
# CUSTOMER DETAILS
# ==================================

col1, col2, col3 = st.columns(3)

with col1:

    gender = st.selectbox(
        "Gender",
        ['Female', 'Male']
    )

    senior = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

    partner = st.selectbox(
        "Partner",
        ['No', 'Yes']
    )

    dependents = st.selectbox(
        "Dependents",
        ['No', 'Yes']
    )

    tenure = st.number_input(
        "Tenure (Months)",
        min_value=0,
        max_value=100,
        value=12
    )

with col2:

    phone = st.selectbox(
        "Phone Service",
        ['No', 'Yes']
    )

    multiple = st.selectbox(
        "Multiple Lines",
        ['No', 'No phone service', 'Yes']
    )

    internet = st.selectbox(
        "Internet Service",
        ['DSL', 'Fiber optic', 'No']
    )

    security = st.selectbox(
        "Online Security",
        ['No', 'No internet service', 'Yes']
    )

    backup = st.selectbox(
        "Online Backup",
        ['No', 'No internet service', 'Yes']
    )

with col3:

    protection = st.selectbox(
        "Device Protection",
        ['No', 'No internet service', 'Yes']
    )

    support = st.selectbox(
        "Tech Support",
        ['No', 'No internet service', 'Yes']
    )

    tv = st.selectbox(
        "Streaming TV",
        ['No', 'No internet service', 'Yes']
    )

    movies = st.selectbox(
        "Streaming Movies",
        ['No', 'No internet service', 'Yes']
    )

    contract = st.selectbox(
        "Contract Type",
        ['Month-to-month', 'One year', 'Two year']
    )

paperless = st.selectbox(
    "Paperless Billing",
    ['No', 'Yes']
)

payment = st.selectbox(
    "Payment Method",
    [
        'Bank transfer (automatic)',
        'Credit card (automatic)',
        'Electronic check',
        'Mailed check'
    ]
)

monthly = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=50.0
)

total = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=500.0
)

# ==================================
# PREDICTION
# ==================================

if st.button("Predict Churn"):

    input_data = pd.DataFrame({
        'gender':[gender],
        'SeniorCitizen':[senior],
        'Partner':[partner],
        'Dependents':[dependents],
        'tenure':[tenure],
        'PhoneService':[phone],
        'MultipleLines':[multiple],
        'InternetService':[internet],
        'OnlineSecurity':[security],
        'OnlineBackup':[backup],
        'DeviceProtection':[protection],
        'TechSupport':[support],
        'StreamingTV':[tv],
        'StreamingMovies':[movies],
        'Contract':[contract],
        'PaperlessBilling':[paperless],
        'PaymentMethod':[payment],
        'MonthlyCharges':[monthly],
        'TotalCharges':[total]
    })

    categorical_cols = [
        'gender',
        'Partner',
        'Dependents',
        'PhoneService',
        'MultipleLines',
        'InternetService',
        'OnlineSecurity',
        'OnlineBackup',
        'DeviceProtection',
        'TechSupport',
        'StreamingTV',
        'StreamingMovies',
        'Contract',
        'PaperlessBilling',
        'PaymentMethod'
    ]

    for col in categorical_cols:
        input_data[col] = encoders[col].transform(
            input_data[col]
        )

    scaled_data = scaler.transform(input_data)

    prediction = model.predict(
        scaled_data
    )

    probability = prediction[0][0]

    st.divider()

    st.subheader("Prediction Result")

    st.metric(
        "Churn Probability",
        f"{probability*100:.2f}%"
    )

    if probability > 0.5:

        st.error(
            "⚠️ High Risk Customer - Likely to Churn"
        )

    else:

        st.success(
            "✅ Customer Likely to Stay"
        )