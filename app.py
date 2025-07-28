import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd

# Load model, scaler, and column list
model = joblib.load("model.pkl")
model_columns=["step", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "balance_change", "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER"]

st.set_page_config(page_title="FraudGuard", page_icon="🛡️")
st.title("🛡️ FraudGuard")
st.subheader("Transaction Fraud Detection")
st.markdown("Fill in the transaction details below:")

# Input form
with st.form("fraud_form"):
    col1, col2 = st.columns(2)

    amount = col1.number_input("Transaction Amount ($)", min_value=0.0, step=0.01)
    step = col2.number_input("Step (Time Step)", min_value=0)

    oldbalanceOrg = col1.number_input("Old Balance (Origin)", min_value=0.0, step=0.01)
    newbalanceOrig = col2.number_input("New Balance (Origin)", min_value=0.0, step=0.01)

    oldbalanceDest = col1.number_input("Old Balance (Destination)", min_value=0.0, step=0.01)
    newbalanceDest = col2.number_input("New Balance (Destination)", min_value=0.0, step=0.01)

    transfer_type = col1.selectbox("Transaction Type", ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"])

    submit = st.form_submit_button("🔍 Predict Fraud")

if submit:
    # Prepare input dictionary
    input_dict = {
        "step": step,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest,
    }

    # One-hot encode transaction type
    for t in ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]:
        input_dict[f"type_{t}"] = 1 if transfer_type == t else 0
        

    # If model uses balance_change, compute it
    if "balance_change" in model_columns:
        input_dict["balance_change"] = oldbalanceOrg - newbalanceOrig

    # Ensure all columns are present and in correct order
    X = pd.DataFrame([[input_dict.get(col, 0) for col in model_columns]], columns=model_columns)
    # X_scaled = scaler.transform(X)
    prob = model.predict_proba(X)[0, 1]
    pred = int(prob > 0.5)

    # Output
    st.markdown("---")
    if pred == 1:
        st.error(f"⚠️ Fraudulent Transaction Detected with {prob*100:.2f}% confidence.")
    else:
        st.success(f"✅ Legitimate Transaction with {(1 - prob)*100:.2f}% confidence.")
    if pred == 1:
        st.error(f"⚠️ Fraudulent Transaction Detected with {prob*100:.2f}% confidence.")
    else:
        st.success(f"✅ Legitimate Transaction with {(1 - prob)*100:.2f}% confidence.")
