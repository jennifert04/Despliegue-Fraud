# prompt: haz el despliegue anterior pero en streamlit, quiero que convierta las entradas amount y newbalanceOrig en float siempre

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Define the path to the scaler and model files
scaler_filename = 'scaler_fraud.pkl'
model_filename = 'best_fraud_detection_model_SVM.pkl'

# Load the scaler and the model
try:
    loaded_scaler = joblib.load(scaler_filename)
    loaded_model = joblib.load(model_filename)
except FileNotFoundError:
    st.error("Error: Model or scaler files not found. Make sure 'scaler_fraud.pkl' and 'best_fraud_detection_model_SVM.pkl' are in the same directory as the script.")
    st.stop()

st.title('Fraud Detection App')

st.write("""
This app predicts whether a transaction is fraudulent or not based on the input features.
""")

# Create input fields for the features
amount = st.number_input('Amount', value=0.0)
newbalanceOrig = st.number_input('New Balance Original', value=0.0)
type_cash_in = st.selectbox('Transaction Type: CASH_IN', [0, 1])
type_cash_out = st.selectbox('Transaction Type: CASH_OUT', [0, 1])
type_debit = st.selectbox('Transaction Type: DEBIT', [0, 1])
type_payment = st.selectbox('Transaction Type: PAYMENT', [0, 1])
type_transfer = st.selectbox('Transaction Type: TRANSFER', [0, 1])

# Create a DataFrame from the input
input_data = pd.DataFrame({
    'amount': [float(amount)],
    'newbalanceOrig': [float(newbalanceOrig)],
    'type_CASH_IN': [type_cash_in],
    'type_CASH_OUT': [type_cash_out],
    'type_DEBIT': [type_debit],
    'type_PAYMENT': [type_payment],
    'type_TRANSFER': [type_transfer]
})

# Ensure all dummy columns are present, even if the type is not selected
# Get the list of columns the scaler/model expects based on the training data
# This assumes the scaler and model were trained on data with these columns
# In a real scenario, you might want to inspect the model/scaler for feature names
expected_cols = ['amount', 'newbalanceOrig', 'type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']
for col in expected_cols:
    if col not in input_data.columns:
        input_data[col] = 0

# Reorder columns to match the training data
input_data = input_data[expected_cols]

# Scale the 'amount' and 'newbalanceOrig' features
try:
    input_data[['amount', 'newbalanceOrig']] = loaded_scaler.transform(input_data[['amount', 'newbalanceOrig']])
except Exception as e:
    st.error(f"Error during scaling: {e}")
    st.stop()

# Make prediction
if st.button('Predict'):
    try:
        prediction = loaded_model.predict(input_data)

        if prediction[0] == 1:
            st.error('Prediction: Fraudulent Transaction')
        else:
            st.success('Prediction: Non-Fraudulent Transaction')
    except Exception as e:
        st.error(f"Error during prediction: {e}")

