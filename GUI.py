# prompt: haz el despliegue anterior pero en streamlit, con los campos para ingresar los datos, no con archivo

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Define the path to the scaler and model files
SCALER_PATH = 'standard_scaler.pkl'
MODEL_PATH = 'best_fraud_detection_model_SVM.pkl'

# Load the scaler and the model
try:
    loaded_scaler = joblib.load(SCALER_PATH)
    loaded_model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    st.error("Scaler or Model file not found. Make sure 'standard_scaler.pkl' and 'best_fraud_detection_model_SVM.pkl' are in the same directory as your Streamlit app.")
    st.stop()


st.title("Predicción de Fraude en Transacciones")

st.write("""
Esta aplicación predice si una transacción es fraudulenta basándose en los datos proporcionados.
Por favor, ingresa los detalles de la transacción a continuación:
""")

# Input fields for the user
amount = st.number_input("Monto de la transacción", min_value=0.0, format="%.2f")
type_of_transaction = st.selectbox("Tipo de transacción", ['CASH_OUT', 'PAYMENT', 'CASH_IN', 'TRANSFER', 'DEBIT'])
newbalanceOrig = st.number_input("Nuevo saldo del originador", min_value=0.0, format="%.2f")

# Create a dictionary with the input data
input_data = {
    'amount': amount,
    'type': type_of_transaction,
    'newbalanceOrig': newbalanceOrig,
}

# Convert input data to DataFrame
input_df = pd.DataFrame([input_data])

# Apply one-hot encoding to the 'type' column
input_df = pd.get_dummies(input_df, columns=['type'], prefix='type', drop_first=False)

# Ensure all type columns from training data are present and in the correct order
# This is important because get_dummies might not create columns for all categories if they aren't in the single input row
# You might need to know all possible categories from your training data here
all_types = ['type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']
for type_col in all_types:
    if type_col not in input_df.columns:
        input_df[type_col] = 0

# Reorder columns to match the order used during model training
# This requires knowing the exact column order your model was trained on
# For simplicity, let's assume the order was amount, newbalanceOrig, followed by the type columns
# You should adjust this based on your actual training data's column order
# A better approach would be to save the list of columns after preprocessing the training data
expected_columns = ['amount', 'newbalanceOrig'] + all_types
input_df = input_df[expected_columns]


# Apply the same scaling as the training data
try:
    input_df[['amount', 'newbalanceOrig']] = loaded_scaler.transform(input_df[['amount', 'newbalanceOrig']])
except Exception as e:
    st.error(f"Error during scaling: {e}")
    st.stop()


# Make prediction when button is clicked
if st.button("Predecir Fraude"):
    try:
        prediction = loaded_model.predict(input_df)

        st.subheader("Resultado de la Predicción:")
        if prediction[0] == 1:
            st.error("¡Alerta! Es probable que esta transacción sea FRAUDULENTA.")
        else:
            st.success("Esta transacción parece ser LEGÍTIMA.")

    except Exception as e:
        st.error(f"Ocurrió un error durante la predicción: {e}")

