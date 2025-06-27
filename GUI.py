
import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler # Import StandardScaler

# Define the path to the model and scaler files
MODEL_PATH = 'best_fraud_detection_model_Naive_Bayes.pkl'
SCALER_PATH = 'standard_scaler.pkl'

# Function to preprocess the data (adapted for the new dataset)
def preprocess_data(df, scaler):
    # Drop specified columns
    df = df.drop(columns=['newbalanceDest', 'oldbalanceDest', 'oldbalanceOrg', 'step', 'nameOrig', 'nameDest', 'isFlaggedFraud'])

    # Convert object type columns to category
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category')

    # Apply One-Hot Encoding to the 'type' column
    df = pd.get_dummies(df, columns=['type'], prefix='type', drop_first=False)

    # Apply Scaling to 'amount' and 'newbalanceOrig'
    # Note: In a real application, the scaler should be fit on the training data
    # and only transformed on the new data. Assuming the provided scaler is already fit.
    df[['amount', 'newbalanceOrig']] = scaler.transform(df[['amount', 'newbalanceOrig']])

    # Handle potential missing columns after one-hot encoding
    # This assumes the training data had the same 'type' categories.
    # If not, you might need a more robust way to handle this (e.g., reindexing).
    expected_columns = ['amount', 'newbalanceOrig', 'type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0 # Add missing columns with a default value of 0

    # Ensure columns are in the same order as the training data
    df = df[expected_columns] # This assumes the expected_columns list is in the correct order

    return df

# Load the pre-trained model and scaler
try:
    loaded_model = joblib.load(MODEL_PATH)
    loaded_scaler = joblib.load(SCALER_PATH)
except FileNotFoundError:
    st.error("Error: Model or scaler files not found. Make sure they are in the correct directory.")
    st.stop()

# Streamlit App Title
st.title("Detección de Fraude en Transacciones")
st.write("""
Esta aplicación predice si una transacción es fraudulenta basado en los datos proporcionados.
""")

# File uploader for the user to upload their data
uploaded_file = st.file_uploader("Sube tu archivo Excel (solo .xlsx) con datos de transacciones", type=["xlsx"])

if uploaded_file is not None:
    try:
        # Read the uploaded Excel file into a pandas DataFrame
        df = pd.read_excel(uploaded_file)
        st.subheader("Datos cargados:")
        st.write(df.head())

        # Preprocess the uploaded data
        st.subheader("Preprocesando datos...")
        df_processed = preprocess_data(df.copy(), loaded_scaler) # Use .copy() to avoid modifying the original DataFrame
        st.write("Datos preprocesados (primeras filas):")
        st.write(df_processed.head())

        # Make predictions
        st.subheader("Realizando predicciones...")
        predictions = loaded_model.predict(df_processed)

        # Add predictions to the original DataFrame (aligning by index)
        df['Fraude_Predicho'] = predictions

        st.subheader("Resultados de la Predicción:")
        st.write(df[['amount', 'type', 'Fraude_Predicho']].head()) # Display some key columns and the prediction

        # You can also display a summary of predictions
        fraud_count = df['Fraude_Predicho'].sum()
        st.write(f"Número total de transacciones predichas como fraudulentas: {fraud_count}")

    except Exception as e:
        st.error(f"Ocurrió un error durante el procesamiento: {e}")
