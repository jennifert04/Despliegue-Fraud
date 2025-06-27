import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Define the path to the model and scaler files
MODEL_PATH = 'best_fraud_detection_model_SVM.pkl'
SCALER_PATH = 'standard_scaler.pkl'

# Function to preprocess the data (adapted for the new dataset)
def preprocess_data(df, scaler):
    # Elimina solo columnas irrelevantes
    df = df.drop(columns=['newbalanceDest', 'oldbalanceDest', 'step', 'nameOrig', 'nameDest', 'isFlaggedFraud'])

    # Convertir columnas tipo object a categoría
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category')

    # One-hot encoding para la columna 'type'
    df = pd.get_dummies(df, columns=['type'], prefix='type', drop_first=False)

    # Asegura que las columnas dummy sean booleanas
    dummy_cols = [col for col in df.columns if col.startswith('type_')]
    df[dummy_cols] = df[dummy_cols].astype(bool)

    # Escalar variables numéricas: incluye oldbalanceOrg
    df[['amount', 'newbalanceOrig', 'oldbalanceOrg']] = scaler.transform(
        df[['amount', 'newbalanceOrig', 'oldbalanceOrg']]
    )

    # Lista de columnas esperadas por el modelo
    expected_columns = ['amount', 'newbalanceOrig', 'oldbalanceOrg',
                        'type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']

    # Añadir columnas faltantes como False
    for col in expected_columns:
        if col not in df.columns:
            df[col] = False

    # Reordenar columnas
    df = df[expected_columns]

    return df

# Load the pre-trained model and scaler
try:
    loaded_model = joblib.load(MODEL_PATH)
    loaded_scaler = joblib.load(SCALER_PATH)
except FileNotFoundError:
    st.error("Error: No se encontraron los archivos del modelo o del escalador. Verifica que estén en el directorio correcto.")
    st.stop()

# Streamlit App Title
st.title("🔍 Detección de Fraude en Transacciones")
st.write("""
Esta aplicación predice si una transacción es fraudulenta, basándose en los datos que cargues en formato Excel.
""")

# File uploader for the user to upload their data
uploaded_file = st.file_uploader("📁 Sube tu archivo Excel (.xlsx) con transacciones", type=["xlsx"])

if uploaded_file is not None:
    try:
        # Read the uploaded Excel file into a pandas DataFrame
        df = pd.read_excel(uploaded_file)
        st.subheader("✅ Datos cargados:")
        st.write(df.head())

        # Preprocess the uploaded data
        st.subheader("⚙️ Preprocesando datos...")
        df_processed = preprocess_data(df.copy(), loaded_scaler)
        st.write("🔎 Vista de datos preprocesados:")
        st.write(df_processed.head())

        # Make predictions
        st.subheader("📊 Realizando predicciones...")
        predictions = loaded_model.predict(df_processed)

        # Add predictions to the original DataFrame
        df['Fraude_Predicho'] = predictions

        st.subheader("📋 Resultados de la Predicción:")
        st.write(df[['amount', 'type', 'Fraude_Predicho']].head())

        # Display summary
        fraud_count = df['Fraude_Predicho'].sum()
        st.success(f"Número total de transacciones predichas como fraudulentas: {fraud_count}")

    except Exception as e:
        st.error(f"Ocurrió un error durante el procesamiento: {e}")
