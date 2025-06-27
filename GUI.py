import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Define the path to the model and scaler files
MODEL_PATH = 'best_fraud_detection_model_SVM.pkl'
SCALER_PATH = 'standard_scaler.pkl'

# Function to preprocess the data
def preprocess_data(df, scaler):
    # Eliminar columnas no usadas por el modelo
    df = df.drop(columns=['newbalanceDest', 'oldbalanceDest', 'step', 'nameOrig', 'nameDest', 'isFlaggedFraud'])

    # Convertir columnas categóricas
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category')

    # One-hot encoding para la columna 'type'
    df = pd.get_dummies(df, columns=['type'], prefix='type', drop_first=False)

    # Asegurar que las columnas dummy sean booleanas
    dummy_cols = [col for col in df.columns if col.startswith('type_')]
    df[dummy_cols] = df[dummy_cols].astype(bool)

    # Escalar columnas numéricas (incluyendo oldbalanceOrg)
    df[['amount', 'newbalanceOrig', 'oldbalanceOrg']] = scaler.transform(
        df[['amount', 'newbalanceOrig', 'oldbalanceOrg']]
    )

    # Añadir columnas dummy faltantes como False (en caso de que falte alguna categoría)
    expected_dummies = ['type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']
    for col in expected_dummies:
        if col not in df.columns:
            df[col] = False

    return df

# Cargar el modelo y el escalador
try:
    loaded_model = joblib.load(MODEL_PATH)
    loaded_scaler = joblib.load(SCALER_PATH)
except FileNotFoundError:
    st.error("❌ No se encontraron los archivos del modelo o del escalador.")
    st.stop()

# Título de la app
st.title("🔍 Detección de Fraude en Transacciones")
st.write("Esta aplicación predice si una transacción es fraudulenta con base en los datos proporcionados.")

# Subida de archivo
uploaded_file = st.file_uploader("📁 Sube tu archivo Excel (.xlsx) con transacciones", type=["xlsx"])

if uploaded_file is not None:
    try:
        # Leer el archivo
        df = pd.read_excel(uploaded_file)
        st.subheader("✅ Datos cargados:")
        st.write(df.head())

        # Preprocesamiento
        st.subheader("⚙️ Preprocesando datos...")
        df_processed = preprocess_data(df.copy(), loaded_scaler)

        # Reordenar columnas en el orden exacto del entrenamiento
        df_processed = df_processed[loaded_model.feature_names_in_]

        st.write("🔎 Datos preprocesados:")
        st.write(df_processed.head())

        # Predicción
        st.subheader("📊 Realizando predicciones...")
        predictions = loaded_model.predict(df_processed)

        # Agregar resultados al DataFrame original
        df['Fraude_Predicho'] = predictions

        st.subheader("📋 Resultados:")
        st.write(df[['amount', 'type', 'Fraude_Predicho']].head())

        # Mostrar resumen
        fraud_count = df['Fraude_Predicho'].sum()
        st.success(f"✅ Total de transacciones predichas como fraudulentas: {fraud_count}")

    except Exception as e:
        st.error(f"❌ Ocurrió un error durante el procesamiento: {e}")
