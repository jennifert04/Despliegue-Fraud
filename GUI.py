import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Paths a los archivos
MODEL_PATH = 'best_fraud_detection_model_SVM.pkl'
SCALER_PATH = 'standard_scaler.pkl'

# Función de preprocesamiento
def preprocess_data(df, scaler):
    # Eliminar columnas irrelevantes
    df = df.drop(columns=['newbalanceDest', 'oldbalanceDest', 'step', 'nameOrig', 'nameDest', 'isFlaggedFraud'])

    # Convertir columnas categóricas
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category')

    # One-hot encoding para la columna 'type'
    df = pd.get_dummies(df, columns=['type'], prefix='type', drop_first=False)

    # Asegurar que las dummies sean tipo bool
    dummy_cols = [col for col in df.columns if col.startswith('type_')]
    df[dummy_cols] = df[dummy_cols].astype(bool)

    # Escalar numéricas (incluye oldbalanceOrg)
    df[['amount', 'newbalanceOrig', 'oldbalanceOrg']] = scaler.transform(
        df[['amount', 'newbalanceOrig', 'oldbalanceOrg']]
    )

    # Agregar columnas dummy faltantes
    expected_dummies = ['type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']
    for col in expected_dummies:
        if col not in df.columns:
            df[col] = False

    return df

# Cargar modelo y scaler
try:
    loaded_model = joblib.load(MODEL_PATH)
    loaded_scaler = joblib.load(SCALER_PATH)
except FileNotFoundError:
    st.error("❌ No se encontraron los archivos del modelo o escalador.")
    st.stop()

# App
st.title("🔍 Detección de Fraude en Transacciones")

uploaded_file = st.file_uploader("📁 Sube tu archivo Excel (.xlsx) con transacciones", type=["xlsx"])

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
        st.subheader("✅ Datos cargados:")
        st.write(df.head())

        st.subheader("⚙️ Preprocesando datos...")
        df_processed = preprocess_data(df.copy(), loaded_scaler)

        # ⛑️ Reordenar columnas EXACTAMENTE como en el entrenamiento
        df_processed = df_processed.loc[:, loaded_model.feature_names_in_]

        st.write("🔎 Datos preprocesados:")
        st.write(df_processed.head())

        st.subheader("📊 Realizando predicciones...")
        predictions = loaded_model.predict(df_processed)

        df['Fraude_Predicho'] = predictions

        st.subheader("📋 Resultados:")
        st.write(df[['amount', 'type', 'Fraude_Predicho']].head())

        fraud_count = df['Fraude_Predicho'].sum()
        st.success(f"✅ Total de transacciones predichas como fraudulentas: {fraud_count}")

    except Exception as e:
        st.error(f"❌ Ocurrió un error durante el procesamiento: {e}")
