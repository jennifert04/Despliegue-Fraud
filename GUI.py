import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler

# Paths del modelo y el escalador
MODEL_PATH = 'best_fraud_detection_model_SVM.pkl'
SCALER_PATH = 'standard_scaler.pkl'

# Preprocesamiento con control total de tipos y orden
def preprocess_data(df, scaler, expected_columns):
    # Eliminar columnas irrelevantes
    df = df.drop(columns=['newbalanceDest', 'oldbalanceDest', 'step', 'nameOrig', 'nameDest', 'isFlaggedFraud'])

    # One-hot encoding en 'type'
    df = pd.get_dummies(df, columns=['type'], prefix='type', drop_first=False)

    # Asegurar columnas dummy como booleanas
    for col in df.columns:
        if col.startswith("type_"):
            df[col] = df[col].astype(bool)

    # Agregar columnas faltantes
    for col in expected_columns:
        if col not in df.columns:
            df[col] = False if "type_" in col else 0.0

    # Escalar numéricas
    df[['amount', 'newbalanceOrig', 'oldbalanceOrg']] = scaler.transform(
        df[['amount', 'newbalanceOrig', 'oldbalanceOrg']]
    )

    # Asegurar orden exacto
    df = df.loc[:, expected_columns]

    return df

# Cargar modelo y scaler
try:
    loaded_model = joblib.load(MODEL_PATH)
    loaded_scaler = joblib.load(SCALER_PATH)
except FileNotFoundError:
    st.error("❌ No se encontraron los archivos del modelo o escalador.")
    st.stop()

# Obtener columnas que el modelo espera (en orden)
expected_columns = list(loaded_model.feature_names_in_)

# App Streamlit
st.title("🔍 Detección de Fraude en Transacciones")

uploaded_file = st.file_uploader("📁 Sube tu archivo Excel (.xlsx) con transacciones", type=["xlsx"])

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
        st.subheader("✅ Datos cargados:")
        st.write(df.head())

        st.subheader("⚙️ Preprocesando datos...")
        df_processed = preprocess_data(df.copy(), loaded_scaler, expected_columns)

        # Confirmar orden y tipo antes de predecir
        assert list(df_processed.columns) == expected_columns, "Orden de columnas inconsistente"
        assert all(df_processed.dtypes == pd.Series({col: loaded_model.feature_names_in_.dtype for col in expected_columns})), "Tipos inconsistentes"

        st.write("🔎 Datos listos para predecir:")
        st.write(df_processed.head())

        # Predicciones
        predictions = loaded_model.predict(df_processed)
        df['Fraude_Predicho'] = predictions

        st.subheader("📋 Resultados:")
        st.write(df[['amount', 'type', 'Fraude_Predicho']].head())
        st.success(f"✅ Total de transacciones predichas como fraudulentas: {df['Fraude_Predicho'].sum()}")

    except Exception as e:
        st.error(f"❌ Ocurrió un error durante el procesamiento: {e}")
