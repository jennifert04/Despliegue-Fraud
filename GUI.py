# prompt: despliegue completo en Streamlit para predicción de fraude con escalado y diagnóstico
import streamlit as st
import pandas as pd
import os
import joblib

# Título de la app
st.title("🛡️ Detección de Fraude en Transacciones")

st.write("""
Esta aplicación predice si una transacción es fraudulenta usando un modelo de Machine Learning.
""")

# Cargar archivos del modelo y scaler
scaler_filename = 'standard_scaler.pkl'  # Asegúrate que este scaler fue entrenado con .fit() y guardado con joblib
model_filename = 'best_fraud_detection_model_SVM.pkl'

if not os.path.exists(scaler_filename) or not os.path.exists(model_filename):
    st.error("❌ No se encuentran los archivos del modelo o del escalador.")
    st.stop()

# Cargar scaler y modelo
scaler = joblib.load(scaler_filename)
model = joblib.load(model_filename)

# Subir archivo
uploaded_file = st.file_uploader("📁 Sube tu archivo Excel (.xlsx) con transacciones", type=['xlsx'])

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)

        st.subheader("📄 Datos cargados:")
        st.write(df.head())

        # Verificar tipos de columnas relevantes
        st.subheader("🔍 Tipos de datos originales:")
        st.write(df.dtypes)

        # Asegurar que columnas sean numéricas
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        df['newbalanceOrig'] = pd.to_numeric(df['newbalanceOrig'], errors='coerce')

        # Mostrar si hay valores nulos
        st.subheader("🧪 Valores nulos tras conversión a numérico:")
        st.write(df[['amount', 'newbalanceOrig']].isnull().sum())

        # Eliminar columnas no necesarias (si existen)
        df = df.drop(columns=['newbalanceDest', 'oldbalanceDest', 'oldbalanceOrg', 'step', 'nameOrig', 'nameDest', 'isFlaggedFraud'], errors='ignore')

        # Convertir columnas categóricas
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype('category')

        # One-hot encoding
        df = pd.get_dummies(df, columns=['type'], prefix='type', drop_first=False)

        # Ver estadísticas antes del escalado
        st.subheader("📊 Estadísticas ANTES del escalado:")
        st.write(df[['amount', 'newbalanceOrig']].describe())

        # Escalar columnas numéricas
        cols_to_scale = ['amount', 'newbalanceOrig']
        if all(col in df.columns for col in cols_to_scale):
            df[cols_to_scale] = scaler.transform(df[cols_to_scale])
        else:
            st.warning("⚠️ Faltan columnas necesarias para escalar.")

        # Ver estadísticas después del escalado
        st.subheader("📊 Estadísticas DESPUÉS del escalado:")
        st.write(df[['amount', 'newbalanceOrig']].describe())

        # Asegurar columnas esperadas por el modelo
        expected_columns = ['amount', 'newbalanceOrig', 'type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']
        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0  # agregar columnas faltantes con 0
        df = df[expected_columns]  # orden correcto

        # Predicción
        st.subheader("🤖 Realizando predicciones...")
        predictions = model.predict(df)
        df['Fraude_Predicho'] = predictions

        # Mostrar resultados
        st.subheader("✅ Resultados:")
        st.write(df.head())

        fraud_count = df['Fraude_Predicho'].sum()
        st.write(f"🔎 Número total de transacciones fraudulentas predichas: {fraud_count}")

    except Exception as e:
        st.error(f"❌ Error al procesar el archivo: {e}")
