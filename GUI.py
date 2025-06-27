# prompt: Haz el despliegue de un modelo de detección de fraude en Streamlit

import streamlit as st
import pandas as pd
import joblib
import os

# Título de la aplicación
st.title("🔍 Detección de Fraude en Transacciones")

st.write("""
Esta aplicación predice si una transacción es fraudulenta con base en un modelo previamente entrenado.
""")

# Carga del archivo
uploaded_file = st.file_uploader("📂 Sube tu archivo Excel (solo .xlsx)", type=["xlsx"])

if uploaded_file is not None:
    try:
        # Cargar archivo en un DataFrame
        df = pd.read_excel(uploaded_file)
        st.subheader("📄 Datos cargados:")
        st.write(df.head())

        # Preprocesamiento
        st.subheader("⚙️ Preprocesando datos...")

        # Eliminar columnas innecesarias si existen
        df = df.drop(columns=['newbalanceDest', 'oldbalanceDest', 'oldbalanceOrg', 'step', 'nameOrig', 'nameDest', 'isFlaggedFraud'], errors='ignore')

        # Convertir columnas categóricas a tipo 'category'
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype('category')

        # One-hot encoding de la columna 'type'
        df = pd.get_dummies(df, columns=['type'], prefix='type', drop_first=False)

        # Cargar scaler y modelo
        scaler_filename = 'standard_scaler_fraud.pkl'  # Asegúrate de usar el nombre correcto
        model_filename = 'best_fraud_detection_model_SVM.pkl'

        if not os.path.exists(scaler_filename):
            st.error(f"❌ Scaler no encontrado: {scaler_filename}")
            st.stop()
        if not os.path.exists(model_filename):
            st.error(f"❌ Modelo no encontrado: {model_filename}")
            st.stop()

        loaded_scaler = joblib.load(scaler_filename)
        loaded_model = joblib.load(model_filename)

        # Escalar las columnas numéricas
        st.write("🔎 Antes de escalar:")
        st.write(df[['amount', 'newbalanceOrig']].head())

        cols_to_scale = ['amount', 'newbalanceOrig']
        present_cols_to_scale = [col for col in cols_to_scale if col in df.columns]
        df[present_cols_to_scale] = loaded_scaler.transform(df[present_cols_to_scale])

        st.write("📉 Después de escalar:")
        st.write(df[['amount', 'newbalanceOrig']].head())

        # Asegurar que las columnas están completas
        expected_columns = ['amount', 'newbalanceOrig', 'type_CASH_IN', 'type_CASH_OUT',
                            'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']
        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0

        df = df[expected_columns]

        # Predicción
        predictions = loaded_model.predict(df)
        df_result = df.copy()
        df_result['Fraude_Predicho'] = predictions

        st.subheader("✅ Resultados de la Predicción:")
        st.write(df_result.head())

        st.subheader("📊 Resumen:")
        fraud_count = df_result['Fraude_Predicho'].sum()
        total_count = len(df_result)
        st.write(f"Total de transacciones: {total_count}")
        st.write(f"Transacciones fraudulentas predichas: {fraud_count}")

    except Exception as e:
        st.error(f"❌ Ocurrió un error al procesar el archivo: {e}")
