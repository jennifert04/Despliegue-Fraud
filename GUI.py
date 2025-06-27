import streamlit as st
import pandas as pd
import os
import joblib
from sklearn.preprocessing import StandardScaler

st.title('Fraud Detection Prediction App con Debug de Escalado')

uploaded_file = st.file_uploader("Sube tu archivo Excel (solo .xlsx)", type=['xlsx'])

scaler_filename = 'standard_scaler.pkl'
model_filename = 'best_fraud_detection_model_SVM.pkl'

if uploaded_file is not None:
    try:
        # Leer archivo Excel
        df = pd.read_excel(uploaded_file)

        st.subheader("Datos originales:")
        st.write(df.head())

        # Convertir columnas a numérico float para evitar problemas en el escalado
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce').astype(float)
        df['newbalanceOrig'] = pd.to_numeric(df['newbalanceOrig'], errors='coerce').astype(float)

        # Mostrar stats antes de escalar
        st.subheader("Estadísticas antes de escalar:")
        st.write(df[['amount', 'newbalanceOrig']].describe())

        # Cargar scaler y modelo si existen
        if not os.path.exists(scaler_filename):
            st.error(f"Error: Scaler file '{scaler_filename}' no encontrado.")
            st.stop()
        if not os.path.exists(model_filename):
            st.error(f"Error: Model file '{model_filename}' no encontrado.")
            st.stop()

        loaded_scaler = joblib.load(scaler_filename)
        loaded_model = joblib.load(model_filename)

        # Mostrar medias y varianzas del scaler cargado
        st.write("Scaler means:", loaded_scaler.mean_)
        st.write("Scaler variances:", loaded_scaler.var_)

        # Mostrar datos antes de escalar (primeras filas)
        st.subheader("Valores 'amount' y 'newbalanceOrig' antes de escalar:")
        st.write(df[['amount', 'newbalanceOrig']].head())

        # Escalar datos
        scaled_values = loaded_scaler.transform(df[['amount', 'newbalanceOrig']])
        df[['amount', 'newbalanceOrig']] = scaled_values

        # Mostrar datos después de escalar (primeras filas)
        st.subheader("Valores 'amount' y 'newbalanceOrig' después de escalar:")
        st.write(df[['amount', 'newbalanceOrig']].head())

        # Escalado de prueba con scaler nuevo (fit_transform)
        st.subheader("Escalado de prueba con nuevo scaler (fit_transform):")
        scaler_test = StandardScaler()
        scaled_test = scaler_test.fit_transform(df[['amount', 'newbalanceOrig']])
        st.write(pd.DataFrame(scaled_test, columns=['amount', 'newbalanceOrig']).head())

        # Aquí continuarías con el resto de preprocesamiento (drop columnas, one-hot encode, etc.)
        df = df.drop(columns=['newbalanceDest', 'oldbalanceDest', 'oldbalanceOrg', 'step', 'nameOrig', 'nameDest', 'isFlaggedFraud'], errors='ignore')

        # Convertir columnas categóricas
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype('category')

        # One-hot encode 'type' columna
        df = pd.get_dummies(df, columns=['type'], prefix='type', drop_first=False)

        # Asegurar columnas que el modelo espera (lista ejemplo)
        expected_columns = ['amount', 'newbalanceOrig', 'type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']
        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[expected_columns]

        # Predicciones
        predictions = loaded_model.predict(df)
        df['prediction'] = predictions

        st.subheader("Predicciones:")
        st.write(df.head())

        st.write(f"Total registros: {len(df)}")
        st.write(f"Transacciones fraudulentas predichas: {df['prediction'].sum()}")

    except Exception as e:
        st.error(f"Error procesando el archivo: {e}")
