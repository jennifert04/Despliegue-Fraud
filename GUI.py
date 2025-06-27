import streamlit as st
import pandas as pd
import os
import joblib

# Título de la app
st.title('Detección de Fraude en Transacciones')

# Carga archivo Excel
uploaded_file = st.file_uploader("Sube tu archivo Excel (.xlsx) con datos de transacciones", type=['xlsx'])

if uploaded_file is not None:
    try:
        # Leer archivo
        df_original = pd.read_excel(uploaded_file)
        st.subheader("Datos originales cargados:")
        st.write(df_original.head())

        # Crear copia para procesar y predecir
        df = df_original.copy()

        # Eliminar columnas no usadas para la predicción
        cols_to_drop = ['newbalanceDest', 'oldbalanceDest', 'oldbalanceOrg', 'step', 'nameOrig', 'nameDest', 'isFlaggedFraud']
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

        # Convertir columnas tipo object a categoría
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].astype('category')

        # One-hot encoding para la columna 'type'
        df = pd.get_dummies(df, columns=['type'], prefix='type', drop_first=False)

        # Asegurar columnas esperadas para el modelo
        expected_columns = ['amount', 'newbalanceOrig', 'type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']
        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0

        df = df[expected_columns]

        # Cargar scaler y modelo
        scaler_path = 'fraud_standard_scaler.pkl'
        model_path = 'best_fraud_detection_model_SVM.pkl'

        if not os.path.exists(scaler_path) or not os.path.exists(model_path):
            st.error("Error: Archivos de scaler o modelo no encontrados.")
            st.stop()

        scaler = joblib.load(scaler_path)
        model = joblib.load(model_path)

        # Escalar columnas para predicción
        df[['amount', 'newbalanceOrig']] = scaler.transform(df[['amount', 'newbalanceOrig']])

        # Realizar predicciones
        preds = model.predict(df)

        # Añadir columna de predicciones al dataframe original sin modificar
        df_original['Fraude_Predicho'] = preds
        df_original['Etiqueta_Fraude'] = df_original['Fraude_Predicho'].map({1: 'Fraude', 0: 'No Fraude'})

        st.subheader("Datos originales con columna de predicción:")
        st.write(df_original[['amount', 'newbalanceOrig', 'Etiqueta_Fraude']].head())

        #st.subheader("Datos originales con columna de predicción:")
        #st.write(df_original.head())

        fraud_count = df_original['Fraude_Predicho'].sum()
        st.write(f"Número total de transacciones predichas como fraudulentas: {fraud_count}")

    except Exception as e:
        st.error(f"Ocurrió un error durante el procesamiento: {e}")
