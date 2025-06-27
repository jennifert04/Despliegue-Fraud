import streamlit as st
import pandas as pd
import joblib
import os

# ======== FUNCIÓN DE PREPROCESAMIENTO ===========
def preprocess_data(df, scaler):
    # Eliminar columnas innecesarias si existen
    df = df.drop(columns=[
        'newbalanceDest', 'oldbalanceDest', 'oldbalanceOrg',
        'step', 'nameOrig', 'nameDest', 'isFlaggedFraud'
    ], errors='ignore')

    # Convertir columnas tipo object a categoría
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category')

    # One-hot encoding de la columna 'type'
    df = pd.get_dummies(df, columns=['type'], prefix='type', drop_first=False)

    # Columnas esperadas por el modelo
    expected_columns = [
        'amount', 'newbalanceOrig', 'type_CASH_IN',
        'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER'
    ]

    # Agregar columnas faltantes
    for col in expected_columns:
        if col not in df.columns:
            df[col] = False if col.startswith('type_') else 0

    # Reordenar columnas
    df = df[expected_columns]

    # Aplicar escalado
    df[['amount', 'newbalanceOrig']] = scaler.transform(df[['amount', 'newbalanceOrig']])

    return df

# ======== CARGA DE MODELO Y SCALER ===========
try:
    scaler = joblib.load('standard_scaler.pkl')
    model = joblib.load('best_fraud_detection_model_SVM.pkl')
except FileNotFoundError:
    st.error("❌ Error: No se encontraron los archivos del modelo o del escalador.")
    st.stop()

# ======== TÍTULO Y DESCRIPCIÓN ===========
st.title("🔍 Predicción de Transacciones Fraudulentas")
st.write("""
Esta aplicación predice si una transacción financiera es fraudulenta, basada en los datos que subas en formato `.xlsx`.
""")

# ======== SUBIR ARCHIVO ===========
uploaded_file = st.file_uploader("📁 Sube tu archivo Excel con los datos (ej: datos_futuros.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    try:
        df_original = pd.read_excel(uploaded_file)

        st.subheader("📄 Datos cargados:")
        st.write(df_original.head())

        # ======== PREPROCESAMIENTO ===========
        st.subheader("⚙️ Datos preprocesados para el modelo:")
        df_processed = preprocess_data(df_original.copy(), scaler)
        st.write(df_processed.head())

        # ======== PREDICCIÓN ===========
        st.subheader("🤖 Predicciones del Modelo:")
        predictions = model.predict(df_processed)
        df_original['Predicción_Fraude'] = predictions

        st.write(df_original[['amount', 'type', 'Predicción_Fraude']].head())

        st.subheader("📊 Resumen de Resultados:")
        st.write(f"Total de transacciones: {len(df_original)}")
        st.write(f"Transacciones predichas como FRAUDULENTAS: {df_original['Predicción_Fraude'].sum()}")

    except Exception as e:
        st.error(f"❌ Ocurrió un error al procesar el archivo: {e}")
