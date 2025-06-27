import streamlit as st
import pandas as pd
import joblib

# Rutas a archivos
MODEL_PATH = 'best_fraud_detection_model_SVM.pkl'
SCALER_PATH = 'standard_scaler.pkl'

# Función de preprocesamiento
def preprocess_data(df, scaler, expected_columns):
    # Borra columnas que no usas en el modelo
    cols_to_drop = ['newbalanceDest', 'oldbalanceDest', 'step', 'nameOrig', 'nameDest', 'isFlaggedFraud']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    # One-hot encoding
    df = pd.get_dummies(df, columns=['type'], prefix='type', drop_first=False)

    # Agrega las columnas faltantes para que coincida con expected_columns
    for col in expected_columns:
        if col not in df.columns:
            # Si es tipo dummy (bool), asigna False
            if col.startswith('type_'):
                df[col] = False
            else:
                df[col] = 0.0

    # Asegúrate que las columnas dummy sean booleanas
    dummy_cols = [c for c in expected_columns if c.startswith('type_')]
    df[dummy_cols] = df[dummy_cols].astype(bool)

    # Escala columnas numéricas (debe incluir todas las que usas)
    numeric_cols = ['amount', 'newbalanceOrig', 'oldbalanceOrg']
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    # Reordena exactamente las columnas esperadas
    df = df.loc[:, expected_columns]

    return df

# Carga modelo y scaler
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except FileNotFoundError:
    st.error("No se encontró el archivo del modelo o escalador.")
    st.stop()

# Obtén las columnas exactas que el modelo espera
expected_cols = list(model.feature_names_in_)

st.title("Detección de Fraude en Transacciones")

uploaded_file = st.file_uploader("Sube archivo Excel (.xlsx) con las transacciones", type="xlsx")

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    st.write("Datos originales:")
    st.write(df.head())

    try:
        df_proc = preprocess_data(df.copy(), scaler, expected_cols)
        st.write("Datos preprocesados:")
        st.write(df_proc.head())

        # Validación extra
        st.write("Columnas esperadas:", expected_cols)
        st.write("Columnas del DataFrame preprocesado:", list(df_proc.columns))
        st.write("Tipos de columnas:")
        st.write(df_proc.dtypes)

        # Predecir
        preds = model.predict(df_proc)
        df['Fraude_Predicho'] = preds

        st.write("Predicciones:")
        st.write(df[['amount', 'type', 'Fraude_Predicho']].head())
        st.success(f"Total predicciones de fraude: {sum(preds)}")

    except Exception as e:
        st.error(f"Error durante el procesamiento: {e}")
