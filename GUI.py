import streamlit as st
import pandas as pd
import joblib
import os

# Streamlit app title
st.title('Fraud Detection Prediction App')

uploaded_file = st.file_uploader("Upload your Excel file (datos_futuros.xlsx)", type=['xlsx'])

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
        st.subheader("Original Data:")
        st.write(df.head())

        # Drop unnecessary columns safely
        drop_cols = ['newbalanceDest', 'oldbalanceDest', 'oldbalanceOrg', 'step', 'nameOrig', 'nameDest', 'isFlaggedFraud']
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

        # Cast to float to avoid scaling issues
        df['amount'] = df['amount'].astype(float)
        df['newbalanceOrig'] = df['newbalanceOrig'].astype(float)

        # Load scaler and model
        scaler_filename = 'fraud_standard_scaler.pkl'
        model_filename = 'best_fraud_detection_model_SVM.pkl'

        if not os.path.exists(scaler_filename):
            st.error(f"Scaler file '{scaler_filename}' not found.")
        elif not os.path.exists(model_filename):
            st.error(f"Model file '{model_filename}' not found.")
        else:
            loaded_scaler = joblib.load(scaler_filename)
            loaded_model = joblib.load(model_filename)

            st.write("Antes de escalar:")
            st.write(df[['amount', 'newbalanceOrig']].dtypes)
            st.write(df[['amount', 'newbalanceOrig']].head())

            # Escalar
            df[['amount', 'newbalanceOrig']] = loaded_scaler.transform(df[['amount', 'newbalanceOrig']])

            st.write("Después de escalar:")
            st.write(df[['amount', 'newbalanceOrig']].dtypes)
            st.write(df[['amount', 'newbalanceOrig']].head())

            # One-hot encode 'type'
            df = pd.get_dummies(df, columns=['type'], prefix='type', drop_first=False)

            # Añadir columnas faltantes para consistencia
            expected_cols = ['amount', 'newbalanceOrig', 'type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']
            for c in expected_cols:
                if c not in df.columns:
                    df[c] = 0

            df = df[expected_cols]

            # Predecir
            predictions = loaded_model.predict(df)
            df['predictions'] = predictions

            st.subheader("Predictions:")
            st.write(df.head())

            fraud_count = df['predictions'].sum()
            st.write(f"Predicted fraud transactions: {fraud_count}")

    except Exception as e:
        st.error(f"Error processing file: {e}")
