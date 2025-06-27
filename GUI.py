# prompt: haz el despliegue anterior pero en streamlit

import streamlit as st
import pandas as pd
import joblib
import os

# Load the pre-trained model and scaler
@st.cache_resource
def load_resources():
    # Adjust these paths if your files are not in the same directory as the script
    scaler_filename = 'scaler_fraud.pkl'
    model_filename = 'best_fraud_detection_model_SVM.pkl'

    if not os.path.exists(scaler_filename):
        st.error(f"Scaler file not found: {scaler_filename}")
        return None, None

    if not os.path.exists(model_filename):
        st.error(f"Model file not found: {model_filename}")
        return None, None

    try:
        loaded_scaler = joblib.load(scaler_filename)
        loaded_model = joblib.load(model_filename)
        return loaded_scaler, loaded_model
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
        return None, None

loaded_scaler, loaded_model = load_resources()

if loaded_scaler is None or loaded_model is None:
    st.stop()

st.title("Fraud Detection Application")

st.write("Upload your Excel file with transaction data for fraud prediction.")

uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
        st.write("Original Data:")
        st.dataframe(df.head())

        # --- Data Preprocessing ---
        # Based on your Colab notebook code
        if 'newbalanceDest' in df.columns and 'oldbalanceDest' in df.columns and 'oldbalanceOrg' in df.columns:
            df = df.drop(columns=['newbalanceDest', 'oldbalanceDest', 'oldbalanceOrg'], errors='ignore')
        if 'step' in df.columns and 'nameOrig' in df.columns and 'nameDest' in df.columns:
            df = df.drop(columns=['step', 'nameOrig', 'nameDest'], errors='ignore')
        if 'isFlaggedFraud' in df.columns:
            df = df.drop(columns=['isFlaggedFraud'], errors='ignore')

        # Convert object columns to category (specifically 'type')
        for col in df.columns:
            if df[col].dtype == 'object':
                 if col == 'type': # Ensure 'type' is converted
                    df[col] = df[col].astype('category')
                 else:
                    # Handle other potential object columns if necessary, or drop them
                    pass # Or df = df.drop(columns=[col], errors='ignore')

        # One-hot encode 'type'
        if 'type' in df.columns:
            df = pd.get_dummies(df, columns=['type'], prefix='type', drop_first=False)
        else:
             st.warning("'type' column not found. One-hot encoding skipped.")


        # Ensure required columns for scaling exist
        required_scaling_cols = ['amount', 'newbalanceOrig']
        missing_scaling_cols = [col for col in required_scaling_cols if col not in df.columns]

        if missing_scaling_cols:
            st.error(f"Missing columns required for scaling: {', '.join(missing_scaling_cols)}. Please check your data file.")
            st.stop()

        # Scale the relevant columns
        df[['amount', 'newbalanceOrig']] = loaded_scaler.transform(df[['amount', 'newbalanceOrig']]) # Use transform, not fit_transform on new data

        # --- Prediction ---
        # Ensure the DataFrame has the same columns as the training data used for the model
        # This is a crucial step for deployment. You need to know the exact columns and their order.
        # For simplicity here, we'll assume the processed 'df' has the right columns.
        # In a real application, you might need to reindex or align columns.

        # Make predictions
        predictions = loaded_model.predict(df)

        # Add predictions to the DataFrame
        df['predictions'] = predictions

        st.write("Data with Predictions (1 indicates potential fraud):")
        st.dataframe(df.head())

        # Optional: Display only potentially fraudulent transactions
        st.write("Potentially Fraudulent Transactions:")
        fraudulent_transactions = df[df['predictions'] == 1]
        st.dataframe(fraudulent_transactions)

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")

