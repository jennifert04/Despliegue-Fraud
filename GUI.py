# prompt: haz el despliegue anterior pero en streamlit

import streamlit as st
import pandas as pd
import joblib
import os

# Define the paths to the scaler and model files
scaler_filename = os.path.join('standard_scaler.pkl')
model_filename = os.path.join('best_fraud_detection_model_SVM.pkl')

# Load the scaler and model
try:
    loaded_scaler = joblib.load(scaler_filename)
    loaded_model = joblib.load(model_filename)
except FileNotFoundError:
    st.error("Scaler or model file not found. Make sure 'standard_scaler.pkl' and 'best_fraud_detection_model_SVM.pkl' are in the same directory as the Streamlit app.")
    st.stop() # Stop the app if files are not found

st.title("Fraud Detection Prediction")

st.write("Upload your Excel file with future transaction data for fraud prediction.")

uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)

        st.subheader("Original Data")
        st.write(df.head())

        # Data cleaning and preprocessing steps (as in the Colab notebook)
        df = df.drop(columns=['newbalanceDest', 'oldbalanceDest', 'oldbalanceOrg'], errors='ignore')
        df = df.drop(columns=['step', 'nameOrig', 'nameDest'], errors='ignore')
        df = df.drop(columns=['isFlaggedFraud'], errors='ignore')

        # Convert object columns to category
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype('category')

        # One-hot encode the 'type' column
        df = pd.get_dummies(df, columns=['type'], prefix='type', drop_first=False)

        # Ensure all required dummy columns are present, add if missing with 0
        expected_type_cols = ['type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']
        for col in expected_type_cols:
            if col not in df.columns:
                df[col] = 0

        # Align columns with the training data used for the model
        # (Assuming the training data had columns in a specific order)
        # This is a critical step to ensure the model gets features in the expected order.
        # You might need to load a sample of your training data to get the correct column order.
        # For now, we'll assume a possible column order based on the dummy encoding.
        # Replace this with the actual column order from your training data if necessary.
        # Example:
        # training_cols = ['amount', 'newbalanceOrig', 'type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']
        # df = df[training_cols]
        # A more robust way is to save the training columns during model training and load them here.

        # For demonstration, let's select the expected numerical and one-hot encoded columns
        # Make sure 'amount' and 'newbalanceOrig' are numerical before scaling
        numerical_cols = ['amount', 'newbalanceOrig']
        for col in numerical_cols:
            if col in df.columns:
                 df[col] = pd.to_numeric(df[col], errors='coerce')
                 df[col].fillna(df[col].mean(), inplace=True) # Handle potential NaNs from coercion

        # Apply the loaded scaler
        if all(col in df.columns for col in numerical_cols):
             df[numerical_cols] = loaded_scaler.transform(df[numerical_cols])
        else:
             st.error("Missing numerical columns required for scaling.")
             st.stop()

        # Select columns for prediction (ensure order matches training)
        # This is a placeholder. You MUST replace this with the exact list of columns
        # and their order that your trained model expects.
        # A common way is to save the feature names after preprocessing the training data.
        feature_cols = numerical_cols + expected_type_cols
        # Ensure all feature columns exist in the processed df
        if not all(col in df.columns for col in feature_cols):
            missing_cols = [col for col in feature_cols if col not in df.columns]
            st.error(f"Missing feature columns required for prediction: {missing_cols}")
            st.stop()

        df_processed = df[feature_cols]


        st.subheader("Processed Data (before prediction)")
        st.write(df_processed.head())

        # Make predictions
        predictions = loaded_model.predict(df_processed)

        # Add predictions to the original DataFrame (or a copy) for display
        # Ensure the index alignment is correct if any rows were dropped during processing
        # In this case, we haven't explicitly dropped rows based on data content,
        # but it's good practice to align by index if necessary.
        # Here, we assume the processed df still aligns with the original df's index.
        df_with_predictions = df.copy()
        df_with_predictions['predictions'] = predictions

        st.subheader("Predictions")
        st.write(df_with_predictions[['amount', 'type', 'predictions']].head()) # Display relevant columns

        # Optional: Display full results or filter by fraud
        st.subheader("Full Data with Predictions")
        st.write(df_with_predictions)

        st.subheader("Fraudulent Transactions Predicted (predictions = 1)")
        fraudulent_transactions = df_with_predictions[df_with_predictions['predictions'] == 1]
        if not fraudulent_transactions.empty:
            st.write(fraudulent_transactions)
        else:
            st.write("No fraudulent transactions predicted.")

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")

