# prompt: haz el despliegue anterior pero en streamlit

import streamlit as st
import pandas as pd
import os
import joblib

# Streamlit app title
st.title('Fraud Detection Prediction App')

# File upload (assuming the user will upload a file now)
uploaded_file = st.file_uploader("Upload your Excel file (datos_futuros.xlsx)", type=['xlsx'])

if uploaded_file is not None:
    try:
        # Read the uploaded file
        df = pd.read_excel(uploaded_file)

        st.subheader("Original Data:")
        st.write(df.head())

        # Data preprocessing steps (as in the original notebook)
        # Drop unnecessary columns
        df = df.drop(columns=['newbalanceDest', 'oldbalanceDest', 'oldbalanceOrg'], errors='ignore')
        df = df.drop(columns=['step', 'nameOrig', 'nameDest'], errors='ignore')
        df = df.drop(columns=['isFlaggedFraud'], errors='ignore')

        # Convert object columns to category (assuming 'type' is the only object column left)
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype('category')

        # One-hot encode 'type' column
        df = pd.get_dummies(df, columns=['type'], prefix='type', drop_first=False)

        # Assuming the scaler and model are in the same directory as the Streamlit app script
        # or in a predefined directory. For simplicity, let's assume they are in the same directory.
        # In a real application, you might need to handle paths more robustly.

        # Define the paths to the scaler and model files
        scaler_filename = 'standard_scaler.pkl'
        model_filename = 'best_fraud_detection_model_SVM.pkl'

        # Check if the scaler and model files exist
        if not os.path.exists(scaler_filename):
            st.error(f"Error: Scaler file '{scaler_filename}' not found. Please ensure it's in the correct directory.")
        elif not os.path.exists(model_filename):
            st.error(f"Error: Model file '{model_filename}' not found. Please ensure it's in the correct directory.")
        else:
            # Load the scaler and model
            loaded_scaler = joblib.load(scaler_filename)
            loaded_model = joblib.load(model_filename)

            # Apply scaling to relevant columns
            # Ensure columns exist before scaling
            cols_to_scale = ['amount', 'newbalanceOrig']
            present_cols_to_scale = [col for col in cols_to_scale if col in df.columns]
            if present_cols_to_scale:
                df[present_cols_to_scale] = loaded_scaler.transform(df[present_cols_to_scale])
            else:
                st.warning("Columns 'amount' or 'newbalanceOrig' not found for scaling")
            
            st.subheader("DataFrame preprocesado para predicción:")
            st.write(df.head())

            

            # Ensure the DataFrame has the same columns as the training data
            # This is a crucial step for consistent predictions.
            # A common approach is to save the list of columns from the training data.
            # For this example, let's assume the columns after dummy encoding are consistent.
            # In a real scenario, you might load a list of expected columns.

            # Make predictions
            try:
                # Ensure columns match the model's expected input
                # A robust way is to align columns based on the training data's columns.
                # Since we don't have the training data columns here, we proceed assuming consistency.
                # In a production app, you would load the list of training columns and reindex/align df.
                predictions = loaded_model.predict(df)

                # Add predictions to the DataFrame
                df['predictions'] = predictions

                st.subheader("Data with Predictions:")
                st.write(df.head())

                st.subheader("Prediction Summary:")
                fraud_count = df['predictions'].sum()
                total_count = len(df)
                st.write(f"Total records: {total_count}")
                st.write(f"Predicted Fraudulent Transactions: {fraud_count}")

            except Exception as e:
                st.error(f"Error during prediction: {e}")
                st.write("Please ensure the input data structure matches the expected format for the loaded model.")

    except Exception as e:
        st.error(f"Error processing the uploaded file: {e}")
        st.write("Please ensure the file is a valid Excel file.")
