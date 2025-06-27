import streamlit as st
import pandas as pd
import os
import joblib

# Streamlit app title
st.title('Fraud Detection Prediction App')

# File upload
uploaded_file = st.file_uploader("Upload your Excel file (e.g., datos_futuros.xlsx)", type=['xlsx'])

if uploaded_file is not None:
    try:
        # Read the uploaded file
        df = pd.read_excel(uploaded_file)

        st.subheader("Original Data:")
        st.write(df.head())

        # --- Data preprocessing steps ---
        # Drop unnecessary columns. Keep 'oldbalanceOrg' as it's a critical feature.
        # Ensure 'errors='ignore'' is used to prevent errors if a column is already missing.
        columns_to_drop = ['newbalanceDest', 'oldbalanceDest', 'step', 'nameOrig', 'nameDest', 'isFlaggedFraud']
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

        # Convert object columns to category (assuming 'type' is the primary object column)
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype('category')

        # One-hot encode 'type' column
        # Use drop_first=False to keep all dummy variables, matching typical model training
        df = pd.get_dummies(df, columns=['type'], prefix='type', drop_first=False)

        # Define the paths to the scaler and model files
        scaler_filename = 'standard_scaler.pkl'
        model_filename = 'best_fraud_detection_model_SVM.pkl' # Ensure this matches your model file name

        # Check if the scaler and model files exist
        # In a real deployment, these files would need to be accessible.
        # For local testing, ensure they are in the same directory as your Streamlit app.
        if not os.path.exists(scaler_filename):
            st.error(f"Error: Scaler file '{scaler_filename}' not found. Please ensure it's in the correct directory.")
            st.stop()
        if not os.path.exists(model_filename):
            st.error(f"Error: Model file '{model_filename}' not found. Please ensure it's in the correct directory.")
            st.stop()

        # Load the scaler and model
        try:
            loaded_scaler = joblib.load(scaler_filename)
            loaded_model = joblib.load(model_filename)
            st.success("Scaler and Model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading scaler or model: {e}")
            st.stop()

        # Define columns to be scaled. 'oldbalanceOrg' is now included.
        cols_to_scale = ['amount', 'oldbalanceOrg', 'newbalanceOrig']
        present_cols_to_scale = [col for col in cols_to_scale if col in df.columns]

        if present_cols_to_scale:
            df[present_cols_to_scale] = loaded_scaler.transform(df[present_cols_to_scale])
            st.info(f"Scaled columns: {present_cols_to_scale}")
        else:
            st.warning("No columns found to scale among 'amount', 'oldbalanceOrg', 'newbalanceOrig'.")


        # IMPORTANT: Ensure that the dataframe has all expected columns in the correct order
        # This list MUST exactly match the features and their order used during model training.
        # Based on the original data and common preprocessing, these are likely candidates.
        # Adjust this list if your model was trained with different features or order.
        expected_columns = [
            'amount',
            'oldbalanceOrg',
            'newbalanceOrig',
            'type_CASH_IN',
            'type_CASH_OUT',
            'type_DEBIT',
            'type_PAYMENT',
            'type_TRANSFER'
        ]

        # Add missing columns with default 0 to match the training data's feature set
        # This is crucial for consistency between training and inference.
        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0
                st.warning(f"Added missing column '{col}' with default value 0.")

        # Reorder columns to match the exact order of features used during model training
        df = df[expected_columns]

        # Show the preprocessed DataFrame before prediction
        st.subheader("Preprocessed DataFrame for Prediction:")
        st.write(df.head())
        st.write(f"Shape of DataFrame before prediction: {df.shape}")
        st.write("Columns and their order before prediction:")
        st.write(df.columns.tolist())


        # Make predictions
        try:
            predictions = loaded_model.predict(df)

            # Add predictions to the DataFrame
            df['isFraud_prediction'] = predictions # Renamed to be more explicit

            st.subheader("Data with Predictions:")
            st.write(df.head())

            st.subheader("Prediction Summary:")
            fraud_count = df['isFraud_prediction'].sum()
            total_count = len(df)
            st.write(f"Total records: {total_count}")
            st.write(f"Predicted Fraudulent Transactions: {fraud_count}")
            st.write(f"Predicted Legitimate Transactions: {total_count - fraud_count}")


        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.write("Please ensure the input data structure and feature order precisely match the model's training requirements.")
            st.write(f"Expected columns by model (from `expected_columns` list): {expected_columns}")
            st.write(f"Columns in preprocessed DataFrame: {df.columns.tolist()}")

    except Exception as e:
        st.error(f"Error processing the uploaded file: {e}")
        st.write("Please ensure the file is a valid Excel file and its content matches the expected format (e.g., column names).")

