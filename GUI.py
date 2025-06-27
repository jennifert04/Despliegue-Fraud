import streamlit as st
import pandas as pd
import os
import joblib
import numpy as np # Import numpy for NaN and inf checks
import sklearn # Import sklearn to check version

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
        # Define columns to drop.
        # Based on your model's reported training features (7 columns, no oldbalanceOrg),
        # we are explicitly dropping 'oldbalanceOrg' from the input data.
        columns_to_drop = [
            'newbalanceDest', 'oldbalanceDest', 'step',
            'nameOrig', 'nameDest', 'isFlaggedFraud',
            'oldbalanceOrg' # Explicitly dropping oldbalanceOrg to match reported training data
        ]
        # Drop only the columns that are present in the current DataFrame
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

        # Convert object columns to category (assuming 'type' is the primary object column)
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype('category')

        # One-hot encode 'type' column
        df = pd.get_dummies(df, columns=['type'], prefix='type', drop_first=False)

        # Define the paths to the scaler and model files
        scaler_filename = 'standard_scaler.pkl'
        model_filename = 'best_fraud_detection_model_SVM.pkl' # Ensure this matches your model file name

        # Check if the scaler and model files exist
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

        # Define columns to be scaled. 'oldbalanceOrg' is *NOT* in this list,
        # as it's being dropped and wasn't part of the reported training features.
        cols_to_scale = ['amount', 'newbalanceOrig']
        present_cols_to_scale = [col for col in cols_to_scale if col in df.columns]

        if present_cols_to_scale:
            df[present_cols_to_scale] = loaded_scaler.transform(df[present_cols_to_scale])
            st.info(f"Scaled columns: {present_cols_to_scale}")
        else:
            st.warning("No columns found to scale among 'amount', 'newbalanceOrig'.")
            st.warning("This could lead to issues if your model expects these features to be scaled.")

        # Display DataFrame head *immediately after scaling* to verify
        st.subheader("DataFrame head after scaling:")
        st.write(df.head())


        # IMPORTANT: Ensure that the dataframe has all expected columns in the correct order.
        # This list NOW EXACTLY MATCHES your provided training DataFrame structure (7 columns).
        expected_columns = [
            'amount',
            'newbalanceOrig',
            'type_CASH_IN',
            'type_CASH_OUT',
            'type_DEBIT',
            'type_PAYMENT',
            'type_TRANSFER'
        ]

        # Add missing columns with default 0 to match the training data's feature set
        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0
                st.warning(f"Added missing column '{col}' with default value 0.")

        # Reorder columns to match the exact order of features used during model training
        df = df[expected_columns]

        # Show the preprocessed DataFrame before prediction
        st.subheader("Final Preprocessed DataFrame for Prediction:")
        st.write(df.head())
        st.write(f"Shape of DataFrame before prediction: {df.shape}")
        st.write("Columns and their order before prediction:")
        st.write(df.columns.tolist())

        # --- Debugging checks before prediction ---
        st.subheader("Pre-prediction Data Checks:")
        # Check for NaN values
        if df.isnull().sum().sum() > 0:
            st.error("Warning: DataFrame contains NaN values after preprocessing!")
            st.write(df.isnull().sum()) # Show count of NaNs per column

        # Check for infinite values
        if np.isinf(df).sum().sum() > 0:
            st.error("Warning: DataFrame contains Infinite values after preprocessing!")
            st.write(np.isinf(df).sum()) # Show count of Infs per column

        # Display data types
        st.write("Data types of preprocessed DataFrame columns:")
        st.write(df.dtypes)

        # Display scikit-learn version
        st.info(f"Scikit-learn version: {sklearn.__version__}")


        # Make predictions
        try:
            predictions = loaded_model.predict(df)

            # Add predictions to the DataFrame
            df['isFraud_prediction'] = predictions

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
            st.write("A prediction error occurred. This is likely due to a scikit-learn version mismatch between when the model was saved and when it's being loaded/used, or an issue with the data itself (e.g., NaNs, Infs).")
            st.write(f"Current scikit-learn version: {sklearn.__version__}")
            st.write(f"Expected columns by model (from `expected_columns` list): {expected_columns}")
            st.write(f"Columns in preprocessed DataFrame: {df.columns.tolist()}")
            st.write("Please double-check the scikit-learn version used during model training and ensure it exactly matches your current environment. If the version matches and the error persists, consider re-saving your model with the current scikit-learn version.")

    except Exception as e:
        st.error(f"Error processing the uploaded file: {e}")
        st.write("Please ensure the file is a valid Excel file and its content matches the expected format (e.g., column names).")

