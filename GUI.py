# prompt: haz el despliegue anterior pero en streamlit

import streamlit as st
import pandas as pd
import joblib
import os

# Mount Google Drive (This is not directly supported in Streamlit cloud)
# You would need to handle file loading differently for a public deployment.
# For local development, you can keep the original drive mount.
# from google.colab import drive
# drive.mount('/content/drive')

# --- Streamlit App ---
st.title("Fraud Detection Prediction")

# Instructions for the user to upload the data file
st.write("Please upload your 'datos_futuros.xlsx' file.")

# File uploader widget
uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

if uploaded_file is not None:
    try:
        # Read the uploaded Excel file into a DataFrame
        df = pd.read_excel(uploaded_file)
        st.write("Original DataFrame head:")
        st.write(df.head())

        # --- Data Preprocessing (from your original script) ---
        # Note: Adjust paths if you are not running this locally with Drive mounted
        # and have moved your scaler and model files.

        # Drop unnecessary columns
        df = df.drop(columns=['newbalanceDest', 'oldbalanceDest', 'oldbalanceOrg'])
        df = df.drop(columns=['step', 'nameOrig', 'nameDest'])
        df = df.drop(columns=['isFlaggedFraud'])

        # Convert 'type' column to category and then one-hot encode
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype('category')
        df = pd.get_dummies(df, columns=['type'], prefix='type', drop_first=False)

        # Handle missing 'type' columns if they weren't present in the uploaded file
        # This is important to ensure the dataframe has the same columns as the one
        # the scaler and model were trained on.
        expected_type_columns = ['type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']
        for col in expected_type_columns:
            if col not in df.columns:
                df[col] = 0 # Add the missing column with default value 0

        # Ensure column order matches the training data
        # This assumes your training data columns (after preprocessing) are known.
        # You might need to save the column order from your training phase.
        # For simplicity here, we'll make an assumption based on the dropped columns
        # and dummy variables.
        # You should replace this with the actual column order if possible.
        expected_columns = ['amount', 'newbalanceOrig'] + expected_type_columns
        # Filter and reorder columns to match the expected columns
        # This handles cases where the input data might have extra columns
        df = df[expected_columns]


        # Load the scaler (adjust the path if needed)
        # Assuming 'standard_scaler.pkl' is in the same directory as the Streamlit app
        # or a specified relative path.
        scaler_filename = 'standard_scaler.pkl' # Update this path as needed
        if os.path.exists(scaler_filename):
            loaded_scaler = joblib.load(scaler_filename)

            # Apply the scaler to the relevant columns
            # Make sure to only transform the columns the scaler was fitted on
            cols_to_scale = ['amount', 'newbalanceOrig']
            # Check if these columns exist in the dataframe before scaling
            if all(col in df.columns for col in cols_to_scale):
                 df[cols_to_scale] = loaded_scaler.transform(df[cols_to_scale]) # Use transform, not fit_transform

            else:
                 st.error("Error: Columns for scaling ('amount', 'newbalanceOrig') not found in the processed data.")
                 st.stop()
        else:
            st.error(f"Error: Scaler file not found at {scaler_filename}")
            st.stop()


        st.write("Processed DataFrame head after scaling and one-hot encoding:")
        st.write(df.head())

        # Load the model (adjust the path if needed)
        # Assuming 'best_fraud_detection_model_SVM.pkl' is in the same directory.
        model_filename = 'best_fraud_detection_model_SVM.pkl' # Update this path as needed
        if os.path.exists(model_filename):
            loaded_model = joblib.load(model_filename)

            # Make predictions
            # Ensure the dataframe used for prediction has the exact same columns
            # and in the same order as the training data.
            # It's crucial to handle this precisely. The previous steps try to
            # match columns, but explicit column ordering is safer.
            # Ensure df has the same columns and order as the data the model was trained on
            # A robust way is to save the training columns and use them here.
            # For this example, assuming the `expected_columns` list is correct.
            if list(df.columns) == expected_columns:
                 predictions = loaded_model.predict(df)

                 # Add predictions to the DataFrame
                 df['predictions'] = predictions

                 st.write("DataFrame with Predictions:")
                 st.write(df)

                 # Display fraud cases
                 fraud_cases = df[df['predictions'] == 1]
                 if not fraud_cases.empty:
                     st.warning("Potential Fraudulent Transactions Detected:")
                     st.write(fraud_cases)
                 else:
                     st.info("No potential fraudulent transactions detected.")

            else:
                 st.error("Error: Column mismatch between processed data and model training data.")
                 st.write("Processed Data Columns:", df.columns.tolist())
                 st.write("Expected Model Columns:", expected_columns)
                 st.stop()


        else:
            st.error(f"Error: Model file not found at {model_filename}")
            st.stop()

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.write("Please ensure the uploaded file is a valid Excel file ('datos_futuros.xlsx') and contains the expected columns.")

